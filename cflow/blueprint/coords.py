#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Blueprint-AD: Coordinate Field Rendering and GPU Prefetching
Utilities for pose caching, pose fusion, coordinate-field rendering, and asynchronous GPU prefetching.
"""

import os
import math
import random
from collections import deque
import numpy as np
from tqdm import tqdm

import torch
from ultralytics import YOLO


# DataLoader helpers

def coordinate_collate(batch):
    """
    Keep each sample intact so the coordinate-field prefetcher can assemble the batch on GPU.
    """
    return batch


def get_truncated_normal_noise(sigma, limit=3.0):
    """
    Sample zero-mean Gaussian noise truncated to +/- limit * sigma.
    """
    if sigma <= 1e-6:
        return 0.0
    noise = random.gauss(0.0, sigma)
    return max(min(noise, limit * sigma), -limit * sigma)


# Pose cache and pose fusion

def build_yolo_cache(yolo_model_path, dataset_path_list, mapping_data):
    """
    Run pose inference once and cache detections for the requested image folders.
    """
    print(f"\n[INFO] Starting YOLO pre-inference and cache building...")
    model = YOLO(yolo_model_path)
    cache = {}

    id_to_name = {}
    for i, (cls_name, meta) in enumerate(mapping_data.items()):
        if isinstance(meta, dict) and 'class_id' in meta:
            id_to_name[meta['class_id']] = cls_name
        else:
            id_to_name[i] = cls_name

    valid_exts = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif')
    all_img_paths = []
    for d in dataset_path_list:
        for root, dirs, files in os.walk(d):
            all_img_paths.extend([os.path.join(root, f) for f in files if f.lower().endswith(valid_exts)])

    for path in tqdm(all_img_paths, desc="YOLO Inference Cache"):
        norm_path = os.path.abspath(path).lower()
        results = model(path, verbose=False, conf=0.3, iou=0.45)
        cache[norm_path] = []

        if not results or len(results[0].boxes) == 0:
            continue

        for det in results[0]:
            box = det.boxes.xyxy[0].cpu().numpy()
            cls_id = int(det.boxes.cls[0].item())
            conf = float(det.boxes.conf[0].item())
            kpts = det.keypoints.data[0].cpu().numpy() if hasattr(det,
                                                                  'keypoints') and det.keypoints is not None else []

            cache[norm_path].append({
                'box': box,
                'cls_id': cls_id,
                'kpts': kpts,
                'conf': conf
            })

    print("[SUCCESS] YOLO Cache built successfully.\n")
    return cache


def compute_fused_pose(box, kpts, calib_params, expected_kpts):
    """
    Fuse the bounding box and keypoints into an instance center and orientation.
    """
    w_c = calib_params.get('weight_center(HBB)', 1.0)
    dx = calib_params.get('offset_x', 0.0)
    dy = calib_params.get('offset_y', 0.0)
    dtheta = calib_params.get('offset_angle', 0.0)

    cx_hbb, cy_hbb = (box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0

    if expected_kpts == 3 and len(kpts) >= 3:
        cx_kpt, cy_kpt = kpts[2][0], kpts[2][1]
        final_cx = w_c * cx_hbb + (1 - w_c) * cx_kpt + dx
        final_cy = w_c * cy_hbb + (1 - w_c) * cy_kpt + dy

        vec_01 = kpts[0][:2] - kpts[1][:2]
        vec_02 = kpts[0][:2] - kpts[2][:2]
        vec_21 = kpts[2][:2] - kpts[1][:2]

        norm_01 = vec_01 / (np.linalg.norm(vec_01) + 1e-6)
        norm_02 = vec_02 / (np.linalg.norm(vec_02) + 1e-6)
        norm_21 = vec_21 / (np.linalg.norm(vec_21) + 1e-6)

        angle_weights = calib_params.get('weight_angle_vectors', {})
        w01 = angle_weights.get('w_Full(KP0-KP1)', 1.0)
        w02 = angle_weights.get('w_Front(KP0-KP2)', 0.0)
        w21 = angle_weights.get('w_Back(KP2-KP1)', 0.0)

        v_fused = w01 * norm_01 + w02 * norm_02 + w21 * norm_21
        final_theta_rad = np.arctan2(v_fused[1], v_fused[0])
        final_theta_deg = np.degrees(final_theta_rad) + dtheta

    elif expected_kpts == 2 and len(kpts) >= 2:
        cx_kpt = (kpts[0][0] + kpts[1][0]) / 2.0
        cy_kpt = (kpts[0][1] + kpts[1][1]) / 2.0
        final_cx = w_c * cx_hbb + (1 - w_c) * cx_kpt + dx
        final_cy = w_c * cy_hbb + (1 - w_c) * cy_kpt + dy

        vec = kpts[0][:2] - kpts[1][:2]
        final_theta_rad = np.arctan2(vec[1], vec[0])
        final_theta_deg = np.degrees(final_theta_rad) + dtheta

    else:
        final_cx = cx_hbb + dx
        final_cy = cy_hbb + dy
        final_theta_deg = dtheta

    return final_cx, final_cy, final_theta_deg


# Coordinate-field renderer

def render_coordinate_fields(pre_x, pre_y, cx, cy, angle_deg, global_extents, expected_kpts=3):
    """
    Render pose-aligned object-centric coordinate fields for a batch of instances.
    """
    cx = cx.view(-1, 1, 1)
    cy = cy.view(-1, 1, 1)
    theta = -angle_deg.view(-1, 1, 1) * np.pi / 180.0

    dx = pre_x.unsqueeze(0) - cx
    dy = pre_y.unsqueeze(0) - cy

    cos_t, sin_t = torch.cos(theta), torch.sin(theta)
    local_x = dx * cos_t - dy * sin_t
    local_y = dx * sin_t + dy * cos_t

    p = global_extents
    p_X_pos = p['beam_X_pos']
    p_X_neg = p['beam_X_neg']
    p_Y_pos = p['beam_Y_pos']
    p_Y_neg = p['beam_Y_neg']

    def create_beam(fwd_val, lat_val, extents_dict):
        c_ratio = extents_dict['c_ratio']
        E_x = extents_dict['fwd_max']
        lat_pos = extents_dict['lat_pos_max']
        lat_neg = extents_dict['lat_neg_max']

        E_y = torch.where(lat_val > 0, lat_pos, lat_neg)
        E_y_mean = (lat_pos + lat_neg) / 2.0
        eps = 1e-3

        u_norm = fwd_val / (E_x + eps)
        v_norm = lat_val / (E_y + eps)

        sigma_fwd, sigma_lat = (0.85, 0.75) if expected_kpts == 2 else (0.8, 0.8)

        internal_decay = torch.exp(-(u_norm ** 2) / (2 * sigma_fwd ** 2)) * \
                         torch.exp(-(v_norm ** 2) / (2 * sigma_lat ** 2))

        R_c = c_ratio * torch.minimum(E_x, E_y_mean)
        S_x = torch.clamp(E_x - R_c, min=0.0)
        S_y = torch.clamp(E_y - R_c, min=0.0)

        dist_x = torch.clamp(fwd_val - S_x, min=0.0)
        dist_y = torch.clamp(torch.abs(lat_val) - S_y, min=0.0)

        dist_outside = torch.sqrt(dist_x ** 2 + dist_y ** 2 + 1e-6)
        d_boundary = torch.clamp(dist_outside - R_c, min=0.0)

        D_excess = d_boundary / (torch.maximum(E_x, E_y_mean) + eps)
        boundary_mask = torch.exp(-(D_excess ** 2) / (2 * 0.12 ** 2))

        u_back = torch.clamp(-fwd_val / (E_x + eps), min=0.0)
        back_decay = torch.exp(-(u_back ** 2) / (2 * 0.05 ** 2))

        raw_beam = internal_decay * boundary_mask
        return torch.where(fwd_val > 0, raw_beam, back_decay * raw_beam)

    beam_x_pos = create_beam(local_x, local_y, p_X_pos)
    beam_x_neg = create_beam(-local_x, local_y, p_X_neg)
    beam_y_pos = create_beam(local_y, local_x, p_Y_pos)
    beam_y_neg = create_beam(-local_y, local_x, p_Y_neg)

    if expected_kpts == 0:
        channels = torch.max(torch.max(beam_x_pos, beam_x_neg), torch.max(beam_y_pos, beam_y_neg)).unsqueeze(1)
    elif expected_kpts == 2:
        channels = torch.stack([torch.max(beam_x_pos, beam_x_neg), torch.max(beam_y_pos, beam_y_neg)], dim=1)
    else:
        channels = torch.stack([beam_x_pos, beam_x_neg, beam_y_pos, beam_y_neg], dim=1)

    return channels


# Asynchronous GPU prefetching and rendering

class CoordinateFieldPrefetcher:
    """
    Asynchronously assemble CFLOW-AD batches and render coordinate fields on a CUDA stream.
    """

    def __init__(self, loader, num_coord_channels, img_size=256, prefetch_depth=3):
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()
        self.num_coord_channels = num_coord_channels
        self.img_size = img_size
        self.prefetch_depth = prefetch_depth
        self.queue = deque()

        for _ in range(self.prefetch_depth):
            self._preload_one()

    def _preload_one(self):
        try:
            batch = next(self.loader)  # coordinate_collate returns a list of samples
        except StopIteration:
            return

        with torch.cuda.stream(self.stream):
            # (image, label, mask, render_params, original_shape, path, logic_violation)
            images = torch.stack([b[0] for b in batch]).cuda(non_blocking=True)  # [B, 3, H, W]
            labels = torch.tensor([b[1] for b in batch]).cuda(non_blocking=True)  # [B]
            masks = torch.stack([b[2] for b in batch]).cuda(non_blocking=True)  # [B, 1, H, W]

            render_params_list = [b[3] for b in batch]
            orig_shapes = [b[4] for b in batch]
            paths = [b[5] for b in batch]
            logic_violations = torch.tensor([b[6] for b in batch]).cuda(non_blocking=True)  # [B]

            B = len(batch)
            coordinate_fields = torch.zeros((B, self.num_coord_channels, self.img_size, self.img_size), device='cuda',
                                          dtype=torch.float32)

            for b_idx in range(B):
                orig_h, orig_w = orig_shapes[b_idx]
                render_params = render_params_list[b_idx]

                if not render_params:
                    continue

                # Pixel grid in image coordinates.
                pre_y, pre_x = torch.meshgrid(
                    torch.linspace(0, orig_h, self.img_size, device='cuda'),
                    torch.linspace(0, orig_w, self.img_size, device='cuda'),
                )

                kpt_groups = {}
                for rp in render_params:
                    k = rp['expected_kpts']
                    if k not in kpt_groups:
                        kpt_groups[k] = {'cx': [], 'cy': [], 'angle': [], 'extents': [], 'sc': [], 'ec': []}
                    kpt_groups[k]['cx'].append(rp['cx'])
                    kpt_groups[k]['cy'].append(rp['cy'])
                    kpt_groups[k]['angle'].append(rp['angle'])
                    kpt_groups[k]['extents'].append(rp['extents'])
                    kpt_groups[k]['sc'].append(rp['start_c'])
                    kpt_groups[k]['ec'].append(rp['end_c'])

                for expected_kpts, data in kpt_groups.items():
                    N = len(data['cx'])
                    cx_t = torch.tensor(data['cx'], device='cuda', dtype=torch.float32)
                    cy_t = torch.tensor(data['cy'], device='cuda', dtype=torch.float32)
                    angle_t = torch.tensor(data['angle'], device='cuda', dtype=torch.float32)

                    batched_extents = {}
                    for beam in ['beam_X_pos', 'beam_X_neg', 'beam_Y_pos', 'beam_Y_neg']:
                        batched_extents[beam] = {}
                        for param in ['fwd_max', 'lat_pos_max', 'lat_neg_max', 'c_ratio']:
                            vals = [ex[beam].get(param, 1.0) if param == 'c_ratio' else ex[beam][param] for ex in
                                    data['extents']]
                            batched_extents[beam][param] = torch.tensor(vals, device='cuda', dtype=torch.float32).view(
                                -1, 1, 1)

                    rendered_fields = render_coordinate_fields(
                        pre_x, pre_y, cx_t, cy_t, angle_t,
                        global_extents=batched_extents, expected_kpts=expected_kpts
                    )

                    for i in range(N):
                        sc, ec = data['sc'][i], data['ec'][i]
                        coordinate_fields[b_idx, sc:ec] = torch.max(coordinate_fields[b_idx, sc:ec], rendered_fields[i])

            done_event = torch.cuda.Event()
            done_event.record(self.stream)

            # Queue the assembled batch.
            self.queue.append((images, labels, masks, coordinate_fields, paths, logic_violations, done_event))

    def next(self):
        if not self.queue:
            return None

        item = self.queue.popleft()
        done_event = item[-1]
        done_event.wait()

        current_stream = torch.cuda.current_stream()
        for t in item[:-1]:
            if isinstance(t, torch.Tensor):
                t.record_stream(current_stream)

        self._preload_one()
        # Return images, labels, masks, coordinate fields, paths, and logic flags.
        return item[:-1]