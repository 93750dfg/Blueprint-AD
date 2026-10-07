#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Coordinate-field construction utilities for Blueprint-AD + EfficientAD."""

import json
import os
import random
from collections import deque

import numpy as np
import torch
from PIL import Image
from torchvision.datasets import ImageFolder
from tqdm import tqdm
from ultralytics import YOLO

def get_truncated_normal_noise(sigma, limit=3.0):
    """Sample zero-mean Gaussian noise truncated at +/- limit * sigma."""
    if sigma <= 1e-6:
        return 0.0
    noise = random.gauss(0.0, sigma)
    return max(min(noise, limit * sigma), -limit * sigma)


class CoordinateFieldPrefetcher:
    """Asynchronously move images to CUDA and render coordinate fields."""

    def __init__(
        self,
        loader,
        num_coord_channels,
        image_size=256,
        prefetch_depth=10,
    ):
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()
        self.num_coord_channels = num_coord_channels
        self.image_size = image_size
        self.prefetch_depth = prefetch_depth
        self.queue = deque()

        for _ in range(self.prefetch_depth):
            self._preload_one()

    def _preload_one(self):
        try:
            batch = next(self.loader)[0]
        except StopIteration:
            return

        with torch.cuda.stream(self.stream):
            is_train = len(batch) == 3

            if is_train:
                (image_st, image_ae), render_params, (orig_h, orig_w) = batch
                image_st = image_st.unsqueeze(0).cuda(non_blocking=True)
                image_ae = image_ae.unsqueeze(0).cuda(non_blocking=True)
            else:
                image, render_params, (orig_h, orig_w), target, path = batch
                image = image.unsqueeze(0).cuda(non_blocking=True)

            coord_fields = torch.zeros(
                (
                    self.num_coord_channels,
                    self.image_size,
                    self.image_size,
                ),
                device='cuda',
                dtype=torch.float32,
            )

            grid_y, grid_x = torch.meshgrid(
                torch.linspace(
                    0, orig_h, self.image_size, device='cuda'
                ),
                torch.linspace(
                    0, orig_w, self.image_size, device='cuda'
                ),
            )

            topology_groups = {}
            for params in render_params:
                expected_kpts = params['expected_kpts']
                if expected_kpts not in topology_groups:
                    topology_groups[expected_kpts] = {
                        'cx': [],
                        'cy': [],
                        'angle': [],
                        'extents': [],
                        'start': [],
                        'end': [],
                    }

                group = topology_groups[expected_kpts]
                group['cx'].append(params['cx'])
                group['cy'].append(params['cy'])
                group['angle'].append(params['angle'])
                group['extents'].append(params['extents'])
                group['start'].append(params['start_c'])
                group['end'].append(params['end_c'])

            for expected_kpts, group in topology_groups.items():
                num_instances = len(group['cx'])

                cx_tensor = torch.tensor(
                    group['cx'], device='cuda', dtype=torch.float32
                )
                cy_tensor = torch.tensor(
                    group['cy'], device='cuda', dtype=torch.float32
                )
                angle_tensor = torch.tensor(
                    group['angle'], device='cuda', dtype=torch.float32
                )

                batched_extents = {}
                for beam in [
                    'beam_X_pos',
                    'beam_X_neg',
                    'beam_Y_pos',
                    'beam_Y_neg',
                ]:
                    batched_extents[beam] = {}
                    for param_name in [
                        'fwd_max',
                        'lat_pos_max',
                        'lat_neg_max',
                        'c_ratio',
                    ]:
                        values = [
                            (
                                extents[beam].get(param_name, 1.0)
                                if param_name == 'c_ratio'
                                else extents[beam][param_name]
                            )
                            for extents in group['extents']
                        ]
                        batched_extents[beam][param_name] = torch.tensor(
                            values,
                            device='cuda',
                            dtype=torch.float32,
                        ).view(-1, 1, 1)

                instance_fields = render_coordinate_fields(
                    grid_x,
                    grid_y,
                    cx_tensor,
                    cy_tensor,
                    angle_tensor,
                    template_extents=batched_extents,
                    expected_kpts=expected_kpts,
                )

                for index in range(num_instances):
                    start_channel = group['start'][index]
                    end_channel = group['end'][index]
                    coord_fields[start_channel:end_channel] = torch.max(
                        coord_fields[start_channel:end_channel],
                        instance_fields[index],
                    )

            coords = coord_fields.unsqueeze(0)

            done_event = torch.cuda.Event()
            done_event.record(self.stream)

            if is_train:
                self.queue.append(
                    (
                        image_st,
                        image_ae,
                        coords,
                        orig_h,
                        orig_w,
                        done_event,
                    )
                )
            else:
                self.queue.append(
                    (
                        image,
                        coords,
                        orig_h,
                        orig_w,
                        target,
                        path,
                        done_event,
                    )
                )

    def next(self):
        if not self.queue:
            return None

        item = self.queue.popleft()
        done_event = item[-1]
        done_event.wait()

        current_stream = torch.cuda.current_stream()
        for value in item[:-1]:
            if isinstance(value, torch.Tensor):
                value.record_stream(current_stream)

        self._preload_one()
        return item[:-1]


def compute_fused_pose(box, keypoints, calib_params, expected_kpts):
    """Compute calibrated instance center and orientation from a box and keypoints."""
    center_weight = calib_params.get('weight_center(HBB)', 1.0)
    offset_x = calib_params.get('offset_x', 0.0)
    offset_y = calib_params.get('offset_y', 0.0)
    offset_angle = calib_params.get('offset_angle', 0.0)

    cx_box = (box[0] + box[2]) / 2.0
    cy_box = (box[1] + box[3]) / 2.0

    if expected_kpts == 3 and len(keypoints) >= 3:
        cx_kpt, cy_kpt = keypoints[2][0], keypoints[2][1]
        final_cx = center_weight * cx_box + (1.0 - center_weight) * cx_kpt + offset_x
        final_cy = center_weight * cy_box + (1.0 - center_weight) * cy_kpt + offset_y

        vec_01 = keypoints[0][:2] - keypoints[1][:2]
        vec_02 = keypoints[0][:2] - keypoints[2][:2]
        vec_21 = keypoints[2][:2] - keypoints[1][:2]

        norm_01 = vec_01 / (np.linalg.norm(vec_01) + 1e-6)
        norm_02 = vec_02 / (np.linalg.norm(vec_02) + 1e-6)
        norm_21 = vec_21 / (np.linalg.norm(vec_21) + 1e-6)

        angle_weights = calib_params.get('weight_angle_vectors', {})
        w01 = angle_weights.get('w_Full(KP0-KP1)', 1.0)
        w02 = angle_weights.get('w_Front(KP0-KP2)', 0.0)
        w21 = angle_weights.get('w_Back(KP2-KP1)', 0.0)

        fused_vector = w01 * norm_01 + w02 * norm_02 + w21 * norm_21
        final_theta = np.degrees(np.arctan2(fused_vector[1], fused_vector[0])) + offset_angle

    elif expected_kpts == 2 and len(keypoints) >= 2:
        cx_kpt = (keypoints[0][0] + keypoints[1][0]) / 2.0
        cy_kpt = (keypoints[0][1] + keypoints[1][1]) / 2.0
        final_cx = center_weight * cx_box + (1.0 - center_weight) * cx_kpt + offset_x
        final_cy = center_weight * cy_box + (1.0 - center_weight) * cy_kpt + offset_y

        axis_vector = keypoints[0][:2] - keypoints[1][:2]
        final_theta = np.degrees(np.arctan2(axis_vector[1], axis_vector[0])) + offset_angle

    else:
        final_cx = cx_box + offset_x
        final_cy = cy_box + offset_y
        final_theta = offset_angle

    return final_cx, final_cy, final_theta


def render_coordinate_fields(
    grid_x,
    grid_y,
    cx,
    cy,
    angle_deg,
    template_extents,
    expected_kpts=3,
):
    """
    Render pose-aligned local coordinate fields for a batch of instances.

    The returned channel count is determined by the pose type:
      - 3 keypoints: 4 directional channels (+x, -x, +y, -y)
      - 2 keypoints: 2 axial channels
      - 0 keypoints: 1 orientation-agnostic channel
    """
    cx = cx.view(-1, 1, 1)
    cy = cy.view(-1, 1, 1)
    theta = -angle_deg.view(-1, 1, 1) * np.pi / 180.0

    dx = grid_x.unsqueeze(0) - cx
    dy = grid_y.unsqueeze(0) - cy

    cos_t = torch.cos(theta)
    sin_t = torch.sin(theta)
    local_x = dx * cos_t - dy * sin_t
    local_y = dx * sin_t + dy * cos_t

    params_x_pos = template_extents['beam_X_pos']
    params_x_neg = template_extents['beam_X_neg']
    params_y_pos = template_extents['beam_Y_pos']
    params_y_neg = template_extents['beam_Y_neg']

    def create_directional_field(forward, lateral, params):
        c_ratio = params['c_ratio']
        extent_forward = params['fwd_max']
        lateral_pos = params['lat_pos_max']
        lateral_neg = params['lat_neg_max']

        extent_lateral = torch.where(lateral > 0, lateral_pos, lateral_neg)
        extent_lateral_mean = (lateral_pos + lateral_neg) / 2.0
        eps = 1e-3

        u_norm = forward / (extent_forward + eps)
        v_norm = lateral / (extent_lateral + eps)

        sigma_forward, sigma_lateral = (
            (0.85, 0.75) if expected_kpts == 2 else (0.8, 0.8)
        )

        internal_decay = (
            torch.exp(-(u_norm ** 2) / (2 * sigma_forward ** 2))
            * torch.exp(-(v_norm ** 2) / (2 * sigma_lateral ** 2))
        )

        corner_radius = c_ratio * torch.minimum(
            extent_forward, extent_lateral_mean
        )
        straight_x = torch.clamp(extent_forward - corner_radius, min=0.0)
        straight_y = torch.clamp(extent_lateral - corner_radius, min=0.0)

        dist_x = torch.clamp(forward - straight_x, min=0.0)
        dist_y = torch.clamp(torch.abs(lateral) - straight_y, min=0.0)

        dist_outside = torch.sqrt(dist_x ** 2 + dist_y ** 2 + 1e-6)
        boundary_distance = torch.clamp(
            dist_outside - corner_radius, min=0.0
        )

        excess = boundary_distance / (
            torch.maximum(extent_forward, extent_lateral_mean) + eps
        )
        boundary_decay = torch.exp(-(excess ** 2) / (2 * 0.12 ** 2))

        backward = torch.clamp(
            -forward / (extent_forward + eps), min=0.0
        )
        backward_decay = torch.exp(-(backward ** 2) / (2 * 0.05 ** 2))

        raw_field = internal_decay * boundary_decay
        return torch.where(
            forward > 0,
            raw_field,
            backward_decay * raw_field,
        )

    field_x_pos = create_directional_field(local_x, local_y, params_x_pos)
    field_x_neg = create_directional_field(-local_x, local_y, params_x_neg)
    field_y_pos = create_directional_field(local_y, local_x, params_y_pos)
    field_y_neg = create_directional_field(-local_y, local_x, params_y_neg)

    if expected_kpts == 0:
        return torch.max(
            torch.max(field_x_pos, field_x_neg),
            torch.max(field_y_pos, field_y_neg),
        ).unsqueeze(1)

    if expected_kpts == 2:
        return torch.stack(
            [
                torch.max(field_x_pos, field_x_neg),
                torch.max(field_y_pos, field_y_neg),
            ],
            dim=1,
        )

    return torch.stack(
        [field_x_pos, field_x_neg, field_y_pos, field_y_neg],
        dim=1,
    )


def build_yolo_cache(yolo_model_path, dataset_path_list, mapping_data):
    """Run the pose estimator once and cache detections for all input images."""
    print("\n[INFO] Building YOLO pose cache...")
    model = YOLO(yolo_model_path)
    cache = {}

    id_to_name = {}
    for index, (class_name, metadata) in enumerate(mapping_data.items()):
        if isinstance(metadata, dict) and 'class_id' in metadata:
            id_to_name[metadata['class_id']] = class_name
        else:
            id_to_name[index] = class_name

    detection_counts = {}
    images_with_objects = 0
    missed_images = []
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif')

    image_paths = []
    for dataset_path in dataset_path_list:
        for root, _, files in os.walk(dataset_path):
            image_paths.extend(
                os.path.join(root, filename)
                for filename in files
                if filename.lower().endswith(valid_extensions)
            )

    total_images = len(image_paths)

    for path in tqdm(image_paths, desc='YOLO pose cache'):
        normalized_path = os.path.abspath(path).lower()
        results = model(path, verbose=False, conf=0.3, iou=0.45)
        cache[normalized_path] = []

        if not results or len(results[0].boxes) == 0:
            missed_images.append(path)
            continue

        images_with_objects += 1

        for detection in results[0]:
            box = detection.boxes.xyxy[0].cpu().numpy()
            class_id = int(detection.boxes.cls[0].item())
            confidence = float(detection.boxes.conf[0].item())
            keypoints = (
                detection.keypoints.data[0].cpu().numpy()
                if hasattr(detection, 'keypoints')
                and detection.keypoints is not None
                else []
            )

            cache[normalized_path].append(
                {
                    'box': box,
                    'cls_id': class_id,
                    'kpts': keypoints,
                    'conf': confidence,
                }
            )

            class_name = id_to_name.get(
                class_id, f'Unknown_ID_{class_id}'
            )
            detection_counts[class_name] = (
                detection_counts.get(class_name, 0) + 1
            )

    print("\n" + "=" * 60)
    print("[REPORT] YOLO Pose Cache Summary")
    print("=" * 60)
    print(f"Total scanned images : {total_images}")
    if total_images > 0:
        ratio = images_with_objects / total_images * 100.0
        print(f"Images with objects  : {images_with_objects} ({ratio:.1f}%)")
    else:
        print("Images with objects  : 0")

    if missed_images:
        print(f"\n[WARNING] {len(missed_images)} images yielded no detections:")
        for index, missed_path in enumerate(missed_images):
            print(f"   [{index + 1:03d}] -> {missed_path}")

    print("\n[STATISTICS] Detections per class:")
    if not detection_counts:
        print("  [WARNING] No objects were detected.")
    else:
        for class_name, count in detection_counts.items():
            print(f"  -> {class_name}: {count}")

    print("=" * 60 + "\n")
    return cache


class CoordinateFieldDataset(ImageFolder):
    """ImageFolder wrapper that prepares instance parameters for coordinate rendering."""

    def __init__(
        self,
        root,
        transform,
        yolo_cache,
        calib_data,
        mapping_data,
        is_train=True,
        val_set_path='',
        conf_multiplier=1.0,
    ):
        super().__init__(root, transform=None)
        self.custom_transform = transform
        self.yolo_cache = yolo_cache
        self.calib_data = calib_data
        self.is_train = is_train
        self.conf_multiplier = conf_multiplier

        self.yolo_val_images = set()
        if self.is_train:
            if val_set_path and os.path.exists(val_set_path):
                with open(val_set_path, 'r', encoding='utf-8') as handle:
                    self.yolo_val_images = set(json.load(handle))
                print(
                    "[INFO] Loaded "
                    f"{len(self.yolo_val_images)} pose-validation images "
                    "that will not receive pose jitter."
                )
            else:
                print(
                    "[WARNING] Pose-validation split was not found. "
                    "All training images may receive pose jitter."
                )

        self.id_to_name = {}
        for index, (class_name, metadata) in enumerate(mapping_data.items()):
            if class_name == 'Unknown_Noise':
                continue
            if isinstance(metadata, dict) and 'class_id' in metadata:
                self.id_to_name[metadata['class_id']] = class_name
            else:
                self.id_to_name[index] = class_name

        self.valid_classes = []
        self.coord_disabled_classes = set()

        for class_name in sorted(self.calib_data.keys()):
            calib = self.calib_data.get(class_name, {})
            mask_ratio_info = calib.get('mask_pixel_ratio_full_image')
            relation_info = calib.get('component_relation')

            disable_coord = False
            disable_reason = None

            if isinstance(mask_ratio_info, dict) and isinstance(
                relation_info, dict
            ):
                role = relation_info.get('component_role')
                median_ratio = mask_ratio_info.get('median')
                try:
                    median_ratio = float(median_ratio)
                except (TypeError, ValueError):
                    median_ratio = None

                if role == 'secondary' and median_ratio is not None:
                    if median_ratio < 0.008:
                        disable_coord = True
                        disable_reason = (
                            "secondary component with median mask ratio "
                            f"{median_ratio:.4f} < 0.008"
                        )
                    elif median_ratio < 0.016:
                        metadata = mapping_data.get(class_name, {})
                        variance = (
                            metadata.get('variance')
                            if isinstance(metadata, dict)
                            else None
                        )
                        try:
                            variance = float(variance)
                        except (TypeError, ValueError):
                            variance = None

                        if variance is None or variance >= 0.06:
                            disable_coord = True
                            variance_text = (
                                'missing'
                                if variance is None
                                else f'{variance:.4f}'
                            )
                            disable_reason = (
                                "secondary component with median mask ratio "
                                f"{median_ratio:.4f} < 0.016 but "
                                f"variance={variance_text} is not < 0.06"
                            )

            if disable_coord:
                self.coord_disabled_classes.add(class_name)
                print(
                    "[INFO] Coordinate field disabled for "
                    f"'{class_name}': {disable_reason}."
                )
            else:
                self.valid_classes.append(class_name)

        self.class_channel_map = {}
        self.name_to_kpt_num = {}

        current_channel = 0
        for class_name in self.valid_classes:
            metadata = mapping_data.get(class_name, {})
            expected_kpts = (
                metadata.get('kpt_num', 3)
                if isinstance(metadata, dict)
                else 3
            )
            self.name_to_kpt_num[class_name] = expected_kpts

            num_channels = (
                4 if expected_kpts == 3
                else 2 if expected_kpts == 2
                else 1
            )
            self.class_channel_map[class_name] = {
                'start': current_channel,
                'end': current_channel + num_channels,
                'expected_kpts': expected_kpts,
            }
            current_channel += num_channels

        self.num_coord_channels = current_channel
        print(
            "[INFO] Dataset initialized with "
            f"{self.num_coord_channels} coordinate-field channels."
        )

        self.processed_extents = {}
        for class_name in self.valid_classes:
            calib = self.calib_data[class_name]
            expected_kpts = self.name_to_kpt_num[class_name]
            extents = calib['global_extents']

            if expected_kpts == 0:
                avg_x = (
                    extents['beam_X_pos']['fwd_max']
                    + extents['beam_X_neg']['fwd_max']
                    + extents['beam_Y_pos']['lat_pos_max']
                    + extents['beam_Y_pos']['lat_neg_max']
                    + extents['beam_Y_neg']['lat_pos_max']
                    + extents['beam_Y_neg']['lat_neg_max']
                ) / 6.0
                avg_y = (
                    extents['beam_Y_pos']['fwd_max']
                    + extents['beam_Y_neg']['fwd_max']
                    + extents['beam_X_pos']['lat_pos_max']
                    + extents['beam_X_pos']['lat_neg_max']
                    + extents['beam_X_neg']['lat_pos_max']
                    + extents['beam_X_neg']['lat_neg_max']
                ) / 6.0
                avg_c = sum(
                    extents[key]['c_ratio'] for key in extents.keys()
                ) / 4.0

                params_x = {
                    'fwd_max': avg_x,
                    'lat_pos_max': avg_y,
                    'lat_neg_max': avg_y,
                    'c_ratio': avg_c,
                }
                params_y = {
                    'fwd_max': avg_y,
                    'lat_pos_max': avg_x,
                    'lat_neg_max': avg_x,
                    'c_ratio': avg_c,
                }
                final_extents = {
                    'beam_X_pos': params_x,
                    'beam_X_neg': params_x,
                    'beam_Y_pos': params_y,
                    'beam_Y_neg': params_y,
                }

            elif expected_kpts == 2:
                avg_x_forward = (
                    extents['beam_X_pos']['fwd_max']
                    + extents['beam_X_neg']['fwd_max']
                ) / 2.0
                avg_x_lateral = (
                    extents['beam_X_pos']['lat_pos_max']
                    + extents['beam_X_pos']['lat_neg_max']
                    + extents['beam_X_neg']['lat_pos_max']
                    + extents['beam_X_neg']['lat_neg_max']
                ) / 4.0
                avg_x_c = (
                    extents['beam_X_pos']['c_ratio']
                    + extents['beam_X_neg']['c_ratio']
                ) / 2.0
                params_x = {
                    'fwd_max': avg_x_forward,
                    'lat_pos_max': avg_x_lateral,
                    'lat_neg_max': avg_x_lateral,
                    'c_ratio': avg_x_c,
                }

                avg_y_forward = (
                    extents['beam_Y_pos']['fwd_max']
                    + extents['beam_Y_neg']['fwd_max']
                ) / 2.0
                avg_y_lateral = (
                    extents['beam_Y_pos']['lat_pos_max']
                    + extents['beam_Y_pos']['lat_neg_max']
                    + extents['beam_Y_neg']['lat_pos_max']
                    + extents['beam_Y_neg']['lat_neg_max']
                ) / 4.0
                avg_y_c = (
                    extents['beam_Y_pos']['c_ratio']
                    + extents['beam_Y_neg']['c_ratio']
                ) / 2.0
                params_y = {
                    'fwd_max': avg_y_forward,
                    'lat_pos_max': avg_y_lateral,
                    'lat_neg_max': avg_y_lateral,
                    'c_ratio': avg_y_c,
                }
                final_extents = {
                    'beam_X_pos': params_x,
                    'beam_X_neg': params_x,
                    'beam_Y_pos': params_y,
                    'beam_Y_neg': params_y,
                }

            else:
                final_extents = extents

            self.processed_extents[class_name] = final_extents

    def __getitem__(self, index):
        path, target = self.samples[index]
        image = Image.open(path).convert('RGB')
        orig_width, orig_height = image.size

        normalized_path = os.path.abspath(path).lower()
        render_params = []

        for detection in self.yolo_cache.get(normalized_path, []):
            class_id = detection['cls_id']
            class_name = self.id_to_name.get(class_id, 'Unknown')

            if (
                class_name not in self.calib_data
                or class_name not in self.class_channel_map
            ):
                continue

            calib = self.calib_data[class_name]
            min_conf = (
                calib.get('min_conf_threshold', 0.0)
                * self.conf_multiplier
            )
            if detection.get('conf', 1.0) < min_conf:
                continue

            layout = self.class_channel_map[class_name]
            expected_kpts = layout['expected_kpts']
            start_channel = layout['start']
            end_channel = layout['end']

            cx, cy, angle = compute_fused_pose(
                detection['box'],
                detection['kpts'],
                calib,
                expected_kpts,
            )

            image_basename = os.path.basename(path)
            is_pose_validation = image_basename in self.yolo_val_images

            if self.is_train and not is_pose_validation:
                x1, y1, x2, y2 = detection['box']
                box_width = max(x2 - x1, 1.0)
                box_height = max(y2 - y1, 1.0)

                jitter = calib.get(
                    'empirical_jitter',
                    {
                        'cx_std': 0.025,
                        'cy_std': 0.025,
                        'theta_std': 2.0,
                    },
                )
                std_cx = max(jitter.get('cx_std', 0.0), 0.018)
                std_cy = max(jitter.get('cy_std', 0.0), 0.018)
                std_theta = max(jitter.get('theta_std', 0.0), 1.18)

                cx += get_truncated_normal_noise(std_cx) * box_width
                cy += get_truncated_normal_noise(std_cy) * box_height
                if expected_kpts >= 2:
                    angle += get_truncated_normal_noise(std_theta)

            render_params.append(
                {
                    'cx': cx,
                    'cy': cy,
                    'angle': angle,
                    'extents': self.processed_extents[class_name],
                    'expected_kpts': expected_kpts,
                    'start_c': start_channel,
                    'end_c': end_channel,
                }
            )

        if self.is_train:
            image_st, image_ae = self.custom_transform(image)
            return (
                (image_st, image_ae),
                render_params,
                (orig_height, orig_width),
            )

        image_transformed = self.custom_transform(image)
        return (
            image_transformed,
            render_params,
            (orig_height, orig_width),
            target,
            path,
        )


def custom_collate(batch):
    """Keep batch items unstacked because render parameters have variable length."""
    return batch


def visualize_coordinate_fields(
    dataloader,
    output_dir,
    num_coord_channels,
    image_size=256,
    num_samples=2,
):
    """Save optional coordinate-field sanity-check visualizations."""
    import matplotlib.pyplot as plt

    os.makedirs(output_dir, exist_ok=True)
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    prefetcher = CoordinateFieldPrefetcher(
        dataloader,
        num_coord_channels,
        image_size=image_size,
        prefetch_depth=2,
    )

    for sample_index in range(num_samples):
        batch = prefetcher.next()
        if batch is None:
            break

        image_st, _, coords, _, _ = batch
        image_array = image_st[0].cpu().numpy().transpose(1, 2, 0)
        image_rgb = np.clip(std * image_array + mean, 0, 1)

        coord_array = coords[0].cpu().numpy()
        fused = (
            np.max(coord_array, axis=0)
            if coord_array.shape[0] > 0
            else np.zeros((image_size, image_size))
        )

        num_channels = coord_array.shape[0]
        fig, axes = plt.subplots(
            1,
            2 + num_channels,
            figsize=(4 * (2 + num_channels), 4),
        )

        axes[0].imshow(image_rgb)
        axes[0].set_title('Original Image')
        axes[0].axis('off')

        axes[1].imshow(fused, cmap='viridis')
        axes[1].set_title('Fused Coordinate Field')
        axes[1].axis('off')

        for channel_index in range(num_channels):
            axes[2 + channel_index].imshow(
                coord_array[channel_index],
                cmap='magma',
            )
            axes[2 + channel_index].set_title(
                f'Channel {channel_index + 1}'
            )
            axes[2 + channel_index].axis('off')

        save_path = os.path.join(
            output_dir,
            f'coordinate_fields_{sample_index + 1}.png',
        )
        plt.tight_layout()
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        plt.close(fig)
