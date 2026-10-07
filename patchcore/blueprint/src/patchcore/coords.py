import os
import json
import torch
import numpy as np
from tqdm import tqdm
from ultralytics import YOLO
from collections import deque

def compute_resize_crop_geometry(
    orig_h,
    orig_w,
    resize,
    crop_size,
):
    """
    Compute the geometry of:

        torchvision.transforms.Resize(resize)
        torchvision.transforms.CenterCrop(crop_size)

    when resize and crop_size are integers.

    Resize(int) keeps the aspect ratio and scales the shorter side
    to `resize`.
    """
    orig_h = int(orig_h)
    orig_w = int(orig_w)
    resize = int(resize)
    crop_size = int(crop_size)

    # Match torchvision Resize(int):
    # the shorter edge becomes `resize`.
    if orig_w <= orig_h:
        resized_w = resize
        resized_h = int(resize * orig_h / orig_w)
    else:
        resized_h = resize
        resized_w = int(resize * orig_w / orig_h)

    # Match torchvision CenterCrop:
    # it uses round(), not simple floor division.
    crop_top = int(round((resized_h - crop_size) / 2.0))
    crop_left = int(round((resized_w - crop_size) / 2.0))

    return {
        "resized_h": resized_h,
        "resized_w": resized_w,
        "crop_top": crop_top,
        "crop_left": crop_left,
    }

def compute_fused_pose(box, kpts, calib_params, expected_kpts):
    """
    [EN] Compute fused object center and orientation based on bounding boxes and keypoints.
    [ZH] 基于边界框和关键点计算融合的目标中心与方向角。
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


def render_coordinate_fields(pre_x, pre_y, cx, cy, angle_deg, global_extents, expected_kpts=3):
    """Render pose-aligned object-centric coordinate fields."""
    cx = cx.view(-1, 1, 1)
    cy = cy.view(-1, 1, 1)
    theta = -angle_deg.view(-1, 1, 1) * np.pi / 180.0

    dx = pre_x.unsqueeze(0) - cx
    dy = pre_y.unsqueeze(0) - cy

    cos_t, sin_t = torch.cos(theta), torch.sin(theta)
    local_x = dx * cos_t - dy * sin_t
    local_y = dx * sin_t + dy * cos_t

    p = global_extents
    p_X_pos, p_X_neg = p['beam_X_pos'], p['beam_X_neg']
    p_Y_pos, p_Y_neg = p['beam_Y_pos'], p['beam_Y_neg']

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


class PatchCoreCoordinatePrefetcher:
    """
    GPU prefetcher that renders Blueprint-AD coordinate fields in exactly the
    same geometry as the RGB preprocessing used by the dataset loader.

    Dataset/preprocessing mode is read directly from ``loader.dataset``:

      MVTec / VisA:
          Resize(short side = resize) -> CenterCrop(image_size)

      MVTec LOCO:
          Resize((image_size, image_size)), no crop

    No dataset-name argument is required at the call site. The ``resize`` and
    ``crop_size`` arguments are retained only as fallbacks for older loaders.
    """

    def __init__(
        self,
        loader,
        num_coord_channels,
        resize=256,
        crop_size=224,
    ):
        self._loader_obj = loader
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()
        self.num_coord_channels = num_coord_channels

        dataset = getattr(loader, "dataset", None)

        # The unified dataset loader stores these attributes after inferring
        # the dataset family from the class name.
        self.dataset_family = getattr(
            dataset,
            "dataset_family",
            None,
        )

        # Dataset-side geometry is the source of truth when available.
        # Explicit arguments remain as backward-compatible fallbacks.
        self.resize = int(
            getattr(
                dataset,
                "resize",
                resize,
            )
        )
        self.crop_size = int(
            getattr(
                dataset,
                "image_size",
                crop_size,
            )
        )

        if self.dataset_family == "loco":
            self.preprocess_mode = "square_resize"
        else:
            # Includes MVTec, VisA, and legacy loaders with no family metadata.
            self.preprocess_mode = "aspect_resize_center_crop"

        self.queue = deque()
        self._preload_one()

    @staticmethod
    def _square_resize_grid(
        orig_h,
        orig_w,
        output_h,
        output_w,
        device,
    ):
        """
        Build an output-grid expressed in original-image coordinates for:

            Resize((output_h, output_w))

        The image is stretched independently along x/y, which is exactly the
        LOCO preprocessing used by the unified dataset loader.
        """
        orig_h = float(orig_h)
        orig_w = float(orig_w)
        output_h = int(output_h)
        output_w = int(output_w)

        # Match the conventional pixel-center mapping used by image resizing.
        # Clamp keeps the outermost sample inside the original image.
        x_1d = (
            (torch.arange(
                output_w,
                device=device,
                dtype=torch.float32,
            ) + 0.5)
            * (orig_w / float(output_w))
            - 0.5
        ).clamp(
            min=0.0,
            max=max(orig_w - 1.0, 0.0),
        )

        y_1d = (
            (torch.arange(
                output_h,
                device=device,
                dtype=torch.float32,
            ) + 0.5)
            * (orig_h / float(output_h))
            - 0.5
        ).clamp(
            min=0.0,
            max=max(orig_h - 1.0, 0.0),
        )

        pre_y, pre_x = torch.meshgrid(
            y_1d,
            x_1d,
        )
        return pre_x, pre_y

    @staticmethod
    def _aspect_resize_center_crop_grid(
        orig_h,
        orig_w,
        resize,
        crop_h,
        crop_w,
        device,
    ):
        """
        Build a cropped output-grid expressed in original-image coordinates for:

            Resize(short side = resize) -> CenterCrop(crop_h, crop_w)

        Current PatchCore experiments use a square crop, but keeping h/w
        separate makes the geometry explicit and robust.
        """
        if int(crop_h) != int(crop_w):
            raise ValueError(
                "The current PatchCore preprocessing expects a square "
                f"CenterCrop, got {(crop_h, crop_w)}."
            )

        geometry = compute_resize_crop_geometry(
            orig_h=orig_h,
            orig_w=orig_w,
            resize=resize,
            crop_size=crop_h,
        )

        resized_h = geometry["resized_h"]
        resized_w = geometry["resized_w"]
        crop_top = geometry["crop_top"]
        crop_left = geometry["crop_left"]

        # Coordinates in the resized image corresponding to the final crop.
        x_resized = (
            torch.arange(
                crop_w,
                device=device,
                dtype=torch.float32,
            )
            + float(crop_left)
        )

        y_resized = (
            torch.arange(
                crop_h,
                device=device,
                dtype=torch.float32,
            )
            + float(crop_top)
        )

        # Inverse-map resized coordinates into the original-image coordinate
        # system, where YOLO boxes/keypoints and calibrated extents live.
        x_1d = x_resized * (
            float(orig_w) / float(resized_w)
        )
        y_1d = y_resized * (
            float(orig_h) / float(resized_h)
        )

        pre_y, pre_x = torch.meshgrid(
            y_1d,
            x_1d,
        )
        return pre_x, pre_y

    def _build_original_coordinate_grid(
        self,
        orig_h,
        orig_w,
        output_h,
        output_w,
        device,
    ):
        if self.preprocess_mode == "square_resize":
            return self._square_resize_grid(
                orig_h=orig_h,
                orig_w=orig_w,
                output_h=output_h,
                output_w=output_w,
                device=device,
            )

        if self.preprocess_mode == "aspect_resize_center_crop":
            return self._aspect_resize_center_crop_grid(
                orig_h=orig_h,
                orig_w=orig_w,
                resize=self.resize,
                crop_h=output_h,
                crop_w=output_w,
                device=device,
            )

        raise RuntimeError(
            f"Unknown coordinate-field preprocessing mode: {self.preprocess_mode}"
        )

    def _preload_one(self):
        try:
            batch_dict = next(self.loader)
        except StopIteration:
            return

        with torch.cuda.stream(self.stream):
            images = batch_dict["image"].cuda(
                non_blocking=True
            )
            B = images.size(0)

            # Do not hard-code 224 here. The coordinate field must match the actual
            # transformed RGB tensor exactly.
            output_h = int(images.shape[-2])
            output_w = int(images.shape[-1])

            all_coords = torch.zeros(
                (
                    B,
                    self.num_coord_channels,
                    output_h,
                    output_w,
                ),
                device=images.device,
                dtype=torch.float32,
            )

            for b in range(B):
                orig_h, orig_w = batch_dict[
                    "orig_size"
                ][b]

                render_params = batch_dict.get(
                    "render_params",
                    [],
                )[b]

                if not render_params:
                    continue

                pre_x, pre_y = (
                    self._build_original_coordinate_grid(
                        orig_h=orig_h,
                        orig_w=orig_w,
                        output_h=output_h,
                        output_w=output_w,
                        device=images.device,
                    )
                )

                kpt_groups = {}

                for rp in render_params:
                    k = rp["expected_kpts"]

                    if k not in kpt_groups:
                        kpt_groups[k] = {
                            "cx": [],
                            "cy": [],
                            "angle": [],
                            "extents": [],
                            "sc": [],
                            "ec": [],
                        }

                    kpt_groups[k]["cx"].append(
                        rp["cx"]
                    )
                    kpt_groups[k]["cy"].append(
                        rp["cy"]
                    )
                    kpt_groups[k]["angle"].append(
                        rp["angle"]
                    )
                    kpt_groups[k]["extents"].append(
                        rp["extents"]
                    )
                    kpt_groups[k]["sc"].append(
                        rp["start_c"]
                    )
                    kpt_groups[k]["ec"].append(
                        rp["end_c"]
                    )

                for expected_kpts, data in (
                    kpt_groups.items()
                ):
                    N = len(data["cx"])

                    cx_t = torch.tensor(
                        data["cx"],
                        device=images.device,
                        dtype=torch.float32,
                    )
                    cy_t = torch.tensor(
                        data["cy"],
                        device=images.device,
                        dtype=torch.float32,
                    )
                    angle_t = torch.tensor(
                        data["angle"],
                        device=images.device,
                        dtype=torch.float32,
                    )

                    batched_extents = {}

                    for beam in [
                        "beam_X_pos",
                        "beam_X_neg",
                        "beam_Y_pos",
                        "beam_Y_neg",
                    ]:
                        batched_extents[beam] = {}

                        for param in [
                            "fwd_max",
                            "lat_pos_max",
                            "lat_neg_max",
                            "c_ratio",
                        ]:
                            vals = [
                                (
                                    ex[beam].get(
                                        param,
                                        1.0,
                                    )
                                    if param == "c_ratio"
                                    else ex[beam][param]
                                )
                                for ex in data["extents"]
                            ]

                            batched_extents[
                                beam
                            ][param] = torch.tensor(
                                vals,
                                device=images.device,
                                dtype=torch.float32,
                            ).view(
                                -1,
                                1,
                                1,
                            )

                    rendered_batch = (
                        render_coordinate_fields(
                            pre_x,
                            pre_y,
                            cx_t,
                            cy_t,
                            angle_t,
                            batched_extents,
                            expected_kpts,
                        )
                    )

                    for i in range(N):
                        sc = data["sc"][i]
                        ec = data["ec"][i]

                        all_coords[
                            b,
                            sc:ec,
                        ] = torch.max(
                            all_coords[
                                b,
                                sc:ec,
                            ],
                            rendered_batch[i],
                        )

            batch_dict["image"] = images
            batch_dict["coords"] = all_coords

            done_event = torch.cuda.Event()
            done_event.record(self.stream)
            self.queue.append(
                (
                    batch_dict,
                    done_event,
                )
            )

    def next(self):
        if not self.queue:
            return None

        item, done_event = self.queue.popleft()
        done_event.wait()
        self._preload_one()
        return item



def build_yolo_cache(yolo_model_path, dataset_path_list, mapping_data):
    print("\n[INFO] Building the YOLO pose-inference cache...")
    model = YOLO(yolo_model_path)
    cache = {}

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
                'box': box, 'cls_id': cls_id, 'kpts': kpts, 'conf': conf
            })
    return cache


class CompositionalConstraintEvaluator:
    def __init__(self, mapping_data, calib_data, conf_multiplier=0.68):
        self.calib_data = calib_data
        self.conf_multiplier = conf_multiplier
        self.id_to_name = {}
        self.name_to_id = {}
        self.logic_rules = {}
        self.last_violation_class = None

        if mapping_data is not None:
            for name, meta in mapping_data.items():
                if name == "Unknown_Noise" or not isinstance(meta, dict): continue
                if meta.get("is_stable", False):
                    cls_id = meta["class_id"]
                    self.id_to_name[cls_id] = name
                    self.name_to_id[name] = cls_id
                    self.logic_rules[cls_id] = meta

    def check_logic_violation(self, yolo_detections):
        self.last_violation_class = None
        if not yolo_detections: return True

        detected_counts = {}
        for det in yolo_detections:
            cls_id = det['cls_id']
            cls_name = self.id_to_name.get(cls_id)
            if not cls_name: continue

            if self.calib_data is not None and cls_name in self.calib_data:
                raw_p5_conf = self.calib_data[cls_name].get('min_conf_threshold', 0.0)
                actual_threshold = raw_p5_conf * self.conf_multiplier
                if det.get('conf', 1.0) < actual_threshold:
                    continue

            detected_counts[cls_id] = detected_counts.get(cls_id, 0) + 1

        processed_joint_ids = set()
        logic_violation_found = False

        for cls_id, rules in self.logic_rules.items():
            count = detected_counts.get(cls_id, 0)

            if cls_id not in processed_joint_ids:
                expected_target = rules.get("target_count_per_image", 1)
                recorded_var = rules.get("variance", 0.0)
                strict_var_limit = 0.06

                if recorded_var <= strict_var_limit:
                    if rules.get("formation_type") == "joint":
                        joint_total = count
                        for j_name in rules.get("joint_members", []):
                            j_id = self.name_to_id.get(j_name)
                            if j_id is not None:
                                joint_total += detected_counts.get(j_id, 0)
                                processed_joint_ids.add(j_id)
                        if joint_total != expected_target:
                            self.last_violation_class = self.id_to_name.get(cls_id, str(cls_id))
                            logic_violation_found = True
                    else:
                        if count != expected_target:
                            self.last_violation_class = self.id_to_name.get(cls_id, str(cls_id))
                            logic_violation_found = True
                else:
                    if rules.get("formation_type") == "joint":
                        for j_name in rules.get("joint_members", []):
                            j_id = self.name_to_id.get(j_name)
                            if j_id is not None: processed_joint_ids.add(j_id)
            if logic_violation_found: break

            if count > 0:
                for aw in rules.get("always_with", []):
                    if aw.get("co_occurrence_prob", 0) >= 0.95:
                        target_id = self.name_to_id.get(aw["target"])
                        if target_id is not None and detected_counts.get(target_id, 0) == 0:
                            self.last_violation_class = self.id_to_name.get(cls_id, str(cls_id))
                            logic_violation_found = True
                for mx in rules.get("mutually_exclusive_with", []):
                    if mx.get("exclusion_reliability", 0) >= 0.95:
                        target_id = self.name_to_id.get(mx["target"])
                        if target_id is not None and detected_counts.get(target_id, 0) > 0:
                            self.last_violation_class = self.id_to_name.get(cls_id, str(cls_id))
                            logic_violation_found = True
            if logic_violation_found: break

        return logic_violation_found