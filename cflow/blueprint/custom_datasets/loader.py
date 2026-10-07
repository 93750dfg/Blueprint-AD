import os
import json
import random
import math
from PIL import Image
import numpy as np
import torch
from torchvision.io import read_video, write_jpeg
from torch.utils.data import Dataset
from torchvision import transforms as T

from coords import build_yolo_cache, compute_fused_pose, get_truncated_normal_noise


__all__ = ('MVTecDataset', 'StcDataset')

MVTEC_CLASS_NAMES = ['bottle', 'cable', 'capsule', 'carpet', 'grid',
               'hazelnut', 'leather', 'metal_nut', 'pill', 'screw',
               'tile', 'toothbrush', 'transistor', 'wood', 'zipper']

STC_CLASS_NAMES = ['01', '02', '03', '04', '05', '06',
                '07', '08', '09', '10', '11', '12']

LOCO_CLASS_NAMES = [
    'breakfast_box',
    'juice_bottle',
    'pushpins',
    'screw_bag',
    'splicing_connectors',
]

IMG_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')
MASK_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')


def is_image_file(filename):
    return filename.lower().endswith(IMG_EXTENSIONS)


def is_mask_file(filename):
    return filename.lower().endswith(MASK_EXTENSIONS)


def find_mask_entry(gt_type_dir, stem):
    """
    Return a mask entry for an abnormal sample.

    Supported formats:
      1) LOCO: ground_truth/type/000/000.png, 000/001.png, ...
      2) VisA converted style: ground_truth/bad/000.png
      3) MVTec style: ground_truth/type/000_mask.png
    """
    # LOCO style: one test image corresponds to one folder of one or more masks.
    folder_candidates = [
        os.path.join(gt_type_dir, stem),
        os.path.join(gt_type_dir, stem + '_mask'),
    ]

    for folder in folder_candidates:
        if os.path.isdir(folder):
            mask_files = sorted([
                os.path.join(folder, f)
                for f in os.listdir(folder)
                if is_mask_file(f)
            ])
            if len(mask_files) > 0:
                return mask_files

            raise FileNotFoundError(
                f'Mask folder exists but contains no mask images: {folder}'
            )

    # Single-file mask styles.
    file_candidates = [
        f'{stem}.png',
        f'{stem}.jpg',
        f'{stem}.jpeg',
        f'{stem}.JPG',
        f'{stem}_mask.png',
        f'{stem}_mask.jpg',
        f'{stem}_mask.jpeg',
        f'{stem}_mask.JPG',
    ]

    for name in file_candidates:
        path = os.path.join(gt_type_dir, name)
        if os.path.exists(path):
            return path

    raise FileNotFoundError(
        f'Cannot find mask for image stem={stem} in {gt_type_dir}. '
        f'Tried folders: {folder_candidates}; files: {file_candidates}'
    )


def load_mask_entry(mask_entry, target_size=None):
    """
    Load a mask entry as a PIL grayscale mask.

    mask_entry can be:
      - str: one mask file
      - list[str]: multiple mask files merged by pixel-wise max

    target_size is (H, W). When provided, each mask is resized before merging,
    which avoids high RAM usage for LOCO high-resolution masks.
    """
    if mask_entry is None:
        raise ValueError('Abnormal sample has None mask_entry.')

    def open_mask_as_array(path):
        img = Image.open(path).convert("L")

        if target_size is not None:
            h, w = target_size
            img = img.resize((w, h), Image.NEAREST)

        return np.array(img, dtype=np.uint8).copy()

    if isinstance(mask_entry, str):
        return Image.fromarray(open_mask_as_array(mask_entry))

    if isinstance(mask_entry, (list, tuple)):
        if len(mask_entry) == 0:
            raise ValueError('Empty mask list for abnormal sample.')

        merged = None
        for p in mask_entry:
            arr = open_mask_as_array(p)

            if merged is None:
                # 关键：copy，保证 merged 可写
                merged = arr.copy()
            else:
                # 更稳：不用 out=merged，避免 read-only 数组报错
                merged = np.maximum(merged, arr)

        return Image.fromarray(merged.astype(np.uint8))

    raise TypeError(f'Unsupported mask entry type: {type(mask_entry)}')


class StcDataset(Dataset):
    def __init__(self, c, is_train=True):
        assert c.class_name in STC_CLASS_NAMES, 'class_name: {}, should be in {}'.format(c.class_name, STC_CLASS_NAMES)
        self.class_name = c.class_name
        self.is_train = is_train
        self.cropsize = c.crp_size
        #
        if is_train:
            self.dataset_path = os.path.join(c.data_path, 'training')
            self.dataset_vid = os.path.join(self.dataset_path, 'videos')
            self.dataset_dir = os.path.join(self.dataset_path, 'frames')
            self.dataset_files = sorted([f for f in os.listdir(self.dataset_vid) if f.startswith(self.class_name)])
            if not os.path.isdir(self.dataset_dir):
                os.mkdir(self.dataset_dir)
            done_file = os.path.join(self.dataset_path, 'frames_{}.pt'.format(self.class_name))
            print(done_file)
            H, W = 480, 856
            if os.path.isfile(done_file):
                assert torch.load(done_file) == len(self.dataset_files), 'train frames are not processed!'
            else:
                count = 0
                for dataset_file in self.dataset_files:
                    print(dataset_file)
                    data = read_video(os.path.join(self.dataset_vid, dataset_file)) # read video file entirely -> mem issue!!!
                    vid = data[0] # weird read_video that returns byte tensor in format [T,H,W,C]
                    fps = data[2]['video_fps']
                    print('video mu/std: {}/{} {}'.format(torch.mean(vid/255.0, (0,1,2)), torch.std(vid/255.0, (0,1,2)), vid.shape))
                    assert [H, W] == [vid.size(1), vid.size(2)], 'same H/W'
                    dataset_file_dir = os.path.join(self.dataset_dir, os.path.splitext(dataset_file)[0])
                    os.mkdir(dataset_file_dir)
                    count = count + 1
                    for i, frame in enumerate(vid):
                        filename = '{0:08d}.jpg'.format(i)
                        write_jpeg(frame.permute((2, 0, 1)), os.path.join(dataset_file_dir, filename), 80)
                torch.save(torch.tensor(count), done_file)
            #
            self.x, self.y, self.mask = self.load_dataset_folder()
        else:
            self.dataset_path = os.path.join(c.data_path, 'testing')
            self.x, self.y, self.mask = self.load_dataset_folder()

        # set transforms
        if is_train:
            self.transform_x = T.Compose([
                T.Resize(c.img_size, Image.Resampling.LANCZOS),
                T.RandomRotation(5),
                T.CenterCrop(c.crp_size),
                T.ToTensor()])
        # test:
        else:
            self.transform_x = T.Compose([
                T.Resize(c.img_size, Image.Resampling.LANCZOS),
                T.CenterCrop(c.crp_size),
                T.ToTensor()])
        # mask
        self.transform_mask = T.Compose([
            T.ToPILImage(),
            T.Resize(c.img_size, Image.NEAREST),
            T.CenterCrop(c.crp_size),
            T.ToTensor()])

        self.normalize = T.Compose([T.Normalize(c.norm_mean, c.norm_std)])

    def __getitem__(self, idx):
        x, y, mask = self.x[idx], self.y[idx], self.mask[idx]
        x = Image.open(x).convert('RGB')
        x = self.normalize(self.transform_x(x))
        if y == 0: #self.is_train:
            mask = torch.zeros([1, self.cropsize[0], self.cropsize[1]])
        else:
            mask = self.transform_mask(mask)
        #
        return x, y, mask

    def __len__(self):
        return len(self.x)

    def load_dataset_folder(self):
        phase = 'train' if self.is_train else 'test'
        x, y, mask = list(), list(), list()
        img_dir = os.path.join(self.dataset_path, 'frames')
        img_types = sorted([f for f in os.listdir(img_dir) if f.startswith(self.class_name)])
        gt_frame_dir = os.path.join(self.dataset_path, 'test_frame_mask')
        gt_pixel_dir = os.path.join(self.dataset_path, 'test_pixel_mask')
        for i, img_type in enumerate(img_types):
            print('Folder:', img_type)
            # load images
            img_type_dir = os.path.join(img_dir, img_type)
            img_fpath_list = sorted([os.path.join(img_type_dir, f) for f in os.listdir(img_type_dir) if f.endswith('.jpg')])
            x.extend(img_fpath_list)
            # labels for every test image
            if phase == 'test':
                gt_pixel = np.load('{}.npy'.format(os.path.join(gt_pixel_dir, img_type)))
                gt_frame = np.load('{}.npy'.format(os.path.join(gt_frame_dir, img_type)))
                if i == 0:
                    m = gt_pixel
                    y = gt_frame
                else:
                    m = np.concatenate((m, gt_pixel), axis=0)
                    y = np.concatenate((y, gt_frame), axis=0)
                #
                mask = [e for e in m] # np.expand_dims(e, axis=0)
                assert len(x) == len(y), 'number of x and y should be same'
                assert len(x) == len(mask), 'number of x and mask should be same'
            else:
                mask.extend([None] * len(img_fpath_list))
                y.extend([0] * len(img_fpath_list))
        #
        return list(x), list(y), list(mask)

class MVTecDataset(Dataset):
    def __init__(self, c, is_train=True):
        if c.dataset == 'mvtec':
            assert c.class_name in MVTEC_CLASS_NAMES, \
                'class_name: {}, should be in {}'.format(c.class_name, MVTEC_CLASS_NAMES)
        elif c.dataset == 'loco':
            assert c.class_name in LOCO_CLASS_NAMES, \
                'class_name: {}, should be in {}'.format(c.class_name, LOCO_CLASS_NAMES)
        elif c.dataset == 'visa':
            class_dir = os.path.join(c.data_path, c.class_name)
            assert os.path.isdir(class_dir), \
                f'VisA class folder not found: {class_dir}'
        else:
            raise NotImplementedError(f'{c.dataset} is not supported by MVTecDataset')
        self.dataset_path = c.data_path
        self.class_name = c.class_name
        self.is_train = is_train
        self.cropsize = c.crp_size

        # load dataset paths
        self.x, self.y, self.mask = self.load_dataset_folder()

                # Cache decoded images in RAM to avoid repeated disk I/O.
        print(f"[INFO] Caching {self.class_name} images in RAM...")
        self.ram_cache_x = []
        self.ram_cache_mask = []

        for p in self.x:
            self.ram_cache_x.append(Image.open(p).copy())

        if not is_train:
            for idx, (mp, y) in enumerate(zip(self.mask, self.y)):
                if y == 0:
                    self.ram_cache_mask.append(None)
                else:
                    if mp is None:
                        raise RuntimeError(
                            f'Abnormal test sample has None mask. '
                            f'idx={idx}, image={self.x[idx]}'
                        )
                    self.ram_cache_mask.append(
                        load_mask_entry(mp, target_size=self.cropsize)
                    )

        print(f"[INFO] Cached {len(self.ram_cache_x)} images.")

        # set transforms
        # Use an explicit transform path so image-space augmentation and pose parameters stay aligned.
        self.resizer = T.Resize(c.img_size, Image.Resampling.LANCZOS)
        self.mask_resizer = T.Resize(c.img_size, Image.NEAREST)
        self.center_crop = T.CenterCrop(c.crp_size)
        self.to_tensor = T.ToTensor()
        self.max_rot = 0.0

        if not is_train:
            self.transform_x = T.Compose([
                T.Resize(c.img_size, Image.Resampling.LANCZOS),
                T.CenterCrop(c.crp_size),
                T.ToTensor()])

        # mask
        self.transform_mask = T.Compose([
            T.Resize(c.img_size, Image.NEAREST),
            T.CenterCrop(c.crp_size),
            T.ToTensor()])

        self.normalize = T.Compose([T.Normalize(c.norm_mean, c.norm_std)])

        # =========================================================
        # Initialize Blueprint-AD coordinate priors and the pose cache.
        # =========================================================
        self.workspace_dir = getattr(c, 'workspace_dir', None)
        self.conf_multiplier = getattr(c, 'conf_multiplier', 0.68)
        self.has_blueprint = False

        if self.workspace_dir:
            self.setup_blueprint_priors(c)

    def setup_blueprint_priors(self, c):
        """
        Initialize per-category pose metadata and coordinate-field parameters.
        """
        current_workspace = os.path.join(self.workspace_dir, self.class_name)

        yolo_root_pt = os.path.join(current_workspace, "best.pt")
        yolo_nested_pt = os.path.join(current_workspace, "yolo_output", "weights", "best.pt")
        yolo_weights = yolo_root_pt if os.path.exists(yolo_root_pt) else yolo_nested_pt

        calibration_json = os.path.join(current_workspace, "final_anisotropic_calibration.json")
        mapping_json = os.path.join(current_workspace, "global_classes_mapping.json")
        val_set_json = os.path.join(current_workspace, "yolo_val_set.json")

        if not os.path.exists(calibration_json) or not os.path.exists(mapping_json):
            print(f"[WARNING] No Blueprint-AD priors for {self.class_name}; using base CFLOW-AD for this category.")
            return

        with open(calibration_json, 'r', encoding='utf-8') as f:
            self.calib_data = json.load(f)
        with open(mapping_json, 'r', encoding='utf-8') as f:
            self.mapping_data = json.load(f)

        # Load the stored validation-image list when available.
        self.yolo_val_images = set()
        if self.is_train and os.path.exists(val_set_json):
            with open(val_set_json, 'r', encoding='utf-8') as f:
                self.yolo_val_images = set(json.load(f))

        # Build the pose-inference cache.
        train_dir = os.path.join(self.dataset_path, self.class_name, 'train')
        test_dir = os.path.join(self.dataset_path, self.class_name, 'test')
        self.yolo_cache = build_yolo_cache(yolo_weights, [train_dir, test_dir], self.mapping_data)

        # Parse the object/component class mapping.
        self.id_to_name = {}
        self.logic_rules = {}
        for i, (cls_name, meta) in enumerate(self.mapping_data.items()):
            if cls_name == "Unknown_Noise": continue
            if isinstance(meta, dict) and 'class_id' in meta:
                self.id_to_name[meta['class_id']] = cls_name
                # Keep stable compositional rules for test-time checking.
                if meta.get("is_stable", False):
                    self.logic_rules[meta['class_id']] = meta
            else:
                self.id_to_name[i] = cls_name

        self.name_to_id = {v: k for k, v in self.id_to_name.items()}
        self.valid_classes = sorted(list(self.calib_data.keys()))

        self.class_channel_map = {}
        self.processed_extents = {}
        current_c = 0

        for cls_name in self.valid_classes:
            meta = self.mapping_data.get(cls_name, {})


            kpts = meta.get('kpt_num', 3) if isinstance(meta, dict) else 3

            num_c = 4 if kpts == 3 else (2 if kpts == 2 else 1)
            self.class_channel_map[cls_name] = {'start': current_c, 'end': current_c + num_c, 'expected_kpts': kpts}
            current_c += num_c

            # Precompute the coordinate-field extents.
            calib = self.calib_data[cls_name]
            p = calib['global_extents']
            if kpts == 0:
                avg_X = (p['beam_X_pos']['fwd_max'] + p['beam_X_neg']['fwd_max'] +
                         p['beam_Y_pos']['lat_pos_max'] + p['beam_Y_pos']['lat_neg_max'] +
                         p['beam_Y_neg']['lat_pos_max'] + p['beam_Y_neg']['lat_neg_max']) / 6.0
                avg_Y = (p['beam_Y_pos']['fwd_max'] + p['beam_Y_neg']['fwd_max'] +
                         p['beam_X_pos']['lat_pos_max'] + p['beam_X_pos']['lat_neg_max'] +
                         p['beam_X_neg']['lat_pos_max'] + p['beam_X_neg']['lat_neg_max']) / 6.0
                avg_c = sum([p[k]['c_ratio'] for k in p.keys()]) / 4.0
                p_X = {'fwd_max': avg_X, 'lat_pos_max': avg_Y, 'lat_neg_max': avg_Y, 'c_ratio': avg_c}
                p_Y = {'fwd_max': avg_Y, 'lat_pos_max': avg_X, 'lat_neg_max': avg_X, 'c_ratio': avg_c}
                self.processed_extents[cls_name] = {'beam_X_pos': p_X, 'beam_X_neg': p_X, 'beam_Y_pos': p_Y,
                                                    'beam_Y_neg': p_Y}
            elif kpts == 2:
                avg_X_fwd = (p['beam_X_pos']['fwd_max'] + p['beam_X_neg']['fwd_max']) / 2.0
                avg_X_lat = (p['beam_X_pos']['lat_pos_max'] + p['beam_X_pos']['lat_neg_max'] +
                             p['beam_X_neg']['lat_pos_max'] + p['beam_X_neg']['lat_neg_max']) / 4.0
                avg_X_c = (p['beam_X_pos']['c_ratio'] + p['beam_X_neg']['c_ratio']) / 2.0
                p_X = {'fwd_max': avg_X_fwd, 'lat_pos_max': avg_X_lat, 'lat_neg_max': avg_X_lat, 'c_ratio': avg_X_c}
                avg_Y_fwd = (p['beam_Y_pos']['fwd_max'] + p['beam_Y_neg']['fwd_max']) / 2.0
                avg_Y_lat = (p['beam_Y_pos']['lat_pos_max'] + p['beam_Y_pos']['lat_neg_max'] +
                             p['beam_Y_neg']['lat_pos_max'] + p['beam_Y_neg']['lat_neg_max']) / 4.0
                avg_Y_c = (p['beam_Y_pos']['c_ratio'] + p['beam_Y_neg']['c_ratio']) / 2.0
                p_Y = {'fwd_max': avg_Y_fwd, 'lat_pos_max': avg_Y_lat, 'lat_neg_max': avg_Y_lat, 'c_ratio': avg_Y_c}
                self.processed_extents[cls_name] = {'beam_X_pos': p_X, 'beam_X_neg': p_X, 'beam_Y_pos': p_Y,
                                                    'beam_Y_neg': p_Y}
            else:
                self.processed_extents[cls_name] = p

        self.num_coord_channels = current_c
        self.has_blueprint = True

    def check_semantic_logic(self, norm_path):
        """
        Evaluate the supplementary compositional constraints during testing.
        """
        detections = self.yolo_cache.get(norm_path, [])
        detected_counts = {}

        for det in detections:
            cls_id = det['cls_id']
            cls_name = self.id_to_name.get(cls_id)
            if not cls_name: continue

            if cls_name in self.calib_data:
                actual_threshold = self.calib_data[cls_name].get('min_conf_threshold', 0.0) * self.conf_multiplier
                if det.get('conf', 1.0) < actual_threshold:
                    continue
            detected_counts[cls_id] = detected_counts.get(cls_id, 0) + 1

        logic_violation_found = False
        processed_joint_ids = set()

        for cls_id, rules in self.logic_rules.items():
            count = detected_counts.get(cls_id, 0)

            if cls_id not in processed_joint_ids:
                expected_target = rules.get("target_count_per_image", 1)
                recorded_var = rules.get("variance", 0.0)

                if recorded_var <= 0.06:
                    if rules.get("formation_type") == "joint":
                        joint_total = count
                        for j_name in rules.get("joint_members", []):
                            j_id = self.name_to_id.get(j_name)
                            if j_id is not None:
                                joint_total += detected_counts.get(j_id, 0)
                                processed_joint_ids.add(j_id)
                        if joint_total != expected_target:
                            logic_violation_found = True
                            break
                    else:
                        if count != expected_target:
                            logic_violation_found = True
                            break
                else:
                    if rules.get("formation_type") == "joint":
                        for j_name in rules.get("joint_members", []):
                            j_id = self.name_to_id.get(j_name)
                            if j_id is not None:
                                processed_joint_ids.add(j_id)

            if count > 0:
                for aw in rules.get("always_with", []):
                    if aw.get("co_occurrence_prob", 0) >= 0.95:
                        target_id = self.name_to_id.get(aw["target"])
                        if target_id is not None and detected_counts.get(target_id, 0) == 0:
                            logic_violation_found = True
                            break
                if logic_violation_found: break

                for mx in rules.get("mutually_exclusive_with", []):
                    if mx.get("exclusion_reliability", 0) >= 0.95:
                        target_id = self.name_to_id.get(mx["target"])
                        if target_id is not None and detected_counts.get(target_id, 0) > 0:
                            logic_violation_found = True
                            break
                if logic_violation_found: break

        return logic_violation_found

    def __getitem__(self, idx):
        path = self.x[idx]
        y = self.y[idx]

        # Read the decoded image from the RAM cache.
        x_img = self.ram_cache_x[idx]
        orig_w, orig_h = x_img.size

        if self.class_name in ['zipper', 'screw', 'grid']:
            x_arr = np.expand_dims(np.array(x_img), axis=2)
            x_arr = np.concatenate([x_arr, x_arr, x_arr], axis=2)
            x_img = Image.fromarray(x_arr.astype('uint8')).convert('RGB')

        # Apply training augmentation explicitly so pose parameters follow the same transform.
        rot_angle = 0.0
        if self.is_train:
            # Sample the image rotation used by the explicit augmentation path.
            rot_angle = random.uniform(-self.max_rot, self.max_rot)
            # Apply resize, rotation, crop, and normalization.
            x_img = self.resizer(x_img)
            x_img = x_img.rotate(rot_angle, resample=Image.Resampling.BICUBIC)
            x_img = self.center_crop(x_img)
            x_img = self.to_tensor(x_img)
            x = self.normalize(x_img)
        else:
            # Deterministic test preprocessing.
            x = self.normalize(self.transform_x(x_img))

        if y == 0:
            mask = torch.zeros([1, self.cropsize[0], self.cropsize[1]])
        else:
            mask_img = self.ram_cache_mask[idx]
            if self.is_train:
                # Keep a non-normal training mask aligned if such a sample is present.
                mask_img = self.mask_resizer(mask_img).rotate(rot_angle, resample=Image.Resampling.NEAREST)
                mask_img = self.center_crop(mask_img)
                mask = self.to_tensor(mask_img)
            else:
                mask = self.transform_mask(mask_img)

        render_params = []
        logic_violation = False

        if self.has_blueprint:
            norm_path = os.path.abspath(path).lower()
            if not self.is_train:
                logic_violation = self.check_semantic_logic(norm_path)

            if norm_path in self.yolo_cache:
                for det in self.yolo_cache[norm_path]:
                    cls_id = det['cls_id']
                    cls_name = self.id_to_name.get(cls_id, "Unknown")

                    if cls_name not in self.calib_data or cls_name not in self.class_channel_map:
                        continue

                    calib = self.calib_data[cls_name]
                    actual_threshold = calib.get('min_conf_threshold', 0.0) * self.conf_multiplier
                    if det.get('conf', 1.0) < actual_threshold:
                        continue

                    layout = self.class_channel_map[cls_name]
                    expected_kpts = layout['expected_kpts']
                    start_c, end_c = layout['start'], layout['end']

                    # Compute the pose in original-image coordinates.
                    cx, cy, angle = compute_fused_pose(det['box'], det['kpts'], calib, expected_kpts)

                    # Apply the training-time pose jitter used for the coordinate fields.
                    if self.is_train:
                        # Apply small instance-level pose jitter.
                        bx1, by1, bx2, by2 = det['box']
                        bw, bh = max(bx2 - bx1, 1.0), max(by2 - by1, 1.0)
                        jitter = calib.get('empirical_jitter', {'cx_std': 0.025, 'cy_std': 0.025, 'theta_std': 2.0})

                        cx += get_truncated_normal_noise(max(jitter.get('cx_std', 0.0), 0.018)) * bw * 0.25
                        cy += get_truncated_normal_noise(max(jitter.get('cy_std', 0.0), 0.018)) * bh * 0.25
                        if expected_kpts >= 2:
                            angle += get_truncated_normal_noise(max(jitter.get('theta_std', 0.0), 1.18)) * 0.25

                        # Reproject the pose only when an image-space rotation is applied.
                        if rot_angle != 0.0:
                            # Rotate the predicted orientation.
                            angle = angle + rot_angle

                            # Rotate the instance center around the image center.
                            mid_h, mid_w = orig_h / 2.0, orig_w / 2.0
                            rad = math.radians(-rot_angle)
                            cos_r, sin_r = math.cos(rad), math.sin(rad)

                            tx = cx - mid_w
                            ty = cy - mid_h

                            cx = tx * cos_r - ty * sin_r + mid_w
                            cy = tx * sin_r + ty * cos_r + mid_h

                    render_params.append({
                        'cx': cx, 'cy': cy, 'angle': angle,
                        'extents': self.processed_extents[cls_name],
                        'expected_kpts': expected_kpts,
                        'start_c': start_c, 'end_c': end_c
                    })

        return x, y, mask, render_params, (orig_h, orig_w), path, logic_violation

    def __len__(self):
        return len(self.x)

    def load_dataset_folder(self):
        phase = 'train' if self.is_train else 'test'
        x, y, mask = [], [], []

        img_dir = os.path.join(self.dataset_path, self.class_name, phase)
        gt_dir = os.path.join(self.dataset_path, self.class_name, 'ground_truth')

        img_types = sorted(os.listdir(img_dir))

        for img_type in img_types:
            img_type_dir = os.path.join(img_dir, img_type)
            if not os.path.isdir(img_type_dir):
                continue

            img_fpath_list = sorted([
                os.path.join(img_type_dir, f)
                for f in os.listdir(img_type_dir)
                if is_image_file(f)
            ])

            x.extend(img_fpath_list)

            if img_type == 'good':
                y.extend([0] * len(img_fpath_list))
                mask.extend([None] * len(img_fpath_list))
            else:
                y.extend([1] * len(img_fpath_list))

                gt_type_dir = os.path.join(gt_dir, img_type)
                img_fname_list = [
                    os.path.splitext(os.path.basename(f))[0]
                    for f in img_fpath_list
                ]

                gt_fpath_list = [
                    find_mask_entry(gt_type_dir, img_fname)
                    for img_fname in img_fname_list
                ]

                mask.extend(gt_fpath_list)

        assert len(x) == len(y), 'number of x and y should be same'
        assert len(x) == len(mask), 'number of x and mask should be same'

        return list(x), list(y), list(mask)
