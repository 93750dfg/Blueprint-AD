#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Blueprint-AD integration for EfficientAD.

The underlying EfficientAD implementation is adapted from the unofficial
reproduction by nelson1425:
https://github.com/nelson1425/EfficientAD

Coordinate-field construction is implemented in coords.py. This file keeps the
training and evaluation flow close to the EfficientAD baseline used in the paper.
"""

import argparse
import itertools
import json
import os
import random
from pathlib import Path

import numpy as np
import tifffile
import torch
from PIL import Image
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from common import (
    ImageFolderWithoutTarget,
    InfiniteDataloader,
    get_coord_autoencoder,
    get_pdn_medium,
    get_pdn_medium_coord,
)
from coords import (
    CoordinateFieldDataset,
    CoordinateFieldPrefetcher,
    build_yolo_cache,
    custom_collate,
)

torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.matmul.allow_tf32 = True

REPO_ROOT = Path(__file__).resolve().parents[2]
EFFICIENTAD_ROOT = Path(__file__).resolve().parents[1]


def get_argparse():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', default='mvtec_ad',
                        choices=['mvtec_ad', 'mvtec_loco', 'visa'])
    parser.add_argument('-s', '--subdataset', default='cable')
    parser.add_argument('-o', '--output_dir', default='output/3')
    parser.add_argument('-m', '--model_size', default='medium', choices=['medium'],
                        help='Blueprint-AD experiments use EfficientAD-M.')
    parser.add_argument('-w', '--weights',
                        default=str(EFFICIENTAD_ROOT / 'models' / 'teacher_medium.pth'))

    # The option name ``imagenet_train_path`` is inherited from the upstream
    # EfficientAD implementation. The original implementation uses ImageNet
    # as the external penalty dataset, while the experiments in our paper use
    # DTD consistently for the EfficientAD-M baseline and all variants.
    parser.add_argument('-i', '--imagenet_train_path', default='data/dtd/images',
                        help='Path to the external penalty dataset. The option name is inherited from the upstream EfficientAD implementation. DTD is used in the experiments reported in our paper. Set to "none" to disable the penalty dataset.')
    parser.add_argument('-a', '--mvtec_ad_path', default='data/mvtec_ad',
                        help='Path to the MVTec AD dataset.')
    parser.add_argument('-b', '--mvtec_loco_path', default='data/mvtec_loco',
                        help='Path to the MVTec LOCO dataset.')
    parser.add_argument('-v', '--visa_path', default='data/visa',
                        help='Path to the VisA dataset.')
    parser.add_argument('-t', '--train_steps', type=int, default=70000)
    parser.add_argument('--workspace_dir', default=str(REPO_ROOT / 'runs'),
                        help='Root directory containing precomputed per-category Blueprint-AD products.')
    return parser.parse_args()


def seed_everything(seed=42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id):
    """Initialize DataLoader workers deterministically."""
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


# Global constants
SEED = 42
ON_GPU = torch.cuda.is_available()
OUT_CHANNELS = 384
IMAGE_SIZE = 256


default_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

transform_ae = transforms.RandomChoice([
    transforms.ColorJitter(brightness=0.2),
    transforms.ColorJitter(contrast=0.2),
    transforms.ColorJitter(saturation=0.2)
])


def train_transform(image):
    return default_transform(image), default_transform(transform_ae(image))


def main(config):
    seed_everything(SEED)

    if not ON_GPU:
        raise RuntimeError(
            'This Blueprint-AD EfficientAD implementation requires CUDA '
            'for asynchronous coordinate-field rendering.'
        )

    if config.dataset == 'mvtec_ad':
        dataset_path = config.mvtec_ad_path
    elif config.dataset == 'mvtec_loco':
        dataset_path = config.mvtec_loco_path
    elif config.dataset == 'visa':
        dataset_path = config.visa_path
    else:
        raise ValueError('Unknown config.dataset')

    pretrain_penalty = config.imagenet_train_path != 'none'

    train_output_dir = os.path.join(
        config.output_dir,
        'trainings',
        config.dataset,
        config.subdataset,
    )
    test_output_dir = os.path.join(
        config.output_dir,
        'anomaly_maps',
        config.dataset,
        config.subdataset,
        'test',
    )
    os.makedirs(train_output_dir, exist_ok=True)
    os.makedirs(test_output_dir, exist_ok=True)

    train_dir = os.path.join(
        dataset_path,
        config.subdataset,
        'train',
    )
    test_dir = os.path.join(
        dataset_path,
        config.subdataset,
        'test',
    )

    current_workspace = os.path.join(
        config.workspace_dir,
        config.subdataset,
    )
    yolo_root_pt = os.path.join(current_workspace, 'best.pt')
    yolo_nested_pt = os.path.join(
        current_workspace,
        'yolo_output',
        'weights',
        'best.pt',
    )
    yolo_weights = (
        yolo_root_pt
        if os.path.exists(yolo_root_pt)
        else yolo_nested_pt
    )

    calibration_json = os.path.join(
        current_workspace,
        'final_anisotropic_calibration.json',
    )
    mapping_json = os.path.join(
        current_workspace,
        'global_classes_mapping.json',
    )
    val_set_json = os.path.join(
        current_workspace,
        'yolo_val_set.json',
    )

    required_files = [calibration_json, mapping_json, yolo_weights]
    missing_files = [path for path in required_files if not os.path.exists(path)]
    if missing_files:
        raise FileNotFoundError(
            'Missing Blueprint-AD preprocessing files:\n  '
            + '\n  '.join(missing_files)
        )

    with open(calibration_json, 'r', encoding='utf-8') as handle:
        calib_data = json.load(handle)
    with open(mapping_json, 'r', encoding='utf-8') as handle:
        mapping_data = json.load(handle)

    yolo_cache = build_yolo_cache(
        yolo_weights,
        [train_dir, test_dir],
        mapping_data,
    )

    conf_multiplier = 0.68

    full_train_set = CoordinateFieldDataset(
        train_dir,
        transform=train_transform,
        yolo_cache=yolo_cache,
        calib_data=calib_data,
        mapping_data=mapping_data,
        is_train=True,
        val_set_path=val_set_json,
        conf_multiplier=conf_multiplier,
    )
    test_set = CoordinateFieldDataset(
        test_dir,
        transform=default_transform,
        yolo_cache=yolo_cache,
        calib_data=calib_data,
        mapping_data=mapping_data,
        is_train=False,
        val_set_path=val_set_json,
        conf_multiplier=conf_multiplier,
    )

    num_coord_channels = full_train_set.num_coord_channels

    train_size = int(0.9 * len(full_train_set))
    validation_size = len(full_train_set) - train_size
    split_generator = torch.Generator().manual_seed(SEED)
    train_set, validation_set = torch.utils.data.random_split(
        full_train_set,
        [train_size, validation_size],
        split_generator,
    )

    train_generator = torch.Generator()
    train_generator.manual_seed(SEED)
    train_loader = DataLoader(
        train_set,
        batch_size=1,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        worker_init_fn=seed_worker,
        generator=train_generator,
        persistent_workers=True,
        collate_fn=custom_collate,
    )

    train_loader_infinite = InfiniteDataloader(train_loader)
    validation_loader = DataLoader(
        validation_set,
        batch_size=1,
        collate_fn=custom_collate,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=1,
        collate_fn=custom_collate,
    )

    penalty_generator = torch.Generator()
    penalty_generator.manual_seed(SEED + 1)

    if pretrain_penalty:
        penalty_transform = transforms.Compose(
            [
                transforms.Resize((2 * IMAGE_SIZE, 2 * IMAGE_SIZE)),
                transforms.RandomGrayscale(0.3),
                transforms.CenterCrop(IMAGE_SIZE),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )
        penalty_set = ImageFolderWithoutTarget(
            config.imagenet_train_path,
            transform=penalty_transform,
        )
        penalty_loader = DataLoader(
            penalty_set,
            batch_size=1,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
            worker_init_fn=seed_worker,
            generator=penalty_generator,
            persistent_workers=True,
        )
        penalty_loader_infinite = InfiniteDataloader(penalty_loader)
    else:
        penalty_loader_infinite = itertools.repeat(None)

    teacher = get_pdn_medium(OUT_CHANNELS)
    student = get_pdn_medium_coord(
        2 * OUT_CHANNELS,
        padding=False,
        coord_channels=num_coord_channels,
    )
    autoencoder = get_coord_autoencoder(
        OUT_CHANNELS,
        coord_channels=num_coord_channels,
    )

    state_dict = torch.load(config.weights, map_location='cpu')
    teacher.load_state_dict(state_dict)

    teacher.eval()
    student.train()
    autoencoder.train()

    teacher.cuda()
    student.cuda()
    autoencoder.cuda()

    teacher_mean, teacher_std = teacher_normalization(
        teacher,
        train_loader,
    )

    base_params = []
    student_coord_params = []
    autoencoder_coord_params = []

    for name, parameter in student.named_parameters():
        if not parameter.requires_grad:
            continue
        if 'coord_embed' in name:
            student_coord_params.append(parameter)
        else:
            base_params.append(parameter)

    for name, parameter in autoencoder.named_parameters():
        if 'embed' in name:
            autoencoder_coord_params.append(parameter)
        else:
            base_params.append(parameter)

    optimizer = torch.optim.Adam(
        [
            {'params': base_params, 'lr': 1e-4},
            {'params': student_coord_params, 'lr': 1e-5},
            {'params': autoencoder_coord_params, 'lr': 1e-5},
        ],
        weight_decay=1e-5,
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=60000,
        gamma=0.1,
    )

    prefetcher = CoordinateFieldPrefetcher(
        train_loader_infinite,
        num_coord_channels,
        image_size=IMAGE_SIZE,
        prefetch_depth=5,
    )

    progress = tqdm(
        range(config.train_steps),
        dynamic_ncols=True,
        position=0,
        leave=True,
    )

    for iteration, image_penalty in zip(
        progress,
        penalty_loader_infinite,
    ):
        batch_data = prefetcher.next()
        if batch_data is None:
            break

        image_st, image_ae, coords, _, _ = batch_data

        if image_penalty is not None:
            image_penalty = image_penalty.cuda()

        student_coords = coords
        if random.random() < 0.10:
            student_coords = torch.zeros_like(student_coords)

        autoencoder_coords = coords
        if random.random() < 0.05:
            autoencoder_coords = torch.zeros_like(autoencoder_coords)

        coord_scale = 1.0

        with torch.no_grad():
            teacher_output_st = teacher(image_st)
            teacher_output_st = (
                teacher_output_st - teacher_mean
            ) / teacher_std

        student_output_st = student(
            image_st,
            student_coords,
            coord_scale,
        )[:, :OUT_CHANNELS]
        distance_st = (
            teacher_output_st - student_output_st
        ) ** 2
        hard_threshold = torch.quantile(distance_st, q=0.999)
        loss_hard = torch.mean(
            distance_st[distance_st >= hard_threshold]
        )

        if image_penalty is not None:
            student_output_penalty = student(
                image_penalty,
                student_coords,
                coord_scale,
            )[:, :OUT_CHANNELS]
            loss_penalty = torch.mean(
                student_output_penalty ** 2
            )
            loss_st = loss_hard + loss_penalty
        else:
            loss_st = loss_hard

        autoencoder_output = autoencoder(
            image_ae,
            autoencoder_coords,
            coord_scale,
        )

        with torch.no_grad():
            teacher_output_ae = teacher(image_ae)
            teacher_output_ae = (
                teacher_output_ae - teacher_mean
            ) / teacher_std

        student_output_ae = student(
            image_ae,
            student_coords,
            coord_scale,
        )[:, OUT_CHANNELS:]

        distance_ae = (
            teacher_output_ae - autoencoder_output
        ) ** 2
        distance_stae = (
            autoencoder_output - student_output_ae
        ) ** 2

        loss_ae = torch.mean(distance_ae)
        loss_stae = torch.mean(distance_stae)
        loss_total = loss_st + loss_ae + loss_stae

        optimizer.zero_grad(set_to_none=True)
        loss_total.backward()
        torch.nn.utils.clip_grad_norm_(
            student.parameters(),
            max_norm=1.0,
        )
        torch.nn.utils.clip_grad_norm_(
            autoencoder.parameters(),
            max_norm=1.0,
        )
        optimizer.step()
        scheduler.step()

        if iteration % 10 == 0:
            progress.set_description(
                f"Loss: {loss_total.item():.4f} | "
                f"ST: {loss_st.item():.4f} | "
                f"AE: {loss_ae.item():.4f} | "
                f"STAE: {loss_stae.item():.4f}"
            )

        if (
            iteration > 0
            and (
                iteration % 10000 == 0
                or iteration % 55000 == 0
                or iteration % 65000 == 0
            )
        ):
            teacher.eval()
            student.eval()
            autoencoder.eval()

            quantiles = map_normalization(
                validation_loader=validation_loader,
                teacher=teacher,
                student=student,
                autoencoder=autoencoder,
                teacher_mean=teacher_mean,
                teacher_std=teacher_std,
                coord_scale=coord_scale,
                desc='Intermediate map normalization',
            )

            test(
                dataset_name=config.dataset,
                test_loader=test_loader,
                teacher=teacher,
                student=student,
                autoencoder=autoencoder,
                teacher_mean=teacher_mean,
                teacher_std=teacher_std,
                q_st_start=quantiles[0],
                q_st_end=quantiles[1],
                q_ae_start=quantiles[2],
                q_ae_end=quantiles[3],
                coord_scale=coord_scale,
                test_output_dir=None,
                desc=f'Intermediate Inference (Iter: {iteration})',
                calib_data=calib_data,
                mapping_data=mapping_data,
                yolo_cache=yolo_cache,
                conf_multiplier=conf_multiplier,
            )

            teacher.eval()
            student.train()
            autoencoder.train()

    teacher.eval()
    student.eval()
    autoencoder.eval()

    torch.save(
        teacher,
        os.path.join(train_output_dir, 'teacher_final.pth'),
    )
    torch.save(
        student,
        os.path.join(train_output_dir, 'student_final.pth'),
    )
    torch.save(
        autoencoder,
        os.path.join(train_output_dir, 'autoencoder_final.pth'),
    )

    quantiles = map_normalization(
        validation_loader=validation_loader,
        teacher=teacher,
        student=student,
        autoencoder=autoencoder,
        teacher_mean=teacher_mean,
        teacher_std=teacher_std,
        coord_scale=1.0,
        desc='Final map normalization',
    )

    test(
        dataset_name=config.dataset,
        test_loader=test_loader,
        teacher=teacher,
        student=student,
        autoencoder=autoencoder,
        teacher_mean=teacher_mean,
        teacher_std=teacher_std,
        q_st_start=quantiles[0],
        q_st_end=quantiles[1],
        q_ae_start=quantiles[2],
        q_ae_end=quantiles[3],
        coord_scale=1.0,
        test_output_dir=test_output_dir,
        desc='Final Inference',
        calib_data=calib_data,
        mapping_data=mapping_data,
        yolo_cache=yolo_cache,
        conf_multiplier=conf_multiplier,
    )

    torch.cuda.empty_cache()


def test(dataset_name, test_loader, teacher, student, autoencoder, teacher_mean, teacher_std,
         q_st_start, q_st_end, q_ae_start, q_ae_end, coord_scale, test_output_dir=None,
         desc='Running inference', calib_data=None, mapping_data=None,
         yolo_cache=None, conf_multiplier=0.9):
    """Evaluate Blueprint-AD + EfficientAD with count consistency."""
    y_true = []

    id_to_name, name_to_id, logic_rules = {}, {}, {}
    if mapping_data is not None:
        for name, meta in mapping_data.items():
            if name == 'Unknown_Noise' or not isinstance(meta, dict):
                continue
            if meta.get('is_stable', False):
                cls_id = meta['class_id']
                id_to_name[cls_id] = name
                name_to_id[name] = cls_id
                logic_rules[cls_id] = meta

    coord_img_scores = []
    final_img_scores = []

    if test_output_dir is not None:
        pixel_y_true = []
        coord_pixel_scores = []
        final_pixel_scores = []

    num_coord_channels = student.coord_embed[0].in_channels
    prefetcher = CoordinateFieldPrefetcher(
        test_loader, num_coord_channels, image_size=IMAGE_SIZE, prefetch_depth=5)

    for _ in tqdm(range(len(test_loader.dataset)), desc=desc, disable=True):
        batch = prefetcher.next()
        if batch is None:
            break

        image, coords, orig_height, orig_width, target, path = batch

        map_combined, _, _ = predict(
            image=image, coords=coords, teacher=teacher, student=student,
            autoencoder=autoencoder, teacher_mean=teacher_mean, teacher_std=teacher_std,
            q_st_start=q_st_start, q_st_end=q_st_end,
            q_ae_start=q_ae_start, q_ae_end=q_ae_end,
            coord_scale=coord_scale
        )

        map_combined = torch.nn.functional.pad(map_combined, (4, 4, 4, 4))
        coord_map = torch.nn.functional.interpolate(
            map_combined, size=(orig_height, orig_width),
            mode='bilinear', align_corners=False
        )[0, 0].cpu().numpy()

        final_map = coord_map.copy()

        # Count, co-occurrence, and mutual-exclusion consistency check.
        logic_violation_found = False

        if torch.max(coords).item() <= 1e-4:
            logic_violation_found = True
        elif mapping_data is not None and yolo_cache is not None:
            norm_path = os.path.abspath(path).lower()
            detections = yolo_cache.get(norm_path, [])

            detected_counts = {}
            for det in detections:
                cls_id = det['cls_id']
                cls_name = id_to_name.get(cls_id)
                if not cls_name:
                    continue

                if calib_data is not None and cls_name in calib_data:
                    raw_threshold = calib_data[cls_name].get('min_conf_threshold', 0.0)
                    actual_threshold = raw_threshold * conf_multiplier
                    if det.get('conf', 1.0) < actual_threshold:
                        continue

                detected_counts[cls_id] = detected_counts.get(cls_id, 0) + 1

            processed_joint_ids = set()

            for cls_id, rules in logic_rules.items():
                count = detected_counts.get(cls_id, 0)

                if cls_id not in processed_joint_ids:
                    expected_target = rules.get('target_count_per_image', 1)
                    recorded_var = rules.get('variance', 0.0)
                    strict_var_limit = 0.06

                    if recorded_var <= strict_var_limit:
                        if rules.get('formation_type') == 'joint':
                            joint_total = count
                            for joint_name in rules.get('joint_members', []):
                                joint_id = name_to_id.get(joint_name)
                                if joint_id is not None:
                                    joint_total += detected_counts.get(joint_id, 0)
                                    processed_joint_ids.add(joint_id)
                            if joint_total != expected_target:
                                logic_violation_found = True
                                break
                        elif count != expected_target:
                            logic_violation_found = True
                            break
                    elif rules.get('formation_type') == 'joint':
                        for joint_name in rules.get('joint_members', []):
                            joint_id = name_to_id.get(joint_name)
                            if joint_id is not None:
                                processed_joint_ids.add(joint_id)

                if count > 0:
                    for relation in rules.get('always_with', []):
                        if relation.get('co_occurrence_prob', 0) >= 0.95:
                            target_id = name_to_id.get(relation['target'])
                            if target_id is not None and detected_counts.get(target_id, 0) == 0:
                                logic_violation_found = True
                                break
                    if logic_violation_found:
                        break

                    for relation in rules.get('mutually_exclusive_with', []):
                        if relation.get('exclusion_reliability', 0) >= 0.95:
                            target_id = name_to_id.get(relation['target'])
                            if target_id is not None and detected_counts.get(target_id, 0) > 0:
                                logic_violation_found = True
                                break
                    if logic_violation_found:
                        break

        if logic_violation_found:
            max_idx = np.unravel_index(np.argmax(final_map, axis=None), final_map.shape)
            final_map[max_idx] += 1.0

        coord_score = np.max(coord_map)
        final_score = np.max(final_map)

        defect_class = os.path.basename(os.path.dirname(path))
        y_true.append(0 if defect_class == 'good' else 1)
        coord_img_scores.append(coord_score)
        final_img_scores.append(final_score)

        if test_output_dir is not None:
            img_name = os.path.splitext(os.path.basename(path))[0]
            out_dir = os.path.join(test_output_dir, defect_class)
            os.makedirs(out_dir, exist_ok=True)
            tifffile.imwrite(os.path.join(out_dir, img_name + '.tiff'), final_map)

            mask = np.zeros((orig_height, orig_width), dtype=np.uint8)
            if defect_class != 'good':
                dataset_dir = os.path.dirname(os.path.dirname(os.path.dirname(path)))

                if dataset_name == 'mvtec_ad':
                    mask_path = os.path.join(
                        dataset_dir, 'ground_truth', defect_class, img_name + '_mask.png')
                    mask = np.array(Image.open(mask_path).convert('L')) > 0
                elif dataset_name == 'visa':
                    mask_path = os.path.join(
                        dataset_dir, 'ground_truth', defect_class, img_name + '.png')
                    mask = np.array(Image.open(mask_path).convert('L')) > 0
                elif dataset_name == 'mvtec_loco':
                    mask_folder = os.path.join(
                        dataset_dir, 'ground_truth', defect_class, img_name)
                    for filename in os.listdir(mask_folder):
                        if filename.endswith('.png'):
                            part_mask = np.array(
                                Image.open(os.path.join(mask_folder, filename)).convert('L')) > 0
                            mask = np.logical_or(mask, part_mask)
                else:
                    raise ValueError(f'Unsupported dataset: {dataset_name}')

                mask = mask.astype(np.uint8)

            pixel_y_true.append(mask.flatten())
            coord_pixel_scores.append(coord_map.flatten())
            final_pixel_scores.append(final_map.flatten())

    def calc_metrics(y_t, y_s):
        auc = roc_auc_score(y_t, y_s) * 100
        ap = average_precision_score(y_t, y_s) * 100
        fpr, tpr, _ = roc_curve(y_t, y_s)
        idx = np.where(tpr >= 0.95)[0]
        fpr95 = (fpr[idx[0]] if len(idx) > 0 else 1.0) * 100
        return auc, ap, fpr95

    coord_img_auc, coord_img_ap, coord_img_fpr = calc_metrics(
        y_true, coord_img_scores)
    final_img_auc, final_img_ap, final_img_fpr = calc_metrics(
        y_true, final_img_scores)

    if test_output_dir is not None:
        pixel_y_true = np.concatenate(pixel_y_true)
        coord_pixel_scores = np.concatenate(coord_pixel_scores)
        final_pixel_scores = np.concatenate(final_pixel_scores)

        coord_pix_auc, coord_pix_ap, _ = calc_metrics(
            pixel_y_true, coord_pixel_scores)
        final_pix_auc, final_pix_ap, _ = calc_metrics(
            pixel_y_true, final_pixel_scores)
    else:
        coord_pix_auc = coord_pix_ap = final_pix_auc = final_pix_ap = 0.0

    print(f"\n{'=' * 50}")
    print(f"[EVALUATION] {desc} Results")
    print(f"{'=' * 50}")
    print('[Image-Level Metrics]')
    print(f" AUROC : Coordinates {coord_img_auc:5.2f}%  ->  Final {final_img_auc:5.2f}%")
    print(f" AP    : Coordinates {coord_img_ap:5.2f}%  ->  Final {final_img_ap:5.2f}%")
    print(f" FPR95 : Coordinates {coord_img_fpr:5.2f}%  ->  Final {final_img_fpr:5.2f}%")

    if test_output_dir is not None:
        print('\n[Pixel-Level Metrics]')
        print(f" AUROC : Coordinates {coord_pix_auc:5.2f}%  ->  Final {final_pix_auc:5.2f}%")
        print(f" AP    : Coordinates {coord_pix_ap:5.2f}%  ->  Final {final_pix_ap:5.2f}%")
    print(f"{'=' * 50}\n")

    return {
        'coord_img_auc': coord_img_auc, 'coord_img_ap': coord_img_ap,
        'final_img_auc': final_img_auc, 'final_img_ap': final_img_ap,
        'coord_pix_auc': coord_pix_auc, 'coord_pix_ap': coord_pix_ap,
        'final_pix_auc': final_pix_auc, 'final_pix_ap': final_pix_ap
    }

@torch.no_grad()
def predict(
    image,
    coords,
    teacher,
    student,
    autoencoder,
    teacher_mean,
    teacher_std,
    q_st_start=None,
    q_st_end=None,
    q_ae_start=None,
    q_ae_end=None,
    coord_scale=1.0,
):
    """Generate anomaly maps from coordinate-conditioned EfficientAD outputs."""
    teacher_output = teacher(image)
    teacher_output = (teacher_output - teacher_mean) / teacher_std

    autoencoder_output = autoencoder(image, coords, coord_scale)
    student_output = student(image, coords, coord_scale)

    map_st = torch.mean(
        (teacher_output - student_output[:, :OUT_CHANNELS]) ** 2,
        dim=1,
        keepdim=True,
    )
    map_ae = torch.mean(
        (
            autoencoder_output
            - student_output[:, OUT_CHANNELS:]
        ) ** 2,
        dim=1,
        keepdim=True,
    )

    if q_st_start is not None:
        map_st = (
            0.1
            * (map_st - q_st_start)
            / (q_st_end - q_st_start)
        )
    if q_ae_start is not None:
        map_ae = (
            0.1
            * (map_ae - q_ae_start)
            / (q_ae_end - q_ae_start)
        )

    map_combined = 0.5 * map_st + 0.5 * map_ae
    return map_combined, map_st, map_ae


@torch.no_grad()
def map_normalization(
    validation_loader,
    teacher,
    student,
    autoencoder,
    teacher_mean,
    teacher_std,
    coord_scale,
    desc='Map normalization',
):
    """Compute validation-set quantiles used to scale anomaly maps."""
    maps_st = []
    maps_ae = []
    num_coord_channels = student.coord_embed[0].in_channels

    prefetcher = CoordinateFieldPrefetcher(
        validation_loader,
        num_coord_channels,
        image_size=IMAGE_SIZE,
        prefetch_depth=5,
    )

    for _ in tqdm(
        range(len(validation_loader.dataset)),
        desc=desc,
        disable=True,
    ):
        batch = prefetcher.next()
        if batch is None:
            break

        image_st, _, coords, _, _ = batch

        _, map_st, map_ae = predict(
            image=image_st,
            coords=coords,
            teacher=teacher,
            student=student,
            autoencoder=autoencoder,
            teacher_mean=teacher_mean,
            teacher_std=teacher_std,
            coord_scale=coord_scale,
        )
        maps_st.append(map_st)
        maps_ae.append(map_ae)

    maps_st = torch.cat(maps_st)
    maps_ae = torch.cat(maps_ae)

    return (
        torch.quantile(maps_st, q=0.9),
        torch.quantile(maps_st, q=0.995),
        torch.quantile(maps_ae, q=0.9),
        torch.quantile(maps_ae, q=0.995),
    )


@torch.no_grad()
def teacher_normalization(teacher, train_loader):
    """Compute channel-wise mean and standard deviation of Teacher features."""
    mean_outputs = []

    for batch_data in tqdm(
        train_loader,
        desc='Computing mean of features',
    ):
        (train_image, _), _, _ = batch_data[0]
        train_image = train_image.unsqueeze(0).cuda()
        mean_outputs.append(
            torch.mean(
                teacher(train_image),
                dim=[0, 2, 3],
            )
        )

    channel_mean = torch.mean(
        torch.stack(mean_outputs),
        dim=0,
    )[None, :, None, None]

    mean_distances = []
    for batch_data in tqdm(
        train_loader,
        desc='Computing std of features',
    ):
        (train_image, _), _, _ = batch_data[0]
        train_image = train_image.unsqueeze(0).cuda()
        distance = (
            teacher(train_image) - channel_mean
        ) ** 2
        mean_distances.append(
            torch.mean(
                distance,
                dim=[0, 2, 3],
            )
        )

    channel_std = torch.sqrt(
        torch.mean(
            torch.stack(mean_distances),
            dim=0,
        )
    )[None, :, None, None]

    return channel_mean, channel_std






if __name__ == '__main__':
    config = get_argparse()

    if config.subdataset.lower() == 'all':
        if config.dataset == 'mvtec_ad':
            dataset_path = config.mvtec_ad_path
        elif config.dataset == 'mvtec_loco':
            dataset_path = config.mvtec_loco_path
        elif config.dataset == 'visa':
            dataset_path = config.visa_path
        else:
            raise ValueError('Unknown config.dataset')

        if not os.path.exists(dataset_path):
            raise FileNotFoundError(
                f'Dataset path does not exist: {dataset_path}'
            )

        subdatasets = [
            directory
            for directory in os.listdir(dataset_path)
            if os.path.isdir(
                os.path.join(dataset_path, directory)
            )
        ]
        print(
            f"[INFO] Found {len(subdatasets)} "
            f"sub-datasets: {subdatasets}"
        )

        for subdataset in subdatasets:
            print(f"\n{'=' * 40}")
            print(
                f"[INFO] Starting training for: {subdataset}"
            )
            print(f"{'=' * 40}\n")
            config.subdataset = subdataset
            main(config)
    else:
        main(config)
