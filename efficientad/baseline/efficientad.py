#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
EfficientAD baseline used in the Blueprint-AD experiments.

This file is adapted from the unofficial EfficientAD implementation by nelson1425:
https://github.com/nelson1425/EfficientAD

The upstream repository is distributed under the Apache License 2.0.
Modifications in this repository include dataset support, evaluation utilities,
and experiment settings used in the Blueprint-AD paper.
"""

import numpy as np
import tifffile
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import argparse
import itertools
import os
import random
from tqdm import tqdm
from PIL import Image
from common import get_autoencoder, get_pdn_small, get_pdn_medium, \
    ImageFolderWithoutTarget, ImageFolderWithPath, InfiniteDataloader
from pathlib import Path

from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve

torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.matmul.allow_tf32 = True
EFFICIENTAD_ROOT = Path(__file__).resolve().parents[1]


def get_argparse():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', default='mvtec_ad',
                        choices=['mvtec_ad', 'mvtec_loco', 'visa'])
    parser.add_argument('-s', '--subdataset', default='cable',
                        help='One of 15 sub-datasets of Mvtec AD or 5 sub-datasets of Mvtec LOCO. Use "all" to train all subsets sequentially.')
    parser.add_argument('-o', '--output_dir', default='output/1')
    parser.add_argument('-m', '--model_size', default='medium',
                        choices=['small', 'medium'])
    parser.add_argument(
        '-w', '--weights',
        default=str(EFFICIENTAD_ROOT / 'models' / 'teacher_medium.pth')
    )
    # The option name ``imagenet_train_path`` is inherited from the upstream
    # EfficientAD implementation. The original implementation uses ImageNet
    # as the external penalty dataset, while the experiments in our paper use
    # DTD consistently for the EfficientAD-M baseline and all variants.
    parser.add_argument(
        '-i', '--imagenet_train_path',
        default='data/dtd/images',
        help=(
            'Path to the external penalty dataset. The option name is inherited '
            'from the upstream EfficientAD implementation. DTD is used in the '
            'experiments reported in our paper. Set to "none" to disable the '
            'penalty dataset.'
        )
    )
    parser.add_argument(
        '-a', '--mvtec_ad_path',
        default='data/mvtec_ad',
        help='Path to the MVTec AD dataset.'
    )
    parser.add_argument(
        '-b', '--mvtec_loco_path',
        default='data/mvtec_loco',
        help='Path to the MVTec LOCO dataset.'
    )
    parser.add_argument(
        '-v', '--visa_path',
        default='data/visa',
        help='Path to the VisA dataset.'
    )
    parser.add_argument('-t', '--train_steps', type=int, default=70000)
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
seed = 42
on_gpu = torch.cuda.is_available()
out_channels = 384
image_size = 256

# Data loading transforms
default_transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
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
    seed_everything(seed)

    # 1. Dataset path configuration
    if config.dataset == 'mvtec_ad':
        dataset_path = config.mvtec_ad_path
    elif config.dataset == 'mvtec_loco':
        dataset_path = config.mvtec_loco_path
    elif config.dataset == 'visa':
        dataset_path = config.visa_path
    else:
        raise Exception('Unknown config.dataset')

    pretrain_penalty = config.imagenet_train_path != 'none'

    # 2. Output directory creation
    train_output_dir = os.path.join(config.output_dir, 'trainings',
                                    config.dataset, config.subdataset)
    test_output_dir = os.path.join(config.output_dir, 'anomaly_maps',
                                   config.dataset, config.subdataset, 'test')
    os.makedirs(train_output_dir, exist_ok=True)
    os.makedirs(test_output_dir, exist_ok=True)

    # 3. Load dataset
    full_train_set = ImageFolderWithoutTarget(
        os.path.join(dataset_path, config.subdataset, 'train'),
        transform=transforms.Lambda(train_transform))
    test_set = ImageFolderWithPath(
        os.path.join(dataset_path, config.subdataset, 'test'))

    if config.dataset in ['mvtec_ad', 'visa']:
        # Use 10% of the training set for validation.
        train_size = int(0.9 * len(full_train_set))
        validation_size = len(full_train_set) - train_size
        rng = torch.Generator().manual_seed(seed)
        train_set, validation_set = torch.utils.data.random_split(
            full_train_set, [train_size, validation_size], rng)
    elif config.dataset == 'mvtec_loco':
        train_set = full_train_set
        validation_set = ImageFolderWithoutTarget(
            os.path.join(dataset_path, config.subdataset, 'validation'),
            transform=transforms.Lambda(train_transform))
    else:
        raise Exception('Unknown config.dataset')

    # 4. DataLoader initialization
    g_train = torch.Generator()
    g_train.manual_seed(seed)
    train_loader = DataLoader(
        train_set, batch_size=1, shuffle=True,
        num_workers=4, pin_memory=True,
        worker_init_fn=seed_worker,
        generator=g_train,
        persistent_workers=True)

    train_loader_infinite = InfiniteDataloader(train_loader)
    validation_loader = DataLoader(validation_set, batch_size=1)

    g_penalty = torch.Generator()
    g_penalty.manual_seed(seed + 1)
    if pretrain_penalty:
        penalty_transform = transforms.Compose([
            transforms.Resize((2 * image_size, 2 * image_size)),
            transforms.RandomGrayscale(0.3),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        penalty_set = ImageFolderWithoutTarget(config.imagenet_train_path,
                                               transform=penalty_transform)
        penalty_loader = DataLoader(
            penalty_set, batch_size=1, shuffle=True,
            num_workers=4, pin_memory=True,
            worker_init_fn=seed_worker,
            generator=g_penalty,
            persistent_workers=True)
        penalty_loader_infinite = InfiniteDataloader(penalty_loader)
    else:
        penalty_loader_infinite = itertools.repeat(None)

    # 5. Model creation
    if config.model_size == 'small':
        teacher = get_pdn_small(out_channels)
        student = get_pdn_small(2 * out_channels)
    elif config.model_size == 'medium':
        teacher = get_pdn_medium(out_channels)
        student = get_pdn_medium(2 * out_channels)
    else:
        raise Exception()

    state_dict = torch.load(config.weights, map_location='cpu')
    teacher.load_state_dict(state_dict)
    autoencoder = get_autoencoder(out_channels)

    teacher.eval()
    student.train()
    autoencoder.train()

    if on_gpu:
        teacher.cuda()
        student.cuda()
        autoencoder.cuda()

    teacher_mean, teacher_std = teacher_normalization(teacher, train_loader)

    optimizer = torch.optim.Adam(itertools.chain(student.parameters(),
                                                 autoencoder.parameters()),
                                 lr=1e-4, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=int(0.95 * config.train_steps), gamma=0.1)

    # 6. Training loop
    tqdm_obj = tqdm(range(config.train_steps), dynamic_ncols=True, position=0, leave=True)
    for iteration, (image_st, image_ae), image_penalty in zip(
            tqdm_obj, train_loader_infinite, penalty_loader_infinite):
        if on_gpu:
            image_st = image_st.cuda()
            image_ae = image_ae.cuda()
            if image_penalty is not None:
                image_penalty = image_penalty.cuda()

        with torch.no_grad():
            teacher_output_st = teacher(image_st)
            teacher_output_st = (teacher_output_st - teacher_mean) / teacher_std

        student_output_st = student(image_st)[:, :out_channels]
        distance_st = (teacher_output_st - student_output_st) ** 2
        d_hard = torch.quantile(distance_st, q=0.999)
        loss_hard = torch.mean(distance_st[distance_st >= d_hard])

        if image_penalty is not None:
            student_output_penalty = student(image_penalty)[:, :out_channels]
            loss_penalty = torch.mean(student_output_penalty ** 2)
            loss_st = loss_hard + loss_penalty
        else:
            loss_st = loss_hard

        ae_output = autoencoder(image_ae)
        with torch.no_grad():
            teacher_output_ae = teacher(image_ae)
            teacher_output_ae = (teacher_output_ae - teacher_mean) / teacher_std

        student_output_ae = student(image_ae)[:, out_channels:]
        distance_ae = (teacher_output_ae - ae_output) ** 2
        distance_stae = (ae_output - student_output_ae) ** 2
        loss_ae = torch.mean(distance_ae)
        loss_stae = torch.mean(distance_stae)

        loss_total = loss_st + loss_ae + loss_stae

        optimizer.zero_grad()
        loss_total.backward()
        optimizer.step()
        scheduler.step()

        if iteration % 10 == 0:
            tqdm_obj.set_description(
                f"Loss Total: {loss_total.item():.4f} | "
                f"ST: {loss_st.item():.4f} | "
                f"AE: {loss_ae.item():.4f} | "
                f"STAE: {loss_stae.item():.4f}"
            )

        if iteration % 1000 == 0:
            torch.save(teacher, os.path.join(train_output_dir, 'teacher_tmp.pth'))
            torch.save(student, os.path.join(train_output_dir, 'student_tmp.pth'))
            torch.save(autoencoder, os.path.join(train_output_dir, 'autoencoder_tmp.pth'))

        # Intermediate evaluation
        if iteration % 10000 == 0 and iteration > 0:
            teacher.eval()
            student.eval()
            autoencoder.eval()

            q_st_start, q_st_end, q_ae_start, q_ae_end = map_normalization(
                validation_loader=validation_loader, teacher=teacher,
                student=student, autoencoder=autoencoder,
                teacher_mean=teacher_mean, teacher_std=teacher_std,
                desc='Intermediate map normalization')

            metrics = test(
                dataset_name=config.dataset,
                test_set=test_set, teacher=teacher, student=student, autoencoder=autoencoder,
                teacher_mean=teacher_mean, teacher_std=teacher_std,
                q_st_start=q_st_start, q_st_end=q_st_end, q_ae_start=q_ae_start, q_ae_end=q_ae_end,
                test_output_dir=None, desc=f'Intermediate Inference (Iter: {iteration})'
            )

            teacher.eval()
            student.train()
            autoencoder.train()

    # 7. Final evaluation
    teacher.eval()
    student.eval()
    autoencoder.eval()

    torch.save(teacher, os.path.join(train_output_dir, 'teacher_final.pth'))
    torch.save(student, os.path.join(train_output_dir, 'student_final.pth'))
    torch.save(autoencoder, os.path.join(train_output_dir, 'autoencoder_final.pth'))

    q_st_start, q_st_end, q_ae_start, q_ae_end = map_normalization(
        validation_loader=validation_loader, teacher=teacher, student=student,
        autoencoder=autoencoder, teacher_mean=teacher_mean,
        teacher_std=teacher_std, desc='Final map normalization')

    metrics = test(
        dataset_name=config.dataset,
        test_set=test_set, teacher=teacher, student=student, autoencoder=autoencoder,
        teacher_mean=teacher_mean, teacher_std=teacher_std,
        q_st_start=q_st_start, q_st_end=q_st_end, q_ae_start=q_ae_start, q_ae_end=q_ae_end,
        test_output_dir=test_output_dir, desc='Final Inference'
    )

    # Release GPU memory
    if on_gpu:
        torch.cuda.empty_cache()


def test(dataset_name, test_set, teacher, student, autoencoder, teacher_mean, teacher_std,
         q_st_start, q_st_end, q_ae_start, q_ae_end, test_output_dir=None,
         desc='Running inference'):
    """
    Evaluate the model on the test set and compute image- and pixel-level metrics.

    Returns:
        dict: AUROC, AP, and FPR95 values for the evaluated outputs.
    """
    y_true = []
    y_score_combined = []
    y_score_st = []
    y_score_ae = []

    # Allocate pixel-level buffers only for the final evaluation.
    if test_output_dir is not None:
        pixel_y_true = []
        pixel_score_combined = []
        pixel_score_st = []
        pixel_score_ae = []

    for image, target, path in tqdm(test_set, desc=desc, disable=True):
        orig_width = image.width
        orig_height = image.height
        image = default_transform(image)
        image = image[None]
        if on_gpu:
            image = image.cuda()

        map_combined, map_st, map_ae = predict(
            image=image, teacher=teacher, student=student,
            autoencoder=autoencoder, teacher_mean=teacher_mean,
            teacher_std=teacher_std, q_st_start=q_st_start, q_st_end=q_st_end,
            q_ae_start=q_ae_start, q_ae_end=q_ae_end)

        # Resize anomaly maps to the original image resolution.
        map_combined = torch.nn.functional.pad(map_combined, (4, 4, 4, 4))
        map_combined = torch.nn.functional.interpolate(
            map_combined, (orig_height, orig_width), mode='bilinear')
        map_combined = map_combined[0, 0].cpu().numpy()

        map_st = torch.nn.functional.pad(map_st, (4, 4, 4, 4))
        map_st = torch.nn.functional.interpolate(
            map_st, (orig_height, orig_width), mode='bilinear')
        map_st = map_st[0, 0].cpu().numpy()

        map_ae = torch.nn.functional.pad(map_ae, (4, 4, 4, 4))
        map_ae = torch.nn.functional.interpolate(
            map_ae, (orig_height, orig_width), mode='bilinear')
        map_ae = map_ae[0, 0].cpu().numpy()

        defect_class = os.path.basename(os.path.dirname(path))

        # Save anomaly maps.
        if test_output_dir is not None:
            img_nm = os.path.split(path)[1].split('.')[0]
            if not os.path.exists(os.path.join(test_output_dir, defect_class)):
                os.makedirs(os.path.join(test_output_dir, defect_class))
            file = os.path.join(test_output_dir, defect_class, img_nm + '.tiff')
            tifffile.imwrite(file, map_combined)

        y_true_image = 0 if defect_class == 'good' else 1
        y_true.append(y_true_image)
        y_score_combined.append(np.max(map_combined))
        y_score_st.append(np.max(map_st))
        y_score_ae.append(np.max(map_ae))

        # Collect pixel-level ground truth and scores only for the final evaluation.
        if test_output_dir is not None:
            img_name = os.path.splitext(os.path.basename(path))[0]
            if defect_class == 'good':
                mask = np.zeros((orig_height, orig_width), dtype=np.uint8)
            else:
                dataset_dir = os.path.dirname(os.path.dirname(os.path.dirname(path)))

                if dataset_name == 'mvtec_ad':
                    mask_path = os.path.join(dataset_dir, 'ground_truth', defect_class, img_name + '_mask.png')
                    mask = np.array(Image.open(mask_path).convert('L'))
                    mask = (mask > 0).astype(np.uint8)

                elif dataset_name == 'visa':
                    mask_path = os.path.join(dataset_dir, 'ground_truth', defect_class, img_name + '.png')
                    mask = np.array(Image.open(mask_path).convert('L'))
                    mask = (mask > 0).astype(np.uint8)

                elif dataset_name == 'mvtec_loco':
                    mask_folder = os.path.join(dataset_dir, 'ground_truth', defect_class, img_name)
                    mask = np.zeros((orig_height, orig_width), dtype=np.uint8)
                    for f in os.listdir(mask_folder):
                        if f.endswith('.png'):
                            part_path = os.path.join(mask_folder, f)
                            part_mask = np.array(Image.open(part_path).convert('L'))
                            mask = np.logical_or(mask, part_mask > 0).astype(np.uint8)
                else:
                    raise ValueError(f"Unsupported dataset format for mask reading: {dataset_name}")

            pixel_y_true.append(mask.flatten())
            pixel_score_combined.append(map_combined.flatten())
            pixel_score_st.append(map_st.flatten())
            pixel_score_ae.append(map_ae.flatten())

    # Compute image-level metrics for every evaluation call.
    auc_img_combined = roc_auc_score(y_true, y_score_combined)
    auc_img_st = roc_auc_score(y_true, y_score_st)
    auc_img_ae = roc_auc_score(y_true, y_score_ae)

    ap_img_combined = average_precision_score(y_true, y_score_combined)
    ap_img_st = average_precision_score(y_true, y_score_st)
    ap_img_ae = average_precision_score(y_true, y_score_ae)

    def get_fpr95(y_true, y_score):
        """Compute FPR at 95% TPR."""
        fpr, tpr, thresholds = roc_curve(y_true, y_score)
        idx = np.where(tpr >= 0.95)[0]
        return fpr[idx[0]] if len(idx) > 0 else 1.0

    fpr95_combined = get_fpr95(y_true, y_score_combined)
    fpr95_st = get_fpr95(y_true, y_score_st)
    fpr95_ae = get_fpr95(y_true, y_score_ae)

    # Compute pixel-level metrics when pixel data are available.
    if test_output_dir is not None:
        pixel_y_true = np.concatenate(pixel_y_true)
        pixel_score_combined = np.concatenate(pixel_score_combined)
        pixel_score_st = np.concatenate(pixel_score_st)
        pixel_score_ae = np.concatenate(pixel_score_ae)

        auc_pix_combined = roc_auc_score(pixel_y_true, pixel_score_combined)
        auc_pix_st = roc_auc_score(pixel_y_true, pixel_score_st)
        auc_pix_ae = roc_auc_score(pixel_y_true, pixel_score_ae)

        ap_pix_combined = average_precision_score(pixel_y_true, pixel_score_combined)
        ap_pix_st = average_precision_score(pixel_y_true, pixel_score_st)
        ap_pix_ae = average_precision_score(pixel_y_true, pixel_score_ae)
    else:
        # Use zero placeholders during intermediate evaluation.
        auc_pix_combined = auc_pix_st = auc_pix_ae = 0.0
        ap_pix_combined = ap_pix_st = ap_pix_ae = 0.0

    # Standard output formatting
    print(f"\n--- {desc} ---")
    print(f"C: {auc_img_combined * 100:.2f} | ST: {auc_img_st * 100:.2f} | AE: {auc_img_ae * 100:.2f}")
    print(f"C: {ap_img_combined * 100:.2f} | ST: {ap_img_st * 100:.2f} | AE: {ap_img_ae * 100:.2f}")
    print(f"C: {fpr95_combined * 100:.2f} | ST: {fpr95_st * 100:.2f} | AE: {fpr95_ae * 100:.2f}")
    print(f"-" * 30)
    if test_output_dir is not None:
        print(f"C: {auc_pix_combined * 100:.2f} | ST: {auc_pix_st * 100:.2f} | AE: {auc_pix_ae * 100:.2f}")
        print(f"C: {ap_pix_combined * 100:.2f} | ST: {ap_pix_st * 100:.2f} | AE: {ap_pix_ae * 100:.2f}")
        print(f"-" * 30)

    return {
        "img_auc": (auc_img_combined * 100, auc_img_st * 100, auc_img_ae * 100),
        "img_ap": (ap_img_combined * 100, ap_img_st * 100, ap_img_ae * 100),
        "img_fpr95": (fpr95_combined * 100, fpr95_st * 100, fpr95_ae * 100),
        "pix_auc": (auc_pix_combined * 100, auc_pix_st * 100, auc_pix_ae * 100),
        "pix_ap": (ap_pix_combined * 100, ap_pix_st * 100, ap_pix_ae * 100)
    }


@torch.no_grad()
def predict(image, teacher, student, autoencoder, teacher_mean, teacher_std,
            q_st_start=None, q_st_end=None, q_ae_start=None, q_ae_end=None):
    """Generate anomaly maps from Teacher-Student and Autoencoder discrepancies."""
    teacher_output = teacher(image)
    teacher_output = (teacher_output - teacher_mean) / teacher_std
    student_output = student(image)
    autoencoder_output = autoencoder(image)

    map_st = torch.mean((teacher_output - student_output[:, :out_channels]) ** 2, dim=1, keepdim=True)
    map_ae = torch.mean((autoencoder_output - student_output[:, out_channels:]) ** 2, dim=1, keepdim=True)

    # Normalize maps using the 0.9 and 0.995 quantiles.
    if q_st_start is not None:
        map_st = 0.1 * (map_st - q_st_start) / (q_st_end - q_st_start)
    if q_ae_start is not None:
        map_ae = 0.1 * (map_ae - q_ae_start) / (q_ae_end - q_ae_start)

    map_combined = 0.5 * map_st + 0.5 * map_ae
    return map_combined, map_st, map_ae


@torch.no_grad()
def map_normalization(validation_loader, teacher, student, autoencoder,
                      teacher_mean, teacher_std, desc='Map normalization'):
    """Compute validation-set quantiles used to scale anomaly maps."""
    maps_st = []
    maps_ae = []
    for image, _ in tqdm(validation_loader, desc=desc, disable=True):
        if on_gpu:
            image = image.cuda()
        _, map_st, map_ae = predict(
            image=image, teacher=teacher, student=student,
            autoencoder=autoencoder, teacher_mean=teacher_mean,
            teacher_std=teacher_std)
        maps_st.append(map_st)
        maps_ae.append(map_ae)

    maps_st = torch.cat(maps_st)
    maps_ae = torch.cat(maps_ae)
    q_st_start = torch.quantile(maps_st, q=0.9)
    q_st_end = torch.quantile(maps_st, q=0.995)
    q_ae_start = torch.quantile(maps_ae, q=0.9)
    q_ae_end = torch.quantile(maps_ae, q=0.995)
    return q_st_start, q_st_end, q_ae_start, q_ae_end


@torch.no_grad()
def teacher_normalization(teacher, train_loader):
    """Compute channel-wise mean and standard deviation of Teacher features."""
    mean_outputs = []
    for train_image, _ in tqdm(train_loader, desc='Computing mean of features', dynamic_ncols=True):
        if on_gpu:
            train_image = train_image.cuda()
        teacher_output = teacher(train_image)
        mean_output = torch.mean(teacher_output, dim=[0, 2, 3])
        mean_outputs.append(mean_output)
    channel_mean = torch.mean(torch.stack(mean_outputs), dim=0)
    channel_mean = channel_mean[None, :, None, None]

    mean_distances = []
    for train_image, _ in tqdm(train_loader, desc='Computing std of features', dynamic_ncols=True):
        if on_gpu:
            train_image = train_image.cuda()
        teacher_output = teacher(train_image)
        distance = (teacher_output - channel_mean) ** 2
        mean_distance = torch.mean(distance, dim=[0, 2, 3])
        mean_distances.append(mean_distance)
    channel_var = torch.mean(torch.stack(mean_distances), dim=0)
    channel_var = channel_var[None, :, None, None]
    channel_std = torch.sqrt(channel_var)

    return channel_mean, channel_std


if __name__ == '__main__':
    config = get_argparse()

    # Train all sub-datasets sequentially when requested.
    if config.subdataset.lower() == 'all':
        if config.dataset == 'mvtec_ad':
            dataset_path = config.mvtec_ad_path
        elif config.dataset == 'mvtec_loco':
            dataset_path = config.mvtec_loco_path
        elif config.dataset == 'visa':
            dataset_path = config.visa_path
        else:
            raise Exception('Unknown config.dataset')

        if not os.path.exists(dataset_path):
            raise Exception(f"Dataset path does not exist: {dataset_path}")

        subdatasets = [d for d in os.listdir(dataset_path)
                       if os.path.isdir(os.path.join(dataset_path, d))]

        print(f"[INFO] Found {len(subdatasets)} sub-datasets: {subdatasets}")

        for sub in subdatasets:
            print(f"\n{'=' * 40}")
            print(f"[INFO] Starting training for: {sub}")
            print(f"{'=' * 40}\n")

            config.subdataset = sub
            main(config)

            print(f"\n{'=' * 40}")
            print(f"[INFO] Finished training for: {sub}")
            print(f"{'=' * 40}\n")
    else:
        main(config)