"""Anomaly detection evaluation metrics."""

import numpy as np
from scipy import ndimage
from sklearn import metrics


def _prepare_image_inputs(
    anomaly_prediction_weights,
    anomaly_ground_truth_labels,
):
    """Convert image-level predictions and labels to aligned 1-D arrays."""

    anomaly_prediction_weights = np.asarray(
        anomaly_prediction_weights,
        dtype=np.float64,
    ).reshape(-1)

    anomaly_ground_truth_labels = np.asarray(
        anomaly_ground_truth_labels,
        dtype=np.uint8,
    ).reshape(-1)

    if (
        anomaly_prediction_weights.shape
        != anomaly_ground_truth_labels.shape
    ):
        raise ValueError(
            "Image score/label shape mismatch: "
            f"scores={anomaly_prediction_weights.shape}, "
            f"labels={anomaly_ground_truth_labels.shape}"
        )

    unique_labels = np.unique(anomaly_ground_truth_labels)
    if unique_labels.size < 2:
        raise ValueError(
            "Image-level AUROC/AP requires both normal and anomalous "
            f"samples, but labels contain only: {unique_labels.tolist()}"
        )

    return anomaly_prediction_weights, anomaly_ground_truth_labels


def _prepare_pixel_inputs(
    anomaly_segmentations,
    ground_truth_masks,
):
    """
    Convert pixel predictions and masks to aligned [N, H, W] arrays.

    PatchCore currently returns:
        anomaly_segmentations: [N, H, W]
        ground_truth_masks:    [N, 1, H, W]

    This function removes the singleton channel dimension.
    """

    if isinstance(anomaly_segmentations, list):
        anomaly_segmentations = np.stack(anomaly_segmentations)
    else:
        anomaly_segmentations = np.asarray(anomaly_segmentations)

    if isinstance(ground_truth_masks, list):
        ground_truth_masks = np.stack(ground_truth_masks)
    else:
        ground_truth_masks = np.asarray(ground_truth_masks)

    # Accept [N, 1, H, W].
    if anomaly_segmentations.ndim == 4:
        if anomaly_segmentations.shape[1] != 1:
            raise ValueError(
                "Expected one anomaly-map channel, but received shape "
                f"{anomaly_segmentations.shape}"
            )
        anomaly_segmentations = anomaly_segmentations[:, 0]

    if ground_truth_masks.ndim == 4:
        if ground_truth_masks.shape[1] != 1:
            raise ValueError(
                "Expected one ground-truth mask channel, but received shape "
                f"{ground_truth_masks.shape}"
            )
        ground_truth_masks = ground_truth_masks[:, 0]

    # Also permit one standalone [H, W] image.
    if anomaly_segmentations.ndim == 2:
        anomaly_segmentations = anomaly_segmentations[None, ...]

    if ground_truth_masks.ndim == 2:
        ground_truth_masks = ground_truth_masks[None, ...]

    if anomaly_segmentations.ndim != 3:
        raise ValueError(
            "Anomaly maps must have shape [N, H, W], but received "
            f"{anomaly_segmentations.shape}"
        )

    if ground_truth_masks.ndim != 3:
        raise ValueError(
            "Ground-truth masks must have shape [N, H, W], but received "
            f"{ground_truth_masks.shape}"
        )

    if anomaly_segmentations.shape != ground_truth_masks.shape:
        raise ValueError(
            "Pixel prediction/mask shape mismatch: "
            f"predictions={anomaly_segmentations.shape}, "
            f"masks={ground_truth_masks.shape}"
        )

    anomaly_segmentations = anomaly_segmentations.astype(
        np.float64,
        copy=False,
    )

    # Resize operations can introduce fractional boundary values.
    # Convert the GT masks back to binary masks.
    ground_truth_masks = (
        ground_truth_masks > 0.5
    ).astype(np.uint8, copy=False)

    return anomaly_segmentations, ground_truth_masks


def compute_imagewise_retrieval_metrics(
    anomaly_prediction_weights,
    anomaly_ground_truth_labels,
    target_tpr=0.95,
):
    """
    Compute image-level AUROC, AP and FPR@95TPR.

    Args:
        anomaly_prediction_weights:
            Array-like [N]. Higher means more anomalous.

        anomaly_ground_truth_labels:
            Array-like [N]. 0 means normal and 1 means anomalous.

        target_tpr:
            Recall/TPR operating point used for FPR95.

    Returns:
        Dictionary containing:
            auroc
            ap
            fpr95
            fpr
            tpr
            threshold
    """

    scores, labels = _prepare_image_inputs(
        anomaly_prediction_weights,
        anomaly_ground_truth_labels,
    )

    fpr, tpr, thresholds = metrics.roc_curve(
        labels,
        scores,
    )

    auroc = metrics.roc_auc_score(
        labels,
        scores,
    )

    average_precision = metrics.average_precision_score(
        labels,
        scores,
    )

    # First ROC operating point that reaches at least 95% TPR.
    valid_indices = np.flatnonzero(tpr >= target_tpr)

    if len(valid_indices) == 0:
        fpr95 = 1.0
    else:
        fpr95 = float(fpr[valid_indices[0]])

    return {
        "auroc": float(auroc),
        "ap": float(average_precision),
        "fpr95": fpr95,
        "fpr": fpr,
        "tpr": tpr,
        "threshold": thresholds,
    }


def compute_pixelwise_retrieval_metrics(
    anomaly_segmentations,
    ground_truth_masks,
):
    """
    Compute pixel-level AUROC and AP over all test-image pixels.

    Args:
        anomaly_segmentations:
            Predicted anomaly maps with shape [N, H, W].

        ground_truth_masks:
            Binary ground-truth masks with shape [N, H, W] or
            [N, 1, H, W].

    Returns:
        Dictionary containing:
            auroc
            ap
            fpr
            tpr
            threshold
    """

    anomaly_segmentations, ground_truth_masks = _prepare_pixel_inputs(
        anomaly_segmentations,
        ground_truth_masks,
    )

    flat_scores = anomaly_segmentations.reshape(-1)
    flat_labels = ground_truth_masks.reshape(-1)

    unique_labels = np.unique(flat_labels)
    if unique_labels.size < 2:
        raise ValueError(
            "Pixel-level AUROC/AP requires both normal and anomalous "
            f"pixels, but labels contain only: {unique_labels.tolist()}"
        )

    fpr, tpr, thresholds = metrics.roc_curve(
        flat_labels,
        flat_scores,
    )

    auroc = metrics.roc_auc_score(
        flat_labels,
        flat_scores,
    )

    average_precision = metrics.average_precision_score(
        flat_labels,
        flat_scores,
    )

    return {
        "auroc": float(auroc),
        "ap": float(average_precision),
        "fpr": fpr,
        "tpr": tpr,
        "threshold": thresholds,
    }


def compute_aupro(
    anomaly_segmentations,
    ground_truth_masks,
    fpr_limit=0.30,
):
    """
    Compute normalized Area Under the Per-Region Overlap curve.

    The procedure is:

        1. Find connected anomalous regions in every GT mask.
        2. For each threshold, compute the mean recall/overlap of all regions.
        3. Compute global pixel FPR over normal/background pixels.
        4. Retain the curve for FPR <= fpr_limit.
        5. Integrate and normalize by fpr_limit.

    Args:
        anomaly_segmentations:
            Predicted anomaly maps [N, H, W].

        ground_truth_masks:
            Ground-truth masks [N, H, W] or [N, 1, H, W].

        fpr_limit:
            Upper FPR integration limit. Standard MVTec-style AUPRO
            commonly uses 0.30.

    Returns:
        Dictionary containing:
            aupro
            fpr
            pro
            fpr_limit
            num_regions
    """

    if not 0.0 < fpr_limit <= 1.0:
        raise ValueError(
            f"fpr_limit must lie in (0, 1], but received {fpr_limit}"
        )

    anomaly_segmentations, ground_truth_masks = _prepare_pixel_inputs(
        anomaly_segmentations,
        ground_truth_masks,
    )

    # Use 8-connectivity for 2-D connected components.
    connectivity = ndimage.generate_binary_structure(
        rank=2,
        connectivity=2,
    )

    component_labels = np.zeros_like(
        ground_truth_masks,
        dtype=np.int64,
    )

    # Make connected-component IDs globally unique across images.
    region_offset = 0

    for image_index, mask in enumerate(ground_truth_masks):
        labels, num_regions = ndimage.label(
            mask,
            structure=connectivity,
        )

        labels = labels.astype(np.int64, copy=False)

        if num_regions > 0:
            foreground = labels > 0
            labels[foreground] += region_offset
            region_offset += int(num_regions)

        component_labels[image_index] = labels

    num_regions = region_offset

    if num_regions == 0:
        raise ValueError(
            "AUPRO cannot be computed because no anomalous GT regions "
            "were found."
        )

    labels_flat = component_labels.reshape(-1)
    scores_flat = anomaly_segmentations.reshape(-1)

    background_mask = labels_flat == 0
    num_background_pixels = int(background_mask.sum())

    if num_background_pixels == 0:
        raise ValueError(
            "AUPRO cannot be computed because no normal/background "
            "pixels were found."
        )

    # Number of pixels in every connected region.
    # Index 0 corresponds to background.
    region_sizes = np.bincount(
        labels_flat,
        minlength=num_regions + 1,
    ).astype(np.float64)

    # Every background pixel contributes equally to global FPR.
    fp_increment = background_mask.astype(np.float64)

    # Every foreground pixel contributes 1 / region_size to its region's
    # overlap. Dividing the cumulative sum by num_regions gives the mean
    # per-region overlap.
    pro_increment = np.zeros_like(
        scores_flat,
        dtype=np.float64,
    )

    foreground_mask = labels_flat > 0
    foreground_region_ids = labels_flat[foreground_mask]

    pro_increment[foreground_mask] = (
        1.0 / region_sizes[foreground_region_ids]
    )

    # Sweep the threshold from highest anomaly score to lowest.
    sorted_indices = np.argsort(
        -scores_flat,
        kind="mergesort",
    )

    sorted_scores = scores_flat[sorted_indices]

    fpr = np.cumsum(
        fp_increment[sorted_indices]
    ) / num_background_pixels

    pro = np.cumsum(
        pro_increment[sorted_indices]
    ) / num_regions

    # Keep one point after all pixels sharing the same score have entered.
    keep = np.ones(
        sorted_scores.shape[0],
        dtype=bool,
    )
    keep[:-1] = sorted_scores[:-1] != sorted_scores[1:]

    fpr = fpr[keep]
    pro = pro[keep]

    # Empty prediction at a threshold above the maximum score.
    fpr = np.concatenate(([0.0], fpr))
    pro = np.concatenate(([0.0], pro))

    # Keep all points up to fpr_limit.
    upper_index = np.searchsorted(
        fpr,
        fpr_limit,
        side="right",
    )

    clipped_fpr = fpr[:upper_index]
    clipped_pro = pro[:upper_index]

    # Add an interpolated point exactly at fpr_limit.
    if clipped_fpr[-1] < fpr_limit:
        next_index = upper_index

        if next_index < len(fpr):
            left_fpr = clipped_fpr[-1]
            left_pro = clipped_pro[-1]

            right_fpr = fpr[next_index]
            right_pro = pro[next_index]

            interpolation_ratio = (
                (fpr_limit - left_fpr)
                / (right_fpr - left_fpr)
            )

            pro_at_limit = (
                left_pro
                + interpolation_ratio
                * (right_pro - left_pro)
            )
        else:
            pro_at_limit = clipped_pro[-1]

        clipped_fpr = np.append(
            clipped_fpr,
            fpr_limit,
        )
        clipped_pro = np.append(
            clipped_pro,
            pro_at_limit,
        )

    # Normalize by the FPR integration interval so the result is [0, 1].
    aupro = metrics.auc(
        clipped_fpr,
        clipped_pro,
    ) / fpr_limit

    return {
        "aupro": float(aupro),
        "fpr": clipped_fpr,
        "pro": clipped_pro,
        "fpr_limit": float(fpr_limit),
        "num_regions": int(num_regions),
    }