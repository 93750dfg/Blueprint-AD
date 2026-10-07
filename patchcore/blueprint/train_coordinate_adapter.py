import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.decomposition import IncrementalPCA
from torch.utils.data import DataLoader

import patchcore.backbones
from patchcore.coords import build_yolo_cache, PatchCoreCoordinatePrefetcher
from patchcore.datasets.mvtec import DatasetSplit, MVTecDataset
from patchcore.patchcore import CoordinatePCAAdapter, PatchCore


ADAPTER_FILENAME = "coordinate_adapter.pth"


def _resolve_yolo_weights(workspace_dir):
    workspace_dir = Path(workspace_dir)

    direct = workspace_dir / "best.pt"
    legacy = workspace_dir / "yolo_output" / "weights" / "best.pt"

    if direct.is_file():
        return direct
    if legacy.is_file():
        return legacy

    raise FileNotFoundError(
        "YOLO pose checkpoint was not found. Tried:\n"
        f"  {direct}\n"
        f"  {legacy}"
    )


def _load_blueprint_metadata(workspace_dir):
    workspace_dir = Path(workspace_dir)

    calibration_path = workspace_dir / "final_anisotropic_calibration.json"
    mapping_path = workspace_dir / "global_classes_mapping.json"

    if not calibration_path.is_file():
        raise FileNotFoundError(
            f"Missing Blueprint-AD calibration file: {calibration_path}"
        )
    if not mapping_path.is_file():
        raise FileNotFoundError(
            f"Missing Blueprint-AD class mapping: {mapping_path}"
        )

    with calibration_path.open("r", encoding="utf-8") as f:
        calibration = json.load(f)
    with mapping_path.open("r", encoding="utf-8") as f:
        mapping = json.load(f)

    return calibration, mapping


def _build_class_channel_map(calibration, mapping):
    class_channel_map = {}
    current_channel = 0

    for class_name in sorted(calibration.keys()):
        meta = mapping.get(class_name, {})
        if isinstance(meta, dict):
            expected_kpts = meta.get("kpt_num", 3)
        else:
            expected_kpts = 3

        num_channels = 4 if expected_kpts == 3 else (2 if expected_kpts == 2 else 1)

        class_channel_map[class_name] = {
            "start": current_channel,
            "end": current_channel + num_channels,
            "expected_kpts": expected_kpts,
        }
        current_channel += num_channels

    return class_channel_map, current_channel


def _coordinate_collate(batch):
    return {
        "image": torch.stack([item["image"] for item in batch]),
        "yolo_detections": [item.get("yolo_detections", []) for item in batch],
        "render_params": [item.get("render_params", []) for item in batch],
        "orig_size": [item.get("orig_size", (0, 0)) for item in batch],
    }


def fit_patchcore_pca(
    patchcore_instance,
    dataloader,
    num_coord_channels,
    device,
    n_components=64,
):
    """Fit PCA on normal 1024-D PatchCore training descriptors."""

    pca = IncrementalPCA(n_components=n_components)

    feature_buffer = []
    buffered_rows = 0
    partial_fit_rows = 8192

    prefetcher = PatchCoreCoordinatePrefetcher(
        dataloader,
        num_coord_channels,
    )

    while True:
        batch = prefetcher.next()
        if batch is None:
            break

        image = batch["image"].to(
            device=device,
            dtype=torch.float32,
        )

        with torch.no_grad():
            features_1024 = patchcore_instance._embed(
                image,
                detach=False,
            )

        features_np = (
            features_1024
            .detach()
            .cpu()
            .numpy()
            .astype(np.float32)
        )

        feature_buffer.append(features_np)
        buffered_rows += features_np.shape[0]

        if buffered_rows >= partial_fit_rows:
            block = np.concatenate(feature_buffer, axis=0)
            pca.partial_fit(block)
            feature_buffer.clear()
            buffered_rows = 0

    if buffered_rows >= n_components:
        block = np.concatenate(feature_buffer, axis=0)
        pca.partial_fit(block)

    if not hasattr(pca, "components_"):
        raise RuntimeError(
            "PCA could not be fitted. The training set produced fewer than "
            f"{n_components} patch descriptors."
        )

    print(
        "[PCA] Explained variance ratio:",
        float(pca.explained_variance_ratio_.sum()),
    )

    return pca


def train_coordinate_adapter(
    dataset_path,
    dataset_name,
    class_name,
    workspace_dir,
    gpu=0,
    epochs=40,
    batch_size=4,
    output_filename=ADAPTER_FILENAME,
):
    """Train the 64-D Blueprint-AD coordinate adapter for one category."""

    if not torch.cuda.is_available():
        raise RuntimeError(
            "Blueprint PatchCore coordinate rendering currently requires CUDA."
        )

    device = torch.device(f"cuda:{gpu}")
    workspace_dir = Path(workspace_dir)
    workspace_dir.mkdir(parents=True, exist_ok=True)

    calibration, mapping = _load_blueprint_metadata(workspace_dir)
    yolo_weights = _resolve_yolo_weights(workspace_dir)

    class_channel_map, num_coord_channels = _build_class_channel_map(
        calibration,
        mapping,
    )

    id_to_name = {
        meta["class_id"]: name
        for name, meta in mapping.items()
        if isinstance(meta, dict) and "class_id" in meta
    }

    train_dir = Path(dataset_path) / class_name / "train"
    yolo_cache = build_yolo_cache(
        str(yolo_weights),
        [str(train_dir)],
        mapping,
    )

    train_dataset = MVTecDataset(
        dataset_path,
        classname=class_name,
        resize=256,
        imagesize=224,
        split=DatasetSplit.TRAIN,
        resize_strategy=dataset_name,
    )

    train_dataset.yolo_cache = yolo_cache
    train_dataset.calib_data = calibration
    train_dataset.class_channel_map = class_channel_map
    train_dataset.id_to_name = id_to_name

    dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        collate_fn=_coordinate_collate,
    )

    # Original PatchCore descriptors provide the supervision target.
    patchcore_instance = PatchCore(device)
    patchcore_instance.load(
        backbone=patchcore.backbones.load("wideresnet50"),
        layers_to_extract_from=["layer2", "layer3"],
        device=device,
        input_shape=(3, 224, 224),
        pretrain_embed_dimension=1024,
        target_embed_dimension=1024,
        patchsize=3,
        patchstride=1,
        anomaly_scorer_num_nn=1,
    )
    patchcore_instance.forward_modules.eval()

    pca = fit_patchcore_pca(
        patchcore_instance=patchcore_instance,
        dataloader=dataloader,
        num_coord_channels=num_coord_channels,
        device=device,
        n_components=64,
    )

    pca_mean = torch.from_numpy(
        pca.mean_.astype(np.float32)
    ).to(device)

    pca_components = torch.from_numpy(
        pca.components_.astype(np.float32)
    ).to(device)

    adapter = CoordinatePCAAdapter(
        in_channels=num_coord_channels,
        embed_dim=64,
    ).to(device)

    optimizer = torch.optim.Adam(
        adapter.parameters(),
        lr=1e-3,
        weight_decay=1e-4,
    )

    for epoch in range(epochs):
        adapter.train()
        total_loss = 0.0
        num_batches = 0

        prefetcher = PatchCoreCoordinatePrefetcher(
            dataloader,
            num_coord_channels,
        )

        while True:
            batch = prefetcher.next()
            if batch is None:
                break

            image = batch["image"].to(
                device=device,
                dtype=torch.float32,
            )
            coord_fields = batch["coords"].to(
                device=device,
                dtype=torch.float32,
            )

            current_batch_size = image.shape[0]

            with torch.no_grad():
                target_1024 = patchcore_instance._embed(
                    image,
                    detach=False,
                )

                target_64 = (
                    target_1024 - pca_mean
                ) @ pca_components.t()

                target_64 = (
                    target_64
                    .reshape(current_batch_size, 28, 28, 64)
                    .permute(0, 3, 1, 2)
                    .contiguous()
                )

            coord_28x28 = F.interpolate(
                coord_fields,
                size=(28, 28),
                mode="area",
            )

            pred_64 = adapter(coord_28x28)

            loss_reg = F.smooth_l1_loss(
                pred_64,
                target_64,
                beta=1.0,
            )

            pred_flat = (
                pred_64
                .permute(0, 2, 3, 1)
                .reshape(-1, 64)
            )
            target_flat = (
                target_64
                .permute(0, 2, 3, 1)
                .reshape(-1, 64)
            )

            loss_cos = (
                1.0
                - F.cosine_similarity(
                    pred_flat,
                    target_flat,
                    dim=1,
                    eps=1e-8,
                ).mean()
            )

            loss = loss_reg + 0.1 * loss_cos

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        mean_loss = total_loss / max(num_batches, 1)
        print(
            f"[Coordinate Adapter] "
            f"Epoch [{epoch + 1}/{epochs}] "
            f"Loss: {mean_loss:.6f}"
        )

    save_path = workspace_dir / output_filename

    torch.save(
        {
            "model_state": adapter.state_dict(),
            "in_channels": num_coord_channels,
            "embed_dim": 64,
            # Stored for reproducibility/analysis. Inference only requires
            # the trained coordinate adapter itself.
            "pca_mean": pca.mean_.astype(np.float32),
            "pca_components": pca.components_.astype(np.float32),
            "explained_variance_ratio": (
                pca.explained_variance_ratio_.astype(np.float32)
            ),
        },
        save_path,
    )

    print(f"[SUCCESS] Coordinate adapter saved to: {save_path}")
    return save_path


if __name__ == "__main__":
    raise SystemExit(
        "Use patchcore/blueprint/run.py so the adapter and PatchCore "
        "pipeline use the same dataset/category configuration."
    )
