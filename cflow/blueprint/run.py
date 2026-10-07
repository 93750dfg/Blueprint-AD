# run.py
# -*- coding: utf-8 -*-

import argparse
import gc
import math
import sys
import traceback
from pathlib import Path
from types import SimpleNamespace

import torch

from main import main


IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")

# Unified Blueprint-AD training budget:
# 320 training images, batch size 32 -> 10 batches/meta-epoch.
# 50 meta-epochs therefore correspond to 500 batch-epochs.
REFERENCE_TRAIN_IMAGES = 320
REFERENCE_META_EPOCHS = 50
BATCH_SIZE = 32


class Tee:
    """Write stdout/stderr to both terminal and a per-class log file."""

    def __init__(self, *files):
        self.files = files

    def write(self, text):
        for file in self.files:
            file.write(text)
            file.flush()

    def flush(self):
        for file in self.files:
            file.flush()


def is_image_file(path: Path):
    return path.suffix.lower() in IMAGE_EXTENSIONS


def count_train_images(root: Path, class_name: str):
    """
    Count training images under:
        root / class_name / train / <subfolder> / image
    """
    train_dir = root / class_name / "train"
    if not train_dir.is_dir():
        raise FileNotFoundError(f"Train folder not found: {train_dir}")

    count = 0
    for subdir in train_dir.iterdir():
        if not subdir.is_dir():
            continue
        count += sum(
            1
            for path in subdir.iterdir()
            if path.is_file() and is_image_file(path)
        )

    return count


def get_num_batches(num_images: int, batch_size: int):
    """CFLOW-AD uses drop_last=True, so only complete batches are counted."""
    return num_images // batch_size


def compute_matched_epochs(
    root: Path,
    class_name: str,
    batch_size: int = BATCH_SIZE,
    reference_train_images: int = REFERENCE_TRAIN_IMAGES,
    reference_meta_epochs: int = REFERENCE_META_EPOCHS,
):
    """
    Match batch-level training iterations to the Blueprint-AD reference:
    320 training images trained for 50 meta-epochs.
    """
    reference_batches = get_num_batches(
        reference_train_images, batch_size
    )
    current_images = count_train_images(root, class_name)
    current_batches = get_num_batches(current_images, batch_size)

    if reference_batches <= 0:
        raise ValueError(
            "Invalid training-budget reference: "
            f"{reference_train_images} images with batch size {batch_size}."
        )

    if current_batches <= 0:
        raise ValueError(
            f"No complete training batch for {class_name}: "
            f"{current_images} images with batch size {batch_size}."
        )

    target_batch_epochs = reference_batches * reference_meta_epochs
    matched_epochs = math.ceil(target_batch_epochs / current_batches)

    print(
        "[Epoch Match] "
        f"reference={reference_train_images} images "
        f"=> {reference_batches} batches x {reference_meta_epochs} epochs "
        f"=> target batch-epochs={target_batch_epochs}"
    )
    print(
        "[Epoch Match] "
        f"current={class_name}: {current_images} images "
        f"=> {current_batches} batches "
        f"=> matched meta_epochs={matched_epochs}"
    )

    return max(1, matched_epochs)


def list_dataset_classes(root: Path):
    """Return classes containing both train/ and test/ directories."""
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset root not found: {root}")

    return sorted(
        path.name
        for path in root.iterdir()
        if path.is_dir()
        and (path / "train").is_dir()
        and (path / "test").is_dir()
    )


def get_dataset_root(args):
    if args.dataset == "mvtec":
        return Path(args.mvtec_ad_path)
    if args.dataset == "loco":
        return Path(args.loco_path)
    if args.dataset == "visa":
        return Path(args.visa_path)
    raise ValueError(f"Unsupported dataset: {args.dataset}")


def make_args(args, class_name: str):
    dataset_root = get_dataset_root(args)

    matched_epochs = compute_matched_epochs(
        root=dataset_root,
        class_name=class_name,
        batch_size=BATCH_SIZE,
    )

    return SimpleNamespace(
        # action
        action_type="norm-train",

        # dataset
        dataset=args.dataset,
        class_name=class_name,
        mvtec_ad_path=args.mvtec_ad_path,
        loco_path=args.loco_path,
        visa_path=args.visa_path,
        video_path="",

        # original CFLOW-AD configuration
        enc_arch="wide_resnet50_2",
        dec_arch="freia-cflow",
        pool_layers=3,
        coupling_blocks=8,
        input_size=256,

        # Base positional encoding dimension.
        # train.py expands this to 128 + coordinate-embedding dimension.
        condition_vec=128,

        # training
        batch_size=BATCH_SIZE,
        meta_epochs=matched_epochs,
        sub_epochs=8,
        lr=2e-4,

        # Blueprint-AD
        workspace_dir=args.workspace_dir,
        conf_multiplier=0.68,
        apply_logic_penalty=True,

        # runtime
        gpu=args.gpu,
        no_cuda=False,
        workers=4,
        seed=0,

        # output / evaluation
        run_name=f"cflow_blueprint_{args.dataset}_256_matched_steps",
        pro=True,
        viz=True,
        checkpoint="",
        save_raw_anomaly_maps=True,
        save_raw_maps_original_size=False,
        raw_anomaly_maps_dir=str(
            Path(args.raw_maps_dir) / args.dataset
        ),
        export_legacy_viz=False,
    )


def run_one_class(
    args,
    class_name: str,
    continue_on_error: bool = True,
):
    log_dir = Path(args.log_dir) / args.dataset
    log_dir.mkdir(parents=True, exist_ok=True)

    log_path = log_dir / f"{class_name}.log"

    old_stdout = sys.stdout
    old_stderr = sys.stderr

    with open(log_path, "w", encoding="utf-8") as log_file:
        sys.stdout = Tee(old_stdout, log_file)
        sys.stderr = Tee(old_stderr, log_file)

        try:
            print("\n" + "=" * 100)
            print(
                "Running Blueprint-AD + CFLOW-AD | "
                f"dataset={args.dataset} | "
                f"class={class_name} | input=256x256"
            )
            print("=" * 100 + "\n")

            config = make_args(args, class_name)
            main(config)

        except Exception:
            print("\n" + "!" * 100)
            print(f"Error while running class: {class_name}")
            print("!" * 100 + "\n")
            traceback.print_exc()

            if not continue_on_error:
                raise

        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    print(f"\nFinished {class_name}. Log saved to: {log_path}\n")


def get_argparse():
    parser = argparse.ArgumentParser(
        description="Unified launcher for Blueprint-AD + CFLOW-AD."
    )
    parser.add_argument(
        "--dataset",
        default="mvtec",
        choices=["mvtec", "loco", "visa"],
        help="Dataset to run."
    )
    parser.add_argument(
        "--classes",
        nargs="+",
        default=["cable"],
        help='Class names to run, or use "all" to run every discovered class.'
    )

    parser.add_argument(
        "--mvtec_ad_path",
        default="data/mvtec_ad",
        help="Path to the MVTec AD dataset."
    )
    parser.add_argument(
        "--loco_path",
        default="data/mvtec_loco",
        help="Path to the MVTec LOCO dataset."
    )
    parser.add_argument(
        "--visa_path",
        default="data/visa",
        help="Path to the VisA dataset."
    )
    parser.add_argument(
        "--workspace_dir",
        default="runs",
        help=(
            "Root directory containing the precomputed Blueprint-AD "
            "products for each category."
        )
    )

    parser.add_argument(
        "--gpu",
        default="0",
        help="GPU index passed to CFLOW-AD."
    )
    parser.add_argument(
        "--log_dir",
        default="logs/cflow_blueprint_256",
        help="Directory for per-class logs."
    )
    parser.add_argument(
        "--raw_maps_dir",
        default="output/raw_anomaly_maps/cflow_blueprint_256",
        help="Directory for raw anomaly maps."
    )
    parser.add_argument(
        "--stop_on_error",
        action="store_true",
        help="Stop immediately if one class fails."
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = get_argparse()
    dataset_root = get_dataset_root(args)

    if len(args.classes) == 1 and args.classes[0].lower() == "all":
        classes_to_run = list_dataset_classes(dataset_root)
    else:
        classes_to_run = args.classes

    print(f"Dataset: {args.dataset}")
    print(f"Classes: {classes_to_run}")

    for class_name in classes_to_run:
        run_one_class(
            args,
            class_name,
            continue_on_error=not args.stop_on_error,
        )
