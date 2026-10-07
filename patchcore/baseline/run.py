import argparse
import sys
from pathlib import Path


BASELINE_ROOT = Path(__file__).resolve().parent
SRC_ROOT = BASELINE_ROOT / "src"
BIN_ROOT = BASELINE_ROOT / "bin"

# Keep the upstream PatchCore package layout without requiring installation.
for path in (SRC_ROOT, BIN_ROOT):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from run_patchcore import main as patchcore_main


BACKBONE = "wideresnet50"
LAYERS = ("layer2", "layer3")

RESIZE = 256
IMAGE_SIZE = 224
BATCH_SIZE = 2
NUM_WORKERS = 0

PRETRAIN_EMBED_DIM = 1024
TARGET_EMBED_DIM = 1024
PATCH_SIZE = 3
NUM_NEAREST_NEIGHBOURS = 1

SAMPLER_NAME = "approx_greedy_coreset"
CORESET_PERCENTAGE = 0.1

DEFAULT_SEED = 42
LOG_PROJECT = "PatchCore_Results"


def dataset_root(args):
    if args.dataset == "mvtec":
        return Path(args.mvtec_ad_path)
    if args.dataset == "visa":
        return Path(args.visa_path)
    if args.dataset == "loco":
        return Path(args.loco_path)
    raise ValueError(f"Unsupported dataset: {args.dataset}")


def discover_classes(root):
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset root not found: {root}")

    classes = sorted(
        item.name
        for item in root.iterdir()
        if item.is_dir()
        and (item / "train").is_dir()
        and (item / "test").is_dir()
    )

    if not classes:
        raise RuntimeError(
            f"No PatchCore-compatible category folders found under: {root}"
        )

    return classes


def build_click_arguments(args, classes):
    root = dataset_root(args)

    click_args = [
        "--gpu",
        str(args.gpu),
        "--seed",
        str(args.seed),
        "--log_group",
        f"baseline_{args.dataset}",
        "--log_project",
        LOG_PROJECT,
    ]

    if args.save_model:
        click_args.append("--save_patchcore_model")

    if args.save_anomaly_maps:
        click_args.append("--save_segmentation_images")

    click_args.append(str(Path(args.results_path)))

    # PatchCore configuration.
    click_args.append("patch_core")
    click_args.extend(["-b", BACKBONE])

    for layer in LAYERS:
        click_args.extend(["-le", layer])

    click_args.extend(
        [
            "--pretrain_embed_dimension",
            str(PRETRAIN_EMBED_DIM),
            "--target_embed_dimension",
            str(TARGET_EMBED_DIM),
            "--anomaly_scorer_num_nn",
            str(NUM_NEAREST_NEIGHBOURS),
            "--patchsize",
            str(PATCH_SIZE),
        ]
    )

    if args.faiss_on_gpu:
        click_args.append("--faiss_on_gpu")

    # Coreset sampler.
    click_args.extend(
        [
            "sampler",
            "-p",
            str(CORESET_PERCENTAGE),
            SAMPLER_NAME,
        ]
    )

    # Unified dataset loader.
    click_args.extend(
        [
            "dataset",
            "--resize",
            str(RESIZE),
            "--imagesize",
            str(IMAGE_SIZE),
            "--batch_size",
            str(BATCH_SIZE),
            "--num_workers",
            str(NUM_WORKERS),
        ]
    )

    for class_name in classes:
        click_args.extend(["-d", class_name])

    click_args.extend([args.dataset, str(root)])
    return click_args


def parse_args():
    parser = argparse.ArgumentParser(
        description="Unified launcher for the PatchCore baseline."
    )

    parser.add_argument(
        "--dataset",
        choices=["mvtec", "visa", "loco"],
        default="mvtec",
    )
    parser.add_argument(
        "--classes",
        nargs="+",
        default=["cable"],
        help='Category names, or "all" to discover every category.',
    )

    parser.add_argument(
        "--mvtec_ad_path",
        default="data/mvtec_ad",
    )
    parser.add_argument(
        "--visa_path",
        default="data/visa",
    )
    parser.add_argument(
        "--loco_path",
        default="data/mvtec_loco",
    )

    parser.add_argument(
        "--results_path",
        default="output/patchcore_baseline",
    )
    parser.add_argument(
        "--gpu",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
    )

    parser.add_argument(
        "--faiss_on_gpu",
        action="store_true",
        help="Use FAISS GPU search. CPU FAISS is the default used in the experiments.",
    )
    parser.add_argument(
        "--save_model",
        action="store_true",
        help="Save the fitted PatchCore memory bank and metadata.",
    )
    parser.add_argument(
        "--save_anomaly_maps",
        action="store_true",
        help="Save .npy anomaly maps and grayscale .png previews.",
    )

    return parser.parse_args()


def main():
    args = parse_args()
    root = dataset_root(args)

    if len(args.classes) == 1 and args.classes[0].lower() == "all":
        classes = discover_classes(root)
    else:
        classes = args.classes

    print("=" * 80)
    print("PatchCore baseline")
    print("=" * 80)
    print("Dataset:", args.dataset)
    print("Dataset path:", root)
    print("Classes:", classes)
    print("Backbone:", BACKBONE)
    print("Layers:", list(LAYERS))
    print("Input size:", IMAGE_SIZE)
    print("Coreset percentage:", CORESET_PERCENTAGE)
    print("Nearest neighbours:", NUM_NEAREST_NEIGHBOURS)
    print("FAISS on GPU:", args.faiss_on_gpu)
    print("Seed:", args.seed)
    print("=" * 80)

    if not root.is_dir():
        raise FileNotFoundError(f"Dataset path does not exist: {root}")

    cli_arguments = build_click_arguments(args, classes)
    patchcore_main(args=cli_arguments, standalone_mode=False)


if __name__ == "__main__":
    main()
