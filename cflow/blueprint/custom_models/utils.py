import math
import os

import numpy as np
import torch


RESULT_DIR = "./results"
WEIGHT_DIR = "./weights"
MODEL_DIR = "./models"

__all__ = (
    "save_results",
    "save_weights",
    "load_weights",
    "adjust_learning_rate",
    "warmup_learning_rate",
)

try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url


def save_results(
    det_roc_obs,
    seg_roc_obs,
    seg_pro_obs,
    model_name,
    class_name,
    run_date,
):
    result = (
        "{:.2f},{:.2f},{:.2f} \t\tfor {:s}/{:s}/{:s} "
        "at epoch {:d}/{:d}/{:d} for {:s}\n"
    ).format(
        det_roc_obs.max_score,
        seg_roc_obs.max_score,
        seg_pro_obs.max_score,
        det_roc_obs.name,
        seg_roc_obs.name,
        seg_pro_obs.name,
        det_roc_obs.max_epoch,
        seg_roc_obs.max_epoch,
        seg_pro_obs.max_epoch,
        class_name,
    )

    os.makedirs(RESULT_DIR, exist_ok=True)
    result_path = os.path.join(
        RESULT_DIR, "{}_{}.txt".format(model_name, run_date)
    )
    with open(result_path, "w", encoding="utf-8") as fp:
        fp.write(result)


def save_weights(
    encoder,
    decoders,
    model_name,
    run_date,
    coord_embedder=None,
):
    os.makedirs(WEIGHT_DIR, exist_ok=True)

    state = {
        "encoder_state_dict": encoder.state_dict(),
        "decoder_state_dict": [
            decoder.state_dict() for decoder in decoders
        ],
    }

    if coord_embedder is not None:
        state["coord_embedder_state_dict"] = coord_embedder.state_dict()

    filename = "{}_{}.pt".format(model_name, run_date)
    path = os.path.join(WEIGHT_DIR, filename)
    torch.save(state, path)
    print("Saving weights to {}".format(filename))


def load_weights(
    encoder,
    decoders,
    filename,
    coord_embedder=None,
):
    state = torch.load(filename)

    encoder.load_state_dict(
        state["encoder_state_dict"], strict=False
    )
    for decoder, decoder_state in zip(
        decoders, state["decoder_state_dict"]
    ):
        decoder.load_state_dict(decoder_state, strict=False)

    if coord_embedder is not None:
        if "coord_embedder_state_dict" not in state:
            raise KeyError(
                "The checkpoint does not contain "
                "'coord_embedder_state_dict'. "
                "A Blueprint-AD checkpoint must include the trained "
                "coordinate adapter."
            )
        coord_embedder.load_state_dict(
            state["coord_embedder_state_dict"], strict=True
        )

    print("Loading weights from {}".format(filename))


def adjust_learning_rate(c, optimizer, epoch):
    lr = c.lr

    if c.lr_cosine:
        eta_min = lr * (c.lr_decay_rate ** 3)
        lr = eta_min + (lr - eta_min) * (
            1 + math.cos(math.pi * epoch / c.meta_epochs)
        ) / 2
    else:
        steps = np.sum(epoch >= np.asarray(c.lr_decay_epochs))
        if steps > 0:
            lr = lr * (c.lr_decay_rate ** steps)

    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def warmup_learning_rate(
    c,
    epoch,
    batch_id,
    total_batches,
    optimizer,
):
    if c.lr_warm and epoch < c.lr_warm_epochs:
        p = (
            batch_id + epoch * total_batches
        ) / (c.lr_warm_epochs * total_batches)

        lr = c.lr_warmup_from + p * (
            c.lr_warmup_to - c.lr_warmup_from
        )

        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

    return optimizer.param_groups[-1]["lr"]
