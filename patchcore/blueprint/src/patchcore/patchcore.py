"""PatchCore and PatchCore detection methods."""
import logging
import os
import pickle

import numpy as np
import torch
import torch.nn.functional as F
import tqdm
from patchcore.coords import PatchCoreCoordinatePrefetcher

import patchcore
import patchcore.backbones
import patchcore.common
import patchcore.sampler
import torch.nn as nn

# ============================================================
# Conservative background suppression
# ============================================================
# Top-K is used only to decide whether a query patch is reliably
# background. It never changes PatchCore's original 1-NN match.
BACKGROUND_CONSENSUS_TOPK = 5

# A PatchCore descriptor is considered clear background only when
# the maximum coordinate response over its local support is <= this.
BACKGROUND_PATCH_THRESHOLD = 0.05

# High-resolution soft suppression range on the 224x224 coordinate field.
BACKGROUND_PIXEL_BG_THRESHOLD = 0.05
BACKGROUND_PIXEL_FG_THRESHOLD = 0.20

# Strongest suppression applied to a fully confirmed background pixel.
# 0.5 means keeping at least 50% of the original anomaly response.
BACKGROUND_MIN_WEIGHT = 0.5

# False: image-level score uses the standard PatchCore raw patch scores.
# True: image-level score also suppresses patches that pass the conservative
#       background-consensus rule before taking the image-wise maximum.
IMAGE_SCORE_USE_BACKGROUND_SUPPRESSION = False


LOGGER = logging.getLogger(__name__)

class CoordinatePCAAdapter(nn.Module):
    def __init__(
        self,
        in_channels,
        embed_dim=64,
    ):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(
                in_channels,
                32,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                32,
                embed_dim,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True),
        )

        # This output is directly supervised.
        # No BatchNorm and no ReLU after this layer,
        # because PCA coefficients are signed.
        self.output_projection = nn.Conv2d(
            embed_dim,
            embed_dim,
            kernel_size=1,
            bias=True,
        )

    def forward(self, coord_fields):
        hidden = self.encoder(coord_fields)
        pred_pca = self.output_projection(hidden)
        return pred_pca

class PatchCore(torch.nn.Module):
    def __init__(self, device):
        """PatchCore anomaly detection class."""
        super(PatchCore, self).__init__()
        self.device = device

    def load(
        self,
        backbone,
        layers_to_extract_from,
        device,
        input_shape,
        pretrain_embed_dimension,
        target_embed_dimension,
        patchsize=3,
        patchstride=1,
        anomaly_scorer_num_nn=1,
        featuresampler=patchcore.sampler.IdentitySampler(),
        nn_method=patchcore.common.FaissNN(False, 4),
        blueprint_checkpoint=None,
        **kwargs,
    ):
        if blueprint_checkpoint:
            checkpoint = torch.load(
                blueprint_checkpoint,
                map_location=self.device,
            )

            self.num_coord_channels = int(
                checkpoint["in_channels"]
            )
            embed_dim = int(
                checkpoint.get("embed_dim", 64)
            )

            self.coordinate_adapter = CoordinatePCAAdapter(
                in_channels=self.num_coord_channels,
                embed_dim=embed_dim,
            ).to(self.device)

            self.coordinate_adapter.load_state_dict(
                checkpoint["model_state"]
            )

            self.coordinate_adapter.eval()
            self.coordinate_adapter.requires_grad_(False)

        self.backbone = backbone.to(device)
        self.layers_to_extract_from = layers_to_extract_from
        self.input_shape = input_shape

        self.device = device
        self.patch_maker = PatchMaker(patchsize, stride=patchstride)

        self.forward_modules = torch.nn.ModuleDict({})

        feature_aggregator = patchcore.common.NetworkFeatureAggregator(
            self.backbone, self.layers_to_extract_from, self.device
        )
        feature_dimensions = feature_aggregator.feature_dimensions(input_shape)
        self.forward_modules["feature_aggregator"] = feature_aggregator

        preprocessing = patchcore.common.Preprocessing(
            feature_dimensions, pretrain_embed_dimension
        )
        self.forward_modules["preprocessing"] = preprocessing

        self.target_embed_dimension = target_embed_dimension
        preadapt_aggregator = patchcore.common.Aggregator(
            target_dim=target_embed_dimension
        )

        _ = preadapt_aggregator.to(self.device)

        self.forward_modules["preadapt_aggregator"] = preadapt_aggregator

        self.anomaly_scorer = patchcore.common.NearestNeighbourScorer(
            n_nearest_neighbours=anomaly_scorer_num_nn, nn_method=nn_method
        )

        self.anomaly_segmentor = patchcore.common.RescaleSegmentor(
            device=self.device, target_size=input_shape[-2:]
        )

        self.featuresampler = featuresampler

    def embed(self, data):
        if isinstance(data, torch.utils.data.DataLoader):
            features = []
            for image in data:
                if isinstance(image, dict):
                    image = image["image"]
                with torch.no_grad():
                    input_image = image.to(torch.float).to(self.device)
                    features.append(self._embed(input_image))
            return features
        return self._embed(data)

    def _embed(self, images, coords=None, detach=True, provide_patch_shapes=False):
        def _detach(features):
            if detach:
                return [x.detach().cpu().numpy() for x in features]
            return features

        _ = self.forward_modules["feature_aggregator"].eval()
        with torch.no_grad():
            features = self.forward_modules["feature_aggregator"](images)

        features = [features[layer] for layer in self.layers_to_extract_from]
        features = [self.patch_maker.patchify(x, return_spatial_info=True) for x in features]
        patch_shapes = [x[1] for x in features]
        features = [x[0] for x in features]
        ref_num_patches = patch_shapes[0]

        for i in range(1, len(features)):
            _features = features[i]
            patch_dims = patch_shapes[i]
            _features = _features.reshape(_features.shape[0], patch_dims[0], patch_dims[1], *_features.shape[2:])
            _features = _features.permute(0, -3, -2, -1, 1, 2)
            perm_base_shape = _features.shape
            _features = _features.reshape(-1, *_features.shape[-2:])
            _features = F.interpolate(_features.unsqueeze(1), size=(ref_num_patches[0], ref_num_patches[1]),
                                      mode="bilinear", align_corners=False)
            _features = _features.squeeze(1)
            _features = _features.reshape(*perm_base_shape[:-2], ref_num_patches[0], ref_num_patches[1])
            _features = _features.permute(0, -2, -1, 1, 2, 3)
            _features = _features.reshape(len(_features), -1, *_features.shape[-3:])
            features[i] = _features

        features = [x.reshape(-1, *x.shape[-3:]) for x in features]
        features = self.forward_modules["preprocessing"](features)

        # The original PatchCore descriptor is 1024-D here.
        features = self.forward_modules["preadapt_aggregator"](features)

        # =================================================================
        # Blueprint-AD: late concatenate the learned 64-D coordinate descriptor.
        # =================================================================
        if coords is not None and hasattr(
                self,
                "coordinate_adapter",
        ):
            H_p, W_p = patch_shapes[0]

            coord_28x28 = F.interpolate(
                coords,
                size=(H_p, W_p),
                mode="area",
            )

            pred_64 = self.coordinate_adapter(
                coord_28x28,
            )

            pred_flat = (
                pred_64
                .permute(0, 2, 3, 1)
                .reshape(-1, pred_64.shape[1])
            )

            pred_flat = pred_flat.to(
                device=features.device,
                dtype=features.dtype,
            )

            alpha = 0.3

            features = torch.cat(
                [
                    features,
                    alpha * pred_flat,
                ],
                dim=1,
            )
        # =================================================================

        if provide_patch_shapes:
            return _detach(features), patch_shapes
        return _detach(features)
        # =================================================================


    def fit(self, training_data):
        """PatchCore training.

        This function computes the embeddings of the training data and fills the
        memory bank of SPADE.
        """
        self._fill_memory_bank(training_data)

    def _clear_background_patch_mask(
            self,
            coords,
            patch_shape,
    ):
        """
        Conservative clear-background mask for PatchCore locations.

        A patch is considered clear background only when the maximum
        coordinate-field response inside its corresponding high-resolution
        region is below BACKGROUND_PATCH_THRESHOLD.

        Note:
            We deliberately DO NOT expand the decision with an additional
            3x3 max-pooling operation. Pixel-level suppression later uses the
            original high-resolution coordinate field again, which already
            protects object boundaries.
        """

        patch_h, patch_w = patch_shape

        # ---------------------------------------------------------
        # 1. Merge coordinate channels.
        # [B, C, H, W] -> [B, 1, H, W]
        # ---------------------------------------------------------
        coord_response = torch.amax(
            coords,
            dim=1,
            keepdim=True,
        )

        # ---------------------------------------------------------
        # 2. Project high-resolution response to PatchCore grid.
        #
        # max pooling is intentionally used instead of area averaging:
        # if any part of this spatial cell has clear object response,
        # the patch should not be treated as clear background.
        # ---------------------------------------------------------
        response_patch = F.adaptive_max_pool2d(
            coord_response,
            output_size=(
                patch_h,
                patch_w,
            ),
        )

        # ---------------------------------------------------------
        # 3. Clear background.
        #
        # No additional 3x3 dilation here.
        # ---------------------------------------------------------
        clear_bg = (
                response_patch[:, 0]
                <= BACKGROUND_PATCH_THRESHOLD
        )

        return clear_bg

    def _fill_memory_bank(self, input_data):
        """Compute support features and aligned clear-background labels."""
        _ = self.forward_modules.eval()

        features_list = []
        background_labels_list = []

        num_c = getattr(self, "num_coord_channels", 4)
        prefetcher = PatchCoreCoordinatePrefetcher(
            input_data,
            num_c,
            resize=256,
            crop_size=224,
        )

        with tqdm.tqdm(
            total=len(input_data.dataset),
            desc="Computing support features...",
            position=1,
            leave=False,
        ) as pbar:
            while True:
                batch = prefetcher.next()
                if batch is None:
                    break

                image = batch["image"].to(
                    torch.float
                ).to(self.device)
                coords = batch["coords"].to(
                    torch.float
                ).to(self.device)

                with torch.no_grad():
                    features, patch_shapes = self._embed(
                        image,
                        coords=coords,
                        provide_patch_shapes=True,
                    )
                    features_list.append(features)

                    clear_bg = self._clear_background_patch_mask(
                        coords,
                        patch_shapes[0],
                    )
                    background_labels_list.append(
                        clear_bg
                        .reshape(-1)
                        .cpu()
                        .numpy()
                        .astype(np.bool_)
                    )

                pbar.update(image.shape[0])

        all_features = np.concatenate(
            features_list,
            axis=0,
        )
        all_background_labels = np.concatenate(
            background_labels_list,
            axis=0,
        )

        # Keep labels exactly aligned with the sampled memory-bank features.
        # This mirrors the existing coreset path while retaining selected indices.
        if isinstance(
            self.featuresampler,
            patchcore.sampler.IdentitySampler,
        ):
            selected_indices = np.arange(
                len(all_features)
            )

        elif isinstance(
            self.featuresampler,
            patchcore.sampler.GreedyCoresetSampler,
        ):
            feature_tensor = torch.from_numpy(
                all_features
            )

            with torch.no_grad():
                reduced_features = self.featuresampler._reduce_features(
                    feature_tensor
                )
                selected_indices = (
                    self.featuresampler
                    ._compute_greedy_coreset_indices(
                        reduced_features
                    )
                )

            selected_indices = np.asarray(
                selected_indices,
                dtype=np.int64,
            )

        else:
            # Preserve the behavior of the previous local implementation for
            # unsupported sampler types.
            selected_indices = np.random.choice(
                len(all_features),
                int(
                    len(all_features)
                    * self.featuresampler.percentage
                ),
                replace=False,
            )

        sampled_features = all_features[
            selected_indices
        ]
        self.memory_clear_background = (
            all_background_labels[
                selected_indices
            ]
        )

        self.anomaly_scorer.fit(
            detection_features=[sampled_features]
        )

    def predict(self, data, semantic_evaluator=None):
        if isinstance(data, torch.utils.data.DataLoader):
            return self._predict_dataloader(data, semantic_evaluator=semantic_evaluator)
        return self._predict(data)

    def _predict_dataloader(self, dataloader, semantic_evaluator=None):
        """This function provides anomaly scores/maps for full dataloaders."""
        _ = self.forward_modules.eval()

        scores = []
        masks = []
        labels_gt = []
        masks_gt = []

        # Render coordinate fields asynchronously on GPU.
        num_c = getattr(self, 'num_coord_channels', 4)
        prefetcher = PatchCoreCoordinatePrefetcher(dataloader, num_c, resize=256, crop_size=224)



        with tqdm.tqdm(total=len(dataloader.dataset), desc="Inferring...", leave=False) as pbar:
            while True:
                batch = prefetcher.next()
                if batch is None: break

                labels_gt.extend(batch["is_anomaly"].numpy().tolist())
                if batch.get("mask") is not None:
                    masks_gt.extend(batch["mask"].numpy().tolist())

                image = batch["image"].to(torch.float).to(self.device)
                coords = batch["coords"].to(torch.float).to(self.device)
                yolo_detections_batch = batch.get("yolo_detections", [[]] * image.shape[0])

                # Predict with RGB features and aligned coordinate fields.
                _scores, _masks = self._predict(image, coords=coords)

                # ========================================================
                # Supplementary compositional-constraint check.
                # ========================================================
                if semantic_evaluator is not None:
                    for b in range(image.size(0)):
                        detections = yolo_detections_batch[b]

                        if semantic_evaluator.check_logic_violation(detections):

                            # Log compositional false positives on normal images.
                            if int(batch["is_anomaly"][b]) == 0:
                                print(
                                    "[FALSE LOGIC]",
                                    batch["image_path"][b],
                                    "| class:",
                                    semantic_evaluator.last_violation_class,
                                )

                            max_idx = np.unravel_index(
                                np.argmax(_masks[b]),
                                _masks[b].shape
                            )
                            _masks[b][max_idx] += 1000.0
                            _scores[b] += 1000.0
                # ========================================================

                scores.extend(_scores)
                masks.extend(_masks)
                pbar.update(image.shape[0])

        return scores, masks, labels_gt, masks_gt

    def _predict(self, images, coords=None):
        """Infer image scores and pixel anomaly maps for a batch.

        PatchCore's original 1-NN score remains the anomaly score. Top-5 nearest
        neighbours are queried only as evidence for conservative background
        suppression. Pixel suppression is performed on the final high-resolution
        anomaly map.
        """
        images = images.to(
            torch.float
        ).to(self.device)
        _ = self.forward_modules.eval()

        batchsize = images.shape[0]

        with torch.no_grad():
            features, patch_shapes = self._embed(
                images,
                coords=coords,
                provide_patch_shapes=True,
            )
            features = np.asarray(features)

            # -------------------------------------------------------------
            # 1. Standard PatchCore anomaly score.
            # -------------------------------------------------------------
            raw_patch_scores = self.anomaly_scorer.predict(
                [features]
            )[0]
            raw_patch_scores = np.asarray(
                raw_patch_scores,
                dtype=np.float32,
            )

            patch_h, patch_w = patch_shapes[0]

            # By default no patch is allowed to be background-suppressed.
            suppress_allowed = np.zeros(
                len(raw_patch_scores),
                dtype=np.bool_,
            )
            query_clear_bg = np.zeros_like(
                suppress_allowed
            )
            memory_bg_consensus = np.zeros_like(
                suppress_allowed
            )

            # -------------------------------------------------------------
            # 2. Conservative background consensus.
            #
            # Suppression is allowed only when:
            #   a) the query patch is clear background in the coordinate field;
            #   b) all Top-5 normal-memory neighbours are also clear background.
            #
            # Top-5 is used only for this binary decision. It never replaces the
            # original PatchCore 1-NN used above for anomaly scoring.
            # -------------------------------------------------------------
            if (
                coords is not None
                and hasattr(
                    self,
                    "memory_clear_background",
                )
            ):
                bank_size = len(
                    self.memory_clear_background
                )

                if bank_size >= BACKGROUND_CONSENSUS_TOPK:
                    _, neighbour_indices = (
                        self.anomaly_scorer
                        .nn_method
                        .run(
                            BACKGROUND_CONSENSUS_TOPK,
                            features,
                        )
                    )

                    neighbour_indices = np.asarray(
                        neighbour_indices,
                        dtype=np.int64,
                    )

                    # Invalid FAISS indices must never count as BG consensus.
                    valid_neighbours = (
                        neighbour_indices >= 0
                    )
                    safe_indices = np.clip(
                        neighbour_indices,
                        0,
                        bank_size - 1,
                    )

                    neighbour_is_bg = (
                        self.memory_clear_background[
                            safe_indices
                        ]
                        & valid_neighbours
                    )

                    memory_bg_consensus = np.all(
                        neighbour_is_bg,
                        axis=1,
                    )

                    query_clear_bg = (
                        self._clear_background_patch_mask(
                            coords,
                            patch_shapes[0],
                        )
                        .reshape(-1)
                        .cpu()
                        .numpy()
                        .astype(np.bool_)
                    )

                    suppress_allowed = (
                        query_clear_bg
                        & memory_bg_consensus
                    )

            # -------------------------------------------------------------
            # 3. Image-level anomaly score.
            #
            # False -> exact standard PatchCore image score.
            # True  -> suppress only patches that passed the conservative
            #          Query-BG + Top5-all-BG rule, then take patch maximum.
            # -------------------------------------------------------------
            image_patch_scores = (
                raw_patch_scores.copy()
            )

            if IMAGE_SCORE_USE_BACKGROUND_SUPPRESSION:
                image_patch_scores[
                    suppress_allowed
                ] *= BACKGROUND_MIN_WEIGHT

            image_scores = (
                self.patch_maker
                .unpatch_scores(
                    image_patch_scores,
                    batchsize=batchsize,
                )
            )
            image_scores = image_scores.reshape(
                *image_scores.shape[:2],
                -1,
            )
            image_scores = self.patch_maker.score(
                image_scores
            )

            # -------------------------------------------------------------
            # 4. Standard high-resolution PatchCore anomaly map.
            # -------------------------------------------------------------
            patch_score_grid = (
                self.patch_maker
                .unpatch_scores(
                    raw_patch_scores,
                    batchsize=batchsize,
                )
            )
            patch_score_grid = patch_score_grid.reshape(
                batchsize,
                patch_h,
                patch_w,
            )

            raw_masks = (
                self.anomaly_segmentor
                .convert_to_segmentation(
                    patch_score_grid
                )
            )

            # -------------------------------------------------------------
            # 5. High-resolution conservative background suppression.
            #
            # The 28x28 consensus map is only a permission map. Actual
            # attenuation is performed at the final anomaly-map resolution
            # using the high-resolution coordinate response.
            # -------------------------------------------------------------
            masks = raw_masks

            if (
                coords is not None
                and np.any(suppress_allowed)
            ):
                target_h, target_w = (
                    raw_masks[0].shape[-2],
                    raw_masks[0].shape[-1],
                )

                # Patch-level permission map -> high resolution. Bilinear
                # interpolation avoids block artifacts at patch boundaries.
                gate_patch = torch.from_numpy(
                    suppress_allowed.reshape(
                        batchsize,
                        patch_h,
                        patch_w,
                    ).astype(np.float32)
                ).to(self.device)

                gate_high = F.interpolate(
                    gate_patch.unsqueeze(1),
                    size=(target_h, target_w),
                    mode="bilinear",
                    align_corners=False,
                )

                # High-resolution coordinate response.
                coord_response_high = torch.amax(
                    coords,
                    dim=1,
                    keepdim=True,
                )

                if (
                    coord_response_high.shape[-2:]
                    != (target_h, target_w)
                ):
                    coord_response_high = F.interpolate(
                        coord_response_high,
                        size=(target_h, target_w),
                        mode="bilinear",
                        align_corners=False,
                    )

                denom = (
                    BACKGROUND_PIXEL_FG_THRESHOLD
                    - BACKGROUND_PIXEL_BG_THRESHOLD
                )
                if denom <= 0:
                    raise ValueError(
                        "BACKGROUND_PIXEL_FG_THRESHOLD must be larger than "
                        "BACKGROUND_PIXEL_BG_THRESHOLD."
                    )

                foreground_confidence = (
                    (
                        coord_response_high
                        - BACKGROUND_PIXEL_BG_THRESHOLD
                    )
                    / denom
                ).clamp(0.0, 1.0)

                # Smoothstep transition between clear BG and clear FG.
                foreground_confidence = (
                    foreground_confidence
                    * foreground_confidence
                    * (
                        3.0
                        - 2.0
                        * foreground_confidence
                    )
                )

                background_confidence = (
                    1.0
                    - foreground_confidence
                )

                suppression_confidence = (
                    gate_high
                    * background_confidence
                )

                suppression_weight = (
                    1.0
                    - (
                        1.0
                        - BACKGROUND_MIN_WEIGHT
                    )
                    * suppression_confidence
                )

                suppression_weight = (
                    suppression_weight[:, 0]
                    .cpu()
                    .numpy()
                    .astype(np.float32)
                )

                masks = [
                    np.asarray(mask, dtype=np.float32)
                    * weight
                    for mask, weight in zip(
                        raw_masks,
                        suppression_weight,
                    )
                ]

        return (
            [score for score in image_scores],
            [mask for mask in masks],
        )

    @staticmethod
    def _params_file(filepath, prepend=""):
        return os.path.join(filepath, prepend + "patchcore_params.pkl")

    def save_to_path(self, save_path: str, prepend: str = "") -> None:
        LOGGER.info("Saving PatchCore data.")
        self.anomaly_scorer.save(
            save_path, save_features_separately=False, prepend=prepend
        )
        patchcore_params = {
            "backbone.name": self.backbone.name,
            "layers_to_extract_from": self.layers_to_extract_from,
            "input_shape": self.input_shape,
            "pretrain_embed_dimension": self.forward_modules[
                "preprocessing"
            ].output_dim,
            "target_embed_dimension": self.forward_modules[
                "preadapt_aggregator"
            ].target_dim,
            "patchsize": self.patch_maker.patchsize,
            "patchstride": self.patch_maker.stride,
            "anomaly_scorer_num_nn": self.anomaly_scorer.n_nearest_neighbours,
        }
        with open(self._params_file(save_path, prepend), "wb") as save_file:
            pickle.dump(patchcore_params, save_file, pickle.HIGHEST_PROTOCOL)

    def load_from_path(
        self,
        load_path: str,
        device: torch.device,
        nn_method: patchcore.common.FaissNN(False, 4),
        prepend: str = "",
    ) -> None:
        LOGGER.info("Loading and initializing PatchCore.")
        with open(self._params_file(load_path, prepend), "rb") as load_file:
            patchcore_params = pickle.load(load_file)
        patchcore_params["backbone"] = patchcore.backbones.load(
            patchcore_params["backbone.name"]
        )
        patchcore_params["backbone"].name = patchcore_params["backbone.name"]
        del patchcore_params["backbone.name"]
        self.load(**patchcore_params, device=device, nn_method=nn_method)

        self.anomaly_scorer.load(load_path, prepend)


# Image handling classes.
class PatchMaker:
    def __init__(self, patchsize, stride=None):
        self.patchsize = patchsize
        self.stride = stride

    def patchify(self, features, return_spatial_info=False):
        """Convert a tensor into a tensor of respective patches.
        Args:
            x: [torch.Tensor, bs x c x w x h]
        Returns:
            x: [torch.Tensor, bs * w//stride * h//stride, c, patchsize,
            patchsize]
        """
        padding = int((self.patchsize - 1) / 2)
        unfolder = torch.nn.Unfold(
            kernel_size=self.patchsize, stride=self.stride, padding=padding, dilation=1
        )
        unfolded_features = unfolder(features)
        number_of_total_patches = []
        for s in features.shape[-2:]:
            n_patches = (
                s + 2 * padding - 1 * (self.patchsize - 1) - 1
            ) / self.stride + 1
            number_of_total_patches.append(int(n_patches))
        unfolded_features = unfolded_features.reshape(
            *features.shape[:2], self.patchsize, self.patchsize, -1
        )
        unfolded_features = unfolded_features.permute(0, 4, 1, 2, 3)

        if return_spatial_info:
            return unfolded_features, number_of_total_patches
        return unfolded_features

    def unpatch_scores(self, x, batchsize):
        return x.reshape(batchsize, -1, *x.shape[1:])

    def score(self, x):
        was_numpy = False
        if isinstance(x, np.ndarray):
            was_numpy = True
            x = torch.from_numpy(x)
        while x.ndim > 1:
            x = torch.max(x, dim=-1).values
        if was_numpy:
            return x.numpy()
        return x
