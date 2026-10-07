import os, time, datetime
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
# Evaluation metrics
from sklearn.metrics import roc_auc_score, auc, precision_recall_curve, average_precision_score
from skimage.measure import label, regionprops
from tqdm import tqdm
from visualize import *
from model import load_decoder_arch, load_encoder_arch, positionalencoding2d, activation
from utils import *
from custom_datasets import *
from custom_models import *

# Coordinate-field rendering and asynchronous GPU prefetching
from coords import CoordinateFieldPrefetcher, coordinate_collate

log_theta = torch.nn.LogSigmoid()

# Cache positional encodings on the active device.
GLOBAL_PE_CACHE = {}


# Lightweight coordinate-field adapter used in the CFLOW condition vector.
class CoordinateEmbedder(nn.Module):
    def __init__(self, in_channels, out_channels=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.net(x)


def train_meta_epoch(c, epoch, loader, encoder, decoders, optimizer_dec, pool_layers, N, coord_embedder=None,
                     optimizer_coord=None):
    P = c.condition_vec
    L = c.pool_layers
    decoders = [decoder.train() for decoder in decoders]
    if coord_embedder is not None:
        coord_embedder.train()

    adjust_learning_rate(c, optimizer_dec, epoch)
    I = len(loader)

    for sub_epoch in range(c.sub_epochs):
        # Accumulate training loss on GPU and synchronize once per sub-epoch.
        gpu_train_loss = torch.tensor(0.0, device=c.device)
        train_count = 0

        prefetcher = CoordinateFieldPrefetcher(loader, c.num_coord_channels, img_size=c.crp_size[0], prefetch_depth=3)

        for i in range(I):
            # Update the decoder learning rate, including warm-up.
            lr = warmup_learning_rate(c, epoch, i + sub_epoch * I, I * c.sub_epochs, optimizer_dec)

            # Keep the coordinate adapter on the same effective learning rate.
            if optimizer_coord is not None:
                for param_group in optimizer_coord.param_groups:
                    param_group['lr'] = lr

            batch = prefetcher.next()
            if batch is None: break
            image, _, _, coord_fields, _, logic_violations = batch

            # Non-blocking transfer where applicable.
            image = image.to(c.device, non_blocking=True)
            with torch.no_grad():
                _ = encoder(image)

            for l, layer in enumerate(pool_layers):
                if 'vit' in c.enc_arch:
                    e = activation[layer].transpose(1, 2)[..., 1:]
                    e_hw = int(np.sqrt(e.size(2)))
                    e = e.reshape(-1, e.size(1), e_hw, e_hw)
                else:
                    e = activation[layer].detach()

                B, C, H, W = e.size()
                S = H * W
                E = B * S

                # The original CFLOW positional encoding remains 128-D.
                pe_key = (128, H, W)
                if pe_key not in GLOBAL_PE_CACHE:
                    GLOBAL_PE_CACHE[pe_key] = positionalencoding2d(128, H, W).to(c.device)

                p = GLOBAL_PE_CACHE[pe_key].unsqueeze(0).repeat(B, 1, 1, 1)

                if coord_fields is not None and coord_embedder is not None:
                    coord_resized = F.interpolate(coord_fields, size=(H, W), mode='area')
                    coord_feat = coord_embedder(coord_resized)
                    cond_map = torch.cat([p, coord_feat], dim=1)
                else:
                    cond_map = p

                c_r = cond_map.reshape(B, P, S).transpose(1, 2).reshape(E, P)

                # =============================================================
                # 截断反向传播 (Detach Trick) - 防止 SDF 多次 Backward 崩溃
                # =============================================================
                if c_r.requires_grad:
                    c_r_detached = c_r.detach().requires_grad_(True)
                else:
                    c_r_detached = c_r

                e_r = e.reshape(B, C, S).transpose(1, 2).reshape(E, C)

                # Generate the fiber permutation on GPU.
                perm = torch.randperm(E, device=c.device)
                decoder = decoders[l]

                FIB = E // N
                assert FIB > 0, 'MAKE SURE WE HAVE ENOUGH FIBERS!'

                for f in range(FIB):
                    # Keep fiber indices on the active GPU.
                    idx = torch.arange(f * N, (f + 1) * N, device=c.device)
                    c_p = c_r_detached[perm[idx]]  # detached condition fibers
                    e_p = e_r[perm[idx]]

                    if 'cflow' in c.dec_arch:
                        z, log_jac_det = decoder(e_p, [c_p, ])
                    else:
                        z, log_jac_det = decoder(e_p)

                    decoder_log_prob = get_logp(C, z, log_jac_det)
                    log_prob = decoder_log_prob / C
                    loss = -log_theta(log_prob)

                    # Update the normalizing-flow decoder for each fiber batch.
                    optimizer_dec.zero_grad()
                    loss.mean().backward()
                    optimizer_dec.step()

                    # Accumulate loss without per-fiber CPU synchronization.
                    gpu_train_loss += loss.sum().detach()
                    train_count += len(loss)

                # =============================================================
                # 将 FIB 循环中攒下来的总梯度，一波流反传给 SDF Embedder
                # =============================================================
                if c_r.requires_grad and c_r_detached.grad is not None:
                    optimizer_coord.zero_grad()
                    c_r.backward(c_r_detached.grad)
                    optimizer_coord.step()

        # Synchronize once for logging.
        mean_train_loss = gpu_train_loss.item() / train_count
        if c.verbose:
            print('Epoch: {:d}.{:d} \t train loss: {:.4f}, lr={:.6f}'.format(epoch, sub_epoch, mean_train_loss, lr))


def test_meta_epoch(c, epoch, loader, encoder, decoders, pool_layers, N, coord_embedder=None):
    if c.verbose:
        print('\nCompute loss and scores on test set:')

    P = c.condition_vec
    decoders = [decoder.eval() for decoder in decoders]
    if coord_embedder is not None:
        coord_embedder.eval()

    height, width = list(), list()
    image_list, gt_label_list, gt_mask_list = list(), list(), list()
    export_legacy_viz = getattr(c, 'export_legacy_viz', False)
    test_dist = [list() for _ in pool_layers]
    all_logic_violations = []

    test_loss = 0.0
    test_count = 0
    start = time.time()

    prefetcher = CoordinateFieldPrefetcher(loader, c.num_coord_channels, img_size=c.crp_size[0], prefetch_depth=2)

    with torch.no_grad():
        for i in tqdm(range(len(loader)), disable=c.hide_tqdm_bar):
            batch = prefetcher.next()
            if batch is None: break
            image, label, mask, coord_fields, _, logic_violations = batch

            if export_legacy_viz:
                image_list.extend(t2np(image))
            gt_label_list.extend(t2np(label))
            gt_mask_list.extend(t2np(mask))
            all_logic_violations.extend(t2np(logic_violations))

            image = image.to(c.device, non_blocking=True)
            _ = encoder(image)

            for l, layer in enumerate(pool_layers):
                if 'vit' in c.enc_arch:
                    e = activation[layer].transpose(1, 2)[..., 1:]
                    e_hw = int(np.sqrt(e.size(2)))
                    e = e.reshape(-1, e.size(1), e_hw, e_hw)
                else:
                    e = activation[layer]

                B, C, H, W = e.size()
                S = H * W
                E = B * S

                if i == 0:
                    height.append(H)
                    width.append(W)

                pe_key = (128, H, W)
                if pe_key not in GLOBAL_PE_CACHE:
                    GLOBAL_PE_CACHE[pe_key] = positionalencoding2d(128, H, W).to(c.device)

                p = GLOBAL_PE_CACHE[pe_key].unsqueeze(0).repeat(B, 1, 1, 1)

                if coord_fields is not None and coord_embedder is not None:
                    coord_resized = F.interpolate(coord_fields, size=(H, W), mode='area')
                    coord_feat = coord_embedder(coord_resized)
                    cond_map = torch.cat([p, coord_feat], dim=1)
                else:
                    cond_map = p

                c_r = cond_map.reshape(B, P, S).transpose(1, 2).reshape(E, P)
                e_r = e.reshape(B, C, S).transpose(1, 2).reshape(E, C)

                m = F.interpolate(mask, size=(H, W), mode='nearest')
                m_r = m.reshape(B, 1, S).transpose(1, 2).reshape(E, 1)

                decoder = decoders[l]
                FIB = E // N + int(E % N > 0)

                for f in range(FIB):
                    if f < (FIB - 1):
                        idx = torch.arange(f * N, (f + 1) * N, device=c.device)
                    else:
                        idx = torch.arange(f * N, E, device=c.device)

                    c_p = c_r[idx]
                    e_p = e_r[idx]

                    if 'cflow' in c.dec_arch:
                        z, log_jac_det = decoder(e_p, [c_p, ])
                    else:
                        z, log_jac_det = decoder(e_p)

                    decoder_log_prob = get_logp(C, z, log_jac_det)
                    log_prob = decoder_log_prob / C
                    loss = -log_theta(log_prob)

                    test_loss += loss.sum().item()
                    test_count += len(loss)
                    test_dist[l] = test_dist[l] + log_prob.detach().cpu().tolist()

    fps = len(loader.dataset) / (time.time() - start)
    mean_test_loss = test_loss / test_count
    if c.verbose:
        print('Epoch: {:d} \t test_loss: {:.4f} and {:.2f} fps'.format(epoch, mean_test_loss, fps))

    return height, width, image_list, test_dist, gt_label_list, gt_mask_list, all_logic_violations


def train(c):
    run_date = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    L = c.pool_layers
    print('Number of pool layers =', L)
    encoder, pool_layers, pool_dims = load_encoder_arch(c, L)
    encoder = encoder.to(c.device).eval()

    # The dataset caches decoded samples in RAM, so worker multiprocessing is disabled.
    kwargs = {'num_workers': 0, 'pin_memory': True} if c.use_cuda else {}
    if c.dataset in ['mvtec', 'visa', 'loco']:
        train_dataset = MVTecDataset(c, is_train=True)
        test_dataset = MVTecDataset(c, is_train=False)
    elif c.dataset == 'stc':
        train_dataset = StcDataset(c, is_train=True)
        test_dataset = StcDataset(c, is_train=False)
    else:
        raise NotImplementedError('{} is not supported dataset!'.format(c.dataset))

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=c.batch_size, shuffle=True, drop_last=True,
                                               collate_fn=coordinate_collate, **kwargs)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=c.batch_size, shuffle=False, drop_last=False,
                                              collate_fn=coordinate_collate, **kwargs)

    c.num_coord_channels = getattr(train_dataset, 'num_coord_channels', 0)
    c.coord_embed_dim = 32 if c.num_coord_channels > 0 else 0
    c.condition_vec = 128 + c.coord_embed_dim

    coord_embedder = None
    optimizer_coord = None
    if c.num_coord_channels > 0:
        coord_embedder = CoordinateEmbedder(
            in_channels=c.num_coord_channels,
            out_channels=c.coord_embed_dim
        ).to(c.device)

        # Initialize with the decoder learning rate; synchronize the effective rate each batch.
        optimizer_coord = torch.optim.Adam(
            coord_embedder.parameters(),
            lr=c.lr
        )

        print(
            f"[INFO] Blueprint-AD coordinate adapter enabled. "
            f"Condition vector expanded to {c.condition_vec}."
        )

    decoders = [load_decoder_arch(c, pool_dim) for pool_dim in pool_dims]
    decoders = [decoder.to(c.device) for decoder in decoders]

    params_dec = list(decoders[0].parameters())
    for l in range(1, L):
        params_dec += list(decoders[l].parameters())

    optimizer_dec = torch.optim.Adam(params_dec, lr=c.lr)

    N = 256
    print('train/test loader length', len(train_loader.dataset), len(test_loader.dataset))
    print('train/test loader batches', len(train_loader), len(test_loader))

    det_roc_obs = Score_Observer('DET_AUROC')
    det_ap_obs = Score_Observer('DET_AP')
    seg_roc_obs = Score_Observer('SEG_AUROC')
    seg_ap_obs = Score_Observer('SEG_AP')
    seg_pro_obs = Score_Observer('SEG_AUPRO')

    c.apply_logic_penalty = getattr(c, 'apply_logic_penalty', False)

    if c.action_type == 'norm-test':
        c.meta_epochs = 1

    for epoch in range(c.meta_epochs):
        if c.action_type == 'norm-test' and c.checkpoint:
            load_weights(encoder, decoders, c.checkpoint, coord_embedder=coord_embedder)
        elif c.action_type == 'norm-train':
            print('Train meta epoch: {}'.format(epoch))
            train_meta_epoch(c, epoch, train_loader, encoder, decoders, optimizer_dec, pool_layers, N, coord_embedder,
                             optimizer_coord)
        else:
            raise NotImplementedError('{} is not supported action type!'.format(c.action_type))

        # =============================================================
        # 提速补丁 4：包裹测试代码，仅每 25 Epoch 评估一次，前 24 Epoch 极速飞过
        # =============================================================
        is_eval_epoch = (epoch + 1) % 25 == 0 or (epoch + 1) == c.meta_epochs or c.action_type == 'norm-test'
        is_last_eval = (epoch + 1) == c.meta_epochs or c.action_type == 'norm-test'

        if is_eval_epoch:
            height, width, test_image_list, test_dist, gt_label_list, gt_mask_list, all_logic_violations = test_meta_epoch(
                c, epoch, test_loader, encoder, decoders, pool_layers, N, coord_embedder)

            print('Heights/Widths', height, width)
            test_map = [list() for p in pool_layers]
            for l, p in enumerate(pool_layers):
                test_norm = torch.tensor(test_dist[l], dtype=torch.double)
                test_norm -= torch.max(test_norm)
                test_prob = torch.exp(test_norm)
                test_mask = test_prob.reshape(-1, height[l], width[l])

                test_map[l] = F.interpolate(test_mask.unsqueeze(1),
                                            size=c.crp_size, mode='bilinear', align_corners=True).squeeze().numpy()

            score_map = np.zeros_like(test_map[0])
            for l, p in enumerate(pool_layers):
                score_map += test_map[l]

            score_mask = score_map
            super_mask = score_mask.max() - score_mask

            if c.apply_logic_penalty and len(all_logic_violations) > 0:
                all_logic_violations = np.array(all_logic_violations, dtype=bool)
                penalty_value = 1.0
                punished_count = 0
                for b in range(len(super_mask)):
                    if all_logic_violations[b]:
                        max_idx = np.unravel_index(np.argmax(super_mask[b], axis=None), super_mask[b].shape)
                        super_mask[b][max_idx] += penalty_value
                        punished_count += 1
                print(f"[INFO] Applied logic penalty to {punished_count} images.")

            if is_last_eval and getattr(c, 'save_raw_anomaly_maps', True):
                export_raw_anomaly_maps_tiff(
                    c,
                    super_mask,
                    image_paths=getattr(test_loader.dataset, 'x', None),
                    epoch=epoch,
                )

            score_label = np.max(super_mask, axis=(1, 2))
            gt_label = np.asarray(gt_label_list, dtype=bool)

            det_roc_auc = roc_auc_score(gt_label, score_label)
            det_ap = average_precision_score(gt_label, score_label)
            _ = det_roc_obs.update(100.0 * det_roc_auc, epoch)
            _ = det_ap_obs.update(100.0 * det_ap, epoch)

            gt_mask = np.squeeze(np.asarray(gt_mask_list, dtype=bool), axis=1)
            seg_roc_auc = roc_auc_score(gt_mask.flatten(), super_mask.flatten())
            seg_ap = average_precision_score(gt_mask.flatten(), super_mask.flatten())
            save_best_seg_weights = seg_roc_obs.update(100.0 * seg_roc_auc, epoch)
            _ = seg_ap_obs.update(100.0 * seg_ap, epoch)

            if save_best_seg_weights and c.action_type != 'norm-test':
                save_weights(encoder, decoders, c.model, run_date, coord_embedder=coord_embedder)

            if c.pro:
                max_step = 1000
                expect_fpr = 0.3
                max_th = super_mask.max()
                min_th = super_mask.min()
                delta = (max_th - min_th) / max_step
                ious_mean, ious_std, pros_mean, pros_std, threds, fprs = [], [], [], [], [], []
                binary_score_maps = np.zeros_like(super_mask, dtype=bool)

                for step in range(max_step):
                    thred = max_th - step * delta
                    binary_score_maps[super_mask <= thred] = 0
                    binary_score_maps[super_mask > thred] = 1
                    pro, iou = [], []

                    for i in range(len(binary_score_maps)):
                        label_map = label(gt_mask[i], connectivity=2)
                        props = regionprops(label_map)
                        for prop in props:
                            x_min, y_min, x_max, y_max = prop.bbox
                            cropped_pred_label = binary_score_maps[i][x_min:x_max, y_min:y_max]
                            cropped_mask = prop.filled_image
                            intersection = np.logical_and(cropped_pred_label, cropped_mask).astype(np.float32).sum()
                            pro.append(intersection / prop.area)

                        intersection = np.logical_and(binary_score_maps[i], gt_mask[i]).astype(np.float32).sum()
                        union = np.logical_or(binary_score_maps[i], gt_mask[i]).astype(np.float32).sum()
                        if gt_mask[i].any() > 0:
                            iou.append(intersection / union)

                    ious_mean.append(np.array(iou).mean() if len(iou) > 0 else 0)
                    ious_std.append(np.array(iou).std() if len(iou) > 0 else 0)
                    pros_mean.append(np.array(pro).mean() if len(pro) > 0 else 0)
                    pros_std.append(np.array(pro).std() if len(pro) > 0 else 0)

                    gt_masks_neg = ~gt_mask
                    fpr = np.logical_and(gt_masks_neg, binary_score_maps).sum() / gt_masks_neg.sum()
                    fprs.append(fpr)
                    threds.append(thred)

                threds, pros_mean, pros_std, fprs, ious_mean, ious_std = map(np.array,
                                                                             [threds, pros_mean, pros_std, fprs,
                                                                              ious_mean,
                                                                              ious_std])
                idx = fprs <= expect_fpr
                if np.sum(idx) > 0:
                    fprs_selected = rescale(fprs[idx])
                    pros_mean_selected = pros_mean[idx]
                    seg_pro_auc = auc(fprs_selected, pros_mean_selected)
                    _ = seg_pro_obs.update(100.0 * seg_pro_auc, epoch)

        # Save the current evaluation summary.
        if is_eval_epoch:
            save_results(det_roc_obs, seg_roc_obs, seg_pro_obs, c.model, c.class_name, run_date)

            if getattr(c, 'export_legacy_viz', False):
                precision, recall, thresholds = precision_recall_curve(gt_label, score_label)
                a = 2 * precision * recall
                b = precision + recall
                f1 = np.divide(a, b, out=np.zeros_like(a), where=b != 0)
                det_threshold = thresholds[np.argmax(f1)]
                print('Optimal DET Threshold: {:.2f}'.format(det_threshold))

                precision, recall, thresholds = precision_recall_curve(gt_mask.flatten(), super_mask.flatten())
                a = 2 * precision * recall
                b = precision + recall
                f1 = np.divide(a, b, out=np.zeros_like(a), where=b != 0)
                seg_threshold = thresholds[np.argmax(f1)]
                print('Optimal SEG Threshold: {:.2f}'.format(seg_threshold))

                export_groundtruth(c, test_image_list, gt_mask)
                export_scores(c, test_image_list, super_mask, seg_threshold)
                export_test_images(c, test_image_list, gt_mask, super_mask, seg_threshold)
                export_hist(c, gt_mask, super_mask, seg_threshold)