import os, time, datetime
import numpy as np
import torch
import torch.nn.functional as F
# Detection and segmentation AP metrics.
from sklearn.metrics import roc_auc_score, auc, precision_recall_curve, average_precision_score
from skimage.measure import label, regionprops
from tqdm import tqdm
from visualize import *
from model import load_decoder_arch, load_encoder_arch, positionalencoding2d, activation
from utils import *
from custom_datasets import *
from custom_models import *


gamma = 0.0
theta = torch.nn.Sigmoid()
log_theta = torch.nn.LogSigmoid()

GLOBAL_PE_CACHE = {}


def train_meta_epoch(c, epoch, loader, encoder, decoders, optimizer, pool_layers, N):
    P = c.condition_vec
    L = c.pool_layers
    decoders = [decoder.train() for decoder in decoders]
    adjust_learning_rate(c, optimizer, epoch)
    I = len(loader)
    iterator = iter(loader)

    for sub_epoch in range(c.sub_epochs):
        # Accumulate training loss on GPU to avoid per-step host synchronization.
        gpu_train_loss = torch.tensor(0.0, device=c.device)
        train_count = 0

        for i in range(I):
            lr = warmup_learning_rate(c, epoch, i + sub_epoch * I, I * c.sub_epochs, optimizer)
            try:
                image, _, _ = next(iterator)
            except StopIteration:
                iterator = iter(loader)
                image, _, _ = next(iterator)

            # Use non-blocking device transfer.
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

                # Reuse cached positional encodings.
                pe_key = (P, H, W)
                if pe_key not in GLOBAL_PE_CACHE:
                    GLOBAL_PE_CACHE[pe_key] = positionalencoding2d(P, H, W).to(c.device)

                p = GLOBAL_PE_CACHE[pe_key].unsqueeze(0).repeat(B, 1, 1, 1)

                c_r = p.reshape(B, P, S).transpose(1, 2).reshape(E, P)
                e_r = e.reshape(B, C, S).transpose(1, 2).reshape(E, C)

                # Generate the random permutation on GPU.
                perm = torch.randperm(E, device=c.device)
                decoder = decoders[l]

                FIB = E // N
                assert FIB > 0, 'MAKE SURE WE HAVE ENOUGH FIBERS, otherwise decrease N or batch-size!'

                for f in range(FIB):
                    # Keep indexing tensors on GPU.
                    idx = torch.arange(f * N, (f + 1) * N, device=c.device)
                    c_p = c_r[perm[idx]]
                    e_p = e_r[perm[idx]]

                    if 'cflow' in c.dec_arch:
                        z, log_jac_det = decoder(e_p, [c_p, ])
                    else:
                        z, log_jac_det = decoder(e_p)

                    decoder_log_prob = get_logp(C, z, log_jac_det)
                    log_prob = decoder_log_prob / C
                    loss = -log_theta(log_prob)

                    optimizer.zero_grad()
                    loss.mean().backward()
                    optimizer.step()

                    # Accumulate loss on GPU without synchronizing each step.
                    gpu_train_loss += loss.sum().detach()
                    train_count += len(loss)

        # Synchronize once per sub-epoch when reporting the mean loss.
        mean_train_loss = gpu_train_loss.item() / train_count
        if c.verbose:
            print('Epoch: {:d}.{:d} \t train loss: {:.4f}, lr={:.6f}'.format(epoch, sub_epoch, mean_train_loss, lr))


def test_meta_epoch(c, epoch, loader, encoder, decoders, pool_layers, N):
    if c.verbose:
        print('\nCompute loss and scores on test set:')

    P = c.condition_vec
    decoders = [decoder.eval() for decoder in decoders]
    height = list()
    width = list()
    image_list = list()
    gt_label_list = list()
    gt_mask_list = list()
    export_legacy_viz = getattr(c, 'export_legacy_viz', False)
    test_dist = [list() for layer in pool_layers]
    test_loss = 0.0
    test_count = 0
    start = time.time()

    with torch.no_grad():
        for i, (image, label, mask) in enumerate(tqdm(loader, disable=c.hide_tqdm_bar)):
            if export_legacy_viz:
                image_list.extend(t2np(image))
            gt_label_list.extend(t2np(label))
            gt_mask_list.extend(t2np(mask))

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

                pe_key = (P, H, W)
                if pe_key not in GLOBAL_PE_CACHE:
                    GLOBAL_PE_CACHE[pe_key] = positionalencoding2d(P, H, W).to(c.device)

                p = GLOBAL_PE_CACHE[pe_key].unsqueeze(0).repeat(B, 1, 1, 1)

                c_r = p.reshape(B, P, S).transpose(1, 2).reshape(E, P)
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

    return height, width, image_list, test_dist, gt_label_list, gt_mask_list


def train(c):
    run_date = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    L = c.pool_layers
    print('Number of pool layers =', L)
    encoder, pool_layers, pool_dims = load_encoder_arch(c, L)
    encoder = encoder.to(c.device).eval()

    decoders = [load_decoder_arch(c, pool_dim) for pool_dim in pool_dims]
    decoders = [decoder.to(c.device) for decoder in decoders]

    params = list(decoders[0].parameters())
    for l in range(1, L):
        params += list(decoders[l].parameters())

    optimizer = torch.optim.Adam(params, lr=c.lr)

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
                                               **kwargs)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=c.batch_size, shuffle=False, drop_last=False,
                                              **kwargs)

    N = 256  # hyperparameter that increases batch size for the decoder model by N
    print('train/test loader length', len(train_loader.dataset), len(test_loader.dataset))
    print('train/test loader batches', len(train_loader), len(test_loader))

    # Track AUROC, AP, and AUPRO metrics.
    det_roc_obs = Score_Observer('DET_AUROC')
    det_ap_obs = Score_Observer('DET_AP')
    seg_roc_obs = Score_Observer('SEG_AUROC')
    seg_ap_obs = Score_Observer('SEG_AP')
    seg_pro_obs = Score_Observer('SEG_AUPRO')

    if c.action_type == 'norm-test':
        c.meta_epochs = 1

    for epoch in range(c.meta_epochs):
        if c.action_type == 'norm-test' and c.checkpoint:
            load_weights(encoder, decoders, c.checkpoint)
        elif c.action_type == 'norm-train':
            print('Train meta epoch: {}'.format(epoch))
            train_meta_epoch(c, epoch, train_loader, encoder, decoders, optimizer, pool_layers, N)
        else:
            raise NotImplementedError('{} is not supported action type!'.format(c.action_type))

        # Evaluate every 25 meta-epochs, in test-only mode, and at the final epoch.
        is_eval_epoch = (epoch + 1) % 25 == 0 or (epoch + 1) == c.meta_epochs or c.action_type == 'norm-test'
        is_last_eval = (epoch + 1) == c.meta_epochs or c.action_type == 'norm-test'

        if is_eval_epoch:
            height, width, test_image_list, test_dist, gt_label_list, gt_mask_list = test_meta_epoch(
                c, epoch, test_loader, encoder, decoders, pool_layers, N)

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

            if is_last_eval and getattr(c, 'save_raw_anomaly_maps', True):
                export_raw_anomaly_maps_tiff(
                    c,
                    super_mask,
                    image_paths=getattr(test_loader.dataset, 'x', None),
                    epoch=epoch,
                )

            # calculate detection AUROC & AP
            score_label = np.max(super_mask, axis=(1, 2))
            gt_label = np.asarray(gt_label_list, dtype=bool)
            det_roc_auc = roc_auc_score(gt_label, score_label)
            det_ap = average_precision_score(gt_label, score_label)
            _ = det_roc_obs.update(100.0 * det_roc_auc, epoch)
            _ = det_ap_obs.update(100.0 * det_ap, epoch)

            # calculate segmentation AUROC & AP
            gt_mask = np.squeeze(np.asarray(gt_mask_list, dtype=bool), axis=1)
            seg_roc_auc = roc_auc_score(gt_mask.flatten(), super_mask.flatten())
            seg_ap = average_precision_score(gt_mask.flatten(), super_mask.flatten())
            save_best_seg_weights = seg_roc_obs.update(100.0 * seg_roc_auc, epoch)
            _ = seg_ap_obs.update(100.0 * seg_ap, epoch)

            if save_best_seg_weights and c.action_type != 'norm-test':
                save_weights(encoder, decoders, c.model, run_date)

                # calculate segmentation AUPRO
            if c.pro:
                max_step = 1000
                expect_fpr = 0.3
                max_th = super_mask.max()
                min_th = super_mask.min()
                delta = (max_th - min_th) / max_step
                ious_mean = []
                ious_std = []
                pros_mean = []
                pros_std = []
                threds = []
                fprs = []
                binary_score_maps = np.zeros_like(super_mask, dtype=bool)
                for step in range(max_step):
                    thred = max_th - step * delta
                    binary_score_maps[super_mask <= thred] = 0
                    binary_score_maps[super_mask > thred] = 1
                    pro = []
                    iou = []
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

                threds = np.array(threds)
                pros_mean = np.array(pros_mean)
                pros_std = np.array(pros_std)
                fprs = np.array(fprs)
                ious_mean = np.array(ious_mean)
                ious_std = np.array(ious_std)

                idx = fprs <= expect_fpr
                if np.sum(idx) > 0:
                    fprs_selected = fprs[idx]
                    fprs_selected = rescale(fprs_selected)
                    pros_mean_selected = pros_mean[idx]
                    seg_pro_auc = auc(fprs_selected, pros_mean_selected)
                    _ = seg_pro_obs.update(100.0 * seg_pro_auc, epoch)

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