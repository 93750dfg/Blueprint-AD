import os
import datetime
import numpy as np
from skimage import morphology
from skimage.segmentation import mark_boundaries
import matplotlib.pyplot as plt
import matplotlib
from PIL import Image

try:
    import tifffile
except ImportError:
    tifffile = None


OUT_DIR = './viz/'

norm = matplotlib.colors.Normalize(vmin=0.0, vmax=255.0)
cm = 1/2.54
dpi = 300

def denormalization(x, norm_mean, norm_std):
    mean = np.array(norm_mean)
    std = np.array(norm_std)
    x = (((x.transpose(1, 2, 0) * std) + mean) * 255.).astype(np.uint8)
    return x


def export_hist(c, gts, scores, threshold):
    print('Exporting histogram...')
    plt.rcParams.update({'font.size': 4})
    image_dirs = os.path.join(OUT_DIR, c.model)
    os.makedirs(image_dirs, exist_ok=True)
    Y = scores.flatten()
    Y_label = gts.flatten()
    fig = plt.figure(figsize=(4*cm, 4*cm), dpi=dpi)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    fig.add_axes(ax)
    plt.hist([Y[Y_label==1], Y[Y_label==0]], 500, density=True, color=['r', 'g'], label=['ANO', 'TYP'], alpha=0.75, histtype='barstacked')
    image_file = os.path.join(image_dirs, 'hist_images_' + datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
    fig.savefig(image_file, dpi=dpi, format='svg', bbox_inches = 'tight', pad_inches = 0.0)
    plt.close()

def export_groundtruth(c, test_img, gts):
    image_dirs = os.path.join(OUT_DIR, c.model, 'gt_images_' + datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
    # images
    if not os.path.isdir(image_dirs):
        print('Exporting grountruth...')
        os.makedirs(image_dirs, exist_ok=True)
        num = len(test_img)
        kernel = morphology.disk(4)
        for i in range(num):
            img = test_img[i]
            img = denormalization(img, c.norm_mean, c.norm_std)
            # gts
            gt_mask = gts[i].astype(np.float64)
            gt_mask = morphology.opening(gt_mask, kernel)
            gt_mask = (255.0*gt_mask).astype(np.uint8)
            gt_img = mark_boundaries(img, gt_mask, color=(1, 0, 0), mode='thick')
            #
            fig = plt.figure(figsize=(2*cm, 2*cm), dpi=dpi)
            ax = plt.Axes(fig, [0., 0., 1., 1.])
            ax.set_axis_off()
            fig.add_axes(ax)
            ax.imshow(gt_img)
            image_file = os.path.join(image_dirs, '{:08d}'.format(i))
            fig.savefig(image_file, dpi=dpi, format='svg', bbox_inches = 'tight', pad_inches = 0.0)
            plt.close()


def export_scores(c, test_img, scores, threshold):
    image_dirs = os.path.join(OUT_DIR, c.model, 'sc_images_' + datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
    # images
    if not os.path.isdir(image_dirs):
        print('Exporting scores...')
        os.makedirs(image_dirs, exist_ok=True)
        num = len(test_img)
        kernel = morphology.disk(4)
        scores_norm = 1.0/scores.max()
        for i in range(num):
            img = test_img[i]
            img = denormalization(img, c.norm_mean, c.norm_std)
            # scores
            score_mask = np.zeros_like(scores[i])
            score_mask[scores[i] >  threshold] = 1.0
            score_mask = morphology.opening(score_mask, kernel)
            score_mask = (255.0*score_mask).astype(np.uint8)
            score_img = mark_boundaries(img, score_mask, color=(1, 0, 0), mode='thick')
            score_map = (255.0*scores[i]*scores_norm).astype(np.uint8)
            #
            fig_img, ax_img = plt.subplots(2, 1, figsize=(2*cm, 4*cm))
            for ax_i in ax_img:
                ax_i.axes.xaxis.set_visible(False)
                ax_i.axes.yaxis.set_visible(False)
                ax_i.spines['top'].set_visible(False)
                ax_i.spines['right'].set_visible(False)
                ax_i.spines['bottom'].set_visible(False)
                ax_i.spines['left'].set_visible(False)
            #
            plt.subplots_adjust(hspace = 0.1, wspace = 0.1)
            ax_img[0].imshow(img, cmap='gray', interpolation='none')
            ax_img[0].imshow(score_map, cmap='jet', norm=norm, alpha=0.5, interpolation='none')
            ax_img[1].imshow(score_img)
            image_file = os.path.join(image_dirs, '{:08d}'.format(i))
            fig_img.savefig(image_file, dpi=dpi, format='svg', bbox_inches = 'tight', pad_inches = 0.0)
            plt.close()


def export_test_images(c, test_img, gts, scores, threshold):
    image_dirs = os.path.join(OUT_DIR, c.model, 'images_' + datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
    cm = 1/2.54
    # images
    if not os.path.isdir(image_dirs):
        print('Exporting images...')
        os.makedirs(image_dirs, exist_ok=True)
        num = len(test_img)
        font = {'family': 'serif', 'color': 'black', 'weight': 'normal', 'size': 8}
        kernel = morphology.disk(4)
        scores_norm = 1.0/scores.max()
        for i in range(num):
            img = test_img[i]
            img = denormalization(img, c.norm_mean, c.norm_std)
            # gts
            gt_mask = gts[i].astype(np.float64)
            print('GT:', i, gt_mask.sum())
            gt_mask = morphology.opening(gt_mask, kernel)
            gt_mask = (255.0*gt_mask).astype(np.uint8)
            gt_img = mark_boundaries(img, gt_mask, color=(1, 0, 0), mode='thick')
            # scores
            score_mask = np.zeros_like(scores[i])
            score_mask[scores[i] >  threshold] = 1.0
            print('SC:', i, score_mask.sum())
            score_mask = morphology.opening(score_mask, kernel)
            score_mask = (255.0*score_mask).astype(np.uint8)
            score_img = mark_boundaries(img, score_mask, color=(1, 0, 0), mode='thick')
            score_map = (255.0*scores[i]*scores_norm).astype(np.uint8)
            #
            fig_img, ax_img = plt.subplots(3, 1, figsize=(2*cm, 6*cm))
            for ax_i in ax_img:
                ax_i.axes.xaxis.set_visible(False)
                ax_i.axes.yaxis.set_visible(False)
                ax_i.spines['top'].set_visible(False)
                ax_i.spines['right'].set_visible(False)
                ax_i.spines['bottom'].set_visible(False)
                ax_i.spines['left'].set_visible(False)
            #
            plt.subplots_adjust(hspace = 0.1, wspace = 0.1)
            ax_img[0].imshow(gt_img)
            ax_img[1].imshow(score_map, cmap='jet', norm=norm)
            ax_img[2].imshow(score_img)
            image_file = os.path.join(image_dirs, '{:08d}'.format(i))
            fig_img.savefig(image_file, dpi=dpi, format='svg', bbox_inches = 'tight', pad_inches = 0.0)
            plt.close()


def _get_test_rel_info(image_path, class_name):
    """
    Infer defect type and image stem from a MVTec-style path:
        root/class_name/test/defect_type/000.png
    Returns:
        defect_type, stem
    """
    norm_path = os.path.normpath(str(image_path))
    parts = norm_path.split(os.sep)

    defect_type = os.path.basename(os.path.dirname(norm_path))
    if 'test' in parts:
        test_idx = len(parts) - 1 - parts[::-1].index('test')
        if test_idx + 1 < len(parts):
            defect_type = parts[test_idx + 1]

    stem = os.path.splitext(os.path.basename(norm_path))[0]
    return defect_type, stem


def _resize_score_to_image_size(score_map, image_path):
    """
    Resize a raw anomaly score map to the original image size while preserving
    floating-point scores. PIL expects size=(W, H).
    """
    with Image.open(image_path) as img:
        width, height = img.size

    if score_map.shape == (height, width):
        return score_map.astype(np.float32, copy=False)

    score_img = Image.fromarray(score_map.astype(np.float32))
    score_img = score_img.resize((width, height), Image.BILINEAR)
    return np.asarray(score_img, dtype=np.float32)


def export_raw_anomaly_maps_tiff(c, scores, image_paths, epoch=None):
    """
    Export raw anomaly score maps as float32 TIFF files.

    Directory layout:
        <raw_root>/<class_name>/<defect_type>/<image_stem>.tiff

    For LOCO, this mirrors:
        <dataset_root>/<class_name>/test/<defect_type>/<image_stem>.png

    The exported values are raw anomaly scores from super_mask: no thresholding,
    no RGB overlay, no GT drawing, no uint8 normalization.
    """
    if image_paths is None:
        raise ValueError('image_paths is required to export raw anomaly maps.')

    if tifffile is None:
        raise ImportError('Please install tifffile first: pip install tifffile')

    scores = np.asarray(scores, dtype=np.float32)
    if len(scores) != len(image_paths):
        raise ValueError(
            f'Number of score maps ({len(scores)}) does not match number of images ({len(image_paths)}).'
        )

    default_dataset_dir = 'mvtec_loco' if getattr(c, 'dataset', '') == 'loco' else getattr(c, 'dataset', 'dataset')
    raw_root = getattr(c, 'raw_anomaly_maps_dir', None)
    if raw_root is None:
        run_name = str(getattr(c, 'run_name', 'run'))
        raw_root = os.path.join('output', run_name, 'anomaly_maps', default_dataset_dir)

    save_original_size = getattr(c, 'save_raw_maps_original_size', getattr(c, 'dataset', '') == 'loco')

    count = 0
    for i, image_path in enumerate(image_paths):
        defect_type, stem = _get_test_rel_info(image_path, getattr(c, 'class_name', 'class'))
        out_dir = os.path.join(raw_root, c.class_name, defect_type)
        os.makedirs(out_dir, exist_ok=True)

        score_map = np.nan_to_num(scores[i].astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        if save_original_size:
            score_map = _resize_score_to_image_size(score_map, image_path)

        out_path = os.path.join(out_dir, f'{stem}.tiff')
        tifffile.imwrite(out_path, score_map.astype(np.float32))
        count += 1

    size_msg = 'original image size' if save_original_size else f'model map size {scores.shape[-2]}x{scores.shape[-1]}'
    print(f'[INFO] Exported {count} raw anomaly maps as float32 TIFF to: {raw_root}')
    print(f'[INFO] Raw anomaly map save size: {size_msg}')
    return raw_root
