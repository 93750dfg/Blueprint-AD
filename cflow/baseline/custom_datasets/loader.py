import os
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms as T
from torchvision.io import read_video, write_jpeg

__all__ = ("MVTecDataset", "StcDataset")


MVTEC_CLASS_NAMES = [
    "bottle", "cable", "capsule", "carpet", "grid",
    "hazelnut", "leather", "metal_nut", "pill", "screw",
    "tile", "toothbrush", "transistor", "wood", "zipper",
]

LOCO_CLASS_NAMES = [
    "breakfast_box",
    "juice_bottle",
    "pushpins",
    "screw_bag",
    "splicing_connectors",
]

STC_CLASS_NAMES = [
    "01", "02", "03", "04", "05", "06",
    "07", "08", "09", "10", "11", "12",
]

IMG_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
MASK_EXTENSIONS = IMG_EXTENSIONS
GRAY_AS_RGB_CLASSES = {"zipper", "screw", "grid"}


def is_image_file(filename: str) -> bool:
    return filename.lower().endswith(IMG_EXTENSIONS)


def is_mask_file(filename: str) -> bool:
    return filename.lower().endswith(MASK_EXTENSIONS)


def _list_images(folder: str):
    if not os.path.isdir(folder):
        return []
    return sorted(
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if is_image_file(f)
    )


def find_mask_entry(gt_type_dir: str, stem: str):
    """
    Return a mask entry for one abnormal test image.

    Supported layouts:
      1. LOCO:  ground_truth/type/000/*.png
      2. VisA:  ground_truth/type/000.png
      3. MVTec: ground_truth/type/000_mask.png

    The returned entry is either a single path or a list of paths. A list is
    merged later by pixel-wise maximum.
    """
    # LOCO style: ground_truth/logical_anomalies/000/*.png
    for folder in [
        os.path.join(gt_type_dir, stem),
        os.path.join(gt_type_dir, stem + "_mask"),
    ]:
        if os.path.isdir(folder):
            mask_files = _list_images(folder)
            if mask_files:
                return mask_files
            raise FileNotFoundError(
                f"Mask folder exists but contains no mask images: {folder}"
            )

    # VisA / MVTec style: ground_truth/type/000.png or 000_mask.png
    for name in [
        f"{stem}.png", f"{stem}.jpg", f"{stem}.jpeg", f"{stem}.bmp", f"{stem}.tif", f"{stem}.tiff",
        f"{stem}_mask.png", f"{stem}_mask.jpg", f"{stem}_mask.jpeg",
        f"{stem}_mask.bmp", f"{stem}_mask.tif", f"{stem}_mask.tiff",
    ]:
        path = os.path.join(gt_type_dir, name)
        if os.path.exists(path):
            return path

    raise FileNotFoundError(
        f"Cannot find mask for image stem={stem} in {gt_type_dir}"
    )


def load_mask_entry(mask_entry, target_size=None):
    """
    Load a mask entry.

    Args:
        mask_entry: str path, or list/tuple of str paths.
        target_size: optional (H, W). If provided, each mask is resized before
            merging. This avoids high RAM usage on LOCO's large masks.
    """
    if mask_entry is None:
        raise ValueError("Abnormal sample has None mask_entry.")

    def open_mask_as_array(path: str):
        img = Image.open(path).convert("L")
        if target_size is not None:
            h, w = target_size
            img = img.resize((w, h), Image.NEAREST)
        return np.asarray(img, dtype=np.uint8)

    if isinstance(mask_entry, str):
        return Image.fromarray(open_mask_as_array(mask_entry))

    if isinstance(mask_entry, (list, tuple)):
        if len(mask_entry) == 0:
            raise ValueError("Empty mask list for abnormal sample.")

        merged = None
        for p in mask_entry:
            arr = open_mask_as_array(p)

            if merged is None:
                # Copy to ensure the merged array is writable.
                merged = arr.copy()
            else:
                # Avoid in-place output on potentially read-only arrays.
                merged = np.maximum(merged, arr)

        return Image.fromarray(merged.astype(np.uint8))

    raise TypeError(f"Unsupported mask entry type: {type(mask_entry)}")


class StcDataset(Dataset):
    def __init__(self, c, is_train=True):
        assert c.class_name in STC_CLASS_NAMES, (
            f"class_name: {c.class_name}, should be in {STC_CLASS_NAMES}"
        )
        self.class_name = c.class_name
        self.is_train = is_train
        self.cropsize = c.crp_size

        if is_train:
            self.dataset_path = os.path.join(c.data_path, "training")
            self.dataset_vid = os.path.join(self.dataset_path, "videos")
            self.dataset_dir = os.path.join(self.dataset_path, "frames")
            self.dataset_files = sorted(
                f for f in os.listdir(self.dataset_vid)
                if f.startswith(self.class_name)
            )
            if not os.path.isdir(self.dataset_dir):
                os.mkdir(self.dataset_dir)

            done_file = os.path.join(
                self.dataset_path, f"frames_{self.class_name}.pt"
            )
            print(done_file)
            H, W = 480, 856
            if os.path.isfile(done_file):
                assert torch.load(done_file) == len(self.dataset_files), (
                    "train frames are not processed!"
                )
            else:
                count = 0
                for dataset_file in self.dataset_files:
                    print(dataset_file)
                    data = read_video(os.path.join(self.dataset_vid, dataset_file))
                    vid = data[0]  # [T, H, W, C]
                    print(
                        "video mu/std: {}/{} {}".format(
                            torch.mean(vid / 255.0, (0, 1, 2)),
                            torch.std(vid / 255.0, (0, 1, 2)),
                            vid.shape,
                        )
                    )
                    assert [H, W] == [vid.size(1), vid.size(2)], "same H/W"
                    dataset_file_dir = os.path.join(
                        self.dataset_dir, os.path.splitext(dataset_file)[0]
                    )
                    os.mkdir(dataset_file_dir)
                    count += 1
                    for i, frame in enumerate(vid):
                        filename = f"{i:08d}.jpg"
                        write_jpeg(
                            frame.permute((2, 0, 1)),
                            os.path.join(dataset_file_dir, filename),
                            80,
                        )
                torch.save(torch.tensor(count), done_file)

            self.x, self.y, self.mask = self.load_dataset_folder()
        else:
            self.dataset_path = os.path.join(c.data_path, "testing")
            self.x, self.y, self.mask = self.load_dataset_folder()

        if is_train:
            self.transform_x = T.Compose([
                T.Resize(c.img_size, Image.Resampling.LANCZOS),
                T.RandomRotation(5),
                T.CenterCrop(c.crp_size),
                T.ToTensor(),
            ])
        else:
            self.transform_x = T.Compose([
                T.Resize(c.img_size, Image.Resampling.LANCZOS),
                T.CenterCrop(c.crp_size),
                T.ToTensor(),
            ])

        self.transform_mask = T.Compose([
            T.ToPILImage(),
            T.Resize(c.img_size, Image.NEAREST),
            T.CenterCrop(c.crp_size),
            T.ToTensor(),
        ])
        self.normalize = T.Compose([T.Normalize(c.norm_mean, c.norm_std)])

    def __getitem__(self, idx):
        x, y, mask = self.x[idx], self.y[idx], self.mask[idx]
        x = Image.open(x).convert("RGB")
        x = self.normalize(self.transform_x(x))
        if y == 0:
            mask = torch.zeros([1, self.cropsize[0], self.cropsize[1]])
        else:
            mask = self.transform_mask(mask)
        return x, y, mask

    def __len__(self):
        return len(self.x)

    def load_dataset_folder(self):
        phase = "train" if self.is_train else "test"
        x, y, mask = [], [], []

        img_dir = os.path.join(self.dataset_path, self.class_name, phase)
        gt_frame_dir = os.path.join(self.dataset_path, "test_frame_mask")
        gt_pixel_dir = os.path.join(self.dataset_path, "test_pixel_mask")

        img_types = sorted(
            f for f in os.listdir(img_dir)
            if os.path.isdir(os.path.join(img_dir, f))
        )

        for i, img_type in enumerate(img_types):
            print("Folder:", img_type)
            img_type_dir = os.path.join(img_dir, img_type)
            img_fpath_list = _list_images(img_type_dir)
            x.extend(img_fpath_list)

            if phase == "test":
                gt_pixel = np.load(f"{os.path.join(gt_pixel_dir, img_type)}.npy")
                gt_frame = np.load(f"{os.path.join(gt_frame_dir, img_type)}.npy")
                if i == 0:
                    m = gt_pixel
                    y = gt_frame
                else:
                    m = np.concatenate((m, gt_pixel), axis=0)
                    y = np.concatenate((y, gt_frame), axis=0)
                mask = [e for e in m]
                assert len(x) == len(y), "number of x and y should be same"
                assert len(x) == len(mask), "number of x and mask should be same"
            else:
                mask.extend([None] * len(img_fpath_list))
                y.extend([0] * len(img_fpath_list))

        return list(x), list(y), list(mask)


class MVTecDataset(Dataset):
    """
    MVTec-style dataset loader for MVTec AD, VisA converted to MVTec layout,
    and MVTec LOCO.

    Expected layout:
        root/class_name/train/good/*.png or *.jpg
        root/class_name/test/good/*
        root/class_name/test/<defect_type>/*
        root/class_name/ground_truth/<defect_type>/<mask files or folders>

    LOCO folder-style ground truth is supported by merging all masks in
    ground_truth/<defect_type>/<image_stem>/ with pixel-wise max.
    """

    def __init__(self, c, is_train=True):
        self.dataset = c.dataset
        self.dataset_path = c.data_path
        self.class_name = c.class_name
        self.is_train = is_train
        self.cropsize = c.crp_size
        self.cache_in_ram = getattr(c, "cache_in_ram", True)
        self.cache_resized_images = getattr(c, "cache_resized_images", True)
        self.disable_random_rotation = getattr(c, "disable_random_rotation", False)

        self._check_dataset_class()
        self.x, self.y, self.mask = self.load_dataset_folder()

        self.transform_x = self._build_image_transform(c, is_train)
        self.transform_mask = T.Compose([
            T.Resize(c.img_size, Image.NEAREST),
            T.CenterCrop(c.crp_size),
            T.ToTensor(),
        ])
        self.normalize = T.Compose([T.Normalize(c.norm_mean, c.norm_std)])

        self.ram_cache_x = None
        self.ram_cache_mask = None
        if self.cache_in_ram:
            self._build_ram_cache(c)

    def _check_dataset_class(self):
        if self.dataset == "mvtec":
            assert self.class_name in MVTEC_CLASS_NAMES, (
                f"class_name: {self.class_name}, should be in {MVTEC_CLASS_NAMES}"
            )
        elif self.dataset == "loco":
            assert self.class_name in LOCO_CLASS_NAMES, (
                f"class_name: {self.class_name}, should be in {LOCO_CLASS_NAMES}"
            )
        elif self.dataset == "visa":
            class_dir = os.path.join(self.dataset_path, self.class_name)
            assert os.path.isdir(class_dir), f"VisA class folder not found: {class_dir}"
        else:
            raise NotImplementedError(
                f"{self.dataset} is not supported by MVTecDataset"
            )

    def _build_image_transform(self, c, is_train):
        transforms = [T.Resize(c.img_size, Image.Resampling.LANCZOS)]
        if is_train and not self.disable_random_rotation:
            transforms.append(T.RandomRotation(5))
        transforms.extend([T.CenterCrop(c.crp_size), T.ToTensor()])
        return T.Compose(transforms)

    def _load_image(self, path: str) -> Image.Image:
        img = Image.open(path)
        if self.class_name in GRAY_AS_RGB_CLASSES:
            arr = np.asarray(img)
            if arr.ndim == 2:
                arr = np.repeat(arr[..., None], 3, axis=2)
            img = Image.fromarray(arr.astype("uint8"))
        return img.convert("RGB")

    def _build_ram_cache(self, c):
        print(f"[INFO] Caching {self.class_name} dataset in memory...")
        self.ram_cache_x = []
        self.ram_cache_mask = []

        for path in self.x:
            img = self._load_image(path)
            # Keep the original operation order equivalent to the original code:
            # Resize first, then RandomRotation, then CenterCrop.
            # Caching the resized image greatly reduces RAM usage on LOCO.
            if self.cache_resized_images:
                h, w = c.img_size
                img = img.resize((w, h), Image.Resampling.LANCZOS)
            self.ram_cache_x.append(img.copy())

        if not self.is_train:
            for idx, (mask_entry, y) in enumerate(zip(self.mask, self.y)):
                if y == 0:
                    self.ram_cache_mask.append(None)
                    continue

                if mask_entry is None:
                    raise RuntimeError(
                        "Abnormal test sample has None mask. "
                        f"idx={idx}, image={self.x[idx]}"
                    )

                self.ram_cache_mask.append(
                    load_mask_entry(mask_entry, target_size=self.cropsize)
                )

        print(f"[INFO] Dataset cache ready: {len(self.x)} images.")

    def __getitem__(self, idx):
        y = self.y[idx]

        if self.cache_in_ram:
            x_img = self.ram_cache_x[idx]
        else:
            x_img = self._load_image(self.x[idx])

        x = self.normalize(self.transform_x(x_img))

        if y == 0:
            mask = torch.zeros([1, self.cropsize[0], self.cropsize[1]])
        else:
            if self.cache_in_ram:
                mask_img = self.ram_cache_mask[idx]
            else:
                mask_img = load_mask_entry(self.mask[idx], target_size=self.cropsize)
            mask = self.transform_mask(mask_img)

        return x, y, mask

    def __len__(self):
        return len(self.x)

    def load_dataset_folder(self):
        phase = "train" if self.is_train else "test"
        x, y, mask = [], [], []

        img_dir = os.path.join(self.dataset_path, self.class_name, phase)
        gt_dir = os.path.join(self.dataset_path, self.class_name, "ground_truth")

        img_types = sorted(
            d for d in os.listdir(img_dir)
            if os.path.isdir(os.path.join(img_dir, d))
        )

        for img_type in img_types:
            img_type_dir = os.path.join(img_dir, img_type)
            img_fpath_list = _list_images(img_type_dir)
            x.extend(img_fpath_list)

            if img_type == "good":
                y.extend([0] * len(img_fpath_list))
                mask.extend([None] * len(img_fpath_list))
                continue

            y.extend([1] * len(img_fpath_list))
            gt_type_dir = os.path.join(gt_dir, img_type)
            img_stems = [
                os.path.splitext(os.path.basename(path))[0]
                for path in img_fpath_list
            ]
            mask.extend(find_mask_entry(gt_type_dir, stem) for stem in img_stems)

        assert len(x) == len(y), "number of x and y should be same"
        assert len(x) == len(mask), "number of x and mask should be same"
        return list(x), list(y), list(mask)
