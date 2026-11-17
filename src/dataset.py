"""
Data pipeline, preprocessing, and dataset loaders for BanglaLekha-Isolated.
"""

import os
import io
import json
import zipfile
import random
from typing import Tuple, List, Optional, Union, Dict

import cv2
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader


def get_label_mapping(labels_path: str = "labels.json") -> Dict[str, str]:
    """Load the 84-class mapping from JSON file."""
    if not os.path.isabs(labels_path):
        labels_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), labels_path)
    with open(labels_path, "r", encoding="utf-8") as f:
        return json.load(f)


def preprocess_image(
    image: Union[np.ndarray, Image.Image],
    target_size: Tuple[int, int] = (64, 64),
    pad_ratio: float = 0.15
) -> np.ndarray:
    """
    Standardize a handwritten character image:
    1. Grayscale conversion
    2. Polarity normalization (white stroke on black background)
    3. Noise filtering & morphological opening
    4. Tight bounding box crop with proportional aspect-ratio padding
    5. Resize to target_size (e.g., 64x64)
    6. Normalize to [0.0, 1.0] float32 with shape (H, W, 1)
    """
    if isinstance(image, Image.Image):
        img_arr = np.array(image.convert("L"))
    elif isinstance(image, np.ndarray):
        if image.ndim == 3:
            if image.shape[2] == 4:  # RGBA
                # If transparent background, composite onto white
                alpha = image[:, :, 3] / 255.0
                rgb = image[:, :, :3]
                white_bg = np.ones_like(rgb, dtype=np.uint8) * 255
                comp = (rgb * alpha[:, :, None] + white_bg * (1.0 - alpha[:, :, None])).astype(np.uint8)
                img_arr = cv2.cvtColor(comp, cv2.COLOR_RGB2GRAY)
            elif image.shape[2] == 3:
                img_arr = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            else:
                img_arr = image.squeeze(axis=2)
        else:
            img_arr = image.copy()
    else:
        raise ValueError(f"Unsupported image type: {type(image)}")

    img_arr = img_arr.astype(np.uint8)

    # Ensure stroke is white (255) on dark (0) background
    # Check if corners are mostly bright -> background is white
    h, w = img_arr.shape
    corner_samples = [
        img_arr[:10, :10],
        img_arr[:10, -10:],
        img_arr[-10:, :10],
        img_arr[-10:, -10:]
    ]
    mean_corner = np.mean([np.mean(c) for c in corner_samples])
    if mean_corner > 127:
        img_arr = cv2.bitwise_not(img_arr)

    # Threshold to separate stroke from paper background
    _, thresh = cv2.threshold(img_arr, 30, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    # Find bounding box of foreground stroke
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        x_min, y_min = w, h
        x_max, y_max = 0, 0
        for cnt in contours:
            x, y, cw, ch = cv2.boundingRect(cnt)
            x_min = min(x_min, x)
            y_min = min(y_min, y)
            x_max = max(x_max, x + cw)
            y_max = max(y_max, y + ch)

        crop = img_arr[y_min:y_max, x_min:x_max]
    else:
        crop = img_arr

    ch, cw = crop.shape
    if ch == 0 or cw == 0:
        crop = img_arr
        ch, cw = crop.shape

    # Fit into square canvas maintaining aspect ratio
    max_dim = max(ch, cw)
    pad = int(max_dim * pad_ratio)
    square_dim = max_dim + 2 * pad
    square = np.zeros((square_dim, square_dim), dtype=np.uint8)

    y_offset = (square_dim - ch) // 2
    x_offset = (square_dim - cw) // 2
    square[y_offset:y_offset + ch, x_offset:x_offset + cw] = crop

    # Resize to target
    resized = cv2.resize(square, target_size, interpolation=cv2.INTER_AREA)

    # Normalize to [0.0, 1.0]
    normalized = resized.astype(np.float32) / 255.0
    return np.expand_dims(normalized, axis=-1)


class BanglaLekhaDataset(Dataset):
    """
    High-performance PyTorch Dataset streaming directly from BanglaLekha zip archive.
    Avoids uncompressing 166,000 files to disk while supporting real-time augmentations.
    """

    def __init__(
        self,
        zip_path: str = "Datasets/BanglaLekha-Isolated (1).zip",
        split: str = "train",
        val_split: float = 0.15,
        target_size: Tuple[int, int] = (64, 64),
        max_samples_per_class: Optional[int] = None,
        augment: bool = False,
        seed: int = 42
    ):
        self.zip_path = zip_path
        self.target_size = target_size
        self.augment = augment
        self._zf = None  # Lazily opened per worker

        if not os.path.exists(self.zip_path):
            raise FileNotFoundError(f"Dataset archive not found: {self.zip_path}")

        # Index the archive
        with zipfile.ZipFile(self.zip_path, "r") as zf:
            all_names = zf.namelist()

        class_entries: Dict[int, List[str]] = {i: [] for i in range(84)}
        for name in all_names:
            if name.endswith(".png") and "/Images/" in name:
                parts = name.split("/")
                try:
                    folder_id = int(parts[2])
                    class_idx = folder_id - 1
                    if 0 <= class_idx < 84:
                        class_entries[class_idx].append(name)
                except (IndexError, ValueError):
                    continue

        rng = random.Random(seed)
        self.samples: List[Tuple[str, int]] = []

        for class_idx in range(84):
            entries = class_entries[class_idx]
            rng.shuffle(entries)
            if max_samples_per_class is not None:
                entries = entries[:max_samples_per_class]

            n_val = int(len(entries) * val_split)
            if split == "train":
                selected = entries[n_val:]
            else:
                selected = entries[:n_val]

            for s in selected:
                self.samples.append((s, class_idx))

        rng.shuffle(self.samples)

    def _get_zip(self):
        if self._zf is None:
            self._zf = zipfile.ZipFile(self.zip_path, "r")
        return self._zf

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        zf = self._get_zip()
        entry_name, label = self.samples[idx]
        img_bytes = zf.read(entry_name)
        img = Image.open(io.BytesIO(img_bytes)).convert("L")
        arr = np.array(img)

        processed = preprocess_image(arr, target_size=self.target_size)

        if self.augment:
            processed = self._apply_augmentation(processed)

        # PyTorch format: (C, H, W)
        tensor_img = torch.from_numpy(processed.transpose(2, 0, 1)).float()
        return tensor_img, label

    def _apply_augmentation(self, img: np.ndarray) -> np.ndarray:
        """Apply random affine rotation and translation."""
        h, w = img.shape[:2]
        angle = random.uniform(-12, 12)
        dx = random.uniform(-4, 4)
        dy = random.uniform(-4, 4)
        scale = random.uniform(0.9, 1.1)

        center = (w / 2, h / 2)
        m = cv2.getRotationMatrix2D(center, angle, scale)
        m[0, 2] += dx
        m[1, 2] += dy

        squeezed = img.squeeze(axis=-1)
        augmented = cv2.warpAffine(squeezed, m, (w, h), borderValue=0)
        return np.expand_dims(augmented, axis=-1)


def create_dataloaders(
    zip_path: str = "Datasets/BanglaLekha-Isolated (1).zip",
    batch_size: int = 64,
    val_split: float = 0.15,
    max_samples_per_class: Optional[int] = None,
    target_size: Tuple[int, int] = (64, 64),
    num_workers: int = 0
) -> Tuple[DataLoader, DataLoader]:
    """Create train and validation PyTorch DataLoaders."""
    train_dataset = BanglaLekhaDataset(
        zip_path=zip_path,
        split="train",
        val_split=val_split,
        target_size=target_size,
        max_samples_per_class=max_samples_per_class,
        augment=True
    )
    val_dataset = BanglaLekhaDataset(
        zip_path=zip_path,
        split="val",
        val_split=val_split,
        target_size=target_size,
        max_samples_per_class=max_samples_per_class,
        augment=False
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    return train_loader, val_loader
