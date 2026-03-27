"""
UrbanSAR - PyTorch Dataset

Dual-input dataset that returns (SAR tensor, optical tensor, height label)
for training the dual-branch CNN.
"""
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import IMG_SIZE, SAR_CHANNELS, OPTICAL_CHANNELS
from data.loader import load_sar_image, load_optical_image
from data.preprocessing import preprocess_pair


class UrbanSARDataset(Dataset):
    """
    PyTorch Dataset for UrbanSAR dual-input training.

    Each sample consists of:
        - SAR image tensor: [SAR_CHANNELS, IMG_SIZE, IMG_SIZE]
        - Optical image tensor: [OPTICAL_CHANNELS, IMG_SIZE, IMG_SIZE]
        - Height label: scalar (building height in meters)

    Supports data augmentation (random flips and rotations).
    """

    def __init__(
        self,
        chip_list: List[Dict],
        img_size: int = IMG_SIZE,
        augment: bool = False,
        preprocess: bool = True,
    ):
        """
        Args:
            chip_list: List of dicts with keys:
                - 'sar_path': path to SAR chip
                - 'optical_path': path to optical chip
                - 'height_m': building height in meters
                - (optional) 'building_id': building identifier
            img_size: Target image size (square)
            augment: Whether to apply data augmentation
            preprocess: Whether to apply SAR/optical preprocessing
        """
        self.chip_list = chip_list
        self.img_size = img_size
        self.augment = augment
        self.preprocess = preprocess

    def __len__(self) -> int:
        return len(self.chip_list)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        chip = self.chip_list[idx]

        # Load images — detect format (.npy vs .tif)
        sar_path = str(chip["sar_path"])
        opt_path = str(chip["optical_path"])

        if sar_path.endswith(".npy"):
            sar = np.load(sar_path).astype(np.float32)
        else:
            sar, _ = load_sar_image(sar_path)

        if opt_path.endswith(".npy"):
            optical = np.load(opt_path).astype(np.float32)
        else:
            optical, _ = load_optical_image(opt_path)

        # Preprocess
        if self.preprocess:
            sar, optical = preprocess_pair(sar, optical)

        # Resize to target size
        sar = self._resize(sar, self.img_size)
        optical = self._resize(optical, self.img_size)

        # Ensure correct channel count
        sar = self._ensure_channels(sar, SAR_CHANNELS)
        optical = self._ensure_channels(optical, OPTICAL_CHANNELS)

        # Data augmentation
        if self.augment:
            sar, optical = self._augment(sar, optical)

        # Convert to tensors
        sar_tensor = torch.from_numpy(sar).float()
        optical_tensor = torch.from_numpy(optical).float()
        height_tensor = torch.tensor(chip["height_m"], dtype=torch.float32)

        return sar_tensor, optical_tensor, height_tensor

    def _resize(self, image: np.ndarray, target_size: int) -> np.ndarray:
        """
        Resize image to target_size x target_size.

        Args:
            image: [C, H, W] array
            target_size: Target spatial dimension

        Returns:
            Resized [C, target_size, target_size] array
        """
        if image.shape[1] == target_size and image.shape[2] == target_size:
            return image

        channels = image.shape[0]
        resized = np.zeros((channels, target_size, target_size), dtype=np.float32)

        for c in range(channels):
            # Use PIL for high-quality resizing
            pil_img = Image.fromarray(image[c])
            pil_resized = pil_img.resize((target_size, target_size), Image.BILINEAR)
            resized[c] = np.array(pil_resized, dtype=np.float32)

        return resized

    def _ensure_channels(self, image: np.ndarray, target_channels: int) -> np.ndarray:
        """
        Ensure image has the correct number of channels.

        Strategies:
            - If image has fewer channels: repeat last channel to fill
            - If image has more channels: take first target_channels
        """
        current_channels = image.shape[0]

        if current_channels == target_channels:
            return image
        elif current_channels < target_channels:
            # Pad by repeating last channel
            padding = np.repeat(
                image[-1:],
                target_channels - current_channels,
                axis=0
            )
            return np.concatenate([image, padding], axis=0)
        else:
            # Take first target_channels
            return image[:target_channels]

    def _augment(
        self, sar: np.ndarray, optical: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply synchronized augmentation to SAR and optical images.

        Both images receive the SAME random transformation to maintain
        spatial alignment.
        """
        # Random horizontal flip
        if np.random.random() > 0.5:
            sar = sar[:, :, ::-1].copy()
            optical = optical[:, :, ::-1].copy()

        # Random vertical flip
        if np.random.random() > 0.5:
            sar = sar[:, ::-1, :].copy()
            optical = optical[:, ::-1, :].copy()

        # Random 90-degree rotation
        k = np.random.randint(0, 4)
        if k > 0:
            sar = np.rot90(sar, k=k, axes=(1, 2)).copy()
            optical = np.rot90(optical, k=k, axes=(1, 2)).copy()

        return sar, optical


def create_data_splits(
    chip_list: List[Dict],
    val_split: float = 0.2,
    seed: int = 42,
) -> Tuple[List[Dict], List[Dict]]:
    """
    Split chip list into training and validation sets.

    Uses stratified-like splitting by sorting by height first
    to ensure both sets have similar height distributions.

    Args:
        chip_list: Full list of chip dicts
        val_split: Fraction for validation
        seed: Random seed

    Returns:
        (train_chips, val_chips)
    """
    rng = np.random.RandomState(seed)

    # Sort by height for stratified-like split
    sorted_chips = sorted(chip_list, key=lambda x: x["height_m"])

    # Interleave selection: every Nth sample goes to validation
    n = int(1.0 / val_split)
    val_chips = []
    train_chips = []

    for i, chip in enumerate(sorted_chips):
        if i % n == 0:
            val_chips.append(chip)
        else:
            train_chips.append(chip)

    # Shuffle both sets
    rng.shuffle(train_chips)
    rng.shuffle(val_chips)

    print(f"[INFO] Data split: {len(train_chips)} train, {len(val_chips)} val")
    return train_chips, val_chips
