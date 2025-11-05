from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from . import config


# Sınıf isimlerini indekslere eşleyerek modeli ve raporlamayı ortak noktada buluşturuyoruz.
LABEL_TO_INDEX = {name: idx for idx, name in enumerate(config.CLASS_NAMES)}
SUPPORTED_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def _canonical_label(folder_name: str) -> str | None:
    """Map inconsistent folder names onto canonical class identifiers."""
    return config.LABEL_CANONICAL_MAP.get(folder_name.lower())


def _gather_samples(split_dir: Path) -> List[Tuple[Path, int]]:
    samples: List[Tuple[Path, int]] = []
    for class_dir in sorted(split_dir.iterdir()):
        if not class_dir.is_dir():
            continue
        canonical = _canonical_label(class_dir.name)
        if canonical is None:
            continue
        target_idx = LABEL_TO_INDEX[canonical]
        for image_path in sorted(class_dir.glob("*")):
            if (
                image_path.is_file()
                and image_path.suffix.lower() in SUPPORTED_EXTENSIONS
            ):
                # Desteklenen görselleri etiketleriyle birlikte veri setine ekliyoruz.
                samples.append((image_path, target_idx))
    return samples


class ASDDataset(Dataset):
    """Simple dataset wrapper around the cleaned folder structure."""

    def __init__(
        self,
        split: str,
        transform: Callable | None = None,
        root: Path | None = None,
    ) -> None:
        self.root = Path(root or config.DATA_ROOT)
        self.split = split
        self.transform = transform

        split_dir = self.root / split
        if not split_dir.exists():
            raise FileNotFoundError(f"Split not found: {split_dir}")

        self.samples = _gather_samples(split_dir)
        if not self.samples:
            raise RuntimeError(f"No samples found under {split_dir}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        image_path, target = self.samples[index]
        with Image.open(image_path) as img:
            img = img.convert("RGB")
        if self.transform:
            img = self.transform(img)
        # DataLoader tarafında ham dosya yolunu döndürmek hata takibini kolaylaştırıyor.
        return img, target, str(image_path)


def default_transforms(train: bool = False) -> transforms.Compose:
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )

    if train:
        augmentations: List[Callable] = [
            transforms.Resize(int(config.IMAGE_SIZE * 1.1)),
            transforms.RandomResizedCrop(
                config.IMAGE_SIZE,
                scale=(0.8, 1.0),
                ratio=(0.9, 1.1),
            ),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply(
                [
                    transforms.ColorJitter(
                        brightness=0.25, contrast=0.25, saturation=0.25, hue=0.02
                    )
                ],
                p=0.5,
            ),
            transforms.RandomApply(
                [
                    transforms.GaussianBlur(
                        kernel_size=3, sigma=(0.1, 2.0)
                    )
                ],
                p=0.15,
            ),
            transforms.RandomRotation(
                8,
                fill=tuple(int(c * 255) for c in (0.485, 0.456, 0.406)),
            ),
            transforms.ToTensor(),
            normalize,
            transforms.RandomErasing(p=0.15, value="random"),
        ]
    else:
        augmentations = [
            transforms.Resize(int(config.IMAGE_SIZE * 1.1)),
            transforms.CenterCrop(config.IMAGE_SIZE),
            transforms.ToTensor(),
            normalize,
        ]
    # Aynı normalize değerleri ile eğitim ve doğrulama dağılımları dengeleniyor.
    return transforms.Compose(augmentations)


@dataclass
class DataLoaders:
    train: DataLoader
    val: DataLoader
    test: DataLoader | None = None


def create_dataloaders(
    batch_size: int | None = None,
    num_workers: int | None = None,
) -> DataLoaders:
    batch_size = batch_size or config.BATCH_SIZE
    num_workers = num_workers or config.NUM_WORKERS

    train_dataset = ASDDataset("Train", transform=default_transforms(train=True))
    val_dataset = ASDDataset("valid", transform=default_transforms())
    test_dataset = None
    test_split_dir = config.DATA_ROOT / "Test"
    if test_split_dir.exists():
        test_dataset = ASDDataset("Test", transform=default_transforms())

    generator = torch_generator()

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        generator=generator,
        persistent_workers=num_workers > 0,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
    )

    test_loader = (
        DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=num_workers > 0,
        )
        if test_dataset
        else None
    )

    # Üç DataLoader'ı tek bir dataclass içinde döndürmek eğitim akışını sadeleştiriyor.
    return DataLoaders(train_loader, val_loader, test_loader)


def torch_generator():
    import torch

    generator = torch.Generator()
    generator.manual_seed(config.SEED)
    return generator


def compute_class_weights(dataset: ASDDataset) -> np.ndarray:
    """Return inverse frequency weights for class balancing."""
    counts = np.zeros(len(config.CLASS_NAMES), dtype=np.float32)
    for _, label in dataset.samples:
        counts[label] += 1
    weights = counts.sum() / (len(config.CLASS_NAMES) * counts)
    return weights


def set_seed(seed: int | None = None) -> None:
    seed = seed or config.SEED
    random.seed(seed)
    np.random.seed(seed)
    import torch

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
