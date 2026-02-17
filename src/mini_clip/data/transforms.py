from __future__ import annotations

from typing import Dict, Any, Tuple, Optional

from torchvision import transforms
from torchvision.models import ResNet50_Weights

"""
Image augmentation config (data.augment):

enabled: bool
    Включает train-аугментации.

rrc_scale: [min, max]
    Диапазон масштаба для RandomResizedCrop.
    (0.8, 1.0) — мягкая аугментация (рекомендуется для CLIP).

hflip_p: float
    Вероятность горизонтального отражения.

color_jitter:
    enabled: bool
    brightness, contrast, saturation, hue — параметры ColorJitter.

random_grayscale_p: float
    Вероятность перевода изображения в grayscale.

Важно:
- Аугментации применяются только к train.
- Val/Test всегда детерминированы (Resize → CenterCrop → Normalize).
- Слишком сильные аугментации могут ухудшить retrieval.
"""


def build_image_transform(
    image_size: int,
    train: bool,
    aug: Optional[Dict[str, Any]] = None,
):
    """
    Train: Resize -> RandomResizedCrop -> (aug...) -> ToTensor -> Normalize
    Val:   Resize -> CenterCrop -> ToTensor -> Normalize

    По умолчанию: мягкие аугментации, безопасные для CLIP retrieval.
    """
    weights = ResNet50_Weights.DEFAULT
    mean = weights.transforms().mean
    std = weights.transforms().std

    aug = aug or {}
    enabled = bool(aug.get("enabled", True))

    # базовые дефолты (мягко)
    rrc_scale = tuple(aug.get("rrc_scale", (0.8, 1.0)))
    hflip_p = float(aug.get("hflip_p", 0.5))

    cj = aug.get("color_jitter", None)  # dict or None
    gray_p = float(aug.get("random_grayscale_p", 0.0))  # 0.0 по умолчанию

    if train:
        ops = [
            transforms.Resize(image_size + 32),
            transforms.RandomResizedCrop(image_size, scale=rrc_scale),
        ]

        if enabled:
            if hflip_p > 0:
                ops.append(transforms.RandomHorizontalFlip(p=hflip_p))

            if isinstance(cj, dict) and cj.get("enabled", True):
                ops.append(
                    transforms.ColorJitter(
                        brightness=float(cj.get("brightness", 0.0)),
                        contrast=float(cj.get("contrast", 0.0)),
                        saturation=float(cj.get("saturation", 0.0)),
                        hue=float(cj.get("hue", 0.0)),
                    )
                )

            if gray_p > 0:
                ops.append(transforms.RandomGrayscale(p=gray_p))

        ops += [
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
        return transforms.Compose(ops)

    # val/test
    return transforms.Compose([
        transforms.Resize(image_size + 32),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
