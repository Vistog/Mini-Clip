from __future__ import annotations

import torch
import torch.nn as nn
import torchvision.models as tvm

from mini_clip.registry import register


@register("image_encoder", "resnet50")
class ResNet50Encoder(nn.Module):
    """
    Берём torchvision ResNet50, отрезаем классификатор (fc),
    выдаём вектор признаков размерности 2048.
    """
    def __init__(self, name: str = "resnet50", pretrained: bool = True, trainable: bool = False, **kwargs):
        super().__init__()
        if name != "resnet50":
            raise ValueError(f"ResNet50Encoder supports only name='resnet50', got: {name}")

        weights = tvm.ResNet50_Weights.DEFAULT if pretrained else None
        m = tvm.resnet50(weights=weights)

        # backbone: всё кроме fc
        self.backbone = nn.Sequential(*list(m.children())[:-1])  # -> [B, 2048, 1, 1]
        self.out_dim = 2048

        self.set_trainable(trainable)

    def set_trainable(self, trainable: bool) -> None:
        for p in self.parameters():
            p.requires_grad = bool(trainable)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(x)              # [B, 2048, 1, 1]
        feats = feats.flatten(1)              # [B, 2048]
        return feats
