from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class MiniCLIP(nn.Module):
    def __init__(
        self,
        image_encoder: nn.Module,
        text_encoder: nn.Module,
        image_proj: nn.Module,
        text_proj: nn.Module,
        logit_scale: nn.Parameter,
    ):
        super().__init__()
        self.image_encoder = image_encoder
        self.text_encoder = text_encoder
        self.image_proj = image_proj
        self.text_proj = text_proj
        self.logit_scale = logit_scale  # exp(logit_scale) = 1/temperature

    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        feats = self.image_encoder(images)
        emb = self.image_proj(feats)
        return F.normalize(emb, dim=-1)

    def encode_text(self, text_batch: dict) -> torch.Tensor:
        feats = self.text_encoder(**text_batch)
        emb = self.text_proj(feats)
        return F.normalize(emb, dim=-1)

    def forward(self, images: torch.Tensor, text_batch: dict):
        image_emb = self.encode_image(images)
        text_emb = self.encode_text(text_batch)

        scale = self.logit_scale.exp().clamp(max=100.0)
        logits_per_image = scale * (image_emb @ text_emb.t())
        logits_per_text = logits_per_image.t()
        return logits_per_image, logits_per_text
