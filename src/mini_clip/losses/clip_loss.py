from __future__ import annotations

import torch
import torch.nn as nn


class CLIPInBatchLoss(nn.Module):
    """
    Классический CLIP loss:
      logits: [B, B]
      labels: [0..B-1]
      loss = (CE(logits, labels) + CE(logits.T, labels)) / 2
    """
    def __init__(self):
        super().__init__()
        self.ce = nn.CrossEntropyLoss()

    def forward(self, logits_per_image: torch.Tensor, logits_per_text: torch.Tensor) -> torch.Tensor:
        b = logits_per_image.size(0)
        labels = torch.arange(b, device=logits_per_image.device)
        loss_i = self.ce(logits_per_image, labels)
        loss_t = self.ce(logits_per_text, labels)
        return 0.5 * (loss_i + loss_t)
