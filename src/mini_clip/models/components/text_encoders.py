from __future__ import annotations

import torch
import torch.nn as nn
from transformers import AutoModel

from mini_clip.registry import register


def mean_pool(last_hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """
    last_hidden: [B, L, H]
    attention_mask: [B, L] (1 for tokens, 0 for pad)
    """
    mask = attention_mask.unsqueeze(-1).type_as(last_hidden)  # [B, L, 1]
    summed = (last_hidden * mask).sum(dim=1)                  # [B, H]
    denom = mask.sum(dim=1).clamp(min=1e-6)                   # [B, 1]
    return summed / denom


@register("text_encoder", "distilbert-base-uncased")
class DistilBertEncoder(nn.Module):
    """
    HuggingFace AutoModel (DistilBERT).
    Возвращает один вектор [B, H] после pooling.
    """
    def __init__(
        self,
        name: str = "distilbert-base-uncased",
        trainable: bool = False,
        pooling: str = "cls",  # "cls" или "mean"
        **kwargs,
    ):
        super().__init__()
        self.model = AutoModel.from_pretrained(name)
        self.pooling = pooling
        self.out_dim = int(self.model.config.hidden_size)
        self.set_trainable(trainable)

        if self.pooling not in ("cls", "mean"):
            raise ValueError(f"pooling must be 'cls' or 'mean', got: {self.pooling}")

    def set_trainable(self, trainable: bool) -> None:
        for p in self.model.parameters():
            p.requires_grad = bool(trainable)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, **kwargs) -> torch.Tensor:
        out = self.model(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden = out.last_hidden_state  # [B, L, H]

        if self.pooling == "cls":
            # У DistilBERT нет token_type_ids. CLS — это позиция 0.
            feats = last_hidden[:, 0, :]     # [B, H]
        else:
            feats = mean_pool(last_hidden, attention_mask)  # [B, H]

        return feats
