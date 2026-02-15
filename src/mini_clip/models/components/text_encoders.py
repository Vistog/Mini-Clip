from __future__ import annotations

import torch
import torch.nn as nn
from transformers import AutoModel

from mini_clip.registry import register


def mean_pool(last_hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).type_as(last_hidden)
    summed = (last_hidden * mask).sum(dim=1)
    denom = mask.sum(dim=1).clamp(min=1e-6)
    return summed / denom


@register("text_encoder", "distilbert-base-uncased")
class DistilBertEncoder(nn.Module):
    def __init__(
        self,
        name: str = "distilbert-base-uncased",
        trainable: bool = False,
        pooling: str = "cls",  # "cls" | "mean"
        unfreeze_last_n: int = 0,  # 0 = как раньше; если >0, то trainable игнорируем и размораживаем только верхние слои
        **kwargs,
    ):
        super().__init__()
        self.model = AutoModel.from_pretrained(name)
        self.pooling = pooling
        self.out_dim = int(self.model.config.hidden_size)

        if self.pooling not in ("cls", "mean"):
            raise ValueError(f"pooling must be 'cls' or 'mean', got: {self.pooling}")

        if unfreeze_last_n and int(unfreeze_last_n) > 0:
            self.freeze_all()
            self.unfreeze_last_layers(int(unfreeze_last_n))
        else:
            self.set_trainable(trainable)

    def freeze_all(self) -> None:
        for p in self.model.parameters():
            p.requires_grad = False

    def set_trainable(self, trainable: bool) -> None:
        for p in self.model.parameters():
            p.requires_grad = bool(trainable)

    def unfreeze_last_layers(self, n: int) -> None:
        """
        DistilBERT: encoder.layer = список слоёв.
        Размораживаем последние n слоёв + LayerNorm/embeddings можно оставить frozen.
        """
        layers = self.model.transformer.layer
        n = max(1, min(n, len(layers)))
        for layer in layers[-n:]:
            for p in layer.parameters():
                p.requires_grad = True

        # обычно полезно разморозить финальный layer norm
        if hasattr(self.model.transformer, "layer_norm"):
            for p in self.model.transformer.layer_norm.parameters():
                p.requires_grad = True

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, **kwargs) -> torch.Tensor:
        out = self.model(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden = out.last_hidden_state

        if self.pooling == "cls":
            feats = last_hidden[:, 0, :]
        else:
            feats = mean_pool(last_hidden, attention_mask)

        return feats
