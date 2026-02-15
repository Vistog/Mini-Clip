from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Dict, Any

import torch
from torch.utils.data import DataLoader

from transformers import AutoTokenizer

from mini_clip.data.datasets import FakeCLIPDataset


@dataclass
class Batch:
    images: torch.Tensor          # [B, 3, H, W]
    text: Dict[str, torch.Tensor] # tokenized


class CLIPCollator:
    def __init__(self, tokenizer_name: str, max_length: int):
        self.tok = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_length = int(max_length)

    def __call__(self, items: List[Tuple[torch.Tensor, str]]) -> Batch:
        images, texts = zip(*items)
        images = torch.stack(list(images), dim=0)

        text_batch = self.tok(
            list(texts),
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        return Batch(images=images, text=dict(text_batch))


def make_train_loader(cfg: dict):
    dcfg = cfg["data"]
    dataset_name = dcfg.get("dataset", "fake")

    if dataset_name in ("fake", "debug"):
        ds = FakeCLIPDataset(size=256, image_size=dcfg["image_size"])
    else:
        raise ValueError(
            f"Dataset '{dataset_name}' not implemented yet. "
            f"Use data.dataset: fake for now."
        )

    collate = CLIPCollator(
        tokenizer_name=cfg["model"]["text_encoder"]["name"],
        max_length=dcfg["max_length"],
    )

    return DataLoader(
        ds,
        batch_size=dcfg["batch_size"],
        shuffle=True,
        num_workers=dcfg["num_workers"],
        pin_memory=True,
        collate_fn=collate,
    )
