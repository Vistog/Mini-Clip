from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Dict

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from mini_clip.data.datasets import FakeCLIPDataset, Flickr8kDataset


@dataclass
class Batch:
    images: torch.Tensor
    text: Dict[str, torch.Tensor]
    image_names: Optional[List[str]] = None


class CLIPCollator:
    def __init__(self, tokenizer_name: str, max_length: int):
        self.tok = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_length = int(max_length)

    def __call__(self, items):
        # items могут быть (img, txt) или (img, txt, name)
        if len(items[0]) == 3:
            images, texts, names = zip(*items)
            names = list(names)
        else:
            images, texts = zip(*items)
            names = None

        images = torch.stack(list(images), dim=0)
        text_batch = self.tok(
            list(texts),
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        return Batch(images=images, text=dict(text_batch), image_names=names)


def _make_loader(ds, cfg: dict, shuffle: bool) -> DataLoader:
    dcfg = cfg["data"]
    collate = CLIPCollator(
        tokenizer_name=cfg["model"]["text_encoder"]["name"],
        max_length=dcfg["max_length"],
    )
    return DataLoader(
        ds,
        batch_size=dcfg["batch_size"],
        shuffle=shuffle,
        num_workers=dcfg["num_workers"],
        pin_memory=True,
        collate_fn=collate,
    )


def make_train_val_loaders(cfg: dict):
    dcfg = cfg["data"]
    name = dcfg.get("dataset", "fake")
    root = dcfg.get("root", "")

    if name in ("fake", "debug"):
        train_ds = FakeCLIPDataset(size=512, image_size=dcfg["image_size"])
        val_ds = FakeCLIPDataset(size=128, image_size=dcfg["image_size"])
    elif name == "flickr8k":
        train_ds = Flickr8kDataset(root=root, split="train", image_size=dcfg["image_size"], train=True)
        val_ds = Flickr8kDataset(root=root, split="val", image_size=dcfg["image_size"], train=False)
    else:
        raise ValueError(f"Unknown dataset: {name}")

    return _make_loader(train_ds, cfg, shuffle=True), _make_loader(val_ds, cfg, shuffle=False)
