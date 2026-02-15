from __future__ import annotations

import os
import random
from typing import Dict, List, Tuple

import torch
from torch.utils.data import Dataset
from PIL import Image

from mini_clip.data.transforms import build_image_transform


class FakeCLIPDataset(Dataset):
    def __init__(self, size: int = 256, image_size: int = 224):
        self.size = int(size)
        self.image_size = int(image_size)
        self.text_pool = [
            "a photo of a dog",
            "a photo of a cat",
            "a red car",
            "a blue car",
            "a green tree",
            "a person riding a bike",
        ]

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, idx: int):
        img = torch.randn(3, self.image_size, self.image_size)
        txt = random.choice(self.text_pool)
        return img, txt


def _read_lines(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        return [ln.strip() for ln in f.readlines() if ln.strip()]


def _load_split_images(split_file: str) -> List[str]:
    return _read_lines(split_file)


def _load_captions(token_file: str) -> Dict[str, List[str]]:
    """
    Flickr8k.token.txt:
      image.jpg#0<TAB>caption text
    Возвращаем dict: image.jpg -> [cap1, cap2, ...]
    """
    caps: Dict[str, List[str]] = {}
    for ln in _read_lines(token_file):
        # пример: 1000268201_693b08cb0e.jpg#0\tA child in a pink dress ...
        left, caption = ln.split("\t", 1)
        img_id = left.split("#", 1)[0]
        caps.setdefault(img_id, []).append(caption)
    return caps


class Flickr8kDataset(Dataset):
    """
    Возвращает пары (image_tensor, caption_str) для заданного split.
    """
    def __init__(self, root: str, split: str, image_size: int, train: bool):
        """
        root: data/flickr8k (как в структуре выше)
        split: "train" | "val" | "test"
        """
        root = os.path.abspath(root)
        images_dir = os.path.join(root, "Images")
        token_file = os.path.join(root, "Flickr8k.token.txt")

        split_map = {
            "train": "Flickr_8k.trainImages.txt",
            "val": "Flickr_8k.devImages.txt",
            "test": "Flickr_8k.testImages.txt",
        }
        if split not in split_map:
            raise ValueError(f"split must be one of {list(split_map)}, got {split}")

        split_file = os.path.join(root, split_map[split])
        if not os.path.isdir(images_dir):
            raise FileNotFoundError(f"Images dir not found: {images_dir}")
        if not os.path.isfile(token_file):
            raise FileNotFoundError(f"Captions file not found: {token_file}")
        if not os.path.isfile(split_file):
            raise FileNotFoundError(f"Split file not found: {split_file}")

        self.images_dir = images_dir
        self.captions = _load_captions(token_file)
        self.image_names = _load_split_images(split_file)

        # Формируем список пар (image_name, caption)
        pairs: List[Tuple[str, str]] = []
        for img in self.image_names:
            caps = self.captions.get(img, [])
            for c in caps:
                pairs.append((img, c))

        if len(pairs) == 0:
            raise RuntimeError("No (image, caption) pairs found. Check paths / files.")

        self.pairs = pairs
        self.transform = build_image_transform(image_size=image_size, train=train)

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int):
        img_name, caption = self.pairs[idx]
        path = os.path.join(self.images_dir, img_name)
        with Image.open(path) as im:
            im = im.convert("RGB")
            x = self.transform(im)
        return x, caption
