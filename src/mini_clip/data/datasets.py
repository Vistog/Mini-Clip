from __future__ import annotations

import random
from typing import List, Dict

import torch
from torch.utils.data import Dataset


class FakeCLIPDataset(Dataset):
    """
    Для отладки. Возвращает:
      image: [3, H, W]
      text: str
    """
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
