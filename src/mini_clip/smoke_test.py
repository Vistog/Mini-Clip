from __future__ import annotations

import torch
from transformers import AutoTokenizer

from mini_clip.utils.config import load_config
from mini_clip.factory import build_model


def main():
    cfg = load_config("configs/base.yaml", "configs/model/clip_resnet_distilbert.yaml")
    model = build_model(cfg)

    device = torch.device("cuda" if (cfg.get("device") == "cuda" and torch.cuda.is_available()) else "cpu")
    model = model.to(device).eval()

    # fake images
    B = 4
    images = torch.randn(B, 3, cfg["data"]["image_size"], cfg["data"]["image_size"], device=device)

    # texts
    tok = AutoTokenizer.from_pretrained(cfg["model"]["text_encoder"]["name"])
    texts = ["a photo of a dog", "a photo of a cat", "a red car", "a blue car"]
    text_batch = tok(
        texts,
        padding=True,
        truncation=True,
        max_length=cfg["data"]["max_length"],
        return_tensors="pt",
    )
    text_batch = {k: v.to(device) for k, v in text_batch.items()}

    with torch.no_grad():
        logits_i, logits_t = model(images, text_batch)

    print("logits_per_image:", logits_i.shape)  # [B, B]
    print("logits_per_text:", logits_t.shape)   # [B, B]
    print("logit_scale(exp):", float(model.logit_scale.exp().cpu()))


if __name__ == "__main__":
    main()
