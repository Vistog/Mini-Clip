from __future__ import annotations

import os
import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# silence transformers / warnings (опционально, но очень чистит вывод)
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
from transformers.utils import logging as hf_logging
hf_logging.set_verbosity_error()
hf_logging.disable_progress_bar()

from mini_clip.utils.config import load_config
from mini_clip.factory import build_model
from mini_clip.data.datamodule import CLIPCollator, Batch
from mini_clip.data.datasets import Flickr8kDataset


def load_checkpoint(model: torch.nn.Module, ckpt_path: str, device: torch.device) -> Dict[str, Any]:
    ckpt_path = str(ckpt_path)
    try:
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    except TypeError:
        # старый torch без weights_only
        ckpt = torch.load(ckpt_path, map_location=device)

    if isinstance(ckpt, dict) and "model" in ckpt:
        model.load_state_dict(ckpt["model"], strict=True)
        return {"ckpt_path": ckpt_path, "epoch": ckpt.get("epoch", None)}
    else:
        model.load_state_dict(ckpt, strict=True)
        return {"ckpt_path": ckpt_path, "epoch": None}


@torch.no_grad()
def encode_dataset(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    amp: bool,
) -> Tuple[torch.Tensor, torch.Tensor, List[str]]:
    """
    Возвращает:
      img_embs: [N, D]  (на уровне пар image-caption; т.е. повторяется 5 раз для одной картинки)
      txt_embs: [N, D]
      image_names: длины N, имя картинки для каждой пары
    """
    from torch.amp import autocast

    model.eval()
    all_img, all_txt = [], []
    all_names: List[str] = []

    for batch in tqdm(loader, desc="encode", leave=True):
        assert isinstance(batch, Batch)
        if batch.image_names is None:
            raise RuntimeError("image_names is None. Ensure __getitem__ returns 3 elements and collator passes names.")

        images = batch.images.to(device, non_blocking=True)
        text = {k: v.to(device, non_blocking=True) for k, v in batch.text.items()}

        with autocast(device_type="cuda", enabled=amp):
            img_emb = model.encode_image(images)   # [B, D]
            txt_emb = model.encode_text(text)      # [B, D]

        all_img.append(img_emb.cpu())
        all_txt.append(txt_emb.cpu())
        all_names.extend(batch.image_names)

    return torch.cat(all_img, dim=0), torch.cat(all_txt, dim=0), all_names


def build_name_to_indices(names: List[str]) -> Dict[str, List[int]]:
    mp: Dict[str, List[int]] = {}
    for i, n in enumerate(names):
        mp.setdefault(n, []).append(i)
    return mp


@torch.no_grad()
def recall_i2t(sim: torch.Tensor, names: List[str], ks=(1, 5, 10)) -> Dict[str, float]:
    """
    Image->Text: у одной картинки 5 captions => multi-positive.
    """
    name2idx = build_name_to_indices(names)
    N = sim.size(0)

    out = {}
    for k in ks:
        k = min(k, N)
        correct = 0
        for i in range(N):
            gt = set(name2idx[names[i]])
            topk = sim[i].topk(k, dim=0).indices.tolist()
            if any(j in gt for j in topk):
                correct += 1
        out[f"i2t_R@{k}"] = correct / N
    return out


@torch.no_grad()
def recall_t2i(sim_t: torch.Tensor, names: List[str], ks=(1, 5, 10)) -> Dict[str, float]:
    """
    Text->Image: правильные изображения = все пары с тем же image_name (та же картинка).
    """
    name2idx = build_name_to_indices(names)
    N = sim_t.size(0)

    out = {}
    for k in ks:
        k = min(k, N)
        correct = 0
        for i in range(N):
            gt = set(name2idx[names[i]])
            topk = sim_t[i].topk(k, dim=0).indices.tolist()
            if any(j in gt for j in topk):
                correct += 1
        out[f"t2i_R@{k}"] = correct / N
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, default=None, help="path to checkpoint .pt (optional)")
    ap.add_argument("--split", type=str, default="val", choices=["train", "val", "test"])
    ap.add_argument("--batch_size", type=int, default=None, help="override batch size for eval")
    ap.add_argument("--out_json", type=str, default=None, help="where to save json results (optional)")
    args = ap.parse_args()

    cfg = load_config("configs/base.yaml", os.environ.get("OVERRIDE_CFG"))
    dcfg = cfg["data"]

    root = dcfg["root"]
    image_size = int(dcfg["image_size"])
    max_length = int(dcfg["max_length"])
    bs = int(args.batch_size) if args.batch_size is not None else int(dcfg["batch_size"])

    ds = Flickr8kDataset(root=root, split=args.split, image_size=image_size, train=False)
    collate = CLIPCollator(tokenizer_name=cfg["model"]["text_encoder"]["name"], max_length=max_length)

    loader = DataLoader(
        ds,
        batch_size=bs,
        shuffle=False,
        num_workers=int(dcfg.get("num_workers", 2)),
        pin_memory=True,
        collate_fn=collate,
    )

    use_cuda = (cfg.get("device") == "cuda") and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    model = build_model(cfg).to(device)

    ckpt_info: Dict[str, Any] = {"ckpt_path": None, "epoch": None}
    if args.ckpt is not None:
        ckpt_info = load_checkpoint(model, args.ckpt, device)

    amp = bool(cfg["train"].get("amp", True)) and use_cuda

    img_embs, txt_embs, names = encode_dataset(model, loader, device, amp)

    # full similarity matrix
    sim_i = img_embs @ txt_embs.t()      # image->text
    sim_t = sim_i.t().contiguous()       # text->image

    m_i2t = recall_i2t(sim_i, names, ks=(1, 5, 10))
    m_t2i = recall_t2i(sim_t, names, ks=(1, 5, 10))

    result = {
        "mode": "full_retrieval",
        "split": args.split,
        "pairs": int(len(names)),
        "batch_size": int(bs),
        "device": str(device),
        "ckpt": ckpt_info,
        **m_i2t,
        **m_t2i,
    }

    # nice console output
    print(f"FULL_RETRIEVAL | split={args.split} | pairs={len(names)}")
    print("i2t:", f"R@1={result['i2t_R@1']:.3f}", f"R@5={result['i2t_R@5']:.3f}", f"R@10={result['i2t_R@10']:.3f}")
    print("t2i:", f"R@1={result['t2i_R@1']:.3f}", f"R@5={result['t2i_R@5']:.3f}", f"R@10={result['t2i_R@10']:.3f}")

    if args.out_json:
        out_path = Path(args.out_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print("Saved:", str(out_path))


if __name__ == "__main__":
    main()
