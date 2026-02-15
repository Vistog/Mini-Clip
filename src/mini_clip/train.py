from __future__ import annotations

import os
import json
import time
import math
import random
from typing import Dict, Any

import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm

from mini_clip.utils.config import load_config, make_run_dir, save_config
from mini_clip.factory import build_model
from mini_clip.data.datamodule import make_train_loader, Batch
from mini_clip.losses.clip_loss import CLIPInBatchLoss
from mini_clip.losses.metrics import retrieval_metrics


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def to_device(batch: Batch, device: torch.device) -> Batch:
    images = batch.images.to(device, non_blocking=True)
    text = {k: v.to(device, non_blocking=True) for k, v in batch.text.items()}
    return Batch(images=images, text=text)


def save_ckpt(model: nn.Module, optim: torch.optim.Optimizer, epoch: int, run_dir: str) -> None:
    path = os.path.join(run_dir, "checkpoints", f"epoch_{epoch:03d}.pt")
    torch.save(
        {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optim.state_dict(),
        },
        path,
    )


def main():
    # 1) config
    cfg = load_config("configs/base.yaml", os.environ.get("OVERRIDE_CFG"))
    if os.environ.get("OVERRIDE_CFG") is None:
        # если не задан override — просто base
        pass

    set_seed(int(cfg["seed"]))

    # 2) run dir
    run_dir = make_run_dir(cfg["run"]["out_dir"], cfg["run"].get("name"))
    save_config(cfg, run_dir)

    # 3) device
    use_cuda = (cfg.get("device") == "cuda") and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # 4) data
    loader = make_train_loader(cfg)

    # 5) model
    model = build_model(cfg).to(device)
    model.train()

    # 6) loss + optim
    criterion = CLIPInBatchLoss()
    tcfg = cfg["train"]
    optim = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=float(tcfg["lr"]),
        weight_decay=float(tcfg["weight_decay"]),
    )

    amp = bool(tcfg.get("amp", True)) and use_cuda
    scaler = GradScaler(enabled=amp)

    # 7) training
    metrics_path = os.path.join(run_dir, "metrics.jsonl")
    log_every = int(tcfg["log_every"])
    epochs = int(tcfg["epochs"])
    save_every = int(tcfg["save_every"])

    global_step = 0
    with open(metrics_path, "a", encoding="utf-8") as f:
        for epoch in range(1, epochs + 1):
            pbar = tqdm(loader, desc=f"epoch {epoch}/{epochs}", leave=True)
            for it, batch in enumerate(pbar, start=1):
                global_step += 1
                batch = to_device(batch, device)

                optim.zero_grad(set_to_none=True)

                with autocast(enabled=amp):
                    logits_i, logits_t = model(batch.images, batch.text)
                    loss = criterion(logits_i, logits_t)

                scaler.scale(loss).backward()
                scaler.step(optim)
                scaler.update()

                if global_step % log_every == 0:
                    m = retrieval_metrics(logits_i.detach(), logits_t.detach(), ks=(1, 5))
                    log = {
                        "time": time.time(),
                        "epoch": epoch,
                        "iter": it,
                        "step": global_step,
                        "loss": float(loss.item()),
                        "logit_scale": float(model.logit_scale.exp().detach().cpu().item()),
                        **m,
                    }
                    f.write(json.dumps(log) + "\n")
                    f.flush()
                    pbar.set_postfix({k: (round(v, 4) if isinstance(v, float) else v) for k, v in log.items() if k in ("loss","i2t_R@1","t2i_R@1","logit_scale")})

            if epoch % save_every == 0:
                save_ckpt(model, optim, epoch, run_dir)

    print(f"Done. Run saved to: {run_dir}")


if __name__ == "__main__":
    main()
