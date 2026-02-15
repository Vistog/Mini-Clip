from __future__ import annotations

import os
import json
import time
import random

import torch
import torch.nn as nn
from torch.amp import autocast, GradScaler
from tqdm import tqdm

from mini_clip.utils.config import load_config, make_run_dir, save_config
from mini_clip.factory import build_model
from mini_clip.data.datamodule import make_train_val_loaders, Batch
from mini_clip.losses.clip_loss import CLIPInBatchLoss
from mini_clip.losses.metrics import retrieval_metrics

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

from transformers.utils import logging as hf_logging
hf_logging.set_verbosity_error()
hf_logging.disable_progress_bar()

def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def to_device(batch: Batch, device: torch.device) -> Batch:
    images = batch.images.to(device, non_blocking=True)
    text = {k: v.to(device, non_blocking=True) for k, v in batch.text.items()}
    return Batch(images=images, text=text)


@torch.no_grad()
def evaluate_epoch(model: nn.Module, loader, device: torch.device, amp: bool) -> dict:
    model.eval()
    all_logs = []
    for batch in tqdm(loader, desc="val", leave=False):
        batch = to_device(batch, device)
        with autocast(device_type="cuda", enabled=amp):
            logits_i, logits_t = model(batch.images, batch.text)
        m = retrieval_metrics(logits_i, logits_t, ks=(1, 5, 10))
        all_logs.append(m)

    # усреднение по батчам (быстро и достаточно для начала)
    out = {}
    for k in all_logs[0].keys():
        out[k] = float(sum(d[k] for d in all_logs) / len(all_logs))
    model.train()
    return out


def save_ckpt(model: nn.Module, optim: torch.optim.Optimizer, epoch: int, run_dir: str) -> None:
    path = os.path.join(run_dir, "checkpoints", f"epoch_{epoch:03d}.pt")
    torch.save({"epoch": epoch, "model": model.state_dict(), "optimizer": optim.state_dict()}, path)


def main():
    cfg = load_config("configs/base.yaml", os.environ.get("OVERRIDE_CFG"))
    set_seed(int(cfg["seed"]))

    run_dir = make_run_dir(cfg["run"]["out_dir"], cfg["run"].get("name"))
    save_config(cfg, run_dir)

    use_cuda = (cfg.get("device") == "cuda") and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    train_loader, val_loader = make_train_val_loaders(cfg)

    model = build_model(cfg).to(device)
    model.train()

    criterion = CLIPInBatchLoss()
    tcfg = cfg["train"]

    optim = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=float(tcfg["lr"]),
        weight_decay=float(tcfg["weight_decay"]),
    )

    amp = bool(tcfg.get("amp", True)) and use_cuda
    scaler = GradScaler("cuda", enabled=amp)

    metrics_path = os.path.join(run_dir, "metrics.jsonl")
    log_every = int(tcfg["log_every"])
    epochs = int(tcfg["epochs"])
    save_every = int(tcfg["save_every"])

    global_step = 0
    with open(metrics_path, "a", encoding="utf-8") as f:
        for epoch in range(1, epochs + 1):
            pbar = tqdm(train_loader, desc=f"train epoch {epoch}/{epochs}", leave=True)
            for it, batch in enumerate(pbar, start=1):
                global_step += 1
                batch = to_device(batch, device)

                optim.zero_grad(set_to_none=True)
                with autocast(device_type="cuda", enabled=amp):
                    logits_i, logits_t = model(batch.images, batch.text)
                    loss = criterion(logits_i, logits_t)

                scaler.scale(loss).backward()
                scaler.step(optim)
                scaler.update()

                if global_step % log_every == 0:
                    m = retrieval_metrics(logits_i.detach(), logits_t.detach(), ks=(1, 5))
                    m = {f"train_batch_{k}": v for k, v in m.items()}  # <— главное

                    log = {
                        "time": time.time(),
                        "split": "train_batch",  # <— чтобы не путать с full val/test
                        "epoch": epoch,
                        "iter": it,
                        "step": global_step,
                        "loss": float(loss.item()),
                        "logit_scale": float(model.logit_scale.exp().detach().cpu().item()),
                        **m,
                    }
                    f.write(json.dumps(log) + "\n")
                    f.flush()

                    pbar.set_postfix({
                        "loss": round(log["loss"], 4),
                        "i2t@1": round(log["train_batch_i2t_R@1"], 4),
                        "t2i@1": round(log["train_batch_t2i_R@1"], 4),
                    })

            # val at end of epoch
            val_m = evaluate_epoch(model, val_loader, device, amp)
            val_log = {"time": time.time(), "split": "val", "epoch": epoch, **val_m}
            f.write(json.dumps(val_log) + "\n")
            f.flush()

            if epoch % save_every == 0:
                save_ckpt(model, optim, epoch, run_dir)

    print(f"Done. Run saved to: {run_dir}")


if __name__ == "__main__":
    main()
