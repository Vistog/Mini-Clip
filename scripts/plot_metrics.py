from __future__ import annotations

import json
import argparse
from pathlib import Path
from typing import Dict, List, Any, Tuple

import matplotlib.pyplot as plt


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def moving_avg(xs: List[float], w: int) -> List[float]:
    if w <= 1:
        return xs
    out = []
    s = 0.0
    q = []
    for x in xs:
        q.append(x)
        s += x
        if len(q) > w:
            s -= q.pop(0)
        out.append(s / len(q))
    return out


def get_series(rows: List[Dict[str, Any]], split: str, x_key: str, y_key: str) -> Tuple[List[float], List[float]]:
    xs, ys = [], []
    for r in rows:
        if r.get("split") != split:
            continue
        if x_key not in r or y_key not in r:
            continue
        xs.append(r[x_key])
        ys.append(r[y_key])
    return xs, ys


def plot_series(ax, xs, ys, title: str, xlabel: str, ylabel: str):
    ax.plot(xs, ys)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", type=str, required=True, help="e.g. runs/exp_20260215_133822")
    ap.add_argument("--smooth", type=int, default=1, help="moving average window for train curves")
    ap.add_argument("--x_train", type=str, default="step", choices=["step", "epoch"], help="x-axis for train")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    metrics_path = run_dir / "metrics.jsonl"
    if not metrics_path.exists():
        raise FileNotFoundError(f"metrics.jsonl not found: {metrics_path}")

    rows = read_jsonl(metrics_path)

    # --- train curves ---
    train_loss_x, train_loss_y = get_series(rows, "train", args.x_train, "loss")
    train_i2t_x, train_i2t_y = get_series(rows, "train", args.x_train, "i2t_R@1")
    train_t2i_x, train_t2i_y = get_series(rows, "train", args.x_train, "t2i_R@1")
    train_scale_x, train_scale_y = get_series(rows, "train", args.x_train, "logit_scale")

    train_loss_y = moving_avg([float(v) for v in train_loss_y], args.smooth)
    train_i2t_y = moving_avg([float(v) for v in train_i2t_y], args.smooth)
    train_t2i_y = moving_avg([float(v) for v in train_t2i_y], args.smooth)
    train_scale_y = moving_avg([float(v) for v in train_scale_y], args.smooth)

    # --- val curves (обычно x = epoch) ---
    val_i2t_x, val_i2t_y = get_series(rows, "val", "epoch", "i2t_R@1")
    val_t2i_x, val_t2i_y = get_series(rows, "val", "epoch", "t2i_R@1")
    val_i2t5_x, val_i2t5_y = get_series(rows, "val", "epoch", "i2t_R@5")
    val_t2i5_x, val_t2i5_y = get_series(rows, "val", "epoch", "t2i_R@5")

    # 1) Loss
    fig, ax = plt.subplots()
    plot_series(ax, train_loss_x, train_loss_y, "Train loss", args.x_train, "loss")
    fig.tight_layout()
    fig.savefig(run_dir / "plot_train_loss.png", dpi=150)

    # 2) Train Recall@1
    fig, ax = plt.subplots()
    ax.plot(train_i2t_x, train_i2t_y, label="i2t_R@1")
    ax.plot(train_t2i_x, train_t2i_y, label="t2i_R@1")
    ax.set_title("Train Recall@1")
    ax.set_xlabel(args.x_train)
    ax.set_ylabel("recall")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(run_dir / "plot_train_recall1.png", dpi=150)

    # 3) Val Recall@K
    fig, ax = plt.subplots()
    if val_i2t_x:
        ax.plot(val_i2t_x, val_i2t_y, label="val i2t_R@1")
    if val_t2i_x:
        ax.plot(val_t2i_x, val_t2i_y, label="val t2i_R@1")
    if val_i2t5_x:
        ax.plot(val_i2t5_x, val_i2t5_y, label="val i2t_R@5")
    if val_t2i5_x:
        ax.plot(val_t2i5_x, val_t2i5_y, label="val t2i_R@5")
    ax.set_title("Validation Recall@K")
    ax.set_xlabel("epoch")
    ax.set_ylabel("recall")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(run_dir / "plot_val_recall.png", dpi=150)

    # 4) Temperature / logit scale
    fig, ax = plt.subplots()
    plot_series(ax, train_scale_x, train_scale_y, "Train logit_scale (exp)", args.x_train, "logit_scale")
    fig.tight_layout()
    fig.savefig(run_dir / "plot_train_logit_scale.png", dpi=150)

    print("Saved plots to:", run_dir)


if __name__ == "__main__":
    main()
