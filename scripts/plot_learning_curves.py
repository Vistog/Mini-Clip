from __future__ import annotations

import json
import argparse
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


def read_jsonl(path: Path) -> pd.DataFrame:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return pd.DataFrame(rows)


def save_lineplot(df: pd.DataFrame, x: str, y: str, title: str, out_path: Path) -> None:
    # drop NaNs so matplotlib doesn’t silently produce weird output
    d = df[[x, y]].dropna()
    if d.empty:
        print(f"[skip] empty after dropna: {title}")
        return

    fig, ax = plt.subplots()
    ax.plot(d[x].values, d[y].values)
    ax.set_title(title)
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", type=str, required=True)
    ap.add_argument("--out_dir", type=str, default=None)
    ap.add_argument("--x", type=str, default="step", choices=["step", "epoch"])
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    metrics_path = run_dir / "metrics.jsonl"
    if not metrics_path.exists():
        raise FileNotFoundError(metrics_path)

    out_dir = Path(args.out_dir) if args.out_dir else (run_dir / "plots")
    out_dir.mkdir(parents=True, exist_ok=True)

    df = read_jsonl(metrics_path)

    if "split" not in df.columns:
        raise ValueError(f"No 'split' column in metrics.jsonl. Columns: {list(df.columns)}")

    train = df[df["split"] == "train_batch"].copy()
    val = df[df["split"] == "val"].copy()

    # --- TRAIN plots ---
    if train.empty:
        print("[warn] train split is empty. Available splits:", df["split"].value_counts().to_dict())
    else:
        x = args.x
        if x not in train.columns:
            raise ValueError(f"Column '{x}' not found in TRAIN rows. Columns: {list(train.columns)}")

        save_lineplot(train, x, "loss", "Train loss", out_dir / f"train_loss_vs_{x}.png")

        # train retrieval metrics names in твоих строках:
        train_metrics = [
            "train_batch_i2t_R@1",
            "train_batch_t2i_R@1",
            "train_batch_i2t_R@5",
            "train_batch_t2i_R@5",
            "logit_scale",
        ]
        for m in train_metrics:
            if m in train.columns:
                save_lineplot(train, x, m, f"Train {m}", out_dir / f"train_{m}_vs_{x}.png")

    # --- VAL plots ---
    if not val.empty:
        if "epoch" in val.columns:
            save_lineplot(val, "epoch", "i2t_R@1", "Val i2t_R@1", out_dir / "val_i2t_R@1_vs_epoch.png")
            save_lineplot(val, "epoch", "t2i_R@1", "Val t2i_R@1", out_dir / "val_t2i_R@1_vs_epoch.png")

            for m in ["i2t_R@5", "t2i_R@5", "i2t_R@10", "t2i_R@10"]:
                if m in val.columns:
                    save_lineplot(val, "epoch", m, f"Val {m}", out_dir / f"val_{m}_vs_epoch.png")
        else:
            print("[warn] VAL rows exist but no 'epoch' column. Columns:", list(val.columns))

    print("Saved plots to:", out_dir)


if __name__ == "__main__":
    main()
