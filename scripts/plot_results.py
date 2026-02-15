from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def save_fig(fig, path: Path):
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print("Saved:", str(path))


def scatter(df: pd.DataFrame, x: str, y: str, hue: str | None, title: str, out: Path):
    fig, ax = plt.subplots()
    if hue and hue in df.columns:
        for val, g in df.dropna(subset=[x, y]).groupby(hue):
            ax.scatter(g[x], g[y], label=str(val))
        ax.legend(title=hue)
    else:
        g = df.dropna(subset=[x, y])
        ax.scatter(g[x], g[y])
    ax.set_title(title)
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.grid(True, alpha=0.3)
    save_fig(fig, out)


def bar_topk(df: pd.DataFrame, metric: str, k: int, out: Path):
    g = df.dropna(subset=[metric]).sort_values(metric, ascending=False).head(k)
    fig, ax = plt.subplots()
    ax.bar(range(len(g)), g[metric].values)
    ax.set_title(f"Top {k} runs by {metric}")
    ax.set_xlabel("run")
    ax.set_ylabel(metric)
    ax.set_xticks(range(len(g)))
    ax.set_xticklabels(g["run"].values, rotation=45, ha="right")
    ax.grid(True, axis="y", alpha=0.3)
    save_fig(fig, out)


def boxplot(df: pd.DataFrame, by: str, metric: str, out: Path):
    g = df.dropna(subset=[by, metric])
    if g.empty:
        return
    groups = [vals[metric].values for _, vals in g.groupby(by)]
    labels = [str(k) for k, _ in g.groupby(by)]
    fig, ax = plt.subplots()
    ax.boxplot(groups, labels=labels)
    ax.set_title(f"{metric} grouped by {by}")
    ax.set_xlabel(by)
    ax.set_ylabel(metric)
    ax.grid(True, axis="y", alpha=0.3)
    save_fig(fig, out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=str, default="runs/results.csv")
    ap.add_argument("--out_dir", type=str, default="runs/plots")
    ap.add_argument("--metric", type=str, default="val_i2t_R@1")
    ap.add_argument("--topk", type=int, default=10)
    args = ap.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}. Run collect_results.py first.")

    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)

    df = pd.read_csv(csv_path)

    metric = args.metric
    if metric not in df.columns:
        raise ValueError(f"Metric '{metric}' not in CSV columns. Available: {list(df.columns)[:30]} ...")

    # 1) Top-k bar
    bar_topk(df, metric=metric, k=args.topk, out=out_dir / f"top_{args.topk}_{metric}.png")

    # 2) Scatter: unfreeze vs metric (colored by proj_type)
    if "txt_unfreeze_last_n" in df.columns:
        scatter(
            df, x="txt_unfreeze_last_n", y=metric, hue="proj_type",
            title=f"{metric} vs txt_unfreeze_last_n",
            out=out_dir / f"scatter_unfreeze_vs_{metric}.png"
        )

    # 3) Scatter: lr_text vs metric (colored by pooling)
    if "lr_text" in df.columns:
        scatter(
            df, x="lr_text", y=metric, hue="txt_pooling",
            title=f"{metric} vs lr_text",
            out=out_dir / f"scatter_lr_text_vs_{metric}.png"
        )

    # 4) Boxplot: proj_type
    if "proj_type" in df.columns:
        boxplot(df, by="proj_type", metric=metric, out=out_dir / f"box_proj_type_{metric}.png")

    # 5) Boxplot: pooling
    if "txt_pooling" in df.columns:
        boxplot(df, by="txt_pooling", metric=metric, out=out_dir / f"box_pooling_{metric}.png")

    print("Done. Plots in:", str(out_dir))


if __name__ == "__main__":
    main()
