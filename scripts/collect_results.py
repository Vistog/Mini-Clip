from __future__ import annotations

import json
import argparse
from pathlib import Path
from typing import Any, Dict, Optional, List

import yaml
import pandas as pd


def read_yaml(path: Path) -> Dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def get_nested(d: Dict[str, Any], keys: List[str], default=None):
    cur: Any = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def flatten_selected(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Вытащим самые важные гиперпараметры в плоский вид (для таблицы).
    """
    out = {}

    # model
    out["embed_dim"] = get_nested(cfg, ["model", "embed_dim"])
    out["img_enc"] = get_nested(cfg, ["model", "image_encoder", "name"])
    out["img_trainable"] = get_nested(cfg, ["model", "image_encoder", "trainable"])
    out["txt_enc"] = get_nested(cfg, ["model", "text_encoder", "name"])
    out["txt_pooling"] = get_nested(cfg, ["model", "text_encoder", "pooling"])
    out["txt_trainable"] = get_nested(cfg, ["model", "text_encoder", "trainable"])
    out["txt_unfreeze_last_n"] = get_nested(cfg, ["model", "text_encoder", "unfreeze_last_n"])

    out["proj_type"] = get_nested(cfg, ["model", "projection", "type"])
    out["proj_hidden_dim"] = get_nested(cfg, ["model", "projection", "hidden_dim"])
    out["proj_dropout"] = get_nested(cfg, ["model", "projection", "dropout"])

    out["temp_learnable"] = get_nested(cfg, ["model", "temperature", "learnable"])
    out["temp_init"] = get_nested(cfg, ["model", "temperature", "init"])

    # data/train
    out["dataset"] = get_nested(cfg, ["data", "dataset"])
    out["batch_size"] = get_nested(cfg, ["data", "batch_size"])
    out["image_size"] = get_nested(cfg, ["data", "image_size"])
    out["max_length"] = get_nested(cfg, ["data", "max_length"])

    out["epochs"] = get_nested(cfg, ["train", "epochs"])
    out["lr"] = get_nested(cfg, ["train", "lr"])
    out["lr_text"] = get_nested(cfg, ["train", "lr_text"])
    out["lr_image"] = get_nested(cfg, ["train", "lr_image"])
    out["weight_decay"] = get_nested(cfg, ["train", "weight_decay"])
    out["amp"] = get_nested(cfg, ["train", "amp"])

    return out


def pick_eval_metrics(eval_json: Dict[str, Any], prefix: str) -> Dict[str, Any]:
    """
    Берём стандартный набор метрик из full_eval_*.json.
    """
    keys = ["i2t_R@1", "i2t_R@5", "i2t_R@10", "t2i_R@1", "t2i_R@5", "t2i_R@10"]
    out = {}
    for k in keys:
        if k in eval_json:
            out[f"{prefix}{k}"] = eval_json[k]
    # доп поля
    ckpt = eval_json.get("ckpt", {})
    if isinstance(ckpt, dict):
        out[f"{prefix}ckpt_epoch"] = ckpt.get("epoch")
        out[f"{prefix}ckpt_path"] = ckpt.get("ckpt_path")
    out[f"{prefix}pairs"] = eval_json.get("pairs")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs_dir", type=str, default="runs", help="folder with exp_* dirs")
    ap.add_argument("--out_csv", type=str, default="runs/results.csv")
    ap.add_argument("--metric", type=str, default="val_i2t_R@1", help="metric to rank by")
    ap.add_argument("--topk", type=int, default=10)
    args = ap.parse_args()

    runs_dir = Path(args.runs_dir)
    if not runs_dir.exists():
        raise FileNotFoundError(runs_dir)

    rows = []
    for run_dir in sorted([p for p in runs_dir.iterdir() if p.is_dir()]):
        cfg_path = run_dir / "config.yaml"
        if not cfg_path.exists():
            continue

        row: Dict[str, Any] = {}
        row["run"] = run_dir.name
        row["run_dir"] = str(run_dir)

        # config
        cfg = read_yaml(cfg_path)
        row.update(flatten_selected(cfg))

        # full evals
        val_path = run_dir / "full_eval_val.json"
        test_path = run_dir / "full_eval_test.json"
        if val_path.exists():
            row.update(pick_eval_metrics(read_json(val_path), prefix="val_"))
        if test_path.exists():
            row.update(pick_eval_metrics(read_json(test_path), prefix="test_"))

        rows.append(row)

    if not rows:
        print("No runs found with config.yaml.")
        return

    df = pd.DataFrame(rows)

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False, encoding="utf-8")
    print("Saved:", str(out_csv))

    # ranking
    metric = args.metric
    if metric not in df.columns:
        print(f"Metric '{metric}' not found. Available metrics (sample):")
        cols = [c for c in df.columns if "val_" in c or "test_" in c]
        print(cols[:50])
        return

    df_rank = df.dropna(subset=[metric]).sort_values(metric, ascending=False)
    print(f"\nTop {args.topk} by {metric}:")
    show_cols = ["run", metric, "proj_type", "txt_pooling", "txt_unfreeze_last_n", "batch_size", "lr", "lr_text", "temp_learnable"]
    show_cols = [c for c in show_cols if c in df_rank.columns]
    print(df_rank[show_cols].head(args.topk).to_string(index=False))


if __name__ == "__main__":
    main()
