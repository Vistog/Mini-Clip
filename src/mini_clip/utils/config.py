from __future__ import annotations

import os
import time
import yaml
from typing import Any, Dict, Optional


def find_project_root(start: Optional[str] = None) -> str:
    """Ищем корень проекта вверх по дереву: pyproject.toml или папка configs/."""
    cur = os.path.abspath(start or os.getcwd())
    while True:
        if os.path.isfile(os.path.join(cur, "pyproject.toml")):
            return cur
        if os.path.isdir(os.path.join(cur, "configs")):
            return cur
        parent = os.path.dirname(cur)
        if parent == cur:
            return os.getcwd()
        cur = parent


def deep_update(base: Dict[str, Any], upd: Dict[str, Any]) -> Dict[str, Any]:
    for k, v in upd.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            base[k] = deep_update(base[k], v)
        else:
            base[k] = v
    return base


def load_yaml(path: str) -> Dict[str, Any]:
    root = find_project_root()
    abs_path = path if os.path.isabs(path) else os.path.join(root, path)
    with open(abs_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_config(base_path: str, override_path: str | None = None) -> Dict[str, Any]:
    cfg = load_yaml(base_path)
    if override_path is not None:
        cfg = deep_update(cfg, load_yaml(override_path))
    return cfg


def make_run_dir(out_dir: str, name: str | None) -> str:
    root = find_project_root()
    out_dir_abs = out_dir if os.path.isabs(out_dir) else os.path.join(root, out_dir)

    os.makedirs(out_dir_abs, exist_ok=True)
    if name is None:
        stamp = time.strftime("%Y%m%d_%H%M%S")
        name = f"exp_{stamp}"
    run_dir = os.path.join(out_dir_abs, name)
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(os.path.join(run_dir, "checkpoints"), exist_ok=True)
    return run_dir


def save_config(cfg: Dict[str, Any], run_dir: str) -> None:
    path = os.path.join(run_dir, "config.yaml")
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)
