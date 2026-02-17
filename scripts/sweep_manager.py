from __future__ import annotations

import os
import re
import json
import time
import itertools
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Iterable, Optional, Tuple

import yaml

from rich.console import Console
from rich.live import Live
from rich.layout import Layout
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.progress import (
    Progress,
    SpinnerColumn,
    BarColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.align import Align


# --------- Regex patterns ----------
RUN_SAVED_RE = re.compile(r"Run saved to:\s*(.+)$")
# Expected line example:
# [PROGRESS] epoch=1/5 iter=20/235 step=20 loss=4.4940 i2t@1=0.0312 t2i@1=0.0312
PROGRESS_RE = re.compile(
    r"\[PROGRESS\]\s+epoch=(\d+)/(\d+)\s+iter=(\d+)/(\d+)\s+step=(\d+)\s+loss=([0-9.]+)\s+i2t@1=([0-9.]+)\s+t2i@1=([0-9.]+)"
)


# --------- Small helpers ----------
def now_stamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def deep_set(d: Dict[str, Any], dotted_key: str, value: Any) -> None:
    keys = dotted_key.split(".")
    cur = d
    for k in keys[:-1]:
        if k not in cur or not isinstance(cur[k], dict):
            cur[k] = {}
        cur = cur[k]
    cur[keys[-1]] = value


def dump_yaml(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")


def product_dict(grid: Dict[str, List[Any]]) -> Iterable[Dict[str, Any]]:
    keys = list(grid.keys())
    for values in itertools.product(*[grid[k] for k in keys]):
        yield dict(zip(keys, values))


def format_tag(k: str, v: Any) -> str:
    k = k.replace(".", "_")
    if isinstance(v, float):
        s = f"{v:.2e}".replace("+0", "").replace("+", "")
    else:
        s = str(v)
    s = s.replace("/", "-").replace(" ", "")
    return f"{k}={s}"


def build_run_name(base: str, params: Dict[str, Any], max_len: int = 140) -> str:
    parts = [base] + [format_tag(k, v) for k, v in params.items()]
    name = "__".join(parts)
    return name[:max_len]


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


# --------- Sparkline (ASCII graph) ----------
SPARK_CHARS = "▁▂▃▄▅▆▇█"


def sparkline(values: List[float], width: int = 60) -> str:
    if not values:
        return ""
    # downsample to width
    if len(values) > width:
        step = len(values) / width
        sampled = []
        for i in range(width):
            j = int(i * step)
            sampled.append(values[j])
        values = sampled

    vmin = min(values)
    vmax = max(values)
    if vmax - vmin < 1e-12:
        return SPARK_CHARS[0] * len(values)

    out = []
    for v in values:
        t = (v - vmin) / (vmax - vmin)
        idx = int(t * (len(SPARK_CHARS) - 1))
        out.append(SPARK_CHARS[idx])
    return "".join(out)


def tail_lines(text: str, n: int = 40) -> str:
    lines = text.splitlines()
    return "\n".join(lines[-n:])


# --------- Sweep config ----------
@dataclass
class SweepConfig:
    runs_dir: str = "runs"
    out_overrides_dir: str = "configs/_sweeps"
    sweep_tag: str = "sweep"
    eval_batch_size: int = 128
    resume: bool = True
    max_failures: int = 5

    # UI tuning
    spark_width: int = 60
    max_points: int = 400  # keep last N progress points


# --------- UI layout ----------
def make_layout() -> Layout:
    layout = Layout()

    layout.split_column(
        Layout(name="top", size=9),
        Layout(name="mid", size=10),
        Layout(name="bottom"),
    )

    layout["top"].split_row(
        Layout(name="progress"),
        Layout(name="status"),
    )
    layout["mid"].split_row(
        Layout(name="sparks"),
        Layout(name="params"),
    )

    return layout


def make_status_panel(title: str, lines: List[str]) -> Panel:
    txt = Text()
    for ln in lines:
        txt.append(ln + "\n")
    return Panel(txt, title=title, border_style="cyan")


def make_params_panel(params: Dict[str, Any]) -> Panel:
    table = Table(show_header=False, box=None, pad_edge=False)
    table.add_column(justify="right")
    table.add_column()
    for k, v in params.items():
        table.add_row(k, str(v))
    return Panel(table, title="Current params", border_style="magenta")


def make_sparks_panel(loss_vals: List[float], i2t_vals: List[float], t2i_vals: List[float], width: int) -> Panel:
    table = Table(show_header=False, box=None, pad_edge=False)
    table.add_column(justify="right", width=10)
    table.add_column()

    table.add_row("loss", sparkline(loss_vals, width))
    table.add_row("i2t@1", sparkline(i2t_vals, width))
    table.add_row("t2i@1", sparkline(t2i_vals, width))
    return Panel(table, title="Live metrics (sparkline)", border_style="green")


# --------- Subprocess runners ----------
def run_cmd_quiet(cmd: List[str], env: Optional[Dict[str, str]], log_file: Path) -> str:
    p = subprocess.run(cmd, capture_output=True, text=True, env=env)
    out = (p.stdout or "") + "\n" + (p.stderr or "")

    ensure_dir(log_file.parent)
    with log_file.open("a", encoding="utf-8") as f:
        f.write("\n" + "=" * 90 + "\n")
        f.write("CMD: " + " ".join(cmd) + "\n")
        f.write(out + "\n")

    if p.returncode != 0:
        raise RuntimeError(f"Command failed ({p.returncode}): {' '.join(cmd)}\n--- tail ---\n{tail_lines(out, 80)}")
    return out


def run_train_stream_rich(
    cmd: List[str],
    env: Dict[str, str],
    log_file: Path,
    ui: "SweepUI",
) -> Tuple[Path, float]:
    """
    Runs training and streams stdout. Updates UI by parsing [PROGRESS] lines.
    Returns: run_dir, train_seconds
    """
    ensure_dir(log_file.parent)
    t0 = time.time()
    run_dir: Optional[Path] = None

    with log_file.open("a", encoding="utf-8") as lf:
        lf.write("\n" + "=" * 90 + "\n")
        lf.write("CMD: " + " ".join(cmd) + "\n")

        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env)
        assert p.stdout is not None

        for line in p.stdout:
            line = line.rstrip("\n")
            lf.write(line + "\n")

            m_run = RUN_SAVED_RE.search(line.strip())
            if m_run:
                run_dir = Path(m_run.group(1).strip())
                ui.set_run_dir(run_dir)

            m = PROGRESS_RE.search(line)
            if m:
                epoch = int(m.group(1))
                epochs = int(m.group(2))
                it = int(m.group(3))
                iters = int(m.group(4))
                step = int(m.group(5))
                loss = float(m.group(6))
                i2t = float(m.group(7))
                t2i = float(m.group(8))

                ui.update_train_progress(epoch, epochs, it, iters, step, loss, i2t, t2i)

        ret = p.wait()

    train_seconds = time.time() - t0

    if ret != 0:
        raise RuntimeError(f"Training process failed ({ret}). See: {log_file}")

    if run_dir is None:
        raise RuntimeError("Could not detect run_dir. Ensure train.py prints: 'Run saved to: ...'")

    return run_dir, train_seconds


# --------- Rich UI controller ----------
class SweepUI:
    def __init__(self, console: Console, total_jobs: int, spark_width: int, max_points: int):
        self.console = console
        self.total_jobs = total_jobs
        self.spark_width = spark_width
        self.max_points = max_points

        self.layout = make_layout()

        self.progress = Progress(
            SpinnerColumn(),
            TextColumn("[bold]{task.description}[/bold]"),
            BarColumn(),
            TextColumn("{task.completed}/{task.total}"),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            expand=True,
        )

        self.task_sweep = self.progress.add_task("SWEEP", total=total_jobs)
        self.task_epoch = self.progress.add_task("EPOCH", total=1)
        self.task_iter = self.progress.add_task("ITER", total=1)

        self.run_name: str = ""
        self.run_dir: Optional[Path] = None
        self.status_lines: List[str] = []

        self.params: Dict[str, Any] = {}
        self.loss_vals: List[float] = []
        self.i2t_vals: List[float] = []
        self.t2i_vals: List[float] = []

        self.epoch = 0
        self.epochs = 0
        self.it = 0
        self.iters = 0
        self.step = 0

    def set_job(self, idx: int, run_name: str, params: Dict[str, Any], ok: int, fail: int):
        self.run_name = run_name
        self.run_dir = None
        self.params = params
        self.loss_vals.clear()
        self.i2t_vals.clear()
        self.t2i_vals.clear()

        # reset epoch/iter tasks (unknown until first PROGRESS)
        self.progress.update(self.task_epoch, completed=0, total=1)
        self.progress.update(self.task_iter, completed=0, total=1)

        self.status_lines = [
            f"job: {idx}/{self.total_jobs}",
            f"ok: {ok}  fail: {fail}",
            f"run: {run_name}",
            "stage: TRAIN",
        ]

        self.refresh()

    def set_stage(self, stage: str):
        # update stage line
        for i, ln in enumerate(self.status_lines):
            if ln.startswith("stage:"):
                self.status_lines[i] = f"stage: {stage}"
                break
        else:
            self.status_lines.append(f"stage: {stage}")
        self.refresh()

    def set_run_dir(self, run_dir: Path):
        self.run_dir = run_dir
        # keep status line updated
        self._set_or_add("run_dir", str(run_dir))
        self.refresh()

    def mark_sweep_done_one(self):
        self.progress.advance(self.task_sweep, 1)
        self.refresh()

    def _set_or_add(self, key: str, value: str):
        prefix = f"{key}:"
        for i, ln in enumerate(self.status_lines):
            if ln.startswith(prefix):
                self.status_lines[i] = f"{prefix} {value}"
                return
        self.status_lines.append(f"{prefix} {value}")

    def update_train_progress(self, epoch: int, epochs: int, it: int, iters: int, step: int, loss: float, i2t: float, t2i: float):
        self.epoch, self.epochs = epoch, epochs
        self.it, self.iters = it, iters
        self.step = step

        # update bars
        self.progress.update(self.task_epoch, total=max(epochs, 1), completed=min(epoch, epochs))
        self.progress.update(self.task_iter, total=max(iters, 1), completed=min(it, iters))

        # store metrics
        self.loss_vals.append(loss)
        self.i2t_vals.append(i2t)
        self.t2i_vals.append(t2i)
        if len(self.loss_vals) > self.max_points:
            self.loss_vals = self.loss_vals[-self.max_points :]
            self.i2t_vals = self.i2t_vals[-self.max_points :]
            self.t2i_vals = self.t2i_vals[-self.max_points :]

        # status lines
        self._set_or_add("epoch", f"{epoch}/{epochs}")
        self._set_or_add("iter", f"{it}/{iters}")
        self._set_or_add("step", str(step))
        self._set_or_add("loss", f"{loss:.4f}")
        self._set_or_add("i2t@1", f"{i2t:.4f}")
        self._set_or_add("t2i@1", f"{t2i:.4f}")

        self.refresh()

    def set_message(self, msg: str):
        self._set_or_add("msg", msg)
        self.refresh()

    def refresh(self):
        # Left: progress bars
        self.layout["progress"].update(Panel(self.progress, title="Progress", border_style="blue"))

        # Right: status
        self.layout["status"].update(make_status_panel("Status", self.status_lines))

        # Sparks
        self.layout["sparks"].update(make_sparks_panel(self.loss_vals, self.i2t_vals, self.t2i_vals, self.spark_width))

        # Params
        self.layout["params"].update(make_params_panel(self.params))

    def renderable(self):
        return self.layout


# --------- Main sweep ----------
def main():
    console = Console()
    cfg = SweepConfig()
    PY = os.sys.executable

    # ---- GRID ----
    grid = {
        "model.text_encoder.pooling": ["mean"],
        "model.text_encoder.unfreeze_last_n": [0, 2, 4],
        "train.lr_text": [1e-5, 5e-5],
        "model.projection.type": ["linear"],
        "train.weight_decay": [0.01],
        "data.batch_size": [128],
        "train.epochs": [10],
        "train.lr": [1e-3],
    }


    sweep_id = f"{cfg.sweep_tag}_{now_stamp()}"
    overrides_root = Path(cfg.out_overrides_dir) / sweep_id
    ensure_dir(overrides_root)

    sweep_log = Path(cfg.runs_dir) / f"{sweep_id}_sweep.log"
    err_log = Path(cfg.runs_dir) / f"{sweep_id}_errors.log"
    ensure_dir(Path(cfg.runs_dir))

    # Prepare jobs list
    jobs: List[Tuple[str, Dict[str, Any], Path]] = []
    for idx, params in enumerate(product_dict(grid), start=1):
        run_name = build_run_name(f"{sweep_id}__{idx:03d}", params)

        override: Dict[str, Any] = {}
        deep_set(override, "run.name", run_name)
        deep_set(override, "run.out_dir", cfg.runs_dir)

        for k, v in params.items():
            deep_set(override, k, v)

        override_path = overrides_root / f"{run_name}.yaml"
        dump_yaml(override_path, override)
        jobs.append((run_name, params, override_path))

    total = len(jobs)
    ui = SweepUI(console, total_jobs=total, spark_width=cfg.spark_width, max_points=cfg.max_points)

    results: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []
    ok = 0
    failed = 0

    start_all = time.time()

    console.print(f"[bold cyan]SWEEP[/bold cyan] id={sweep_id} jobs={total} python={PY}")
    console.print(f"overrides: {overrides_root}")
    console.print(f"log: {sweep_log}")
    console.print(f"errors: {err_log}")
    console.print()

    with Live(ui.renderable(), console=console, refresh_per_second=12):
        for i, (run_name, params, override_path) in enumerate(jobs, start=1):
            # Resume check
            run_dir_guess = Path(cfg.runs_dir) / run_name
            if cfg.resume and run_dir_guess.exists() and (run_dir_guess / "full_eval_val.json").exists():
                ui.set_job(i, run_name, params, ok, failed)
                ui.set_stage("SKIP (resume)")
                ui.set_message("full_eval_val.json exists")
                ui.mark_sweep_done_one()

                results.append({
                    "idx": i,
                    "run_name": run_name,
                    "override_cfg": str(override_path),
                    "run_dir": str(run_dir_guess),
                    "status": "skipped",
                })
                continue

            ui.set_job(i, run_name, params, ok, failed)

            env = os.environ.copy()
            env["OVERRIDE_CFG"] = str(override_path)

            status = "ok"
            run_dir: Optional[Path] = None

            # ---- TRAIN (stream + UI updates) ----
            try:
                ui.set_stage("TRAIN")
                run_dir, train_sec = run_train_stream_rich(
                    [PY, "-m", "mini_clip.train"],
                    env=env,
                    log_file=sweep_log,
                    ui=ui,
                )
                ui.set_message(f"train_done={train_sec:.1f}s")
            except Exception as e:
                status = "train_failed"
                failed += 1
                failures.append({"run_name": run_name, "stage": "train", "error": repr(e)})
                ensure_dir(err_log.parent)
                with err_log.open("a", encoding="utf-8") as f:
                    f.write(f"\n[{run_name}] TRAIN FAILED\n{repr(e)}\n")

            # ---- EVAL ----
            if status == "ok" and run_dir is not None:
                try:
                    ui.set_stage("EVAL (full) val+test")
                    run_cmd_quiet(
                        [PY, "scripts/eval_run.py", "--run_dir", str(run_dir), "--batch_size", str(cfg.eval_batch_size)],
                        env=env,
                        log_file=sweep_log,
                    )
                    ui.set_message("eval_done")
                except Exception as e:
                    status = "eval_failed"
                    failed += 1
                    failures.append({"run_name": run_name, "stage": "eval", "error": repr(e)})
                    ensure_dir(err_log.parent)
                    with err_log.open("a", encoding="utf-8") as f:
                        f.write(f"\n[{run_name}] EVAL FAILED\n{repr(e)}\n")

            if status == "ok":
                ok += 1

            results.append({
                "idx": i,
                "run_name": run_name,
                "override_cfg": str(override_path),
                "run_dir": str(run_dir) if run_dir else str(run_dir_guess),
                "status": status,
            })

            ui.set_stage(f"DONE ({status})")
            ui.mark_sweep_done_one()

            if failed >= cfg.max_failures:
                ui.set_message("Stopping: max_failures reached")
                break

        # finalize: collect + plots
        ui.set_stage("COLLECT RESULTS")
        try:
            run_cmd_quiet(
                [PY, "scripts/collect_results.py", "--runs_dir", cfg.runs_dir, "--out_csv", "runs/results.csv", "--metric", "val_i2t_R@1"],
                env=None,
                log_file=sweep_log,
            )
            ui.set_message("saved runs/results.csv")
        except Exception as e:
            ui.set_message(f"collect_failed: {e}")

        ui.set_stage("PLOTS")
        try:
            run_cmd_quiet(
                [PY, "scripts/plot_results.py", "--csv", "runs/results.csv", "--out_dir", "runs/plots", "--metric", "val_i2t_R@1", "--topk", "15"],
                env=None,
                log_file=sweep_log,
            )
            ui.set_message("saved runs/plots/*.png")
        except Exception as e:
            ui.set_message(f"plot_failed: {e}")

    elapsed = time.time() - start_all

    manifest = Path(cfg.runs_dir) / f"{sweep_id}_manifest.json"
    manifest.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    if failures:
        fail_path = Path(cfg.runs_dir) / f"{sweep_id}_failures.json"
        fail_path.write_text(json.dumps(failures, ensure_ascii=False, indent=2), encoding="utf-8")

    console.print()
    console.print(f"[bold green]DONE[/bold green] ok={ok} failed={failed} elapsed={elapsed:.1f}s")
    console.print(f"manifest: {manifest}")
    if failures:
        console.print(f"failures: {Path(cfg.runs_dir) / f'{sweep_id}_failures.json'}")


if __name__ == "__main__":
    main()
