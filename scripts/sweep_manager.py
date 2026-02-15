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


# -------- regex patterns --------
RUN_SAVED_RE = re.compile(r"Run saved to:\s*(.+)$")
PROGRESS_RE = re.compile(r"\[PROGRESS\]\s+epoch=(\d+)/(\d+)\s+step=(\d+)\s+loss=([0-9.]+)")


# -------- formatting --------
def now_stamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def fmt_seconds(s: Optional[float]) -> str:
    if s is None:
        return "??:??"
    s = max(0.0, float(s))
    h = int(s // 3600)
    m = int((s % 3600) // 60)
    sec = int(s % 60)
    if h > 0:
        return f"{h:02d}:{m:02d}:{sec:02d}"
    return f"{m:02d}:{sec:02d}"


def fmt_eta(elapsed: float, done: int, total: int) -> Optional[float]:
    if done <= 0:
        return None
    avg = elapsed / done
    remaining = (total - done) * avg
    return remaining


def print_line(msg: str) -> None:
    print(msg, flush=True)


# -------- config helpers --------
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


def product_dict(grid: Dict[str, List[Any]]) -> Iterable[Dict[str, Any]]:
    keys = list(grid.keys())
    for values in itertools.product(*[grid[k] for k in keys]):
        yield dict(zip(keys, values))


# -------- subprocess runners --------
def run_cmd_quiet(cmd: List[str], env: Optional[Dict[str, str]], log_file: Optional[Path]) -> str:
    """
    Run and capture output (used for eval/collect/plots).
    """
    p = subprocess.run(cmd, capture_output=True, text=True, env=env)
    out = (p.stdout or "") + "\n" + (p.stderr or "")

    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        with log_file.open("a", encoding="utf-8") as f:
            f.write("\n" + "=" * 90 + "\n")
            f.write("CMD: " + " ".join(cmd) + "\n")
            f.write(out + "\n")

    if p.returncode != 0:
        tail = "\n".join(out.splitlines()[-80:])
        raise RuntimeError(f"Command failed ({p.returncode}): {' '.join(cmd)}\n--- tail ---\n{tail}")

    return out


def run_train_stream(
    cmd: List[str],
    env: Optional[Dict[str, str]],
    log_file: Path,
    sweep_start: float,
    done_before: int,
    total_jobs: int,
) -> Tuple[Path, float]:
    """
    Stream stdout of training process in real-time.
    Returns: (run_dir, train_seconds)
    """
    log_file.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    run_dir: Optional[Path] = None
    last_epoch: Optional[int] = None
    total_epochs: Optional[int] = None
    last_step: Optional[int] = None
    last_loss: Optional[float] = None

    with log_file.open("a", encoding="utf-8") as lf:
        lf.write("\n" + "=" * 90 + "\n")
        lf.write("CMD: " + " ".join(cmd) + "\n")

        p = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env=env,
        )

        assert p.stdout is not None
        for line in p.stdout:
            line = line.rstrip("\n")
            lf.write(line + "\n")

            # detect run dir
            m_run = RUN_SAVED_RE.search(line.strip())
            if m_run:
                run_dir = Path(m_run.group(1).strip())

            # parse progress
            m = PROGRESS_RE.search(line)
            if m:
                e = int(m.group(1))
                E = int(m.group(2))
                step = int(m.group(3))
                loss = float(m.group(4))

                last_epoch, total_epochs = e, E
                last_step, last_loss = step, loss

                elapsed_run = time.time() - t0
                eta_run = None
                if e > 0:
                    sec_per_epoch = elapsed_run / e
                    eta_run = (E - e) * sec_per_epoch

                elapsed_all = time.time() - sweep_start
                eta_all = fmt_eta(elapsed_all, done_before, total_jobs)

                print_line(
                    f"    [train] epoch {e}/{E} | step={step} | loss={loss:.4f} | "
                    f"eta_run={fmt_seconds(eta_run)} | sweep_ETA={fmt_seconds(eta_all)}"
                )

        ret = p.wait()
        train_seconds = time.time() - t0

    if ret != 0:
        raise RuntimeError(f"Training process failed ({ret}). See log: {log_file}")

    if run_dir is None:
        raise RuntimeError("Could not detect run_dir from train output. Ensure train.py prints 'Run saved to: ...'")

    return run_dir, train_seconds


# -------- sweep config --------
@dataclass
class SweepConfig:
    runs_dir: str = "runs"
    eval_batch_size: int = 128
    sweep_tag: str = "sweep"
    out_overrides_dir: str = "configs/_sweeps"
    resume: bool = True
    max_failures: int = 5


def main():
    cfg = SweepConfig()

    # IMPORTANT: use current interpreter
    PY = os.sys.executable

    # -------- GRID --------
    grid = {
        "model.text_encoder.pooling": ["cls", "mean"],
        "model.text_encoder.unfreeze_last_n": [0, 2, 4],
        "model.projection.type": ["linear", "mlp"],
        "train.lr_text": [2e-5, 5e-5],
        "data.batch_size": [128],
        "train.epochs": [5],
    }

    sweep_id = f"{cfg.sweep_tag}_{now_stamp()}"
    overrides_root = Path(cfg.out_overrides_dir) / sweep_id
    overrides_root.mkdir(parents=True, exist_ok=True)

    # Prepare jobs
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
    start_all = time.time()

    sweep_log = Path(cfg.runs_dir) / f"{sweep_id}_sweep.log"
    err_log = Path(cfg.runs_dir) / f"{sweep_id}_errors.log"

    results: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []
    ok = 0
    failed = 0
    done = 0

    print_line("\n=== SWEEP START ===")
    print_line(f"sweep_id: {sweep_id}")
    print_line(f"jobs: {total}")
    print_line(f"python: {PY}")
    print_line(f"overrides: {overrides_root}")
    print_line(f"log: {sweep_log}")
    print_line(f"errors: {err_log}\n")

    for i, (run_name, params, override_path) in enumerate(jobs, start=1):
        elapsed = time.time() - start_all
        eta_all = fmt_eta(elapsed, done, total)

        header = (
            f"[{i:03d}/{total:03d}] "
            f"elapsed={fmt_seconds(elapsed)} "
            f"avg/run={fmt_seconds(elapsed/done) if done>0 else '??:??'} "
            f"ETA={fmt_seconds(eta_all)} | "
            f"ok={ok} fail={failed} | "
            f"{run_name}"
        )
        print_line(header)

        env = os.environ.copy()
        env["OVERRIDE_CFG"] = str(override_path)

        run_dir_guess = Path(cfg.runs_dir) / run_name
        if cfg.resume and run_dir_guess.exists() and (run_dir_guess / "full_eval_val.json").exists():
            print_line(f"  └─ SKIP (resume): found {run_dir_guess/'full_eval_val.json'}\n")
            results.append({
                "idx": i, "run_name": run_name, "override_cfg": str(override_path),
                "run_dir": str(run_dir_guess), "status": "skipped",
            })
            done += 1
            continue

        status = "ok"
        run_dir: Optional[Path] = None

        # ---- TRAIN (stream) ----
        try:
            print_line("  ├─ TRAIN ...")
            run_dir, train_sec = run_train_stream(
                [PY, "-m", "mini_clip.train"],
                env=env,
                log_file=sweep_log,
                sweep_start=start_all,
                done_before=done,
                total_jobs=total,
            )
            print_line(f"  │  ✓ TRAIN done in {fmt_seconds(train_sec)} | run_dir={run_dir}")
        except Exception as e:
            status = "train_failed"
            failed += 1
            print_line(f"  │  ✗ TRAIN FAILED | {e}\n")
            err_log.parent.mkdir(parents=True, exist_ok=True)
            with err_log.open("a", encoding="utf-8") as f:
                f.write(f"\n[{run_name}] TRAIN FAILED\n{repr(e)}\n")
            failures.append({"run_name": run_name, "stage": "train", "error": repr(e)})
            results.append({
                "idx": i, "run_name": run_name, "override_cfg": str(override_path),
                "run_dir": str(run_dir_guess), "status": status,
            })
            done += 1
            if failed >= cfg.max_failures:
                print_line("Stopping: reached max_failures.\n")
                break
            continue

        # ---- EVAL ----
        try:
            print_line("  ├─ EVAL (full) val+test ...")
            run_cmd_quiet(
                [PY, "scripts/eval_run.py", "--run_dir", str(run_dir), "--batch_size", str(cfg.eval_batch_size)],
                env=env,
                log_file=sweep_log,
            )
            print_line("  │  ✓ EVAL done")
        except Exception as e:
            status = "eval_failed"
            failed += 1
            print_line(f"  │  ✗ EVAL FAILED | {e}")
            err_log.parent.mkdir(parents=True, exist_ok=True)
            with err_log.open("a", encoding="utf-8") as f:
                f.write(f"\n[{run_name}] EVAL FAILED\n{repr(e)}\n")
            failures.append({"run_name": run_name, "stage": "eval", "error": repr(e)})

        if status == "ok":
            ok += 1

        results.append({
            "idx": i,
            "run_name": run_name,
            "override_cfg": str(override_path),
            "run_dir": str(run_dir),
            "status": status,
        })
        done += 1
        print_line(f"  └─ DONE status={status}\n")

    # ---- finalize ----
    elapsed_total = time.time() - start_all
    print_line("\n=== SWEEP FINISH ===")
    print_line(f"ok={ok} failed={failed} done={done}/{total} elapsed={fmt_seconds(elapsed_total)}")

    manifest = Path(cfg.runs_dir) / f"{sweep_id}_manifest.json"
    manifest.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print_line(f"manifest: {manifest}")

    if failures:
        fail_path = Path(cfg.runs_dir) / f"{sweep_id}_failures.json"
        fail_path.write_text(json.dumps(failures, ensure_ascii=False, indent=2), encoding="utf-8")
        print_line(f"failures: {fail_path}")

    # collect + plots
    try:
        print_line("\n=== COLLECT RESULTS ===")
        run_cmd_quiet(
            [PY, "scripts/collect_results.py", "--runs_dir", cfg.runs_dir, "--out_csv", "runs/results.csv", "--metric", "val_i2t_R@1"],
            env=None,
            log_file=sweep_log,
        )
        print_line("Saved: runs/results.csv")

        print_line("\n=== PLOTS ===")
        run_cmd_quiet(
            [PY, "scripts/plot_results.py", "--csv", "runs/results.csv", "--out_dir", "runs/plots", "--metric", "val_i2t_R@1", "--topk", "15"],
            env=None,
            log_file=sweep_log,
        )
        print_line("Saved: runs/plots/*.png")
    except Exception as e:
        print_line(f"Collect/plots skipped due to error: {e}")

    print_line("\nDone.\n")


if __name__ == "__main__":
    main()
