from __future__ import annotations

import re
import argparse
from pathlib import Path
import subprocess
import sys


def find_last_ckpt(run_dir: Path) -> Path:
    ckpt_dir = run_dir / "checkpoints"
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"No checkpoints dir: {ckpt_dir}")

    pts = list(ckpt_dir.glob("epoch_*.pt"))
    if not pts:
        raise FileNotFoundError(f"No epoch_*.pt found in: {ckpt_dir}")

    def epoch_num(p: Path) -> int:
        m = re.search(r"epoch_(\d+)\.pt$", p.name)
        return int(m.group(1)) if m else -1

    pts.sort(key=epoch_num)
    return pts[-1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", type=str, required=True)
    ap.add_argument("--ckpt", type=str, default=None, help="optional path to checkpoint; otherwise uses last epoch_*.pt")
    ap.add_argument("--batch_size", type=int, default=128)
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        raise FileNotFoundError(run_dir)

    ckpt = Path(args.ckpt) if args.ckpt else find_last_ckpt(run_dir)

    # call: python -m mini_clip.eval ...
    for split in ["val", "test"]:
        out_json = run_dir / f"full_eval_{split}.json"
        cmd = [
            sys.executable, "-m", "mini_clip.eval",
            "--split", split,
            "--ckpt", str(ckpt),
            "--batch_size", str(args.batch_size),
            "--out_json", str(out_json),
        ]
        print("Running:", " ".join(cmd))
        subprocess.check_call(cmd)

    print("Done. Saved full eval JSONs in:", str(run_dir))


if __name__ == "__main__":
    main()
