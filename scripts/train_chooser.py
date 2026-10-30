"""Train the member chooser.

Usage:
    uv run python scripts/train_chooser.py --config configs/chooser_v22.yaml --out results/chooser_v22 --device cuda:1
    uv run python scripts/train_chooser.py --steps 500 --out results/chooser_smoke
"""
from __future__ import annotations

import argparse

from chai.train.chooser import ChooserConfig, train_chooser


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/chooser_v22.yaml")
    ap.add_argument("--out", default="results/chooser_v22")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--steps", type=int, default=None, help="override the config's step count")
    ap.add_argument("--workers", type=int, default=None)
    args = ap.parse_args()
    cfg = ChooserConfig.from_yaml(args.config)
    if args.steps is not None:
        cfg.steps = args.steps
    if args.workers is not None:
        cfg.num_workers = args.workers
    train_chooser(cfg, device=args.device, out_dir=args.out, log=lambda *a: print(*a, flush=True))


if __name__ == "__main__":
    main()
