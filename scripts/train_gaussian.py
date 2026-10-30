"""Train a head (the flow head of a shipped member, or the Gaussian head), and print what it learned.

Usage:
    uv run python scripts/train_gaussian.py --config configs/train_flow_v22_100.yaml --out results/flow_v22_100 --device cuda:0
    uv run python scripts/train_gaussian.py --device cuda:1                # the config's defaults
    uv run python scripts/train_gaussian.py --mode report                  # tables from results/
    uv run python scripts/train_gaussian.py --steps 2000 --out results/gaussian_smoke
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from chai.eval.calibration import MODALITIES
from chai.train.loop import TrainConfig, train

REPORT_TIMES = (10.0, 30.0, 60.0, 100.0)


def _nearest(t_grid: np.ndarray, t: float) -> int:
    return int(np.argmin(np.abs(t_grid - t)))


def report(out: Path) -> None:
    d = dict(np.load(out / "diagnostics.npz", allow_pickle=True))
    hist = [json.loads(line) for line in (out / "history.jsonl").read_text().splitlines()]
    mods = list(d["modalities"])
    t_grid = d["t_grid"]
    nodes = [_nearest(t_grid, t) for t in REPORT_TIMES]

    best = min(hist, key=lambda r: r["val_nll"])
    print(f"\nBest validation: step {best['step']} (epoch {best['epoch']:.1f}), "
          f"NLL {best['val_nll']:.3f} nats/row against the prior baseline "
          f"{best['val_nll_prior']:.3f}; {hist[-1]['elapsed_s'] / 60:.1f} min in total.")

    print("\nNLL by modality (nats/row; lower is more information from the data)")
    print(f"{'modality':12s} {'rows':>8s} {'NLL':>9s} {'prior':>9s} {'gain':>8s}")
    for i, m in enumerate(mods):
        print(f"{m:12s} {int(d['n'][i]):8d} {d['nll'][i]:9.3f} {d['nll_prior'][i]:9.3f} "
              f"{d['nll_prior'][i] - d['nll'][i]:8.2f}")

    print("\nPosterior sd in degC at the report nodes, per modality (prior sd in the header)")
    head = " ".join(f"{t:>6.0f}Ma" for t in REPORT_TIMES)
    print(f"{'modality':12s} {head}")
    if "prior_sd" in d:
        print(f"{'prior':12s} " + " ".join(f"{d['prior_sd'][k]:8.1f}" for k in nodes))
    for i, m in enumerate(mods):
        print(f"{m:12s} " + " ".join(f"{d['post_sd'][i, k]:8.1f}" for k in nodes))

    print("\nRMS error of the posterior mean, degC")
    for i, m in enumerate(mods):
        print(f"{m:12s} " + " ".join(f"{d['rmse_mean'][i, k]:8.1f}" for k in nodes))

    print("\nCoverage of the truth by the central 68% / 95% of clamped draws (ideal 0.68 / 0.95)")
    for i, m in enumerate(mods):
        print(f"{m:12s} " + " ".join(f"{d['cov68'][i, k]:.2f}/{d['cov95'][i, k]:.2f}"
                                     for k in nodes))

    print("\nStandardised residual in z (ideal var 1.00, coverage 0.68 / 0.95): "
          "hyperparameters, then the same nodes")
    # A GP prior's latent leads with three hyperparameters; a pulse prior's carries the path alone.
    n_h = d["resid_var"].shape[1] - int(d["post_sd"].shape[1])
    hyp = ["tau", "rate_sd", "drift"][:n_h]
    cols = list(range(n_h)) + [n_h + k for k in nodes]
    names = hyp + [f"{t:.0f}Ma" for t in REPORT_TIMES]
    print(f"{'modality':12s} " + " ".join(f"{n:>14s}" for n in names))
    for i, m in enumerate(mods):
        print(f"{m:12s} " + " ".join(
            f"{d['resid_var'][i, c]:5.2f} {d['resid_cov68'][i, c]:.2f}/{d['resid_cov95'][i, c]:.2f}"
            for c in cols))
    a = mods.index("all")
    print(f"\nAll rows, all {d['resid_var'].shape[1]} coordinates: residual variance median "
          f"{np.median(d['resid_var'][a]):.3f}, range [{d['resid_var'][a].min():.2f}, "
          f"{d['resid_var'][a].max():.2f}]; 68% coverage range "
          f"[{d['resid_cov68'][a].min():.2f}, {d['resid_cov68'][a].max():.2f}].")
    _ = MODALITIES


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/train_flow_v22_100.yaml")
    ap.add_argument("--out", default="results/flow_v22_100")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--mode", choices=("train", "report"), default="train")
    ap.add_argument("--steps", type=int, default=None, help="override the config's step count")
    ap.add_argument("--workers", type=int, default=None)
    args = ap.parse_args()
    out = Path(args.out)
    if args.mode == "train":
        cfg = TrainConfig.from_yaml(args.config)
        if args.steps is not None:
            cfg.steps = args.steps
        if args.workers is not None:
            cfg.num_workers = args.workers
        train(cfg, device=args.device, out_dir=out)
    if (out / "diagnostics.npz").exists():             # an unconditional run writes none
        report(out)


if __name__ == "__main__":
    main()
