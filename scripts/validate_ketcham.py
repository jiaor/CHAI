"""Validate the batched PyTorch Ketcham AFT model against the native C oracle.

Reports the age, the c-axis-projected track-length distribution, and throughput.
Where the two disagree at the oracle's default grid, refining the *oracle*
(``--pct``) moves it onto the torch answer, so the reference here is a refined
oracle rather than the default one.

Usage:
    uv run python scripts/validate_ketcham.py [--n 2000] [--n-sub 128]
                                              [--model 1999|2007] [--pct 0.05]
                                              [--device cpu|cuda]
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import time

import numpy as np
import torch

from chai.forward.native import calculate_aft
from chai.forward.native.ketcham import set_time_step_pct
from chai.forward.torchimpl.ketcham import KETCHAM_1999, KETCHAM_2007, aft_forward

N_CTRL = 10
T_MAX, T_HOT, T_PRES_MAX = 100.0, 125.0, 30.0
BIN_WIDTH = 20.0 / 200

MODELS = {"1999": (KETCHAM_1999, 1), "2007": (KETCHAM_2007, 2)}


def sample_paths(n: int, seed: int = 0):
    """Random control-point paths: a mix of monotonic cooling and free paths."""
    rng = np.random.default_rng(seed)
    t = np.zeros((n, N_CTRL))
    t[:, 1:] = np.sort(rng.uniform(1.0, T_MAX, size=(n, N_CTRL - 1)), axis=1)
    # enforce a minimum spacing so segments are never degenerate
    t[:, 1:] = np.maximum.accumulate(t[:, 1:] + 1e-3 * np.arange(N_CTRL - 1), axis=1)

    T = np.empty((n, N_CTRL))
    T[:, 0] = rng.uniform(0.0, T_PRES_MAX, size=n)
    free = rng.uniform(0.0, T_HOT, size=(n, N_CTRL - 1))
    mono = np.sort(rng.uniform(0.0, T_HOT, size=(n, N_CTRL - 1)), axis=1)
    is_mono = rng.random(n) < 0.5
    T[:, 1:] = np.where(is_mono[:, None], mono, free)
    T[:, 1:] = np.maximum(T[:, 1:], T[:, 0:1])  # never cooler than the present day

    dpar = rng.uniform(1.5, 3.5, size=n)
    return t, T, dpar


_JOB: dict = {}


def _oracle(args):
    t, T, d = args
    set_time_step_pct(_JOB["pct"])
    r = calculate_aft(t, T, kinetic_par=float(d), annealing_model=_JOB["model_id"])
    return r.age, r.mean_length, r.length_pdf


def _init(pct, model_id):
    _JOB.update(pct=pct, model_id=model_id)


def _summary(name, x, unit="", fmt="{:.3e}"):
    q = np.percentile(x, [50, 99, 100])
    print(f"  {name:<24s} median {fmt.format(q[0])}  p99 {fmt.format(q[1])}  "
          f"max {fmt.format(q[2])} {unit}")


def main(n_paths=2000, n_sub=128, model="1999", pct=0.05, device="cpu", procs=32):
    anneal, model_id = MODELS[model]
    t, T, dpar = sample_paths(n_paths)

    t0 = time.perf_counter()
    with mp.Pool(procs, initializer=_init, initargs=(pct, model_id)) as pool:
        res = pool.map(_oracle, zip(t, T, dpar), chunksize=16)
    t_oracle = time.perf_counter() - t0
    ref_age = np.array([r[0] for r in res])
    ref_mean = np.array([r[1] for r in res])
    ref_pdf = np.array([r[2] for r in res])

    args = [torch.as_tensor(x, dtype=torch.float64, device=device) for x in (t, T, dpar)]
    if device == "cuda":
        aft_forward(*args, n_sub=n_sub, model=anneal)
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    out = aft_forward(*args, n_sub=n_sub, model=anneal)
    if device == "cuda":
        torch.cuda.synchronize()
    t_torch = time.perf_counter() - t0
    age = out["age"].cpu().numpy()
    mean = out["mean_length"].cpu().numpy()
    pdf = out["length_pdf"].cpu().numpy()

    live = ref_pdf.max(axis=1) > 0
    ok = ref_age > 0.1  # relative error is meaningless for essentially-zero ages
    print(f"Ketcham {model}  paths={n_paths}  n_sub={n_sub}  oracle pct={pct}  "
          f"device={device}")
    print(f"  oracle {t_oracle:7.2f} s ({procs} procs)   torch {t_torch:6.2f} s "
          f"-> {n_paths / t_torch:,.0f} paths/s (age + lengths)")
    print(f"  nonzero ages {ok.sum()}/{n_paths}   surviving tracks {live.sum()}/{n_paths}"
          f"   NaNs {np.isnan(age).sum() + np.isnan(pdf).sum()}")

    print(" age")
    _summary("abs diff", np.abs(age - ref_age), "Ma")
    _summary("rel diff", np.abs(age - ref_age)[ok] / ref_age[ok])
    print(" track lengths")
    # L1 between two densities that each integrate to 1: the fraction of
    # probability mass in the wrong bin.
    _summary("pdf L1 distance", np.abs(pdf - ref_pdf).sum(axis=1)[live] * BIN_WIDTH)
    _summary("pdf max abs / peak", (np.abs(pdf - ref_pdf).max(axis=1)[live]
                                    / ref_pdf.max(axis=1)[live]))
    _summary("mean length diff", np.abs(mean - ref_mean)[live], "um")

    rel = np.abs(age - ref_age) / np.maximum(ref_age, 1e-9)
    print("  worst relative age offenders:")
    for i in np.argsort(-rel * ok)[:5]:
        print(f"    ref={ref_age[i]:9.4f}  got={age[i]:9.4f}  rel={rel[i]:.2e}  "
              f"dpar={dpar[i]:.2f}  Tmax={T[i].max():6.1f}")
    return ref_age, age


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--n-sub", type=int, default=128)
    ap.add_argument("--model", choices=sorted(MODELS), default="1999")
    ap.add_argument("--pct", type=float, default=0.05, help="oracle quadrature step")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--procs", type=int, default=32)
    a = ap.parse_args()
    main(a.n, a.n_sub, a.model, a.pct, a.device, a.procs)
