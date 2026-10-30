"""Validate the batched PyTorch RDAAM model against the C++ oracle.

The oracle's default answer carries its own discretisation error, and unlike
KETCH.c there is no single knob for it.  There are two:

* ``--precision`` (GOOD/BETTER/BEST) sets the radial nodes and the interpolator's
  maximum temperature step;
* ``--oracle-refine K`` hands the oracle the *same* piecewise-linear path
  pre-subdivided K times.  Its interpolator forces at least five steps per input
  segment, so this refines its internal grid without changing the physics.

Use ``--oracle-refine`` to establish which side is converging before trusting
any agreement number.

Usage:
    uv run python scripts/validate_rdaam.py [--n 200] [--n-sub 64] [--rdim 512]
                                            [--oracle-refine 1] [--device cpu|cuda]
"""
from __future__ import annotations

import argparse
import itertools
import multiprocessing as mp
import time

import numpy as np
import torch

from chai.forward.native import calculate_rdaam_debug, eU
from chai.forward.torchimpl.rdaam import rdaam_age
from chai.paths import refine

N_CTRL = 6


def sample_grains(n: int, seed: int = 0):
    """Paths and grains spanning the apatite He partial-retention zone."""
    rng = np.random.default_rng(seed)
    t = np.zeros((n, N_CTRL))
    t[:, 1:] = np.sort(rng.uniform(1.0, 100.0, size=(n, N_CTRL - 1)), axis=1)
    t[:, 1:] = np.maximum.accumulate(t[:, 1:] + 1e-3 * np.arange(N_CTRL - 1), axis=1)

    T = np.empty((n, N_CTRL))
    T[:, 0] = rng.uniform(0.0, 30.0, size=n)
    mono = np.sort(rng.uniform(0.0, 120.0, size=(n, N_CTRL - 1)), axis=1)
    free = rng.uniform(0.0, 120.0, size=(n, N_CTRL - 1))
    T[:, 1:] = np.where((rng.random(n) < 0.5)[:, None], mono, free)
    T[:, 1:] = np.maximum(T[:, 1:], T[:, 0:1])

    return (t, T, rng.uniform(30.0, 100.0, size=n),
            rng.uniform(5.0, 150.0, size=n), rng.uniform(1.0, 200.0, size=n))


_JOB: dict = {}


def _init(precision, oracle_refine):
    _JOB.update(precision=precision, refine=oracle_refine)


def _oracle(args):
    t, T, rad, U, Th = args
    k = _JOB["refine"]
    if k > 1:
        tf, Tf = refine(torch.as_tensor(t[None]), torch.as_tensor(T[None]), k)
        t, T = tf[0].numpy(), Tf[0].numpy()
    r = calculate_rdaam_debug(t, T, float(rad), float(U), float(Th),
                              precision=_JOB["precision"])
    return r.age, r.corrected_age, r.n_nodes


def _by_covariate(name, cov, rel, edges):
    print(f"  by {name}:")
    for lo, hi in itertools.pairwise(edges):
        m = (cov >= lo) & (cov < hi)
        if m.sum() > 2:
            print(f"    [{lo:5g}, {hi:5g})  n={m.sum():4d}   median {np.median(rel[m]):.2e}"
                  f"   p90 {np.percentile(rel[m], 90):.2e}   max {rel[m].max():.2e}")


def main(n=200, n_sub=64, rdim=512, precision=2, oracle_refine=1, device="cpu", procs=32):
    t, T, rad, U, Th = sample_grains(n)

    t0 = time.perf_counter()
    with mp.Pool(procs, initializer=_init, initargs=(precision, oracle_refine)) as pool:
        res = pool.map(_oracle, zip(t, T, rad, U, Th), chunksize=2)
    t_oracle = time.perf_counter() - t0
    ref = np.array([r[0] for r in res])
    ref_corr = np.array([r[1] for r in res])
    nodes = np.array([r[2] for r in res])

    args = [torch.as_tensor(x, dtype=torch.float64, device=device)
            for x in (t, T, rad, U, Th)]
    with torch.no_grad():
        if device == "cuda":
            rdaam_age(*args, n_sub=n_sub, rdim=rdim, use_checkpoint=False)
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = rdaam_age(*args, n_sub=n_sub, rdim=rdim, use_checkpoint=False)
        if device == "cuda":
            torch.cuda.synchronize()
        t_torch = time.perf_counter() - t0
    age = out["age"].cpu().numpy()
    corr = out["corrected_age"].cpu().numpy()

    print(f"RDAAM  grains={n}  n_sub={n_sub} (N={(N_CTRL - 1) * n_sub + 1})  rdim={rdim}  "
          f"device={device}")
    print(f"  oracle precision={precision} refine={oracle_refine} -> {nodes.min()}-{nodes.max()} "
          f"internal nodes, {t_oracle:.1f} s ({procs} procs)")
    print(f"  torch {t_torch:.2f} s -> {n / t_torch:,.0f} grains/s     "
          f"NaNs {np.isnan(age).sum()}")

    rel = np.abs(age - ref) / np.maximum(ref, 1e-9)
    relc = np.abs(corr - ref_corr) / np.maximum(ref_corr, 1e-9)
    print(f"  raw age       median {np.median(rel):.2e}  p90 {np.percentile(rel, 90):.2e}"
          f"  max {rel.max():.2e}")
    print(f"  corrected age median {np.median(relc):.2e}  p90 {np.percentile(relc, 90):.2e}"
          f"  max {relc.max():.2e}")

    print(" error vs covariate (relative age error):")
    _by_covariate("age (Ma)", ref, rel, [0, 1, 5, 20, 50, 200])
    _by_covariate("eU (ppm)", eU(U, Th), rel, [0, 30, 60, 100, 300])
    _by_covariate("radius (um)", rad, rel, [30, 50, 70, 100])

    print("  worst offenders:")
    for i in np.argsort(-rel)[:5]:
        print(f"    ref={ref[i]:8.3f}  got={age[i]:8.3f}  rel={rel[i]:.2e}  "
              f"r={rad[i]:5.1f}  eU={eU(U[i], Th[i]):6.1f}  Tmax={T[i].max():5.1f}")
    return ref, age


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--n-sub", type=int, default=64)
    ap.add_argument("--rdim", type=int, default=512)
    ap.add_argument("--precision", type=int, default=2, choices=(0, 1, 2))
    ap.add_argument("--oracle-refine", type=int, default=1,
                    help="pre-subdivide the path handed to the oracle, to refine its grid")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--procs", type=int, default=32)
    a = ap.parse_args()
    main(a.n, a.n_sub, a.rdim, a.precision, a.oracle_refine, a.device, a.procs)
