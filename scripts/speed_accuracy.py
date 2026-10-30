"""Speed and accuracy of every forward-model implementation, on one workload. (PLAN.md Stage 7; docs/speed-accuracy.md)

``benchmark_forward.py`` and the two ``validate_*.py`` scripts each measured one
half of the question on their own synthetic paths -- 6 or 10 random control
points, re-subdivided -- and at settings chosen for that one question.  This
measures both halves at once, for every implementation, on the workload the
project actually runs: 64-node uniform paths drawn from the shipped prior, with
grains from the shipped grain prior.  The same paths are timed and scored.

Accuracy is always against a **same-model reference** that is converged past
the thing being measured:

* AFT: ``KETCH.c`` at ``pctPerTimeStep = 0.02`` (50x its default).  The
  reference's own floor is reported as ``pct = 0.02`` against ``pct = 0.01``.
* RDAAM: ``RDAAM.cpp`` at BEST, fed the path pre-subdivided 8x, which forces
  its interpolator onto ~2 500 time nodes.  Floor: 8x against 16x.
* Mad_He: it has no port and a different model, so its row is its own time
  convergence (path refined 8x); its fixed -1.5% radial quadrature floor is
  measured in ``docs/validation.md`` and quoted, not re-measured.

Throughput is measured, not scaled: the oracles on one core and across a
process pool of ``--procs`` workers, the ports at a stated batch on a stated
device.  Load average is recorded with every block, because the machine is
shared.

Usage:
    uv run python scripts/speed_accuracy.py [--device cuda:1] [--procs 32]
                                            [--threads 16] [--n-acc 1000]
                                            [--out results/speed_accuracy]
    uv run python scripts/speed_accuracy.py --mode report   # print the tables
"""
from __future__ import annotations

import os

# The oracles and the CPU port are timed at a stated thread count; nothing may
# quietly fan out behind them.  Must precede the numpy/torch imports.
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import ctypes
import json
import multiprocessing as mp
import platform
import time
from pathlib import Path

import numpy as np
import torch

from chai.data.priors import GrainPrior, ThermalHistoryPrior
from chai.forward.native import (
    PREC_BEST,
    calculate_aft,
    calculate_madhe,
    calculate_rdaam,
)
from chai.forward.native.ketcham import set_time_step_pct
from chai.forward.torchimpl.ketcham import KETCHAM_2007, aft_age, aft_forward
from chai.forward.torchimpl.rdaam import rdaam_age
from chai.paths import fixed_grid, refine

N_GRID, T_MAX = 64, 100.0
BIN_WIDTH = 0.1          # um; the oracle's 200-bin length axis
AGE_FLOOR = 0.1          # Ma; relative error is meaningless below it
SEED = 2026

GRID = fixed_grid(N_GRID, T_MAX, "uniform").numpy()


def load() -> float:
    return os.getloadavg()[0]


def workload(n: int, seed: int = SEED) -> dict[str, np.ndarray]:
    """``n`` prior histories on the canonical grid, one grain each."""
    rng = np.random.default_rng(seed)
    T, _ = ThermalHistoryPrior().sample_on(n, rng, GRID)
    g = GrainPrior().sample(n, rng)
    return {"T": T, **g}


def _refined(T: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    if k == 1:
        return GRID, T
    tf, Tf = refine(torch.as_tensor(GRID[None]), torch.as_tensor(T[None]), k)
    return tf[0].numpy(), Tf[0].numpy()


# --------------------------------------------------------------------------
# Oracle jobs.  Module level so a process pool can pickle them.
# --------------------------------------------------------------------------

def _aft_job(a):
    T, dpar, pct = a
    set_time_step_pct(pct)
    r = calculate_aft(GRID, T, kinetic_par=float(dpar))
    return r.age, r.mean_length, r.length_pdf


def _rdaam_job(a):
    T, rad, U, Th, k = a
    t, TT = _refined(T, k)
    r = calculate_rdaam(t, TT, float(rad), float(U), float(Th), precision=PREC_BEST)
    return r.age, r.corrected_age


def _madhe_job(a):
    T, rad, k = a
    t, TT = _refined(T, k)
    return calculate_madhe(t, TT, grain_size=float(rad))


_POOL_JOBS: list = []        # inherited by forked workers; tasks carry only indices


def _block(a):
    """A block of histories in one task, as a corpus worker runs one: the inputs
    are already in the worker and only a count comes back, so the pool is timed
    on compute and not on pickling."""
    fn, start, count = a
    for i in range(start, start + count):
        fn(_POOL_JOBS[i % len(_POOL_JOBS)])
    return count


def warm_malloc() -> None:
    """Raise glibc's mmap threshold, as a corpus worker's own prior draw does.

    ``KETCH.c`` allocates its work arrays on every call.  In a process that has
    never freed a large block, glibc serves them with ``mmap`` and the call pays
    for fresh pages each time -- 2.1x slower (4 800 against 10 200 paths/s on one
    core, measured 2026-09-22).  Any process that has drawn a block of prior
    histories has already freed a large array and raised the threshold, so the
    corpus build never paid it; a fresh pool worker does.  Timing every process
    in the same state keeps the one-core and pool columns comparable.
    """
    try:
        ctypes.CDLL("libc.so.6").mallopt(-3, 64 << 20)       # M_MMAP_THRESHOLD
    except OSError:
        pass


def _pool_rate(fn, jobs, procs, per_task):
    """Histories/s across ``procs`` workers, two blocks of ``per_task`` each."""
    _POOL_JOBS[:] = jobs
    blocks = [(fn, b * per_task, per_task) for b in range(2 * procs)]
    with mp.get_context("fork").Pool(procs, initializer=warm_malloc) as pool:
        pool.map(_block, [(fn, 0, 2)] * procs, chunksize=1)      # every worker up
        t0 = time.perf_counter()
        done = sum(pool.map(_block, blocks, chunksize=1))
        return done / (time.perf_counter() - t0)


def _pool_map(fn, jobs, procs, chunksize=8):
    with mp.Pool(procs) as pool:
        t0 = time.perf_counter()
        out = pool.map(fn, jobs, chunksize=chunksize)
        return out, time.perf_counter() - t0


def _serial(fn, jobs, repeats=2):
    best = float("inf")
    for _ in range(repeats):
        t0 = time.perf_counter()
        for j in jobs:
            fn(j)
        best = min(best, time.perf_counter() - t0)
    return best


# --------------------------------------------------------------------------
# Error summaries
# --------------------------------------------------------------------------

def _q(x, signed=None) -> dict[str, float]:
    """Quantiles of an error magnitude, and the median of its signed form.

    The signed median says whether an error is a bias or scatter -- the
    difference matters to a likelihood, which absorbs scatter into its noise
    model and cannot absorb a bias.
    """
    x = np.asarray(x, dtype=float)
    out = {"median": float(np.median(x)), "p99": float(np.percentile(x, 99)),
           "max": float(x.max()), "n": int(x.size)}
    if signed is not None:
        out["bias"] = float(np.median(signed))
    return out


def aft_errors(age, mean, pdf, ref) -> dict:
    r_age, r_mean, r_pdf = ref
    ok = r_age > AGE_FLOOR
    live = r_pdf.max(axis=1) > 0
    return {
        "age_rel": _q(np.abs(age - r_age)[ok] / r_age[ok], ((age - r_age) / r_age)[ok]),
        **({} if mean is None else {
            "mean_length_um": _q(np.abs(mean - r_mean)[live]),
            "pdf_l1": _q(np.abs(pdf - r_pdf).sum(axis=1)[live] * BIN_WIDTH),
        }),
        "nan": int(np.isnan(age).sum()),
    }


def he_errors(corr, ref_corr) -> dict:
    ok = ref_corr > AGE_FLOOR
    return {"age_rel": _q(np.abs(corr - ref_corr)[ok] / ref_corr[ok],
                          ((corr - ref_corr) / ref_corr)[ok]),
            "nan": int(np.isnan(corr).sum())}


# --------------------------------------------------------------------------
# Timing the ports
# --------------------------------------------------------------------------

def _sync(dev):
    if dev.type == "cuda":
        torch.cuda.synchronize(dev)


def _time(fn, dev, repeats=3):
    fn()                                    # warm-up: allocator, kernels, caches
    best = float("inf")
    for _ in range(repeats):
        _sync(dev)
        t0 = time.perf_counter()
        fn()
        _sync(dev)
        best = min(best, time.perf_counter() - t0)
    return best


def _tile(x: np.ndarray, b: int) -> np.ndarray:
    reps = -(-b // len(x))
    return np.concatenate([x] * reps)[:b]


def _tensors(w, idx_or_b, dev, dtype, keys):
    if isinstance(idx_or_b, int):
        arrs = [_tile(w[k], idx_or_b) for k in keys]
    else:
        arrs = [w[k][idx_or_b] for k in keys]
    return [torch.as_tensor(a, dtype=dtype, device=dev) for a in arrs]


def _t_grid(n, dev, dtype):
    return torch.as_tensor(GRID, dtype=dtype, device=dev).expand(n, -1).contiguous()


def run_aft_port(w, ref, dev, dtype, n_sub, batch, age_only=False, chunk=4096):
    fn = aft_age if age_only else aft_forward
    # accuracy, on the accuracy set
    n = len(ref[0])
    ages, means, pdfs = [], [], []
    with torch.no_grad():
        for lo in range(0, n, chunk):
            T, d = _tensors(w, slice(lo, min(n, lo + chunk)), dev, dtype, ("T", "dpar"))
            out = fn(_t_grid(len(T), dev, dtype), T, d, n_sub=n_sub, model=KETCHAM_2007)
            if age_only:
                ages.append(out.double().cpu().numpy())
            else:
                ages.append(out["age"].double().cpu().numpy())
                means.append(out["mean_length"].double().cpu().numpy())
                pdfs.append(out["length_pdf"].double().cpu().numpy())
    got = (np.concatenate(ages), None if age_only else np.concatenate(means),
           None if age_only else np.concatenate(pdfs))
    err = aft_errors(*got, ref)
    # throughput, at the stated batch
    T, d = _tensors(w, batch, dev, dtype, ("T", "dpar"))
    t = _t_grid(batch, dev, dtype)
    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats(dev)
    with torch.no_grad():
        secs = _time(lambda: fn(t, T, d, n_sub=n_sub, model=KETCHAM_2007), dev)
    return err, batch / secs, _peak(dev), got


def run_rdaam_port(w, ref_corr, dev, dtype, n_sub, rdim, batch, n_acc=None, chunk=1024):
    n = len(ref_corr) if n_acc is None else n_acc
    corr = []
    keys = ("T", "radius", "U", "Th")
    with torch.no_grad():
        for lo in range(0, n, chunk):
            T, r, U, Th = _tensors(w, slice(lo, min(n, lo + chunk)), dev, dtype, keys)
            out = rdaam_age(_t_grid(len(T), dev, dtype), T, r, U, Th,
                            n_sub=n_sub, rdim=rdim, use_checkpoint=False)
            corr.append(out["corrected_age"].double().cpu().numpy())
    got = np.concatenate(corr)
    err = he_errors(got, ref_corr[:n])
    T, r, U, Th = _tensors(w, batch, dev, dtype, keys)
    t = _t_grid(batch, dev, dtype)
    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats(dev)
    with torch.no_grad():
        secs = _time(lambda: rdaam_age(t, T, r, U, Th, n_sub=n_sub, rdim=rdim,
                                       use_checkpoint=False), dev, repeats=2)
    return err, batch / secs, _peak(dev), got


def _peak(dev):
    if dev.type != "cuda":
        return None
    return torch.cuda.max_memory_allocated(dev) / 2 ** 30


def grad_aft(w, dev, dtype, n_sub, batch):
    T, d = _tensors(w, batch, dev, dtype, ("T", "dpar"))
    t = _t_grid(batch, dev, dtype)
    T.requires_grad_(True)

    def fb():
        out = aft_forward(t, T, d, n_sub=n_sub, model=KETCHAM_2007)
        (out["age"].sum() + out["mean_length"].sum()).backward()
        T.grad = None

    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats(dev)
    return batch / _time(fb, dev, repeats=2), _peak(dev)


def grad_rdaam(w, dev, dtype, n_sub, rdim, batch):
    T, r, U, Th = _tensors(w, batch, dev, dtype, ("T", "radius", "U", "Th"))
    t = _t_grid(batch, dev, dtype)
    T.requires_grad_(True)

    def fb():
        out = rdaam_age(t, T, r, U, Th, n_sub=n_sub, rdim=rdim, use_checkpoint=True)
        out["corrected_age"].sum().backward()
        T.grad = None

    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats(dev)
    return batch / _time(fb, dev, repeats=2), _peak(dev)


# --------------------------------------------------------------------------
# The study
# --------------------------------------------------------------------------

AFT_PORTS = [  # (device kind, dtype, n_sub, age_only)
    ("cuda", "float32", 4, False), ("cuda", "float32", 8, False),
    ("cuda", "float32", 16, False), ("cuda", "float32", 32, False),
    ("cuda", "float64", 8, False), ("cuda", "float64", 32, False),
    ("cuda", "float32", 8, True),
    ("cpu", "float32", 8, False), ("cpu", "float64", 8, False),
]
RDAAM_PORTS = [  # (device kind, dtype, n_sub, rdim)
    ("cuda", "float32", 8, 128), ("cuda", "float32", 8, 256), ("cuda", "float32", 8, 512),
    ("cuda", "float32", 4, 128), ("cuda", "float32", 16, 128),
    ("cuda", "float64", 8, 128), ("cuda", "float64", 8, 512),
    ("cpu", "float32", 8, 128), ("cpu", "float64", 8, 128),
]


def study(args) -> dict:
    warm_malloc()
    torch.set_num_threads(args.threads)
    gpu = torch.device(args.device)
    cpu = torch.device("cpu")
    w = workload(args.n_acc)
    n = args.n_acc
    rows: list[dict] = []
    meta = {
        "date": time.strftime("%Y-%m-%d"), "host": platform.node(),
        "cores": os.cpu_count(), "procs": args.procs, "threads": args.threads,
        "torch": torch.__version__, "n_acc": n, "seed": SEED,
    }
    print(json.dumps(meta))

    def emit(row):
        row["load"] = round(load(), 1)
        rows.append(row)
        print(json.dumps(row), flush=True)

    # ---- references --------------------------------------------------------
    jobs = [(w["T"][i], w["dpar"][i]) for i in range(n)]
    res, _ = _pool_map(_aft_job, [j + (0.02,) for j in jobs], args.procs)
    aft_ref = tuple(np.array([r[k] for r in res]) for k in range(3))
    res, _ = _pool_map(_aft_job, [j + (0.01,) for j in jobs], args.procs)
    finer = tuple(np.array([r[k] for r in res]) for k in range(3))
    emit({"system": "AFT", "impl": "reference floor", "setting": "pct 0.02 vs 0.01",
          "accuracy": aft_errors(*aft_ref, finer)})

    hjobs = [(w["T"][i], w["radius"][i], w["U"][i], w["Th"][i]) for i in range(n)]
    res, _ = _pool_map(_rdaam_job, [j + (8,) for j in hjobs], args.procs, chunksize=2)
    he_ref = np.array([r[1] for r in res])
    n_floor = min(n, args.n_floor)
    res, _ = _pool_map(_rdaam_job, [j + (16,) for j in hjobs[:n_floor]], args.procs,
                       chunksize=1)
    emit({"system": "AHe (RDAAM)", "impl": "reference floor",
          "setting": "BEST, path x8 vs x16",
          "accuracy": he_errors(he_ref[:n_floor], np.array([r[1] for r in res]))})

    # ---- oracles at production settings ------------------------------------
    # The pool is timed over enough work (``--pool-work`` per worker) that
    # start-up and pickling are amortised the way they are in the corpus build.
    n_ser = min(n, args.n_serial)
    per_task = args.pool_work       # histories per block; two blocks per worker
    for pct in (1.0, 0.2):
        res, _ = _pool_map(_aft_job, [j + (pct,) for j in jobs], args.procs)
        o = tuple(np.array([r[k] for r in res]) for k in range(3))
        ser = _serial(_aft_job, [j + (pct,) for j in jobs[:n_ser]])
        pool_rate = _pool_rate(_aft_job, [j + (pct,) for j in jobs], args.procs,
                               round(per_task * pct))
        emit({"system": "AFT", "impl": "KETCH.c", "setting": f"pct {pct}",
              "nodes": round(100 / pct), "device": "cpu", "dtype": "float32 in, float64",
              "rate_1core": n_ser / ser, "rate_pool": pool_rate,
              "accuracy": aft_errors(*o, aft_ref)})

    res, _ = _pool_map(_rdaam_job, [j + (1,) for j in hjobs], args.procs, chunksize=2)
    n_he = n_ser // 2              # per-grain cost varies with the path; not too few
    ser = _serial(_rdaam_job, [j + (1,) for j in hjobs[:n_he]], repeats=2)
    pool_rate = _pool_rate(_rdaam_job, [j + (1,) for j in hjobs], args.procs,
                           per_task // 80)
    emit({"system": "AHe (RDAAM)", "impl": "RDAAM.cpp", "setting": "BEST (corpus)",
          "device": "cpu", "dtype": "float64",
          "rate_1core": n_he / ser, "rate_pool": pool_rate,
          "accuracy": he_errors(np.array([r[1] for r in res]), he_ref)})

    mjobs = [(w["T"][i], w["radius"][i]) for i in range(n)]
    base, _ = _pool_map(_madhe_job, [j + (1,) for j in mjobs], args.procs)
    fine, _ = _pool_map(_madhe_job, [j + (8,) for j in mjobs], args.procs)
    ser = _serial(_madhe_job, [j + (1,) for j in mjobs[:n_ser]])
    pool_rate = _pool_rate(_madhe_job, [j + (1,) for j in mjobs], args.procs,
                           per_task // 8)
    emit({"system": "AHe (Mad_He)", "impl": "Mad_He.f90", "setting": "default, 100 radial",
          "device": "cpu", "dtype": "float32",
          "rate_1core": n_ser / ser, "rate_pool": pool_rate,
          "accuracy": he_errors(np.array(base), np.array(fine)),
          "note": "vs itself on the path refined 8x; -1.5% radial floor in docs/validation.md"})

    # ---- how the oracles scale across the machine ---------------------------
    # One core runs at single-core turbo; a full pool does not, so the pool is
    # measured at several sizes rather than extrapolated from one core.
    for p in args.pool_sweep:
        emit({"system": "AFT", "impl": "KETCH.c, pool sweep", "setting": "pct 1.0",
              "procs": p, "rate": _pool_rate(_aft_job, [j + (1.0,) for j in jobs], p,
                                            per_task)})
        emit({"system": "AHe (RDAAM)", "impl": "RDAAM.cpp, pool sweep",
              "setting": "BEST (corpus)", "procs": p,
              "rate": _pool_rate(_rdaam_job, [j + (1,) for j in hjobs], p, per_task // 80)})

    if args.keep_ports:
        # The machine is shared: when the GPUs are busy, the oracle half can be
        # re-measured on its own and the port rows carried over from the last
        # full run, each with the load it was measured under.
        old = json.loads((args.out / "table.json").read_text())
        rows += [r for r in old["rows"]
                 if r["impl"].startswith(("torch", "float32 vs"))]
        meta["ports_from"] = old["meta"]["date"] + " (an earlier full run)"
        meta["gpu"] = old["meta"]["gpu"]
        return {"meta": meta, "rows": rows}

    # ---- the ports ---------------------------------------------------------
    kept = {}
    for kind, dt, n_sub, age_only in AFT_PORTS:
        dev = gpu if kind == "cuda" else cpu
        batch = args.aft_batch if kind == "cuda" else args.cpu_batch
        err, rate, peak, got = run_aft_port(w, aft_ref, dev, getattr(torch, dt), n_sub,
                                            batch, age_only=age_only)
        kept[("aft", kind, dt, n_sub, age_only)] = got
        emit({"system": "AFT", "impl": "torch" + (" (age only)" if age_only else ""),
              "setting": f"n_sub {n_sub}", "nodes": (N_GRID - 1) * n_sub + 1,
              "device": kind, "dtype": dt, "batch": batch, "rate": rate,
              "peak_gib": peak, "accuracy": err})
        if kind == "cuda":
            torch.cuda.empty_cache()

    for kind, dt, n_sub, rdim in RDAAM_PORTS:
        dev = gpu if kind == "cuda" else cpu
        batch = args.rdaam_batch if kind == "cuda" else args.cpu_batch // 8
        err, rate, peak, got = run_rdaam_port(w, he_ref, dev, getattr(torch, dt), n_sub,
                                              rdim, batch,
                                              n_acc=None if kind == "cuda" else 128)
        kept[("he", kind, dt, n_sub, rdim)] = got
        emit({"system": "AHe (RDAAM)", "impl": "torch",
              "setting": f"n_sub {n_sub}, rdim {rdim}", "nodes": (N_GRID - 1) * n_sub + 1,
              "device": kind, "dtype": dt, "batch": batch, "rate": rate,
              "peak_gib": peak, "accuracy": err})
        if kind == "cuda":
            torch.cuda.empty_cache()

    # ---- float32 against float64, same device and refinement ---------------
    for n_sub in (8, 32):
        a32, a64 = kept[("aft", "cuda", "float32", n_sub, False)], \
            kept[("aft", "cuda", "float64", n_sub, False)]
        emit({"system": "AFT", "impl": "float32 vs float64", "setting": f"n_sub {n_sub}",
              "accuracy": aft_errors(*a32, a64)})
    for rdim in (128, 512):
        emit({"system": "AHe (RDAAM)", "impl": "float32 vs float64",
              "setting": f"n_sub 8, rdim {rdim}",
              "accuracy": he_errors(kept[("he", "cuda", "float32", 8, rdim)],
                                    kept[("he", "cuda", "float64", 8, rdim)])})

    # ---- batch size: where the ports stop being launch-bound ---------------
    # Timing only, at the production settings.  Both ports are a sequential
    # loop over ~500 internal nodes, so a small batch pays the whole loop.
    for age_only in (False, True):
        fn = aft_age if age_only else aft_forward
        for b in args.sweep_aft:
            T, d = _tensors(w, b, gpu, torch.float32, ("T", "dpar"))
            t = _t_grid(b, gpu, torch.float32)
            torch.cuda.reset_peak_memory_stats(gpu)
            with torch.no_grad():
                secs = _time(lambda t=t, T=T, d=d, fn=fn: fn(t, T, d, n_sub=8,
                                                             model=KETCHAM_2007), gpu)
            emit({"system": "AFT",
                  "impl": "torch" + (" (age only)" if age_only else "") + ", batch sweep",
                  "setting": "n_sub 8", "device": "cuda", "dtype": "float32", "batch": b,
                  "rate": b / secs, "latency_s": secs, "peak_gib": _peak(gpu)})
            torch.cuda.empty_cache()
    for rdim in (128, 512):
        for b in args.sweep_rdaam:
            T, r, U, Th = _tensors(w, b, gpu, torch.float32, ("T", "radius", "U", "Th"))
            t = _t_grid(b, gpu, torch.float32)
            torch.cuda.reset_peak_memory_stats(gpu)
            with torch.no_grad():
                secs = _time(lambda t=t, T=T, r=r, U=U, Th=Th, rdim=rdim:
                             rdaam_age(t, T, r, U, Th, n_sub=8, rdim=rdim,
                                       use_checkpoint=False), gpu, repeats=2)
            emit({"system": "AHe (RDAAM)", "impl": "torch, batch sweep",
                  "setting": f"n_sub 8, rdim {rdim}", "device": "cuda", "dtype": "float32",
                  "batch": b, "rate": b / secs, "latency_s": secs, "peak_gib": _peak(gpu)})
            torch.cuda.empty_cache()

    # ---- gradients ---------------------------------------------------------
    for dt in ("float32", "float64"):
        b = args.grad_batch if dt == "float32" else args.grad_batch // 2
        rate, peak = grad_aft(w, gpu, getattr(torch, dt), 8, b)
        emit({"system": "AFT", "impl": "torch fwd+bwd", "setting": "n_sub 8",
              "device": "cuda", "dtype": dt, "batch": b, "rate": rate,
              "peak_gib": peak, "what": "d(age + mean length)/dT"})
        torch.cuda.empty_cache()
        # The damage sweep is not checkpointed: it stores O(B * N^2) activations,
        # 16 GiB at B = 1024 and N = 505.  Its share, not the diffusion's, sets
        # the batch here.
        b = args.grad_batch_rdaam if dt == "float32" else args.grad_batch_rdaam // 2
        rate, peak = grad_rdaam(w, gpu, getattr(torch, dt), 8, 128, b)
        emit({"system": "AHe (RDAAM)", "impl": "torch fwd+bwd", "setting": "n_sub 8, rdim 128",
              "device": "cuda", "dtype": dt, "batch": b, "rate": rate,
              "peak_gib": peak, "what": "d(corrected age)/dT, checkpointed"})
        torch.cuda.empty_cache()

    # Asked only now: initialising CUDA before the oracle pools fork is asking for trouble.
    meta["gpu"] = torch.cuda.get_device_name(gpu) if gpu.type == "cuda" else None
    return {"meta": meta, "rows": rows}


# --------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------

def _fmt_rate(x):
    return "—" if x is None else f"{x:,.0f}"


def _fmt_err(row, key="age_rel"):
    a = row.get("accuracy", {}).get(key)
    if a is None:
        return "—"
    b = f" ({a['bias']:+.1e})" if "bias" in a else ""
    return f"{a['median']:.1e} / {a['p99']:.1e} / {a['max']:.1e}{b}"


def report(data: dict) -> None:
    m = data["meta"]
    print(f"{m['date']}  {m['host']}  {m['cores']} cores, pool {m['procs']}, "
          f"torch threads {m['threads']}, {m['gpu']}, n_acc {m['n_acc']}")
    for system in ("AFT", "AHe (RDAAM)", "AHe (Mad_He)"):
        print(f"\n### {system}\n")
        extra = system == "AFT"
        print("| Implementation | Setting | Device, dtype | Batch | Samples/s | "
              "Age rel. err, median / p99 / max (signed median) |"
              + (" Mean length, µm (median / p99) |" if extra else "") + " Load |")
        print("|---|---|---|---|---|---|" + ("---|" if extra else "") + "---|")
        for r in data["rows"]:
            if r["system"] != system:
                continue
            if "rate_1core" in r:
                rate = f"{_fmt_rate(r['rate_1core'])} (1 core); {_fmt_rate(r['rate_pool'])} ({m['procs']} procs)"
                dev = f"CPU, {r['dtype']}"
                batch = "1"
            elif "procs" in r:
                rate = f"{_fmt_rate(r['rate'])} ({r['procs']} procs)"
                dev, batch = "CPU", "1"
            elif "rate" in r:
                rate = _fmt_rate(r["rate"])
                dev = ("GPU" if r["device"] == "cuda" else f"CPU {m['threads']} threads") + f", {r['dtype']}"
                if r.get("peak_gib"):
                    dev += f", {r['peak_gib']:.1f} GiB"
                batch = f"{r['batch']:,}"
                if "latency_s" in r:
                    batch += f" ({r['latency_s']:.2f} s/call)"
            else:
                rate, dev, batch = "—", "—", "—"
            ml = ""
            if extra:
                a = r.get("accuracy", {}).get("mean_length_um")
                ml = " " + ("—" if a is None else f"{a['median']:.1e} / {a['p99']:.1e}") + " |"
            print(f"| {r['impl']} | {r['setting']} | {dev} | {batch} | {rate} | "
                  f"{_fmt_err(r)} |{ml} {r['load']} |")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", choices=("run", "report"), default="run")
    p.add_argument("--device", default="cuda:1" if torch.cuda.device_count() > 1 else "cuda")
    p.add_argument("--procs", type=int, default=32, help="oracle process-pool size")
    p.add_argument("--threads", type=int, default=16, help="torch CPU threads")
    p.add_argument("--n-acc", type=int, default=1000, help="histories scored")
    p.add_argument("--n-floor", type=int, default=250,
                   help="histories for the RDAAM reference's own floor (x16 is slow)")
    p.add_argument("--n-serial", type=int, default=400, help="histories timed on one core")
    p.add_argument("--pool-work", type=int, default=10000,
                   help="AFT histories per pool block (about 1 s); RDAAM and Mad_He "
                        "blocks are scaled to match")
    p.add_argument("--keep-ports", action="store_true",
                   help="re-measure only the oracles; carry the port rows over from "
                        "the existing table.json")
    p.add_argument("--pool-sweep", type=int, nargs="*", default=[1, 8, 16, 32, 56])
    p.add_argument("--sweep-aft", type=int, nargs="+", default=[1, 64, 1024, 16384, 65536, 131072])
    p.add_argument("--sweep-rdaam", type=int, nargs="+", default=[1, 64, 1024, 4096, 8192, 16384, 32768])
    p.add_argument("--aft-batch", type=int, default=16384)
    p.add_argument("--rdaam-batch", type=int, default=16384)
    p.add_argument("--cpu-batch", type=int, default=4096)
    p.add_argument("--grad-batch", type=int, default=4096)
    p.add_argument("--grad-batch-rdaam", type=int, default=512)
    p.add_argument("--out", type=Path, default=Path("results/speed_accuracy"))
    args = p.parse_args()

    path = args.out / "table.json"
    if args.mode == "run":
        data = study(args)
        args.out.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=1))
        print(f"wrote {path}")
    report(json.loads(path.read_text()))


if __name__ == "__main__":
    main()
