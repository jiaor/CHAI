"""Throughput of the PyTorch ports against their native oracles.

Accuracy is settled elsewhere (``validate_ketcham.py``, ``validate_rdaam.py``).
This script answers the other half: what does each implementation cost, and at
what batch size does the GPU port overtake a machine's worth of oracle cores?

Both sides are reported per thermal-history node, because the two codes do not
discretise the same way.  The oracle picks its own internal grid -- ``pct`` for
AFT, ``precision`` plus the interpolator's five-steps-per-segment floor for
RDAAM -- while the port is told its grid via ``n_sub``.  Comparing raw
wall-clock at whatever grid each happens to land on flatters whichever is
coarser, so the rightmost column is the one to read.

The oracle is timed on one core and scaled by ``--cores``.  That is generous to
it: it is embarrassingly parallel across samples, but perfect scaling ignores
memory-bandwidth contention, and the AFT oracle in particular is called through
ctypes, which the port is not.

Usage:
    uv run python scripts/benchmark_forward.py [--model aft|rdaam|both]
                                               [--cores 112] [--quick]
"""
from __future__ import annotations

import argparse
import pathlib
import sys
import time

import torch

from chai.forward.native import calculate_aft, calculate_rdaam, calculate_rdaam_debug
from chai.forward.native.ketcham import set_time_step_pct
from chai.forward.torchimpl.ketcham import KETCHAM_2007, aft_forward
from chai.forward.torchimpl.rdaam import rdaam_age

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from validate_ketcham import sample_paths
from validate_rdaam import sample_grains

AFT_CTRL, RDAAM_CTRL = 10, 6
DTYPE = torch.float64  # set from --dtype in main()


def _time(fn, repeats=3, warmup=1, cuda=False):
    """Best-of-`repeats` wall-clock in seconds, excluding `warmup` runs."""
    for _ in range(warmup):
        fn()
    best = float("inf")
    for _ in range(repeats):
        if cuda:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        if cuda:
            torch.cuda.synchronize()
        best = min(best, time.perf_counter() - t0)
    return best


def _row(label, n, secs, nodes):
    rate = n / secs
    print(f"    {label:<30s} {secs:8.3f} s {rate:12,.1f} /s {rate * nodes / 1e6:9.3f} Mnode/s")


def _header(what):
    print(f"  {what}")


def bench_aft(args):
    n_big = max(args.sizes)
    t, T, dpar = sample_paths(max(n_big, args.oracle_n))
    print(f"\n=== AFT  (Ketcham 2007 published, {AFT_CTRL} control points) ===")

    set_time_step_pct(args.pct)
    o_nodes = round(100.0 / args.pct)
    n = args.oracle_n
    secs = _time(lambda: [calculate_aft(t[i], T[i], dpar[i]) for i in range(n)],
                 repeats=2, warmup=0)
    _header(f"KETCH.c, 1 core  (pct={args.pct} -> ~{o_nodes} internal steps)")
    _row("serial loop", n, secs, o_nodes)
    o_rate = n / secs
    print(f"    -> {o_rate * args.cores:,.0f} paths/s across {args.cores} cores"
          f"  ({o_rate * args.cores * o_nodes / 1e6:.3f} Mnode/s)")

    for dev in args.devices:
        cuda = dev == "cuda"
        base = [torch.as_tensor(x[:n_big], dtype=DTYPE, device=dev)
                for x in (t, T, dpar)]
        for n_sub in args.n_sub_aft:
            nodes = (AFT_CTRL - 1) * n_sub + 1
            _header(f"torch {dev}  (n_sub={n_sub} -> {nodes} nodes)")
            for b in args.sizes:
                a = [x[:b] for x in base]
                with torch.no_grad():
                    secs = _time(lambda a=a, ns=n_sub:
                                 aft_forward(*a, n_sub=ns, model=KETCHAM_2007),
                                 cuda=cuda)
                _row(f"batch {b}", b, secs, nodes)
                if b == args.sizes[-1]:
                    print(f"    {'':30s} {'':8s} {'':12s}    "
                          f"= {(b / secs) / (o_rate * args.cores):.3f}x the "
                          f"{args.cores}-core oracle (per path)")

    # Gradients.  The oracle has no analogue: the only route to a Jacobian
    # through it is finite differences, 2*(2*AFT_CTRL+1) = 42 forward evals per
    # path for a central-difference gradient w.r.t. the path and Dpar.
    dev = args.devices[-1]
    cuda = dev == "cuda"
    b, n_sub = args.grad_batch, args.n_sub_aft[-1]
    nodes = (AFT_CTRL - 1) * n_sub + 1
    tt, TT, dd = (torch.as_tensor(x[:b], dtype=DTYPE, device=dev)
                  for x in (t, T, dpar))
    TT.requires_grad_(True)
    dd.requires_grad_(True)

    def fwd_bwd():
        out = aft_forward(tt, TT, dd, n_sub=n_sub, model=KETCHAM_2007)
        (out["age"].sum() + out["length_pdf"].sum()).backward()
        TT.grad = dd.grad = None

    secs = _time(fwd_bwd, cuda=cuda)
    _header(f"torch {dev} forward+backward  (batch {b}, {nodes} nodes)")
    _row("d(age, pdf)/d(T, Dpar)", b, secs, nodes)
    fd = 2 * (AFT_CTRL + 1)
    print(f"    finite differences through the oracle would need {fd} forward evals"
          f" -> {fd / (o_rate * args.cores) * b:,.3f} s for the same batch")


def bench_rdaam(args):
    sizes = [s for s in args.sizes if s <= args.rdaam_max]
    n_big = max(sizes)
    t, T, rad, U, Th = sample_grains(max(n_big, args.oracle_n_rdaam))
    print(f"\n=== RDAAM  ({RDAAM_CTRL} control points) ===")

    d = calculate_rdaam_debug(t[0], T[0], rad[0], U[0], Th[0], precision=args.precision)
    o_nodes = d.n_nodes
    n = args.oracle_n_rdaam
    secs = _time(lambda: [calculate_rdaam(t[i], T[i], rad[i], U[i], Th[i],
                                          precision=args.precision)
                          for i in range(n)], repeats=2, warmup=0)
    _header(f"RDAAM.cpp, 1 core  (precision={args.precision} -> ~{o_nodes} nodes)")
    _row("serial loop", n, secs, o_nodes)
    o_rate = n / secs
    print(f"    -> {o_rate * args.cores:,.0f} grains/s across {args.cores} cores"
          f"  ({o_rate * args.cores * o_nodes / 1e6:.3f} Mnode/s)")

    nodes = (RDAAM_CTRL - 1) * args.n_sub_rdaam + 1
    for dev in args.devices:
        cuda = dev == "cuda"
        base = [torch.as_tensor(x[:n_big], dtype=DTYPE, device=dev)
                for x in (t, T, rad, U, Th)]
        _header(f"torch {dev}  (n_sub={args.n_sub_rdaam} -> {nodes} nodes, "
                f"rdim={args.rdim})")
        for b in sizes:
            a = [x[:b] for x in base]
            with torch.no_grad():
                secs = _time(lambda a=a: rdaam_age(*a, n_sub=args.n_sub_rdaam,
                                                   rdim=args.rdim,
                                                   use_checkpoint=False), cuda=cuda)
            _row(f"batch {b}", b, secs, nodes)
            if b == sizes[-1]:
                print(f"    {'':30s} {'':8s} {'':12s}    "
                      f"= {(b / secs) / (o_rate * args.cores):.3f}x the "
                      f"{args.cores}-core oracle (per grain)")

    dev = args.devices[-1]
    cuda = dev == "cuda"
    b = min(args.grad_batch, sizes[-1])
    tt, TT, rr, uu, hh = (torch.as_tensor(x[:b], dtype=DTYPE, device=dev)
                          for x in (t, T, rad, U, Th))
    TT.requires_grad_(True)
    for ckpt in (False, True):
        def fwd_bwd(c=ckpt):
            out = rdaam_age(tt, TT, rr, uu, hh, n_sub=args.n_sub_rdaam,
                            rdim=args.rdim, use_checkpoint=c)
            out["corrected_age"].sum().backward()
            TT.grad = None
        try:
            secs = _time(fwd_bwd, repeats=2, cuda=cuda)
        except torch.cuda.OutOfMemoryError:
            print(f"    {'checkpoint=' + str(ckpt):<30s}   out of memory")
            torch.cuda.empty_cache()
            continue
        _header(f"torch {dev} forward+backward, checkpoint={ckpt}  (batch {b})")
        _row("d(corrected_age)/dT", b, secs, nodes)
        if cuda:
            print(f"    {'':30s} peak {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB")
            torch.cuda.reset_peak_memory_stats()


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", choices=("aft", "rdaam", "both"), default="both")
    p.add_argument("--cores", type=int, default=112,
                   help="cores to credit the oracle with when scaling it up")
    p.add_argument("--pct", type=float, default=1.0, help="AFT oracle time step (%%)")
    p.add_argument("--precision", type=int, default=2, help="RDAAM oracle precision")
    p.add_argument("--oracle-n", type=int, default=500)
    p.add_argument("--oracle-n-rdaam", type=int, default=30)
    p.add_argument("--rdaam-max", type=int, default=4096,
                   help="largest RDAAM batch (memory-bound)")
    p.add_argument("--n-sub-aft", type=int, nargs="+", default=[12, 128])
    p.add_argument("--n-sub-rdaam", type=int, default=64)
    p.add_argument("--rdim", type=int, default=512)
    p.add_argument("--sizes", type=int, nargs="+",
                   default=[1, 64, 1024, 16384])
    p.add_argument("--devices", nargs="+", default=None)
    p.add_argument("--dtype", choices=("float64", "float32"), default="float64")
    p.add_argument("--grad-batch", type=int, default=1024,
                   help="batch for the forward+backward timings")
    p.add_argument("--quick", action="store_true")
    args = p.parse_args()

    if args.devices is None:
        args.devices = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
    if args.quick:
        args.sizes = [1, 1024]
        args.n_sub_aft = [128]
        args.oracle_n, args.oracle_n_rdaam = 100, 10

    global DTYPE
    DTYPE = getattr(torch, args.dtype)

    print(f"torch {torch.__version__}  threads={torch.get_num_threads()}  "
          f"dtype={args.dtype}  "
          f"devices={args.devices}")
    if torch.cuda.is_available():
        print(f"gpu: {torch.cuda.get_device_name(0)}")
    print("columns: wall-clock, samples/s, and samples/s x nodes (the "
          "grid-independent rate)")

    if args.model in ("aft", "both"):
        bench_aft(args)
    if args.model in ("rdaam", "both"):
        bench_rdaam(args)


if __name__ == "__main__":
    main()
