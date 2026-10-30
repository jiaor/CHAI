"""Generate the simulation corpus a member is trained on.

A row is one thermal history, its AFT observation, and ``--grains`` AHe grains
drawn on that same history -- see ``chai.data.simulate`` for why the corpus is
joint rather than split by observable.

The run is resumable: rerun the identical command and it fills in whatever blocks
are missing, with byte-identical results.

Usage (the shipped family's two corpora, 3 M histories each):
    uv run python scripts/simulate.py --out data/corpus_v22_100.h5 --n 3000000 --grains 12 \
        --prior-config configs/prior_v22.yaml --seed 2120
    uv run python scripts/simulate.py --out data/corpus_v22_250.h5 --n 3000000 --grains 12 \
        --prior-config configs/prior_v22_250.yaml --seed 2270
    uv run python scripts/simulate.py ... --dry-run                     # the plan and its cost, no rows

A config's ``kind`` names the history prior (``PulsePrior`` for the shipped family); absent, it is
the rate-integrated GP prior (``ThermalHistoryPrior``).

A prior config with a ``grid:`` section (``n_grid``, ``spacing``) sets the stored
grid and horizon, overriding ``--n-grid``, ``--spacing`` and ``--t-max``: the grid
a member was sized for is part of what was pinned with it.
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
from pathlib import Path

import yaml

from chai.data.priors import EpisodicPrior, GrainPrior, ThermalHistoryPrior, prior_from_config
from chai.data.simulate import CorpusSpec, provenance, simulate

#: Single-core, idle-machine cost at n_grid = 64 (docs/prior.md).  The prior draw
#: dominates the AFT oracle by ~50x, so most of the first number is prior cost.
MS_PER_HISTORY, MS_PER_GRAIN = 11.5, 7.9

#: What those numbers cost in practice, once 112 workers are contending for cache
#: and memory bandwidth and the clocks come off turbo.  Calibrated on the 10^6 x 4
#: run: 25.7 min wall on 112 workers is 173 ms of worker time per history against
#: the 43 ms the single-core figures predict.  The 2x10^6 x 4 run held it -- 49.8 min
#: is 167 ms, a factor of 3.9 -- so the contention is a property of the machine at
#: full width, not of the run length.  Projecting without it understates the run by
#: 4x, which is the difference between "go and get a coffee" and "leave it going".
LOAD_FACTOR = 4.0


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, required=True, help="HDF5 corpus path")
    p.add_argument("--n", type=int, default=2_000_000, help="number of histories")
    p.add_argument("--grains", type=int, default=4, help="AHe grains per history")
    p.add_argument("--n-grid", type=int, default=64, help="path nodes (settled: 64)")
    p.add_argument("--t-max", type=float, default=100.0, help="horizon, Ma")
    p.add_argument("--spacing", default="uniform", help="node placement (settled: uniform)")
    p.add_argument("--block-size", type=int, default=512, help="histories per worker task")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--workers", type=int, default=mp.cpu_count())
    p.add_argument("--compression", default="lzf", help="HDF5 filter, or 'none'")
    p.add_argument("--prior-config", type=Path, default=Path("configs/prior_v22.yaml"),
                   help="pinned prior; the dataclass defaults are used if absent")
    p.add_argument("--episodic", action="store_true",
                   help="draw histories from EpisodicPrior with the pinned prior's walls, rate cap, "
                        "present-day and drift ranges -- Stage 6's out-of-family truth (Stage 13)")
    p.add_argument("--dry-run", action="store_true", help="print the plan and stop")
    args = p.parse_args()

    if args.prior_config.exists():
        cfg = yaml.safe_load(args.prior_config.read_text())
        prior = prior_from_config(cfg["history"], cfg.get("kind", "ThermalHistoryPrior"))
        grain_prior = GrainPrior.from_config(cfg["grain"])
        source = str(args.prior_config)
        if "grid" in cfg:
            args.n_grid, args.spacing = int(cfg["grid"]["n_grid"]), cfg["grid"]["spacing"]
            args.t_max = prior.t_max
    else:
        prior, grain_prior, source = ThermalHistoryPrior(t_max=args.t_max), GrainPrior(), "defaults"
    if args.episodic:
        base = prior
        prior = EpisodicPrior(t_max=base.t_max, T_min=base.T_min, T_max=base.T_max, rate_max=base.rate_max,
                              T_present_range=base.T_present_range, drift_total_range=base.drift_total_range)
        source += " (walls, rate cap, T_0 and drift; histories from EpisodicPrior)"
    spec = CorpusSpec(n_histories=args.n, n_grains=args.grains, n_grid=args.n_grid,
                      t_max=args.t_max, spacing=args.spacing,
                      block_size=args.block_size, seed=args.seed)

    # The prior draw scales with its solver steps, RDAAM's damage sweep with the square of the
    # path nodes.  A 2 M-history 250 Ma corpus (5 120 steps, 96 nodes, 12 grains) projects 4.8 h and
    # took 4.0 (239 min on 112 workers; the flat per-history rate it ran under had said 2.1).
    ms_hist = MS_PER_HISTORY * prior.n_steps / 2048
    ms_grain = MS_PER_GRAIN * (args.n_grid / 64) ** 2
    core_s = args.n * (ms_hist + args.grains * ms_grain) * LOAD_FACTOR / 1e3
    # Uncompressed; lzf took the 2x10^6 x 4 corpus from 1.58 to 1.08 GB.
    gb = args.n * (args.n_grid * 4 + 200 * 2 + args.grains * 6 * 4 + 40) / 1e9
    print(prior.describe())
    print(f"\ncorpus   {args.n:,} histories x {args.grains} grains, "
          f"{spec.n_blocks:,} blocks of {args.block_size}")
    print(f"prior    {source}")
    print(f"grid     {args.n_grid} nodes, {args.spacing}, 0-{args.t_max:g} Ma")
    print(f"estimate {core_s / 60 / args.workers:.1f} min on {args.workers} workers "
          f"· {gb:.2f} GB before compression")
    print(f"git      {provenance()['git_sha'][:12]}"
          f"{' (dirty)' if provenance()['git_dirty'] else ''}")
    if args.dry_run:
        return

    simulate(args.out, spec, prior, grain_prior, workers=args.workers,
             compression=None if args.compression == "none" else args.compression)
    print(f"wrote {args.out} ({args.out.stat().st_size / 1e9:.2f} GB)")


if __name__ == "__main__":
    main()
