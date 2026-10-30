"""Calibration of a trained head, and the acceptance criterion.

Simulation-based calibration per node, per functional and per stratum on val
rows the checkpoint was not selected on; the deep-past conditional test
against the corpus by nearest neighbours; and the criterion, evaluated.

Works on any member of the family: report times, the
near-horizon stratum and the deep-past offsets follow the run's own grid, and
``--data-edge`` moves the fixed deep-past boundary and the functionals' data
window (67 Ma on the 100 Ma member).

A member given the present day (``Latent.present_input``) is
scored the way it is used: each row gets a present-day range ``--present-width``
wide around its true ``T_0`` and each draw its own ``T_0`` on it, as
``api.Model.infer`` does with the user's ``T_present`` (``sbc``'s
``present_width``).  ``--present-width 0`` scores the head given the true
``T_0`` instead, on nodes 1 on.  ``z``, the NLL and the deep-past test are
always given the true ``T_0``: the first two need it for their coordinates,
and the deep-past reference is conditional on the draws' own recent nodes,
however their ``T_0`` was chosen.

Usage:
    uv run python scripts/calibration_study.py --device cuda:0        # ~3 min
    uv run python scripts/calibration_study.py --mode report          # tables from results/
    uv run python scripts/calibration_study.py --run results/flow_v22_250 --out results/flow_v22_250/rf \
        --draws 1000 --deep-draws 400 --remove-floor --data-edge 67 --device cuda:0   # the shipped 250 Ma member
"""
from __future__ import annotations

import argparse
import itertools
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from chai.eval.calibration import LEVELS, MODALITIES, pit_summary
from chai.eval.deep_past import deep_past_test

REPORT_TIMES = (10.0, 30.0, 60.0, 100.0)
#: K: the present-day range a member given the present day is scored with, unless told.
#: Provisional (2026-09-27): a user who knows the surface temperature to about ±2.5 K.
PRESENT_WIDTH = 5.0
#: Past 100 Ma, on a member whose horizon reaches them.
LONG_TIMES = (150.0, 240.0)
REPORT_OFFSETS_MA = (1.6, 4.8, 9.5, 19.0, 31.7, 47.6)
LONG_OFFSETS_MA = (79.0, 119.0)
STRATA = ("all",) + MODALITIES + ("touches_wall", "off_walls", "near_horizon", "below_horizon",
                                  "late_exhumation", "fast", "fast_true")


def report_times(t_grid: np.ndarray) -> tuple[float, ...]:
    return REPORT_TIMES + tuple(x for x in LONG_TIMES if x < t_grid[-1])


def _offset_index(r: dict[str, Any], ma: float) -> int:
    """The deep-past offset nearest ``ma`` Ma past the boundary, on any grid."""
    past = r.get("offset_ma", np.asarray(r["offsets"]) * r["dt"])
    return int(np.argmin(np.abs(np.asarray(past) - ma)))

#: The acceptance criterion, in coverage units.  Monte Carlo noise on a
#: coverage over the smallest stratum here (~7 000 rows) is 0.006, so the
#: thresholds are five sigma above noise and mean what they say: no credible
#: interval the head reports is off by more than three points at 68% or two
#: at 95%, its tails are no lighter than one point, and past the data it
#: agrees with the prior's conditional to the same tolerance the reference
#: itself meets.
TOL68, TOL95, TOL_TAIL, TOL_ECDF, DEEP_MIN_ROWS = 0.03, 0.02, 0.01, 0.03, 500


def _nearest(t_grid: np.ndarray, t: float) -> int:
    return int(np.argmin(np.abs(t_grid - t)))


def run(run_dir: Path, out: Path, *, device: str, rows: int, first_row: int | None, seed: int,
        n_draws: int, deep_rows: int, deep_draws: int, K: int, data_edge: float = 67.0,
        split: str = "val", present_width: float | None = None, remove_floor: bool = False) -> None:
    from chai.data.observe import ObservationModel
    from chai.eval.calibration import sbc
    from chai.models.gaussian import AmortizedPosterior
    from chai.train.data import CorpusSplit, NoisyBatches, fixed_batches
    from chai.train.loop import TrainConfig

    raw = yaml.safe_load((run_dir / "config.yaml").read_text())
    cfg = TrainConfig.from_config({k: v for k, v in raw.items() if k != "git_sha"})
    model = AmortizedPosterior.load(run_dir / "model.pt", map_location=device)
    if remove_floor:
        # Stage 19e: the training floor taken back out of every draw (remove_floor); needs >= 2 draws per node.
        model.floor_sd = float(cfg.dequantise.noise_sd)
    if not model.latent.present_input:
        if present_width:
            raise SystemExit("--present-width is for a member given the present day; this one draws it")
        present_width = None
    elif present_width is None:
        present_width = PRESENT_WIDTH
    if first_row is None:
        # A cooling-only split is ~40% of the val rows, too short to start at 100 000; it
        # starts right after the rows the checkpoint was selected on.
        first_row = cfg.val_rows if cfg.max_reheat is not None else 100_000
    t0 = time.perf_counter()
    # A cooling-only member (max_reheat) is scored on, and finds neighbours among, the rows it trained on.
    val = CorpusSplit(cfg.corpus, split, latent=model.latent, max_rows=rows, first_row=first_row,
                      max_reheat=cfg.max_reheat)
    if len(val) == 0:
        raise SystemExit(f"no {split} rows from row {first_row:,}: the split is shorter")
    with open(cfg.observation) as f:
        observation = ObservationModel.from_config(yaml.safe_load(f))
    batches = fixed_batches(NoisyBatches(val, observation, cfg.effort, seed=seed,
                                         dequantise=cfg.dequantise), 4096, seed=seed)
    only = f" (reheating <= {cfg.max_reheat:g} K)" if cfg.max_reheat is not None else ""
    print(f"observed {len(val):,} {split} rows{only} from row {first_row:,} with seed {seed} in "
          f"{time.perf_counter() - t0:.1f} s")

    t0 = time.perf_counter()
    # The late-exhumation stratum reads the true helium ages, as the corpus stores them.
    he_true = np.median(val.he_age, axis=1)
    r = sbc(model, batches, device=device, n_draws=n_draws, seed=seed, aft_true=val.aft_age,
            he_true=he_true, data_edge=data_edge, present_width=present_width)
    how = ("" if present_width is None else
           f", present day {'exact' if not present_width else f'on a {present_width:g} K range'}")
    print(f"SBC with {n_draws} draws per row{how}: {time.perf_counter() - t0:.1f} s")
    flat = {k: v for k, v in r.items() if isinstance(v, np.ndarray)}
    flat["interval_cov"] = np.stack([r["in68"].mean(0), r["in95"].mean(0)])[None]   # (1, 2, n_grid)
    flat.update({f"strata__{k}": v for k, v in r["strata"].items()})
    flat["meta"] = json.dumps({"run": str(run_dir), "rows": rows, "first_row": first_row, "split": split,
                               "max_reheat": cfg.max_reheat,
                               "present_width": present_width, "first_node": r["first_node"],
                               "n_hyper": model.latent.n_hyper, "z_first_node": int(model.latent.present_input),
                               "seed": seed, "n_draws": n_draws, "data_edge": data_edge, "hyper_names": r["hyper_names"],
                               "floor_removed": model.floor_sd,
                               "functionals": r["functionals"], "git_sha": raw.get("git_sha", "")})
    np.savez(out / "calibration.npz", **flat)

    if split == "train" or not deep_rows:
        # The neighbour pool is the train split, so every query would find itself; or asked for SBC only.
        print("deep past skipped: " + ("the queries are train rows" if split == "train" else "--deep-rows 0"))
        (out / "deep_past.npy").unlink(missing_ok=True)
        return
    t0 = time.perf_counter()
    train = CorpusSplit(cfg.corpus, "train", max_reheat=cfg.max_reheat)
    print(f"train paths loaded in {time.perf_counter() - t0:.1f} s")
    k67 = _nearest(val.t_grid, data_edge)
    variants = {"per_sample": {"margin": 1.0}, "per_sample_x1.5": {"margin": 1.5},
                f"fixed_{data_edge:g}Ma": {"fixed_boundary": k67}}
    deep: dict[str, Any] = {"meta": {"rows": deep_rows, "n_draws": deep_draws, "K": K, "k67": k67,
                                     "data_edge": data_edge}}
    for name, kw in variants.items():
        t0 = time.perf_counter()
        deep[name] = deep_past_test(model, batches, train.T, device=device, n_draws=deep_draws,
                                    K=K, seed=seed, max_rows=deep_rows, **kw)
        print(f"deep past [{name}]: {time.perf_counter() - t0:.1f} s, "
              f"{int(deep[name]['usable'].sum()):,} rows usable, "
              f"{torch.cuda.max_memory_allocated() / 1e9 if device.startswith('cuda') else 0:.1f} GB peak")
    np.save(out / "deep_past.npy", deep, allow_pickle=True)


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def _load(out: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    d = dict(np.load(out / "calibration.npz", allow_pickle=True))
    d["strata"] = {k.split("__", 1)[1]: v for k, v in d.items() if k.startswith("strata__")}
    d["meta"] = json.loads(str(d["meta"]))
    f = out / "deep_past.npy"
    deep = np.load(f, allow_pickle=True).item() if f.exists() else {}
    return d, deep


def _n_hyper(d) -> int:
    """Hyperparameters leading ``z``: a GP prior's three, a pulse prior's none."""
    if "n_hyper" in d["meta"]:
        return int(d["meta"]["n_hyper"])
    return d["pit_z"].shape[1] - d["pit_T"].shape[1]       # files from before 2026-09-27


def _z_col(d, k: int) -> int | None:
    """The ``pit_z`` column of ``pit_T`` column ``k``'s node, or ``None`` where ``z`` has none.

    ``pit_T`` starts at node ``first_node``; ``z``'s node coordinates start at
    ``z_first_node`` (1 for a latent given the present day).
    """
    node = int(d["meta"].get("first_node", 0)) + k
    z0 = int(d["meta"].get("z_first_node", 0))
    return None if node < z0 else _n_hyper(d) + node - z0


def _present_text(meta: dict[str, Any]) -> str:
    w = meta.get("present_width")
    if w is None:
        return ""
    if not w:
        return "  Present day: the true T_0, given exactly (node 0 is the input, not scored)."
    return (f"  Present day: a {w:g} K range around the true T_0, one T_0 per draw on it (as the API "
            f"does); z and the NLL are given the true T_0.")

def _cov(u: np.ndarray, cols: list[int] | None = None) -> np.ndarray:
    """``(2, k)`` PIT-band coverage at 68 and 95%."""
    s = pit_summary(u if cols is None else u[:, cols])
    return s["coverage"][[LEVELS.index(0.68), LEVELS.index(0.95)]]


def report(out: Path) -> None:
    d, deep = _load(out)
    t = d["t_grid"]
    times = report_times(t)
    nodes = [_nearest(t, x) for x in times]
    strata = d["strata"]
    meta = d["meta"]
    print(f"\nSBC on {d['pit_T'].shape[0]:,} {meta.get('split', 'val')} rows from row {meta['first_row']:,}, "
          f"observation seed {meta['seed']}, {meta['n_draws']} draws per row.  Ideal: 0.68 / 0.95."
          + _present_text(meta))

    head = "  ".join(f"{x:>5.0f} Ma  " for x in times)
    ci = d["interval_cov"]
    print("\nCoverage of the truth by the central 68% / 95% *interval* of clamped draws (Stage 11's "
          "statistic), all rows:\n" + " " * 23
          + "  ".join(f"{ci[m, 0, k]:.3f}/{ci[m, 1, k]:.3f}" for m in [0] for k in nodes)
          + "   -- an interval that lands on an atom holds the whole atom")
    print("\nPIT-band coverage 68% / 95% of the true path, in degrees from clamped draws, by stratum")
    print(f"{'stratum':14s} {'rows':>7s}  {head}")
    for name in STRATA:
        if name not in strata or not strata[name].any():
            continue
        m = strata[name]
        c = _cov(d["pit_T"][m], nodes)
        print(f"{name:14s} {int(m.sum()):7d}  " + "  ".join(f"{c[0, i]:.3f}/{c[1, i]:.3f}"
                                                            for i in range(len(nodes))))
    c = _cov(d["pit_T_exact"], nodes)
    print(f"{'all, unclamped':14s} {len(d['pit_T']):7d}  " + "  ".join(
        f"{c[0, i]:.3f}/{c[1, i]:.3f}" for i in range(len(nodes))) + "   (exact Gaussian marginal)")
    zc = [_z_col(d, k) for k in nodes]
    c = _cov(d["pit_z"], [j for j in zc if j is not None])
    cells, i = [], 0
    for j in zc:
        if j is None:
            cells.append("     —     ")
        else:
            cells.append(f"{c[0, i]:.3f}/{c[1, i]:.3f}")
            i += 1
    print(f"{'all, in z':14s} {len(d['pit_z']):7d}  " + "  ".join(cells) + "   (innovation coordinate)")
    if "p_fast" in d and "fast_true" in strata:
        # A reliability check chosen from the data: rows binned by the head's own P(a segment faster than
        # 50 K/Ma); the share whose true path has one should match.  ``fast`` is the rows from 0.1 up.
        p, f = d["p_fast"], strata["fast_true"]
        print(f"\nFast segments (> 50 K/Ma on the grid): head's mean P {p.mean():.4f}, true share {f.mean():.4f}; "
              "by the head's P (rows / mean P / true share)")
        edges = (0.0, 0.01, 0.05, 0.1, 0.2, 0.5, 1.0001)
        print("  " + "   ".join(f"{lo:g}–{min(hi, 1):g}: {int(m.sum())} / {p[m].mean():.3f} / {f[m].mean():.3f}"
                                for lo, hi in itertools.pairwise(edges)
                                for m in [(p >= lo) & (p < hi)] if m.any()))

    print("\nShape of the PIT per node, all rows (ideal 0 everywhere; ECDF deviation in probability)")
    s = pit_summary(d["pit_T"])
    sx = pit_summary(d["pit_T_exact"])
    print(f"{'':14s} " + head)
    for key, lab in (("ecdf_dev", "ECDF deviation"), ("centre_excess", "centre excess"),
                     ("tail_excess", "tail excess"), ("bias", "bias (truth above median)")):
        print(f"{lab:26s} " + "  ".join(f"{s[key][k]:+.3f}    " for k in nodes)
              + "   unclamped: " + " ".join(f"{sx[key][k]:+.3f}" for k in nodes))
    print(f"over all {t.size} nodes: ECDF deviation {s['ecdf_dev'].min():.3f}–{s['ecdf_dev'].max():.3f}, "
          f"tail excess {s['tail_excess'].min():+.3f}–{s['tail_excess'].max():+.3f}, "
          f"68% coverage {s['coverage'][1].min():.3f}–{s['coverage'][1].max():.3f}, "
          f"95% coverage {s['coverage'][3].min():.3f}–{s['coverage'][3].max():.3f}")

    print("\nHyperparameters (exact, in z)" if _n_hyper(d) else "\nHyperparameters: none in this latent")
    sz = pit_summary(d["pit_z"][:, :max(_n_hyper(d), 1)])
    for i, name in enumerate(meta["hyper_names"]):
        print(f"  {name:8s} 68/95 {sz['coverage'][1, i]:.3f}/{sz['coverage'][3, i]:.3f}  "
              f"ECDF {sz['ecdf_dev'][i]:.3f}  tail {sz['tail_excess'][i]:+.3f}  bias {sz['bias'][i]:+.3f}")

    print("\nFunctionals of the path (correlations), in degrees from clamped draws")
    sf = pit_summary(d["pit_fn"])
    for i, name in enumerate(meta["functionals"]):
        print(f"  {name:24s} 68/95 {sf['coverage'][1, i]:.3f}/{sf['coverage'][3, i]:.3f}  "
              f"ECDF {sf['ecdf_dev'][i]:.3f}  centre {sf['centre_excess'][i]:+.3f}  "
              f"tail {sf['tail_excess'][i]:+.3f}  bias {sf['bias'][i]:+.3f}")

    if not deep:
        print("\nDeep past: not run (train-split queries would find themselves in the neighbour pool)")
    else:
        print(f"\nDeep past: {deep['meta']['rows']:,} rows, {deep['meta']['n_draws']} draws, "
              f"K = {deep['meta']['K']} neighbours in 1.6 M train paths.  68% / 95% PIT-band coverage "
              f"of a deep node, by how far past the boundary it is.")
    for name, r in deep.items():
        if name == "meta":
            continue
        c, h = r["control"], r.get("head_given_truth")      # the latter: Gaussian heads only
        tg = r.get("t_grid", t)
        edge = deep["meta"].get("data_edge", 67.0)
        bnd = f"{edge:g} Ma for every row" if r["fixed_boundary"] is not None else \
            f"oldest observed age x {r['margin']:g}"
        b_us = r["boundary"][r["usable"]]
        print(f"\n[{name}] boundary = first node past {bnd}; {r['usable'].sum():,} rows usable; "
              f"boundary node median {np.median(tg[b_us]):.0f} Ma (5–95%: "
              f"{np.percentile(tg[b_us], 5):.0f}–{np.percentile(tg[b_us], 95):.0f})")
        past = r.get("offset_ma", np.asarray(r["offsets"]) * r["dt"])
        print(f"{'Ma past':>8s} {'n':>7s} | {'head | own recent':^17s} {'prior | truth':^17s} "
              f"{'head | truth':^17s} {'regression | truth':^17s} | {'sd: head':>8s} {'kNN(head)':>9s} "
              f"{'kNN(truth)':>10s} {'head|tr':>8s} {'regr':>6s} | shift head-kNN")
        offsets_ma = REPORT_OFFSETS_MA + (LONG_OFFSETS_MA if t[-1] > 100.0 else ())
        for ma in offsets_ma:
            j = _offset_index(r, ma)
            if ma > past[-1] + 1.0 or c["n_at_offset"][j] < DEEP_MIN_ROWS:
                continue
            cn = _cov(r["pit_knn"][j])[:, 0]
            cc = _cov(c["pit_knn"][j])[:, 0]
            cr = _cov(c["pit_reg"][j])[:, 0]
            if h is not None:
                ch = _cov(h["pit_head"][j])[:, 0]
                ch_txt, sdh_txt = f"{ch[0]:.3f}/{ch[1]:.3f}", f"{h['sd_head'][j]:8.1f}"
            else:
                ch_txt, sdh_txt = "     —     ", "       —"
            print(f"{past[j]:8.1f} {r['n_at_offset'][j]:7d} |   {cn[0]:.3f}/{cn[1]:.3f}     "
                  f"  {cc[0]:.3f}/{cc[1]:.3f}       {ch_txt}       {cr[0]:.3f}/{cr[1]:.3f}    "
                  f"| {r['sd_net'][j]:8.1f} {r['sd_knn'][j]:9.1f} {c['sd_knn'][j]:10.1f} "
                  f"{sdh_txt} {c['sd_reg'][j]:6.1f} | {r['shift_knn'][j]:5.1f} K "
                  f"(truth {c['shift_knn'][j]:.1f})")
        nd, nb = r["nn_dist"], tg[r["nn_dist_boundary"]]
        print(f"  neighbour distance, rms per node: {np.interp(10, nb, nd):.1f} K at a 10 Ma "
              f"boundary, {np.interp(40, nb, nd):.1f} K at 40 Ma, {np.interp(67, nb, nd):.1f} K at 67 Ma")

    print("\nAcceptance criterion")
    for ok, text in acceptance(d, deep):
        print(f"  [{'pass' if ok else 'FAIL'}] {text}")


def acceptance(d: dict[str, Any], deep: dict[str, Any]) -> list[tuple[bool, str]]:
    """The criterion as a list of ``(passed, statement)``; every item must pass."""
    out: list[tuple[bool, str]] = []
    t = d["t_grid"]
    strata = d["strata"]

    def worst(u: np.ndarray, level: float, nominal: float, label: str, tol: float
              ) -> tuple[bool, str]:
        cov = pit_summary(u)["coverage"][LEVELS.index(level)]
        k = int(np.argmax(np.abs(cov - nominal)))
        dev = float(cov[k] - nominal)
        return abs(dev) <= tol, (f"{label}: every node within ±{tol:.2f} of {nominal:.2f} at "
                                 f"{level * 100:.0f}% — worst {dev:+.3f} at {t[k]:.0f} Ma")

    # ``fast`` (rows where the head itself puts at least 10% of its draws on a segment faster than 50 K/Ma) is
    # empty under a 50 K/Ma cap and opens under the shipped prior's 100 K/Ma one.  ``fast_true`` selects on the true path
    # and is not a line: an exact posterior under-covers there too.
    for name in ("all",) + MODALITIES + ("near_horizon", "late_exhumation", "fast"):
        if name not in strata or not strata[name].any():     # late exhumation is empty at 100 Ma
            continue
        m = strata[name]
        out.append(worst(d["pit_T"][m], 0.68, 0.68, f"coverage, {name}", TOL68))
        out.append(worst(d["pit_T"][m], 0.95, 0.95, f"coverage, {name}", TOL95))
    s = pit_summary(d["pit_T"])
    k = int(np.argmax(s["tail_excess"]))
    out.append((s["tail_excess"][k] <= TOL_TAIL,
                (f"tails: no node puts more than {TOL_TAIL:.2f} extra in the outer 5% — worst "
                 f"{s['tail_excess'][k]:+.3f} at {t[k]:.0f} Ma")))
    sf = pit_summary(d["pit_fn"])
    i = int(np.argmax(sf["ecdf_dev"]))
    out.append((sf["ecdf_dev"][i] <= TOL_ECDF,
                (f"functionals: ECDF deviation ≤ {TOL_ECDF:.2f} for every one — worst "
                 f"{sf['ecdf_dev'][i]:.3f} for '{d['meta']['functionals'][i]}'")))
    n_h = _n_hyper(d)
    if n_h:                                   # a pulse prior's latent has none to test
        sz = pit_summary(d["pit_z"][:, :n_h])
        i68 = int(np.argmax(np.abs(sz["coverage"][1] - 0.68)))
        i95 = int(np.argmax(np.abs(sz["coverage"][3] - 0.95)))
        out.append((abs(sz["coverage"][1, i68] - 0.68) <= TOL68 and abs(sz["coverage"][3, i95] - 0.95) <= TOL95,
                    (f"hyperparameters: within ±{TOL68:.2f} / ±{TOL95:.2f} — worst "
                     f"{sz['coverage'][1, i68] - 0.68:+.3f} at 68% ({d['meta']['hyper_names'][i68]}), "
                     f"{sz['coverage'][3, i95] - 0.95:+.3f} at 95% ({d['meta']['hyper_names'][i95]})")))
    if not deep:
        return out
    r = deep["per_sample"]
    c = r["control"]
    worst68 = worst95 = (0.0, 0)
    for j in range(len(r["offsets"])):
        if c["n_at_offset"][j] < DEEP_MIN_ROWS:          # rows, not draws
            continue
        cn, cc = _cov(r["pit_knn"][j])[:, 0], _cov(c["pit_knn"][j])[:, 0]
        if abs(cn[0] - cc[0]) > abs(worst68[0]):
            worst68 = (float(cn[0] - cc[0]), j)
        if abs(cn[1] - cc[1]) > abs(worst95[0]):
            worst95 = (float(cn[1] - cc[1]), j)
    past = r.get("offset_ma", np.asarray(r["offsets"]) * r["dt"])
    out.append((abs(worst68[0]) <= TOL68 and abs(worst95[0]) <= TOL95,
                (f"deep past: head-vs-neighbours coverage within ±{TOL68:.2f} / ±{TOL95:.2f} of the "
                 f"truth-vs-neighbours control at every offset — worst {worst68[0]:+.3f} at 68% "
                 f"({past[worst68[1]]:.0f} Ma past), {worst95[0]:+.3f} at 95% ({past[worst95[1]]:.0f} Ma past)")))
    return out


def compare(outs: list[Path]) -> None:
    """Several runs' calibration side by side, on the lines the 250 member fails (Stage 16 follow-up).

    Each column is one run directory holding ``calibration.npz`` (and ``deep_past.npy``,
    ``history.jsonl`` when there): val NLL, the worst 68% node over all rows, coverage at the
    report nodes on all / wall-touching / off-wall rows, the PIT shape at the oldest node, the
    largest-rate functional, and the deep-past test at its largest offsets.
    """
    loaded = [(o, *_load(o)) for o in outs]
    names = [o.name for o in outs]
    w = max(12, max(len(n) for n in names) + 2)
    def row(label: str, cells: list[str]) -> None:
        print(f"{label:34s} " + " ".join(f"{c:>{w}s}" for c in cells))
    row("run", names)
    cells = []
    for o, d, _ in loaded:
        h = o / "history.jsonl"
        if h.exists():
            hist = [json.loads(x) for x in h.read_text().splitlines()]
            cells.append(f"{min(r['val_nll'] for r in hist):.3f} @{max(r['step'] for r in hist) // 1000}k")
        else:
            cells.append("—")
    row("best val NLL, steps", cells)
    t = loaded[0][1]["t_grid"]
    times = report_times(t)
    nodes = [_nearest(t, x) for x in times]
    row("worst 68% node, all rows", [f"{pit_summary(d['pit_T'])['coverage'][LEVELS.index(0.68)].min():.3f}"
                                     for _, d, _ in loaded])
    for stratum in ("all", "touches_wall", "off_walls", "he_only", "late_exhumation"):
        for i, k in enumerate(nodes):
            cells = []
            for _, d, _ in loaded:
                m = d["strata"].get(stratum)
                cells.append("—" if m is None or not m.any() else f"{_cov(d['pit_T'][m], [k])[0, 0]:.3f}")
            row(f"68% {stratum} at {times[i]:.0f} Ma", cells)
    k = nodes[-1]
    for key, lab in (("centre_excess", "centre excess"), ("tail_excess", "tail excess")):
        row(f"PIT {lab} at {times[-1]:.0f} Ma", [f"{pit_summary(d['pit_T'])[key][k]:+.3f}" for _, d, _ in loaded])
    row("68% in z at oldest node", [f"{_cov(d['pit_z'], [_z_col(d, k)])[0, 0]:.3f}" for _, d, _ in loaded])
    row("largest rate: 68% / ECDF", [f"{pit_summary(d['pit_fn'])['coverage'][1, -1]:.3f} / "
                                     f"{pit_summary(d['pit_fn'])['ecdf_dev'][-1]:.3f}" for _, d, _ in loaded])
    cells = []
    for _, d, deep in loaded:
        if not deep:
            cells.append("—")
            continue
        r = deep["per_sample"]
        past = r.get("offset_ma", np.asarray(r["offsets"]) * r["dt"])
        ok = np.flatnonzero(np.asarray(r["control"]["n_at_offset"]) >= DEEP_MIN_ROWS)
        j = int(ok[-1])                                    # the farthest offset the report shows
        cn, cc = _cov(r["pit_knn"][j])[0, 0], _cov(r["control"]["pit_knn"][j])[0, 0]
        cells.append(f"{cn - cc:+.3f} at {past[j]:.0f} Ma")
    row("deep past: head − control, 68%, farthest offset", cells)
    row("criterion lines failed", [f"{sum(not ok for ok, _ in acceptance(d, deep))} of "
                                   f"{len(acceptance(d, deep))}" for _, d, deep in loaded])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", default="results/flow_v22_100")
    ap.add_argument("--out", default="results")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--mode", choices=("run", "report", "compare"), default="run")
    ap.add_argument("--runs", nargs="*", default=[], help="compare: run directories holding calibration.npz")
    ap.add_argument("--rows", type=int, default=50_000)
    ap.add_argument("--split", default="val", choices=("train", "val", "test"),
                    help="train scores the rows the head was fitted on: a data-limit check")
    ap.add_argument("--first-row", type=int, default=None,
                    help="skip the val rows the checkpoint was selected on (the first 50 000); "
                         "default 100 000, or the run's val_rows for a cooling-only (max_reheat) run")
    ap.add_argument("--seed", type=int, default=2024)
    ap.add_argument("--draws", type=int, default=100)
    ap.add_argument("--deep-rows", type=int, default=4096)
    ap.add_argument("--deep-draws", type=int, default=32)
    ap.add_argument("--K", type=int, default=100)
    ap.add_argument("--present-width", type=float, default=None,
                    help=f"K: the present-day range a member given the present day is scored with "
                         f"(default {PRESENT_WIDTH:g}); 0 gives it the true T_0 exactly")
    ap.add_argument("--data-edge", type=float, default=67.0,
                    help="Ma: the fixed deep-past boundary and the functionals' data window")
    ap.add_argument("--remove-floor", action="store_true",
                    help="take the run's training noise floor back out of its draws (remove_floor); "
                         "--draws and --deep-draws must then be at least twice the node count")
    args = ap.parse_args()
    if args.mode == "compare":
        compare([Path(r) for r in args.runs])
        return
    out = Path(args.out)
    if args.mode == "run":
        run(Path(args.run), out, device=args.device, rows=args.rows, first_row=args.first_row,
            seed=args.seed, n_draws=args.draws, deep_rows=args.deep_rows,
            deep_draws=args.deep_draws, K=args.K, data_edge=args.data_edge, split=args.split,
            present_width=args.present_width, remove_floor=args.remove_floor)
    report(out)


if __name__ == "__main__":
    main()
