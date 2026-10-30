"""Export the shipped family for the browser app, and the fixtures its port is checked against.

    uv run --group web python scripts/export_onnx.py                  # web/models/: graphs + meta.json
    uv run --group web python scripts/export_onnx.py --fixtures web/test/fixtures.json
    uv run --group web python scripts/export_onnx.py --members 100=results/<run_100> 250=results/<run_250> \
        --chooser results/<chooser> --out /tmp/models --fixtures /tmp/fixtures.json    # any family

The graphs are :mod:`chai.export`'s: per member a condition graph (once per
sample) and a sampler graph (per batch of base draws), and the chooser.  With
``--fixtures`` each of :func:`chai.export.fixture_samples` is also run through
``api.load()`` and what the port must reproduce is written beside it: the
encoder inputs, each member's notes, the log Bayes factor, the member that
answers and its warnings, and the quantiles of one reference run of ``--reference-n`` draws.  The
port's draws cannot match Python's (different generators), so its quantiles are
checked in distribution against those.  A family given the present day is run at each case's
``T_present`` (:func:`chai.export.fixture_samples`).

The weights are stored as float16 and cast to float32 on load (``--float32`` keeps them
float32): half the download, and posterior quantiles move 15-20x less than between two
float32 runs of 20 000 draws (``docs/browser.md`` §1).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

from chai import api
from chai.export import export_family, fixture_samples, sample_from_dict

ROOT = Path(__file__).resolve().parents[1]


def expected(family: api.Family, case: dict, reference_n: int) -> dict:
    """What the API makes of one fixture case."""
    try:
        sample = sample_from_dict(case.get("sample") or case["raw"])
        boxes = [api.Box(**{k: tuple(v) if k in ("t", "T") else v for k, v in b.items()}) for b in case["boxes"]]
        obs, _ = family.long.to_aliquot(sample)
        # Model.infer's order: the sample's refusals, then the present day's.
        family.long.present_range(case.get("T_present"))
    except ValueError as err:
        return {"error": str(err)}
    T_present = case.get("T_present")
    b = family.long.batch(obs)
    out: dict = {
        "features": {k: v.cpu().numpy().astype(float if k != "grain_mask" else bool).tolist() for k, v in b.items()},
        "notes": {str(h): m.to_aliquot(sample)[1] for h, m in family.members.items()},
        "log_bayes_factor": family.log_bayes_factor(sample, T_present),
        "oldest_age": sample.oldest_age,
    }
    kw = {"prior_odds": case["prior_odds"], "boxes": boxes, "member": case["member"],
          "cooling_only": case.get("cooling_only", False), "T_present": T_present}
    try:
        first = family.infer(sample, case["n"], seed=0, **kw)
    except api.InfeasibleBoxes as err:
        out["infer_error"] = str(err)
        return out
    ref = family.infer(sample, reference_n, seed=1, **kw)
    out["result"] = {
        "member": first.choice.member,
        "p": first.choice.p,
        "reason": first.choice.reason,
        "t": first.t.tolist(),
        "warnings": first.warnings,
        "n": len(first),
        "seconds": first.seconds,
        # The reference the port's quantiles are checked against: one large run, so that a
        # quantile on a wall's atom is on it (a mean of small runs' quantiles falls short of it).
        "reference_n": len(ref),
        "reference_quantiles": ref.quantiles().tolist(),
        "reference_member": ref.choice.member,
        "reference_p": ref.choice.p,
    }
    return out


def forward_fixtures(family: api.Family) -> list[dict]:
    """A few posterior paths per member, what api.predict makes of them, and each sample's sensitive window."""
    from chai.export import example

    ex = example(family, n=1)["examples"][0]["sample"]
    old = {"aft": {"age": 180.0, "n_spontaneous": 600, "lengths": [12.5] * 60, "dpar": 1.6},
           "grains": [{"age": 150.0, "radius": 45.0, "eU": 80.0}, {"age": 120.0, "radius": 90.0, "eU": 8.0}]}
    out = []
    for name, d, h in (("example", ex, 100), ("example_250", ex, 250), ("old", old, 250)):
        sample = sample_from_dict(d)
        post = family.infer(sample, 6, member=h, seed=0, T_present=_present(family))
        pred = api.predict(post, sample, n=6)
        w = api.sensitive_window(sample)
        out.append({"name": name, "sample": d, "t": post.t.tolist(), "T": post.T.tolist(),
                    "window": {"lo": w.lo, "hi": w.hi, "systems": w.systems},
                    "aft_age": pred.aft_age.tolist(), "length_pdf": pred.length_pdf.tolist(),
                    "he_age": pred.he_age.tolist()})
    return out


#: The floor the page's correction is checked at when a member carries none (Stage 19e's).
FLOOR_FIXTURE_SD = 1.0


def floor_fixtures(family: api.Family) -> list[dict]:
    """Per member given the present day, one batch of its unclamped draws at the fewest the correction takes
    (two per node, and ten more), and what :meth:`AmortizedPosterior.paths` makes of them with the floor
    taken out and the clamp applied: ``web/js/floor.js`` is checked on exactly these numbers, draw for draw.
    At the member's own floor if it carries one, else :data:`FLOOR_FIXTURE_SD`."""
    from chai.export import FIXTURE_PRESENT, example
    from chai.models.gaussian import draw_z

    sample = sample_from_dict(example(family, n=1)["examples"][0]["sample"])
    out = []
    for h, m in sorted(family.members.items()):
        if not m.present_input:
            continue
        net, sd = m.net, m.net.floor_sd or FLOOR_FIXTURE_SD
        S = 2 * (len(m.t) - 1) + 10
        g = torch.Generator(device=m.device).manual_seed(h)
        lo, hi = FIXTURE_PRESENT
        T0 = lo + (hi - lo) * torch.rand(S, generator=g, device=m.device, dtype=torch.float64)
        feats = m.batch(m.to_aliquot(sample)[0])
        rows = {k: v.expand(S, *v.shape[1:]) for k, v in feats.items()}
        was = net.floor_sd
        try:
            with torch.no_grad():
                z = draw_z(net({**rows, "T0": T0.to(torch.float32)}), 1, g)[0][:, None]
                net.floor_sd = 0.0
                raw, _ = net.paths(z, {"T0": T0[:, None]}, clamp=False)
                net.floor_sd = sd
                ref, _ = net.paths(z, {"T0": T0[:, None]})
        finally:
            net.floor_sd = was
        out.append({"member": h, "sd": sd, "S": S, "T0": T0.tolist(), "raw": raw[:, 0].double().cpu().numpy().ravel().tolist(),
                    "paths": ref[:, 0].double().cpu().numpy().ravel().tolist()})
    return out


#: The fixture cases the port's fit check is compared on, number for number (web/test/fit.test.mjs).
FIT_CASES = ("slow_cooling", "reheated", "old_cratonic", "misfit_inverted_he", "misfit_old_aft", "aft_only", "he_only",
             "no_lengths", "age_sd_only", "few_lengths", "zero_tracks", "uncorrected_one", "past_250")


def fit_fixtures(family: api.Family, cases: list[dict], n: int = 20) -> list[dict]:
    """``n`` paths' predictions per case, rounded, and what :func:`chai.fit.check` makes of exactly those numbers."""
    from chai import fit

    out = []
    for c in cases:
        if c["name"] not in FIT_CASES:
            continue
        sample = sample_from_dict(c["sample"])
        post = family.infer(sample, 200, seed=0, prior_odds=c["prior_odds"], member=c["member"],
                            T_present=c.get("T_present"))
        pred = api.predict(post, sample, n=n)
        r = lambda x: None if x is None else np.array([float(f"{v:.6g}") for v in np.ravel(x)]).reshape(np.shape(x))  # noqa: E731
        pred = api.Prediction(paths=pred.paths, aft_age=r(pred.aft_age), length_pdf=r(pred.length_pdf),
                              he_age=r(pred.he_age), bins=pred.bins)
        f = fit.check(sample, pred, family[post.time_span].observation)
        out.append({"name": c["name"], "sample": c["sample"], "member": post.time_span, "T_present": c.get("T_present"),
                    "aft_age": None if pred.aft_age is None else pred.aft_age.tolist(),
                    "length_pdf": None if pred.length_pdf is None else pred.length_pdf.tolist(),
                    "he_age": pred.he_age.tolist(),
                    "expect": {"level": f.level, "warnings": f.warnings, "caution": f.caution,
                               "data": [{"label": d.label, "observed": d.observed, "p": d.p, "F": d.F,
                                         "interval": list(d.interval), "outside": d.outside, "unit": d.unit}
                                        for d in f.data]}})
    return out


def _present(family: api.Family) -> list[float] | None:
    """The fixtures' present-day range for a family given it, else ``None``."""
    from chai.export import FIXTURE_PRESENT

    return list(FIXTURE_PRESENT) if family.long.present_input else None


def load(members: list[str] | None, chooser: str | None) -> api.Family:
    """``api.load()``, or the family named by ``--members H=run ...`` and ``--chooser run``."""
    if not members and not chooser:
        return api.load()
    if not (members and chooser):
        raise SystemExit("--members and --chooser go together")
    runs = {}
    for m in members:
        h, sep, run = m.partition("=")
        if not sep:
            raise SystemExit(f"--members takes HORIZON=RUN, got {m!r}")
        runs[int(h)] = run
    return api.load_family(runs, chooser)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, default=ROOT / "web" / "models")
    ap.add_argument("--float32", action="store_true",
                    help="store weights as float32 (default float16, cast to float32 on load: half the download)")
    ap.add_argument("--fixtures", type=Path, default=None, help="also write the port's fixtures here")
    ap.add_argument("--reference-n", type=int, default=20_000)
    ap.add_argument("--skip-export", action="store_true")
    ap.add_argument("--members", nargs="+", default=None, metavar="H=RUN",
                    help="the members, e.g. 100=results/flow_v22_100 250=results/flow_v22_250 (a run or its model.pt); "
                         "default: the family api.load() ships")
    ap.add_argument("--chooser", default=None, help="the chooser's run or model.pt, with --members")
    args = ap.parse_args(argv)
    torch.set_num_threads(8)
    family = load(args.members, args.chooser)
    if not args.skip_export:
        t0 = time.perf_counter()
        info = export_family(family, args.out, float16=not args.float32)
        total = sum(info["bytes"].values())
        print(f"exported to {args.out} in {time.perf_counter() - t0:.0f} s ({info['weights']}, "
              f"{total / 1e6:.1f} MB): " + ", ".join(f"{k} {v / 1e6:.1f} MB" for k, v in info["bytes"].items()))
    if args.fixtures:
        t0 = time.perf_counter()
        cases = fixture_samples(family)
        for c in cases:
            c["expect"] = expected(family, c, args.reference_n)
            print(f"  {c['name']:28s} " + ("error" if "error" in c["expect"] else
                  "refused" if "infer_error" in c["expect"] else "ok"), flush=True)
        args.fixtures.parent.mkdir(parents=True, exist_ok=True)
        args.fixtures.write_text(json.dumps({"reference_n": args.reference_n, "cases": cases,
                                             "forward": forward_fixtures(family),
                                             "fit": fit_fixtures(family, cases),
                                             "floor": floor_fixtures(family)}))
        print(f"{len(cases)} fixtures to {args.fixtures} in {time.perf_counter() - t0:.0f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
