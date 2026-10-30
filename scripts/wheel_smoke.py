"""An installed ``chai``, away from the checkout: measurements in, histories out.

Run by ``make wheel-check`` from a fresh venv in a temporary directory, so nothing
here can reach the repository -- not ``results/``, not ``configs/``, not the
compiled oracles in ``native/build/``.  It checks that the wheel carries what
:func:`chai.api.load` needs, and that the thin API runs on the torch ports alone.
Prints one JSON line.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

t0 = time.perf_counter()
import chai
from chai import api

here = Path(chai.__file__).resolve()
assert "site-packages" in here.parts, f"imported from {here}, not an installed wheel"
t_import = time.perf_counter() - t0

t0 = time.perf_counter()
family = api.load()
t_load = time.perf_counter() - t0
assert isinstance(family, api.Family), "the wheel should carry both members and the chooser"
model = family[100]

# A history cooling through both systems, measured through the model's own forward models and noise.
t_ctrl, T_ctrl = [0.0, 20.0, 60.0, 100.0], [15.0, 50.0, 95.0, 130.0]
sample, (tg, Tg) = api.simulate_sample(model, t_ctrl, T_ctrl, seed=1)
TP = (10.0, 20.0)                  # the present day as a range: both histories start at 15 °C
post = model.infer(sample, n=4000, seed=0, T_present=TP)
q = post.quantiles((0.16, 0.5, 0.84))
k = int(np.argmin(np.abs(post.t - 10.0)))
boxed = model.infer(sample, n=1000, seed=0, T_present=TP, boxes=[api.Box(t=(55, 65), T=(80, 110))])
# An old sample, through the family: the chooser should hand it to the 250 Ma member.
old, _ = api.simulate_sample(family, [0.0, 30.0, 120.0, 250.0], [15.0, 40.0, 70.0, 140.0], seed=2)
fam = family.infer(old, n=2000, seed=0, T_present=TP)

try:
    from chai.forward.native import calculate_aft  # noqa: F401
    native = "found"
except FileNotFoundError:
    native = "absent (expected: the wheel does not build the oracles)"

print(json.dumps({
    "chai": chai.__version__, "python": sys.version.split()[0], "module": str(here.parent),
    "import_s": round(t_import, 2), "load_s": round(t_load, 2), "infer_ms": round(post.seconds * 1e3, 1),
    "n": len(post), "T10_true": round(float(np.interp(10.0, tg, Tg)), 1),
    "T10_post": [round(float(v), 1) for v in q[:, k]],
    "box_acceptance": round(boxed.acceptance, 4), "warnings": post.warnings,
    "old_member": fam.choice.member, "old_p": round(fam.choice.p, 4), "old_T150": [round(float(v), 1) for v in fam.quantiles((0.16, 0.5, 0.84))[
        :, int(np.argmin(np.abs(fam.t - 150.0)))]], "family_ms": round(fam.seconds * 1e3, 1), "native": native,
}))
