# Using the forward models

The forward models underneath CHAI are a handful of functions and one convention. Two PyTorch
ports (Ketcham AFT and RDAAM) take a batch on any device and return exact gradients; the
reference implementations they were validated against (`KETCH.c`, `RDAAM.cpp`, `Mad_He.f90`)
take one path per call on a CPU and are not distributed with CHAI (they are third-party code). Everything else in the package, the prior, the
corpus, the observation model and the trained members, is built on these.

Every code block on this page is executed by `tests/test_api_docs.py`, on the CPU, so the page
cannot drift from the code (the blocks that call a reference implementation are skipped where it
is not built).

Setup: `uv sync`. The reference-implementation blocks need their sources under `native/` and
`make native`.

## 1. The convention

A thermal history is a pair of arrays `(t, T)`:

- `t` in **Ma before present**, `t[0] == 0` (the present), **strictly increasing** into the past;
- `T` in **°C**, one value per node;
- the path between nodes is **piecewise linear**. Refining it (`chai.paths.refine`) changes
  the quadrature, never the history.

Every binding validates this and raises on anything else. Mad_He wants its arrays oldest first;
the binding reverses them internally, and no caller ever should.

The shipped members work on 96 uniform nodes over 100 Ma, and on the same 96 followed by 64
more to 250 Ma. The functions accept any strictly increasing grid.

```python
import numpy as np
import torch
from chai.paths import fixed_grid, refine, resample

t = fixed_grid(64, t_max=100.0, spacing="uniform")          # (64,) float64 tensor
t_ctrl = torch.tensor([[0.0, 10.0, 40.0, 100.0]])            # (B, N) control points
T_ctrl = torch.tensor([[15.0, 45.0, 90.0, 120.0]])
T = resample(t_ctrl, T_ctrl, t)                              # (1, 64): the same path on the grid
t_fine, T_fine = refine(t[None], T, n_sub=4)                 # (1, 253): same path, finer sampling
assert torch.allclose(resample(t_fine, T_fine, t), T)
```

## 2. The reference implementations — `chai.forward.native`

Exact, one path per call, CPU, no gradients. They generated the training corpora and they are
the reference every port is validated against. Their sources are third-party code and are not
distributed with CHAI; each function raises `FileNotFoundError` naming the library unless
`native/` holds the source and `make native` built it (or `CHAI_NATIVE_DIR` points at a build).

```python
import numpy as np
from chai.forward.native import (KETCHAM_1999, PREC_BEST, calculate_aft,
                                   calculate_madhe, calculate_rdaam, eU)

t = np.array([0.0, 10.0, 40.0, 100.0])
T = np.array([15.0, 45.0, 90.0, 120.0])

aft = calculate_aft(t, T, kinetic_par=2.0)           # Dpar 2.0 um; Ketcham et al. (2007)
aft.age, aft.oldest_age, aft.mean_length, aft.dfmeanp, aft.length_pdf.shape  # Ma, Ma, um, um, (200,)

old = calculate_aft(t, T, kinetic_par=0.84, kinpar_id=4, annealing_model=KETCHAM_1999)  # by rmr0

he = calculate_rdaam(t, T, radius=60.0, U=40.0, Th=80.0, precision=PREC_BEST)
he.age, he.corrected_age                              # raw and alpha-ejection-corrected, Ma

mad = calculate_madhe(t, T, grain_size=60.0)          # Farley (2000), no damage, no Ft; Ma
print(f"AFT {aft.age:.2f}  AHe {he.corrected_age:.2f}  Mad_He {mad:.2f}  eU {eU(40.0, 80.0):.1f}")
```

| Function | Returns | The knobs that matter |
|---|---|---|
| `calculate_aft(t, T, kinetic_par, kinpar_id=0, annealing_model=KETCHAM_2007)` | `AFTResult(age, oldest_age, mean_length, dfmeanp, length_pdf)`; the pdf is 200 bins of 0.1 µm | `kinpar_id`: 0 Dpar µm, 1 Cl apfu, 2 OH apfu, 3 Cl wt%, 4 rmr0. `set_time_step_pct(pct)` sets the internal step (default 1% of the path); process-global |
| `calculate_rdaam(t, T, radius, U, Th, Sm=0, precision=PREC_BEST)` | `HeResult(age, corrected_age)` | `precision` GOOD/BETTER/BEST. Its internal grid is `max(5 × nodes, TV-driven)`; `calculate_rdaam_debug` also returns that count, the total He and the Ft factors |
| `calculate_madhe(t, T, grain_size=100, method="AHe")` | age, Ma (float32 inside) | `method` also AHe/ZHe/KfAr/BiAr/MuAr/HoAr. A fixed 100-node radial grid, which puts a −1.5% floor under it |

`KETCHAM_2007` is the Ketcham et al. (2007) calibration **as published** — the default here and
in the port, so the two never compute different models. `KETCHAM_1999` is 8.4% older at the
median. Mad_He is a different physical model from RDAAM (damage-free Farley 2000 kinetics),
not a second implementation of it; the two differ by 9–24% where their kinetics differ.

**The reference implementations are not thread-safe** (`RDAAM.cpp` mutates file-scope state,
and the AFT step setting is process-global). Parallelise with processes, as `chai.data.simulate`
does.

## 3. The ports — `chai.forward.torchimpl`

Exact ports, not learned approximations: batched `(B, N)` in, any device, any float dtype,
differentiable with respect to every input.

```python
import torch
from chai.forward.torchimpl.ketcham import KETCHAM_2007, aft_age, aft_forward
from chai.forward.torchimpl.rdaam import rdaam_age
from chai.paths import fixed_grid

device = "cuda" if torch.cuda.is_available() else "cpu"
t = fixed_grid(64, 100.0, "uniform", dtype=torch.float32, device=device).expand(8, -1)
T = torch.linspace(120.0, 15.0, 64, device=device).expand(8, -1).contiguous()
dpar = torch.full((8,), 2.0, device=device)

out = aft_forward(t, T, dpar, n_sub=8, model=KETCHAM_2007)   # one annealing pass, every observable
out["age"].shape, out["length_pdf"].shape, out["mean_length"].shape, out["std_length"].shape
age = aft_age(t, T, dpar, n_sub=8)                            # the age alone: 2-14x cheaper

r, U, Th = (torch.full((8,), v, device=device) for v in (60.0, 40.0, 80.0))
he = rdaam_age(t, T, r, U, Th, n_sub=8, rdim=512, use_checkpoint=False)
he["age"].shape, he["corrected_age"].shape, he["total_he"].shape
```

| Function | Returns | Knobs |
|---|---|---|
| `aft_forward(t, T, dpar, n_sub=64, model=KETCHAM_2007)` | `{"age", "length_pdf", "bins", "mean_length", "std_length"}` | `n_sub` sub-intervals per segment |
| `aft_age(t, T, dpar, n_sub=64, model=...)` | `(B,)` age | as above |
| `track_length_pdf(t, T, dpar, n_sub=64, bins=None, model=...)` | `(pdf, bins)` | `bins` any `(n_bins,)` axis in µm |
| `rdaam_age(t, T, radius, U, Th, Sm=None, n_sub=16, rdim=512, use_checkpoint=True)` | `{"age", "corrected_age", "total_he"}` | `n_sub`, `rdim` radial nodes, `use_checkpoint` |

**Choosing the refinement.** The defaults (`n_sub=64` for AFT, `16` for RDAAM) were set on
paths of 6–10 control points. On a 64-node or finer grid they are far finer than needed and
cost 8× (AFT) or 4× (RDAAM) for nothing. Measured on 1 000 prior paths against converged
reference runs:

| Use | AFT | RDAAM | Error it buys (median relative age) |
|---|---|---|---|
| training, screening, likelihoods at scale | `n_sub=8` | `n_sub=8, rdim=128` | AFT 4e-5; AHe 4.7e-3 |
| validating, or anything quoted | `n_sub=32`, float64 | `n_sub=8, rdim=512` | AFT 6e-6; AHe 2e-4 |

For helium, `rdim` is the knob: `n_sub` past 8 buys nothing at `rdim=128`, because the radial
grid is what is left — and its error is a **bias**, not scatter: at `rdim=128` every age comes out
0.3–1.0% low. All of these are one to three orders of magnitude inside the 5–10% a laboratory
reports, but a likelihood cannot absorb a bias the way it absorbs noise; use `rdim=512` where
that matters.

**Gradients.** Everything is ordinary autograd:

```python
import torch
from chai.forward.torchimpl.ketcham import aft_forward
from chai.forward.torchimpl.rdaam import rdaam_age
from chai.paths import fixed_grid

t = fixed_grid(64, 100.0, "uniform").expand(2, -1)
T = torch.linspace(120.0, 15.0, 64, dtype=torch.float64).expand(2, -1).clone().requires_grad_(True)
dpar = torch.tensor([1.8, 2.4], dtype=torch.float64, requires_grad=True)
out = aft_forward(t, T, dpar, n_sub=8)
(out["age"].sum() + out["mean_length"].sum()).backward()
T.grad.shape, dpar.grad                       # d/dT at every node, and d/dDpar
assert torch.isfinite(T.grad).all()

T.grad = None
r, U, Th = (torch.tensor([v, v], dtype=torch.float64) for v in (60.0, 40.0, 80.0))
he = rdaam_age(t, T, r, U, Th, n_sub=4, rdim=64)   # use_checkpoint=True by default
he["corrected_age"].sum().backward()
assert torch.isfinite(T.grad).all()
```

`use_checkpoint=True` (the default) recomputes the diffusion steps in the backward pass. Without
it RDAAM's activations are `O(B · rdim · steps · log2 rdim)`, gigabytes at any useful batch;
with it the forward-and-backward costs about 6× the forward. On the 64-node grid the damage
sweep, which is not checkpointed, then dominates memory — `O(B · N²)`, 8 GiB at batch 512 and
`n_sub=8` — so size gradient batches in the hundreds. Pass `False` when no gradient is needed. The length distribution's `z ≤ 4 sd` and `L ≥ 2.15 µm` truncations are hard cutoffs
inherited from the C, so its derivative is that of the surviving terms. Gradients are finite in
float32 as well as float64.

**Devices, dtypes, batches.** The ports run in whatever dtype and on whatever device their
inputs are. float32 costs a median ≤5e-6 against float64 at `n_sub=8` (2e-5 at 32, where
rounding accumulates) and is the right choice on a GPU, which runs float64 at a fraction of the
rate. Both ports are a sequential loop over ~`63 · n_sub` internal nodes, so a call has a fixed
latency — 0.09 s for AFT and 1.1 s for RDAAM at `n_sub=8` on the GPU — and throughput only
arrives with batch: 16 384 AFT paths with lengths (131 072 for the age alone), and about
2 × 10⁶ / `rdim` helium grains; wider RDAAM batches get slower, not faster.

## 4. What sits on top

One line each; the module docstrings are the reference.

| Module | What it is |
|---|---|
| `chai.api`, `chai.inputs` | measurements in, posterior thermal histories out (the README); input files shared with the browser page |
| `chai.paths` | `fixed_grid`, `resample`, `refine`, `validate` (§1 above) |
| `chai.data.priors` | `PulsePrior`, the shipped family's prior ([family.md](family.md)); `ThermalHistoryPrior` and `EpisodicPrior`, two alternatives; `GrainPrior` |
| `chai.data.simulate` | the resumable, provenance-stamped corpus builder (`simulate`, `reproduce`), run by `scripts/simulate.py` |
| `chai.data.observe` | `ObservationModel`: draws laboratory data (NumPy) and scores it (`log_prob`, torch, differentiable) |
| `chai.data.aliquot` | `Aliquot`: one AFT analysis plus a masked, shuffled set of AHe grains |
| `chai.data.projection` | c-axis projection of confined track lengths |
| `chai.eval.closure` | `dodson_tc`, `effective_tc`: the external check against Reiners & Brandon (2006) |
| `chai.models`, `chai.train`, `chai.eval.calibration` | the whitened latent, the encoder and flow head ([architecture.md](architecture.md)), the chooser, training, and simulation-based calibration |
| `chai.export` | the family as ONNX graphs for the browser page, and the fixtures its port is checked against |
