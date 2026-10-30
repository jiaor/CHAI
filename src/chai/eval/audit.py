"""Audit the simulation corpus.

Stage 1's acceptance criterion is not "the file exists" -- it is that the file
covers observable space with no dead regions, and that the observables in it
actually say something about the history that produced them.  Those are two
different questions and this module answers them separately:

**Coverage** (:func:`marginals`, :func:`coverage`) reads the corpus as it
stands.  It asks whether the prior-predictive distribution reaches everywhere a
real sample might land, and where it piles up.  A dead region is a place where
a Goal 2 network would be extrapolating without ever being told so.

**Degeneracy** (:func:`conditional_spread`) asks the inverse question: given
observables, how much of the prior's spread in ``T(t)`` survives?  That is the
ceiling on what any inversion -- amortized, MCMC or HeFTy -- can resolve, and
it is worth knowing before building one.

**Sensitivity** (:func:`temperature_jacobian`) is the derivative-based dual of
the degeneracy check.  Where ``d(observable)/d(T_i)`` is flat, the data cannot
see node ``i``, and the degeneracy check should show the prior spread surviving
there.  Two different methods agreeing is the point.

Why the degeneracy and sensitivity work re-runs the forward models instead of
reading the corpus columns
--------------------------------------------------------------------------
Every row of the corpus has its *own* grain: radius, U, Th and Dpar are drawn
per history.  So two rows with the same AFT age may differ because their
histories differ or because their Dpar differs, and conditioning on the column
alone confounds the two.  Grain properties are nuisance parameters that a real
measurement *knows* -- you measure your grain's radius and eU -- so the
question that matters is identifiability of ``T(t)`` at fixed grain.

Holding the grain fixed and re-running the (validated, Stage 2-3) torch ports
over the corpus's own ``T`` paths answers exactly that, and costs seconds on a
GPU.  The paths are still the corpus's paths, so this audits the prior that was
actually shipped.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Self

import h5py
import numpy as np
import torch

from chai.forward.torchimpl.ketcham import aft_forward, length_bin_centres
from chai.forward.torchimpl.rdaam import rdaam_age

if TYPE_CHECKING:  # the audit does not need the observation model to run
    from chai.data.observe import ObservationModel

__all__ = [
    "DT_INVISIBLE",
    "REFERENCE_GRAINS",
    "Corpus",
    "Grain",
    "Tolerance",
    "aliquot_pool",
    "axis_edges",
    "conditional_spread",
    "coverage",
    "forward_pool",
    "marginals",
    "measured_tolerance",
    "mismatch_to",
    "temperature_jacobian",
    "tolerance_at",
]


# ---------------------------------------------------------------------------
# Reading the corpus
# ---------------------------------------------------------------------------

#: Observable columns the audit treats as "the data", with display units.
OBSERVABLES = {
    "aft/age": ("AFT age", "Ma"),
    "aft/mean_length": ("mean track length", "um"),
    "he/corrected_age": ("AHe corrected age", "Ma"),
}

#: Columns whose occupancy is read on a log axis.  eU runs 1 to 500 ppm with a
#: median of 18, so 44% of grains fall in the first fortieth of its linear
#: range: linear bins would leave most of its columns nearly empty and report
#: that emptiness as holes in the corpus.  The rest are within a factor of a
#: few of uniform and bin linearly.
LOG_AXES = frozenset({"he/eU"})

#: Per-row covariates a real measurement also knows.
COVARIATES = {
    "aft/dpar": ("Dpar", "um"),
    "he/eU": ("eU", "ppm"),
    "he/radius": ("grain radius", "um"),
}


@dataclass
class Corpus:
    """A read-only view on the HDF5 corpus, with the split masks resolved.

    Columns are read on demand rather than up front: ``T`` alone is 256 MB and
    ``aft/length_pdf`` is 400 MB, and most of the audit touches neither in full.
    """

    path: str
    _f: h5py.File = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._f = h5py.File(self.path, "r")
        if not self._f["block_done"][:].all():
            n = int((~self._f["block_done"][:]).sum())
            raise ValueError(f"{self.path} has {n} unfinished blocks; audit a complete corpus")

    def __len__(self) -> int:
        return int(self._f["valid"].shape[0])

    @property
    def attrs(self) -> dict[str, Any]:
        return dict(self._f.attrs)

    @property
    def t_grid(self) -> np.ndarray:
        return self._f["t_grid"][:]

    @property
    def length_bins(self) -> np.ndarray:
        return self._f["length_bins"][:]

    def split_index(self, split: str) -> np.ndarray:
        """Row indices of one split, ascending (h5py needs sorted fancy indices)."""
        names = list(self._f.attrs["split_names"])
        code = names.index(split)
        return np.flatnonzero(self._f["split"][:] == code)

    def column(self, name: str, index: np.ndarray | slice | None = None) -> np.ndarray:
        """One column, as float64 -- ``length_pdf`` is stored float16 and must widen."""
        ds = self._f[name]
        raw = ds[:] if index is None else ds[index]
        return np.asarray(raw, dtype=np.float64)

    def paths(self, index: np.ndarray) -> np.ndarray:
        """``(n, n_grid)`` temperature histories for the given rows."""
        return self.column("T", index)

    def close(self) -> None:
        self._f.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def _const(v: float, b: int, dtype: torch.dtype, dev: torch.device) -> torch.Tensor:
    """``(b,)`` filled with ``v`` -- the grain scalars the ports want per row."""
    return torch.full((b,), v, dtype=dtype, device=dev)


def _flat(x: np.ndarray) -> np.ndarray:
    """Per-grain columns are ``(n, 4)``; flatten so every grain is one sample."""
    return x.reshape(-1) if x.ndim > 1 else x


# ---------------------------------------------------------------------------
# 1. Marginals
# ---------------------------------------------------------------------------

_QUANTILES = (0.0, 0.05, 0.25, 0.50, 0.75, 0.95, 1.0)


def marginals(corpus: Corpus, index: np.ndarray,
              quantiles: tuple[float, ...] = _QUANTILES) -> dict[str, dict[str, Any]]:
    """Quantile summary of every observable and covariate over ``index``.

    Per-grain columns are flattened, so the AHe rows describe grains rather
    than histories -- there are four times as many of them.
    """
    out: dict[str, dict[str, Any]] = {}
    for name, (label, unit) in {**OBSERVABLES, **COVARIATES}.items():
        v = _flat(corpus.column(name, index))
        out[name] = {
            "label": label,
            "unit": unit,
            "n": int(v.size),
            "quantiles": dict(zip(quantiles, np.quantile(v, quantiles))),
            "mean": float(v.mean()),
        }
    return out


# ---------------------------------------------------------------------------
# 2. Coverage
# ---------------------------------------------------------------------------

def axis_edges(v: np.ndarray, name: str, bins: int) -> np.ndarray:
    """Bin edges for one axis, log-spaced for the columns that need it."""
    lo, hi = float(v.min()), float(v.max())
    if name in LOG_AXES:
        return np.geomspace(max(lo, 1e-6), hi, bins + 1)
    return np.linspace(lo, hi, bins + 1)


def coverage(x: np.ndarray, y: np.ndarray, bins: Any = 40, min_count: int = 30,
             tail: float = 0.005) -> dict[str, Any]:
    """Occupancy of the ``(x, y)`` plane, and where it is too thin to train on.

    ``bins`` is anything :func:`numpy.histogram2d` accepts -- a count, or a
    pair of explicit edge arrays from :func:`axis_edges`.

    "Dead region" needs a definition that is not just "zero counts", because
    the corner of a rectangle over two correlated variables is empty for a
    perfectly good reason -- no history produces it, and no measurement will
    land there either.  What matters for Goal 2 is a cell that is *reachable*
    yet under-sampled: the network will be asked for an answer there and will
    have seen almost no examples.

    Reachable is the **central ``1 - 2*tail`` of each column**, not the span
    between its lowest and highest occupied cell.  The min/max rule is the
    obvious one and it is wrong: the extreme cells of any distribution are its
    sparsest, so they are flagged as holes in every dataset, and a rule that
    fires everywhere reports nothing.  Clipping the tails leaves the cells that
    are genuinely enclosed by data, which is where an unexplained hole means
    something.
    """
    H, xe, ye = np.histogram2d(x, y, bins=bins)
    interior = np.zeros_like(H, dtype=bool)
    for i in range(H.shape[0]):
        total = H[i].sum()
        if total == 0:
            continue
        cdf = np.cumsum(H[i]) / total
        lo = int(np.searchsorted(cdf, tail, side="left"))
        hi = int(np.searchsorted(cdf, 1.0 - tail, side="left"))
        interior[i, lo:hi + 1] = True

    thin = interior & (H < min_count)
    return {
        "counts": H,
        "x_edges": xe,
        "y_edges": ye,
        "interior": interior,
        "thin": thin,
        "n_interior": int(interior.sum()),
        "n_thin": int(thin.sum()),
        "thin_fraction": float(thin.sum() / max(interior.sum(), 1)),
        "min_interior_count": int(H[interior].min()) if interior.any() else 0,
    }


# ---------------------------------------------------------------------------
# 3. Forward-modelling a pool at fixed grain
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Grain:
    """One apatite grain, held fixed across a pool of histories."""

    name: str
    radius: float      # um
    U: float           # ppm
    Th: float          # ppm
    dpar: float        # um

    @property
    def eU(self) -> float:
        """Effective uranium, the usual ``U + 0.235 Th`` proxy for damage dose."""
        return self.U + 0.235 * self.Th


#: Grains spanning the corners of the damage/diffusion-domain space, so the
#: degeneracy result is not an artifact of one convenient grain.  A low-eU
#: small grain is the most retentive-poor end and a high-eU large grain the
#: most damaged; the middle one is close to the corpus median.
REFERENCE_GRAINS = (
    Grain("low eU, small", radius=40.0, U=8.0, Th=20.0, dpar=1.75),
    Grain("median", radius=60.0, U=30.0, Th=60.0, dpar=1.75),
    Grain("high eU, large", radius=95.0, U=120.0, Th=200.0, dpar=2.40),
)


def forward_pool(T: np.ndarray, t_grid: np.ndarray, grain: Grain,
                 device: str = "cuda", n_sub_aft: int = 64, n_sub_he: int = 16,
                 rdim: int = 512, batch: int = 4096,
                 with_he: bool = True) -> dict[str, np.ndarray]:
    """Every observable for ``T`` at one fixed grain, batched over the GPU.

    float32 throughout: Stages 2 and 3 measured the single-precision penalty at
    a median 6.3e-7 (AFT) and 4.5e-6 (RDAAM) relative, far below the tolerances
    the degeneracy check then applies.

    ``n_sub_he`` is the port's default rather than the validation setting --
    RDAAM's damage sweep is ``O((N n_sub)^2)``, and at ``n_sub=16`` a 64-node
    path already carries 1024 steps.
    """
    dev = torch.device(device)
    dtype = torch.float32
    t = torch.as_tensor(t_grid, dtype=dtype, device=dev)
    bins = length_bin_centres(dtype=dtype, device=dev)

    chunks: list[dict[str, np.ndarray]] = []
    for lo in range(0, T.shape[0], batch):
        Tb = torch.as_tensor(T[lo:lo + batch], dtype=dtype, device=dev)
        b = Tb.shape[0]
        tb = t.expand(b, -1)
        with torch.no_grad():
            aft = aft_forward(tb, Tb, torch.full((b,), grain.dpar, dtype=dtype, device=dev),
                              n_sub=n_sub_aft, bins=bins)
            row = {
                "aft_age": aft["age"],
                "mean_length": aft["mean_length"],
                "length_pdf": aft["length_pdf"],
            }
            if with_he:
                he = rdaam_age(tb, Tb, _const(grain.radius, b, dtype, dev),
                               _const(grain.U, b, dtype, dev),
                               _const(grain.Th, b, dtype, dev),
                               n_sub=n_sub_he, rdim=rdim, use_checkpoint=False)
                row["he_age"] = he["corrected_age"]
        chunks.append({k: v.double().cpu().numpy() for k, v in row.items()})

    return {k: np.concatenate([c[k] for c in chunks]) for k in chunks[0]}


def aliquot_pool(T: np.ndarray, t_grid: np.ndarray, model: ObservationModel,
                 grains: tuple[Grain, ...] = REFERENCE_GRAINS, device: str = "cuda",
                 n_sub_he: int = 16, verbose: bool = False) -> dict[str, np.ndarray]:
    """One multi-grain aliquot per history: AFT from one grain, helium from all.

    The AFT observables come from the **median-eU grain alone** -- a rock has
    one Dpar and one pooled fission-track age, whichever grains were dated for
    helium -- and the helium columns are ordered by eU, so ``he_age_0``,
    ``he_age_1``, ... is a low-to-high eU suite of the *same* history.  That is
    the aliquot :mod:`chai.data.aliquot` holds and PLAN.md Stage 5 asks for,
    assembled from the corpus's own paths.

    The length distribution comes back **smeared by the measurement error**,
    because the object a laboratory hands over is the smeared one.
    :meth:`~chai.data.observe.ObservationModel.smeared` returns bin
    probabilities where the torch port returns a density; :func:`_pool_tensors`
    normalises either, so the two conventions meet without a factor of ``dx``
    between them.

    Written for ``scripts/observation_study.py`` and shared with
    ``scripts/prior_sensitivity.py``, which conditions on the same aliquot
    under a family of priors and must not be comparing a different one.
    """
    ordered = sorted(grains, key=lambda g: g.eU)
    mid = len(ordered) // 2
    pool: dict[str, np.ndarray] = {}
    for i, g in enumerate(ordered):
        t0 = time.perf_counter()
        p = forward_pool(T, t_grid, g, device=device, n_sub_he=n_sub_he)
        pool[f"he_age_{i}"] = p["he_age"]
        if i == mid:
            pool["aft_age"] = p["aft_age"]
            pool["mean_length"] = p["mean_length"]
            pool["length_pdf"] = model.smeared(p["length_pdf"])
        if verbose:
            print(f"  grain {i} ({g.name}, eU={g.eU:.0f}) forward "
                  f"{time.perf_counter() - t0:.1f}s", flush=True)
    return pool


def measured_tolerance(pool: dict[str, np.ndarray], model: ObservationModel,
                       U_ppm: float) -> Tolerance:
    """Per-row sigmas from the observation model, in place of the stand-ins.

    ``U_ppm`` is the aliquot's uranium content, taken as the reference grain's:
    the degeneracy check is at fixed grain by construction (see the module
    docstring), and the counting precision has to be evaluated at *some*
    uranium content.
    """
    from chai.data.observe import length_w1_sigma, mean_length_sigma

    bins = np.asarray(model.length_bins)
    n_l = float(model.aft.n_lengths)
    return Tolerance(
        aft_age_rel=model.aft.age_rel_sd(pool["aft_age"], U_ppm),
        mean_length_um=mean_length_sigma(pool["length_pdf"], n_l, bins),
        length_w1_um=length_w1_sigma(pool["length_pdf"], n_l, model.bin_width),
        he_age_rel=np.full(np.shape(pool["aft_age"]), model.he.rel_sd),
    )


# ---------------------------------------------------------------------------
# 4. Degeneracy
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Tolerance:
    """Analytical uncertainty, one sigma, per observable.

    The defaults are a **stand-in**: typical reported uncertainties, flat
    across the corpus, which is what the Stage 1 audit ran on and said so.
    Stage 5 replaces them -- :mod:`chai.data.observe` derives the same four
    numbers from counting statistics and single-grain reproducibility, and
    finds them neither flat nor centred on these values (``docs/observation.md``).

    Every field may therefore be **either a scalar or an array over the pool
    rows**, so a measured, per-sample sigma can be passed straight in.  Array
    fields are indexed by the query row inside :func:`conditional_spread` and
    must align with ``T_pool``; in :func:`temperature_jacobian` they must align
    with the paths being differentiated.

    These set the scale on which two measurements count as "the same", so they
    decide how hard the degeneracy check squeezes: every number it reports is
    conditional on them, and halving them tightens the posterior.
    """

    aft_age_rel: float | np.ndarray = 0.05     # pooled AFT age, counting statistics
    mean_length_um: float | np.ndarray = 0.10  # mean c-axis-projected track length
    length_w1_um: float | np.ndarray = 0.15    # transport distance between length pdfs
    he_age_rel: float | np.ndarray = 0.06      # AHe single-grain reproducibility


#: The conditioning ladder: what a sample tells you, one observable at a time.
#: Each rung adds to the one before, so the shrinkage is cumulative.
LADDER = (
    ("AFT age", ("aft_age",)),
    ("+ mean length", ("aft_age", "mean_length")),
    ("+ length pdf", ("aft_age", "mean_length", "length_pdf")),
    ("+ 1 AHe age", ("aft_age", "mean_length", "length_pdf", "he_age")),
)


def _pool_tensors(pool: dict[str, np.ndarray], bin_width: float,
                  dev: torch.device) -> dict[str, torch.Tensor]:
    """Move the pool to the device, with the length CDFs formed once.

    The transport distance between two length distributions is
    ``|CDF_p - CDF_q|`` integrated, so the CDFs are what the comparison
    actually needs.  Building them per query would repeat a 200-bin cumsum over
    the whole pool for every one of them, and that single change is the
    difference between seconds and an hour.
    """
    out = {k: torch.as_tensor(v, dtype=torch.float32, device=dev)
           for k, v in pool.items() if k != "length_pdf"}
    if "length_pdf" in pool:
        pdf = torch.as_tensor(pool["length_pdf"], dtype=torch.float32, device=dev)
        # Normalised by its own sum rather than multiplied by ``bin_width``,
        # so that either convention in circulation gives the same CDF: the
        # torch port returns a density (``sum * dx = 1``) and the corpus
        # stores bin probabilities (``sum = 1``).  Multiplying by ``dx``
        # is correct for the first and silently 10x wrong for the second.
        out["length_cdf"] = torch.cumsum(pdf, dim=1) / pdf.sum(dim=1, keepdim=True)
    return out


def _at(field: float | np.ndarray, row: int) -> float:
    """A tolerance field at one query row: scalars pass through, arrays index."""
    return float(field[row]) if isinstance(field, np.ndarray) and field.ndim else float(field)


def tolerance_at(tol: Tolerance, row: int) -> Tolerance:
    """``tol`` with every field resolved to the scalar that applies at ``row``."""
    return Tolerance(*(_at(getattr(tol, f), row) for f in
                       ("aft_age_rel", "mean_length_um", "length_w1_um", "he_age_rel")))


def _query_values(cols: dict[str, torch.Tensor], keys: tuple[str, ...],
                  row: int) -> dict[str, torch.Tensor]:
    """The observables of pool member ``row``, in the form a query takes."""
    q = {k: cols[k][row] for k in keys if k != "length_pdf"}
    if "length_pdf" in keys:
        q["length_cdf"] = cols["length_cdf"][row]
    return q


def _mismatch(cols: dict[str, torch.Tensor], row: int, keys: tuple[str, ...],
              tol: Tolerance, bin_width: float) -> torch.Tensor:
    """Sum of squared standardised residuals against pool member ``row``.

    The query is a member of the pool, which is what
    :func:`conditional_spread` needs: every query then has a guaranteed perfect
    match and the weights can never all vanish.  :func:`mismatch_to` is the
    same distance against a query given as values, which is what a query drawn
    from somewhere else -- a held-out split, or another prior -- needs.
    """
    return mismatch_to(cols, _query_values(cols, keys, row), keys,
                       tolerance_at(tol, row), bin_width)


def mismatch_to(cols: dict[str, torch.Tensor], query: dict[str, torch.Tensor],
                keys: tuple[str, ...], tol: Tolerance,
                bin_width: float) -> torch.Tensor:
    """Sum of squared standardised residuals of every pool member against ``query``.

    ``query`` holds the query's own observables -- ``length_pdf`` arrives as
    ``length_cdf``, already cumulated -- and ``tol`` must be scalar-valued,
    because a per-sample sigma belongs to the *measurement* being scored and
    there is only one of those here.

    Any key named ``he_age...`` is a helium grain and is scaled by
    ``he_age_rel``, so a multi-grain aliquot conditions on ``he_age_0``,
    ``he_age_1``, ... -- one term per grain, which is what makes the eU-age
    *dispersion* visible to the check rather than just one more age.

    > **Fixed 2026-09-20 (Stage 5).**  The length-pdf term dropped the bin
    > width: ``sum |F - F'|`` is a transport distance divided by ``dx``, so
    > with 0.1 um bins every length comparison ran **ten times tighter** than
    > the tolerance it was given, and the Stage 1 audit's ``+ length pdf`` rung
    > was effectively squeezing to 0.015 um rather than 0.15.  The sibling
    > implementation in :func:`chai.eval.study.w1_length` had the width and
    > the two disagreed, which is how it surfaced.  Corrected numbers are in
    > ``docs/observation.md``; the affected rungs widen by roughly a factor of
    > two in posterior sd.
    """
    z2 = torch.zeros(cols["aft_age"].shape[0], device=cols["aft_age"].device)
    for k in keys:
        if k == "length_pdf":
            c = cols["length_cdf"]
            d = (c - query["length_cdf"]).abs().sum(dim=1) * bin_width
            z2 += (d / float(tol.length_w1_um)) ** 2
            continue
        v = cols[k]
        if k.startswith("he_age"):
            scale = float(tol.he_age_rel) * query[k]
        elif k == "aft_age":
            scale = float(tol.aft_age_rel) * query[k]
        elif k == "mean_length":
            scale = float(tol.mean_length_um)
        else:
            raise KeyError(f"no tolerance defined for observable {k!r}")
        z2 += ((v - query[k]) / scale) ** 2
    return z2


def conditional_spread(T_pool: np.ndarray, pool: dict[str, np.ndarray],
                       query_rows: np.ndarray, tol: Tolerance,
                       bin_width: float, ladder: tuple = LADDER,
                       device: str = "cuda") -> dict[str, Any]:
    """How much of the prior's spread in ``T(t)`` survives conditioning.

    Rejection sampling would answer this by keeping only the pool members
    inside tolerance, but the count collapses as rungs are added and the last
    rung returns nothing.  Kernel weighting degrades gracefully instead: each
    pool member gets ``w = exp(-z^2/2)`` for the standardised mismatch ``z``,
    which is the same acceptance rule with a soft edge.  The **effective sample
    size** ``(sum w)^2 / sum w^2`` reports how much pool the weights actually
    rest on, and a result standing on an ESS of 5 is noise and is labelled so.

    The queries are drawn from the pool itself, so every query is guaranteed a
    perfect match (itself) and the weights can never all be zero.

    Returns per-rung arrays over the ``n_grid`` nodes:
      ``ratio``   -- weighted sd of ``T_i`` / prior sd of ``T_i``, median over
                     queries.  1 means the data said nothing about node ``i``;
                     0 means it pinned it.
      ``ess``     -- effective sample size per query.
      ``n_hard``  -- how many pool members sit within a mean 1 sigma across the
                     conditioned observables, for interpretation.
    """
    dev = torch.device(device)
    Tp = torch.as_tensor(T_pool, dtype=torch.float32, device=dev)
    prior_sd = Tp.std(dim=0, unbiased=False)
    cols = _pool_tensors(pool, bin_width, dev)
    results: dict[str, Any] = {"prior_sd": prior_sd.cpu().numpy(), "rungs": []}

    for label, keys in ladder:
        ratios = torch.empty((query_rows.size, Tp.shape[1]), device=dev)
        ess = torch.empty(query_rows.size, device=dev)
        n_hard = torch.empty(query_rows.size, device=dev)

        for j, r in enumerate(query_rows):
            z2 = _mismatch(cols, int(r), keys, tol, bin_width)
            w = torch.exp(-0.5 * (z2 - z2.min()))
            sw = w.sum()
            ess[j] = sw * sw / (w * w).sum()
            n_hard[j] = (z2 <= len(keys)).sum()

            mean = (w[:, None] * Tp).sum(dim=0) / sw
            var = (w[:, None] * (Tp - mean) ** 2).sum(dim=0) / sw
            ratios[j] = var.sqrt() / prior_sd

        r_np = ratios.cpu().numpy()
        results["rungs"].append({
            "label": label,
            "keys": keys,
            "ratio": np.median(r_np, axis=0),
            "ratio_p25": np.quantile(r_np, 0.25, axis=0),
            "ratio_p75": np.quantile(r_np, 0.75, axis=0),
            "ess": ess.cpu().numpy(),
            "n_hard": n_hard.cpu().numpy(),
        })
    return results


# ---------------------------------------------------------------------------
# 5. Sensitivity
# ---------------------------------------------------------------------------

#: Ceiling on the "kelvin per sigma" sensitivity, in K.  Two orders above the
#: 200 K prior box, so reaching it means the node is invisible to that
#: observable rather than merely insensitive.
DT_INVISIBLE = 2e4


def temperature_jacobian(T: np.ndarray, t_grid: np.ndarray, grain: Grain,
                         tol: Tolerance, device: str = "cuda",
                         n_sub_aft: int = 64, n_sub_he: int = 16,
                         rdim: int = 512, batch: int = 256) -> dict[str, np.ndarray]:
    """``d(observable)/d(T_i)``, expressed as how far ``T_i`` must move to matter.

    The raw derivative is in mixed units (Ma/K, um/K) and is not comparable
    across observables.  Dividing the analytical uncertainty by it gives a
    single interpretable number per node:

        ``dT_i`` = sigma_obs / |d(obs)/d(T_i)|   in kelvin

    the perturbation to node ``i`` alone that moves the observable by one
    reported sigma.  Small means the data sees that node; a value larger than
    the prior's own spread at that node means it does not, and the degeneracy
    check should show the prior surviving there.

    float64, because the whole quantity is a derivative and there is no batch
    pressure here to justify anything else.
    """
    dev = torch.device(device)
    dtype = torch.float64
    t = torch.as_tensor(t_grid, dtype=dtype, device=dev)
    bins = length_bin_centres(dtype=dtype, device=dev)
    names = ("aft_age", "mean_length", "he_age")
    grads: dict[str, list[np.ndarray]] = {k: [] for k in names}
    values: dict[str, list[np.ndarray]] = {k: [] for k in names}

    for lo in range(0, T.shape[0], batch):
        Tb = torch.as_tensor(T[lo:lo + batch], dtype=dtype, device=dev).requires_grad_(True)
        b = Tb.shape[0]
        tb = t.expand(b, -1)

        aft = aft_forward(tb, Tb, _const(grain.dpar, b, dtype, dev),
                          n_sub=n_sub_aft, bins=bins)
        he = rdaam_age(tb, Tb, _const(grain.radius, b, dtype, dev),
                       _const(grain.U, b, dtype, dev), _const(grain.Th, b, dtype, dev),
                       n_sub=n_sub_he, rdim=rdim)
        outputs = {"aft_age": aft["age"], "mean_length": aft["mean_length"],
                   "he_age": he["corrected_age"]}

        # Each output is (B,) and independent across the batch, so summing
        # before backward gives every row its own gradient in one pass.
        for i, (k, out) in enumerate(outputs.items()):
            g, = torch.autograd.grad(out.sum(), Tb, retain_graph=i < len(outputs) - 1)
            grads[k].append(g.detach().cpu().numpy())
            values[k].append(out.detach().cpu().numpy())

    sigma = {
        "aft_age": lambda v: tol.aft_age_rel * v,
        "mean_length": lambda v: np.full_like(v, tol.mean_length_um),
        "he_age": lambda v: tol.he_age_rel * v,
    }
    out: dict[str, np.ndarray] = {}
    for k in names:
        J = np.concatenate(grads[k])                       # (n, n_grid)
        v = np.concatenate(values[k])                      # (n,)
        out[f"{k}_jacobian"] = J
        # The observable itself, so a caller aggregating the Jacobian over a
        # window can form that path's own sigma rather than a pooled stand-in.
        out[f"{k}_value"] = v
        with np.errstate(divide="ignore", invalid="ignore"):
            dT = sigma[k](v)[:, None] / np.abs(J)
        # Where the gradient underflows the ratio runs to infinity, which is
        # the right answer -- the observable cannot see that node at all -- but
        # it is not a number anything downstream can plot or average.  The
        # prior box is 200 K wide, so anything past DT_INVISIBLE means the same
        # thing as infinity and says so without overflowing.
        out[f"{k}_dT"] = np.minimum(np.nan_to_num(dT, nan=DT_INVISIBLE,
                                                  posinf=DT_INVISIBLE), DT_INVISIBLE)
    return out
