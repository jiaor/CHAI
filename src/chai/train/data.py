"""The training data: one corpus split in RAM, observed afresh on every batch.

What a batch is
---------------
The corpus stores *truth*: a path, its hyperparameters, and the modelled
observables of one rock (an AFT age and length distribution, twelve helium
grains).  Nothing in it is data.  A batch is made by drawing one laboratory's
measurement of a block of rows with :meth:`chai.data.observe.ObservationModel.observe`
-- and, because the draw costs 41 ms per 4 096 rows on one core against a
corpus that took an hour to simulate, it is drawn in the loader's worker
processes and never stored.  A training run therefore sees ``p(x | theta)``,
not one sample of it, and re-drawing is what makes fifty epochs over
1.6 x 10^6 rows fifty different datasets.

Two things the plan says are not optional, both done here rather than in the
network:

**Laboratory effort is drawn per sample.**  ``ObservationModel`` measures every
sample with the same 20 grains and 100 lengths, and a network trained on that
reads ``age_sd`` and ``n_lengths`` as inputs it has only ever seen at one
value (``docs/observation.md`` open item 4).  :class:`EffortPrior` draws the
counted grains, the counted area, the number of confined lengths and the
number of dated helium grains for every row, over the ranges laboratories
actually report, and hands them to ``observe`` as its per-sample ``effort``.

**Observables drop out.**  Real samples arrive with no lengths, or with AFT
only, or as a handful of helium grains and nothing else.  The same
:class:`EffortPrior` makes an effort of *zero* a possibility -- no lengths
measured, no fission-track analysis, no grains dated -- so that a sample
missing an observable is an ordinary member of the training distribution
rather than something the network meets for the first time on real data.
Both are never absent at once: a sample with no data at all has the prior for
its posterior, which the head already knows in these coordinates.

**The walls can be dequantised** (:class:`WallDequantiser`, Stage 11 flow
head).  38% of stored node values sit exactly on a wall of the temperature
box, and no continuous density has an atom.  With a dequantiser the loader
pushes every wall node a random distance past its wall -- one offset per row
and wall, so a rock that sat at the surface for 30 Ma becomes a smooth path
just below 0 degC, plus a small per-node jitter -- and whitens *that*; the
head learns a continuous density whose mass beyond the wall is the atom's,
and ``Latent.inverse(z, clamp=True)`` projects it back onto the wall at
inference.  ``T`` in the batch is always the truth as stored.

**Shape comes from the file.**  ``n_grid``, ``t_max``, the number of stored
grains and the length bins are read from the corpus attributes, so the Stage
9a/9b corpora load through the same class with no edits here.
"""
from __future__ import annotations

import copy
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import BatchSampler, DataLoader, Dataset, RandomSampler, SequentialSampler

from chai.data.aliquot import Aliquot
from chai.data.observe import ObservationModel
from chai.models.latent import HYPER_NAMES, Latent

__all__ = [
    "AFT_INPUTS",
    "GRAIN_INPUTS",
    "CorpusSplit",
    "EffortPrior",
    "NoisyBatches",
    "WallDequantiser",
    "empty_features",
    "features_from_aliquot",
    "fixed_batches",
    "make_loader",
    "mix_splits",
]

#: Columns of the ``aft`` input, in order.  Logs where the quantity is positive
#: and spans decades; a flag beside every quantity that can be absent, and the
#: quantity itself zero where it is.  ``dpar_sd`` is left out: it is a constant
#: of the observation model today and would standardise to nothing.
AFT_INPUTS = ("log_age", "log_age_sd", "log1p_n_spontaneous", "has_age",
              "has_lengths", "log1p_n_lengths", "mean_length", "dpar", "has_aft")

#: Columns of the per-grain input, in order.  All positive, all logged.
GRAIN_INPUTS = ("log_age", "log_age_sd", "log_radius", "log_eU")

#: Which ``aft`` columns are 0/1 flags, and so are not standardised.
AFT_FLAGS = tuple(i for i, name in enumerate(AFT_INPUTS) if name.startswith("has_"))


# ---------------------------------------------------------------------------
# Laboratory effort, as a distribution
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class EffortPrior:
    """How much work a laboratory did on a sample, drawn per sample.

    The ranges are the ones a reader of the AFT and AHe literature meets in
    data tables; they are *assumptions about practice*, not physics, and they
    are recorded in the training config for that reason.  Every range is
    inclusive; the counted area is log-uniform because it spans a factor of
    several and its effect on the age's precision is through its logarithm.

    Args:
        n_grains_counted: apatite grains counted for the pooled age.
        area_per_grain_um2: area counted per grain, log-uniform.
        n_lengths: confined lengths measured, *given that any were*.
        n_he_grains: helium grains dated, *given that any were*.  ``None``
            means ``(1, G_max)`` with ``G_max`` read from the corpus.
        p_no_lengths: probability that a sample with an AFT analysis has no
            confined-length data (too few tracks, or nobody measured them).
        p_no_aft: probability that a sample has no fission-track analysis.
        p_no_he: probability that a sample has no helium grains.  When both
            the AFT and the helium draws come up absent, one of the two is
            restored at random, so no sample is empty.
    """

    n_grains_counted: tuple[int, int] = (10, 40)
    area_per_grain_um2: tuple[float, float] = (1000.0, 6000.0)
    n_lengths: tuple[int, int] = (20, 150)
    n_he_grains: tuple[int, int] | None = None
    p_no_lengths: float = 0.20
    p_no_aft: float = 0.15
    p_no_he: float = 0.15

    def __post_init__(self) -> None:
        for name in ("n_grains_counted", "area_per_grain_um2", "n_lengths", "n_he_grains"):
            r = getattr(self, name)
            if r is None:
                continue
            lo, hi = r
            if not lo <= hi:
                raise ValueError(f"{name} range must be ordered, got {r}")
            if lo < (0 if name == "n_lengths" else 1) and name != "area_per_grain_um2":
                raise ValueError(f"{name} lower bound too small: {lo}")
        if self.area_per_grain_um2[0] <= 0:
            raise ValueError("area_per_grain_um2 must be positive")
        for name in ("p_no_lengths", "p_no_aft", "p_no_he"):
            p = getattr(self, name)
            if not 0.0 <= p <= 1.0:
                raise ValueError(f"{name} must be a probability, got {p}")
        if self.p_no_aft >= 1.0 and self.p_no_he >= 1.0:
            raise ValueError("every sample would be empty")

    def draw(self, n: int, rng: np.random.Generator, g_max: int) -> dict[str, np.ndarray]:
        """Per-sample ``effort`` for :meth:`ObservationModel.observe`, ``n`` rows."""
        lo, hi = self.n_grains_counted
        counted = rng.integers(lo, hi + 1, size=n)
        a_lo, a_hi = self.area_per_grain_um2
        area = np.exp(rng.uniform(np.log(a_lo), np.log(a_hi), size=n))
        l_lo, l_hi = self.n_lengths
        n_len = rng.integers(l_lo, l_hi + 1, size=n)
        h_lo, h_hi = self.n_he_grains if self.n_he_grains is not None else (1, g_max)
        n_he = rng.integers(h_lo, min(h_hi, g_max) + 1, size=n)

        no_aft = rng.random(n) < self.p_no_aft
        no_he = rng.random(n) < self.p_no_he
        both = no_aft & no_he
        if both.any():
            restore_aft = rng.random(n) < 0.5
            no_aft &= ~(both & restore_aft)
            no_he &= ~(both & ~restore_aft)
        no_len = (rng.random(n) < self.p_no_lengths) | no_aft
        return {
            "n_grains_counted": counted,
            "area_per_grain_um2": area,
            "n_lengths": np.where(no_len, 0, n_len),
            "n_he_grains": np.where(no_he, 0, n_he),
            "aft_present": ~no_aft,
        }

    def to_config(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> EffortPrior:
        cfg = dict(cfg)
        for k in ("n_grains_counted", "area_per_grain_um2", "n_lengths", "n_he_grains"):
            if cfg.get(k) is not None:
                cfg[k] = tuple(cfg[k])
        return cls(**cfg)

    def describe(self) -> str:
        he = self.n_he_grains or "(1, G_max)"
        return "\n".join([
            "Laboratory effort, per sample",
            f"  grains counted      U{self.n_grains_counted}",
            f"  area per grain      logU{self.area_per_grain_um2} um^2",
            f"  confined lengths    U{self.n_lengths}, none with p = {self.p_no_lengths:g}",
            f"  helium grains dated U{he}, none with p = {self.p_no_he:g}",
            f"  no AFT analysis     p = {self.p_no_aft:g}  (never both absent)",
        ])


# ---------------------------------------------------------------------------
# One split, in RAM
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class WallDequantiser:
    """Make the stored truth a density: walls pushed past, every node given a floor.

    Args:
        width: the shared per-row offset is ``U(0, width)`` kelvin, past the wall.
        jitter: each wall node adds its own ``U(0, jitter)`` on top.
        noise_sd: Gaussian noise, kelvin, added to *every* node afterwards.  The
            prior's rate is a 64-feature GP, so for a long correlation length a
            64-node path is fixed by a few numbers and the joint ``(h, T)`` lies
            near a low-dimensional manifold; a density's NLL on it is unbounded
            below and a flow will chase it (``docs/flow-head.md``).  The floor
            says how well a stored path is known -- the simulator's own error
            is 0.17 K (``ThermalHistoryPrior``, ``n_steps``) -- and bounds the
            objective.  Zero turns it off.
    """

    width: float = 10.0
    jitter: float = 1.0
    noise_sd: float = 0.0

    def apply(self, T: np.ndarray, lo: np.ndarray, hi: np.ndarray, rng: np.random.Generator
              ) -> np.ndarray:
        """``T`` ``(n, n_grid)`` with its wall nodes moved outside ``[lo, hi]``; float64."""
        T = np.asarray(T)
        on_lo, on_hi = T <= lo, T >= hi
        n = T.shape[0]
        off_lo = self.width * rng.random((n, 1)) + self.jitter * rng.random(T.shape)
        off_hi = self.width * rng.random((n, 1)) + self.jitter * rng.random(T.shape)
        out = T.astype(np.float64)
        out = np.where(on_lo, lo - off_lo, out)
        out = np.where(on_hi, hi + off_hi, out)
        if self.noise_sd > 0:
            out = out + self.noise_sd * rng.standard_normal(T.shape)
        return out

    def to_config(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_config(cls, cfg: dict[str, Any] | None) -> WallDequantiser | None:
        return None if cfg is None else cls(**cfg)


class CorpusSplit:
    """The rows of one split of a corpus file, held in memory.

    Whole-column reads then a NumPy take, as :meth:`Latent.fit` does, because
    h5py's fancy indexing walks the selection row by row.  The train split of
    a 2 M-history corpus is 410 MB of paths, 640 MB of float16 length histograms
    and 230 MB of helium grains -- one read, a few seconds, and then the
    loader never touches the disk again.

    Args:
        path: the corpus file.
        split: ``"train"``, ``"val"`` or ``"test"``.
        latent: if given, ``z = latent(T, hyper)`` is computed for every row
            once, here, in float64, and stored in float32 as the training
            target.
        max_rows: the first rows of the split only, for quick looks and tests.
        first_row: skip this many rows of the split first -- Stage 12 scores
            calibration on val rows the checkpoint was not selected on.
        max_reheat: keep only the rows whose stored path reheats by at most
            this many kelvin (:func:`chai.constraints.reheating`), before
            ``first_row`` and ``max_rows`` apply: the corpus restricted to
            paths that only cool (``docs/cooling-coverage.md``).
    """

    def __init__(self, path: str | Path, split: str = "train", *,
                 latent: Latent | None = None, max_rows: int | None = None,
                 first_row: int = 0, max_reheat: float | None = None):
        self.path = Path(path)
        self.split = split
        with h5py.File(self.path, "r") as f:
            names = list(f.attrs["split_names"])
            if split not in names:
                raise ValueError(f"split {split!r} not in {names}")
            idx = np.flatnonzero(f["split"][:] == names.index(split))
            if max_reheat is not None:
                T_all = f["T"][:][idx]
                heat = (np.maximum.accumulate(T_all, axis=1) - T_all).max(axis=1)
                idx = idx[heat <= max_reheat]
            idx = idx[first_row:]
            if max_rows is not None:
                idx = idx[:max_rows]
            self.index = idx
            self.t_grid = f["t_grid"][:].astype(np.float64)
            self.length_bins = f["length_bins"][:].astype(np.float64)
            self.prior_config = json.loads(f.attrs["prior_config"])
            self.spec = json.loads(f.attrs["spec"])
            self.g_max = int(f.attrs.get("n_grains_total", f["he/corrected_age"].shape[1]))
            self.T = f["T"][:][idx]
            # A corpus from another prior (an episodic one, Stage 13) stores other
            # hyperparameters; its paths and observables are still readable, but
            # it cannot be whitened without the latent's own.
            self.hyper = {k: f[f"hyper/{k}"][:][idx] for k in HYPER_NAMES if f"hyper/{k}" in f}
            self.aft_age = f["aft/age"][:][idx]
            self.dpar = f["aft/dpar"][:][idx]
            self.length_pdf = f["aft/length_pdf"][:][idx]          # float16, as stored
            self.he_age = f["he/corrected_age"][:][idx]
            self.he_radius = f["he/radius"][:][idx]
            self.he_eU = f["he/eU"][:][idx]
            # The aliquot's uranium content is the mean over its grains, for
            # the reason ``truth_from_corpus`` gives.
            self.U = f["he/U"][:][idx].mean(axis=1, dtype=np.float64).astype(np.float32)
            self.meta = {"corpus": str(self.path), "split": split,
                         "corpus_git_sha": str(f.attrs.get("git_sha", "")),
                         "n_rows": int(idx.size), "first_row": int(first_row),
                         "max_reheat": max_reheat}
        if self.he_age.shape[1] != self.g_max:
            raise ValueError(f"he/corrected_age has {self.he_age.shape[1]} grains against "
                             f"n_grains_total = {self.g_max}")
        self.z: np.ndarray | None = None
        self._latent_cpu: Latent | None = None
        if latent is not None:
            missing = [k for k in latent.hyper_names if k not in self.hyper]
            if missing:
                raise ValueError(f"{self.path} stores no {missing}: it cannot be whitened by this latent")
            if latent.n_grid != self.n_grid:
                raise ValueError(f"latent has {latent.n_grid} nodes, corpus {self.n_grid}")
            self.z = self._whiten(latent)
            # A CPU copy for the loader's workers: the latent handed in may later
            # move to the GPU inside a model, and a forked worker cannot use it there.
            self._latent_cpu = copy.deepcopy(latent).to("cpu")

    def _whiten(self, latent: Latent, chunk: int = 200_000) -> np.ndarray:
        out = np.empty((len(self), latent.dim), dtype=np.float32)
        dev = latent.mean.device            # wherever the latent lives, e.g. inside a GPU model
        with torch.no_grad():
            for s in range(0, len(self), chunk):
                sl = slice(s, s + chunk)
                z = latent(torch.as_tensor(self.T[sl], dtype=torch.float64, device=dev),
                           {k: v[sl] for k, v in self.hyper.items()})
                out[sl] = z.to(torch.float32).cpu().numpy()
        return out

    def whiten(self, T: np.ndarray, rows: np.ndarray) -> Tensor:
        """``z`` for paths ``T`` standing in for ``rows`` (their hyperparameters), float32, CPU."""
        if self._latent_cpu is None:
            raise ValueError("this split was loaded without a latent")
        with torch.no_grad():
            z = self._latent_cpu(torch.as_tensor(np.asarray(T), dtype=torch.float64),
                                 {k: v[rows] for k, v in self.hyper.items()})
        return z.to(torch.float32)

    @property
    def present_input(self) -> bool:
        """Whether the latent this split was whitened by takes ``T_0`` as an input."""
        return self._latent_cpu is not None and self._latent_cpu.present_input

    @property
    def box(self) -> tuple[np.ndarray, np.ndarray]:
        """The temperature box per node, from the latent."""
        if self._latent_cpu is None:
            raise ValueError("this split was loaded without a latent")
        return self._latent_cpu.box_lo.numpy(), self._latent_cpu.box_hi.numpy()

    def __len__(self) -> int:
        return int(self.index.size)

    @property
    def n_grid(self) -> int:
        return int(self.t_grid.size)

    @property
    def t_max(self) -> float:
        return float(self.t_grid[-1])

    def truth(self, rows: np.ndarray) -> dict[str, np.ndarray]:
        """What :meth:`ObservationModel.observe` needs for some rows, in float64."""
        return {
            "aft_age": self.aft_age[rows].astype(np.float64),
            "length_pdf": self.length_pdf[rows].astype(np.float64),
            "dpar": self.dpar[rows].astype(np.float64),
            "U": self.U[rows].astype(np.float64),
            "he_age": self.he_age[rows].astype(np.float64),
            "he_radius": self.he_radius[rows].astype(np.float64),
            "he_eU": self.he_eU[rows].astype(np.float64),
        }

    def prior_sd(self) -> np.ndarray:
        """``(n_grid,)`` the prior's marginal sd per node, from this split."""
        return self.T.std(axis=0, dtype=np.float64)


# ---------------------------------------------------------------------------
# From an aliquot to the network's inputs
# ---------------------------------------------------------------------------

def _safe_log(x: np.ndarray, where: np.ndarray) -> np.ndarray:
    return np.log(np.where(where, x, 1.0), dtype=np.float64) * where


def features_from_aliquot(obs: Aliquot) -> dict[str, np.ndarray]:
    """The data vector as the encoder reads it, in raw (unstandardised) units.

    Returns ``aft`` ``(n, 9)`` per :data:`AFT_INPUTS`, ``length_pdf``
    ``(n, n_bins)`` as bin probabilities (zero rows where none were measured),
    ``grains`` ``(n, G, 4)`` per :data:`GRAIN_INPUTS` and ``grain_mask``
    ``(n, G)``.  Absent quantities are zero and flagged, never NaN: the flag
    is the input, the zero is just a number the network can ignore.
    """
    has_aft = obs.aft_mask
    has_age = obs.has_age
    n_len = obs.n_lengths
    has_len = has_aft & (n_len > 0)
    aft = np.stack([
        _safe_log(obs.aft_age, has_age),
        _safe_log(obs.aft_age_sd, has_aft),
        np.log1p(obs.n_spontaneous.astype(np.float64)) * has_aft,
        has_age.astype(np.float64),
        has_len.astype(np.float64),
        np.log1p(n_len.astype(np.float64)) * has_len,
        obs.mean_length * has_len,
        obs.dpar * has_aft,
        has_aft.astype(np.float64),
    ], axis=1)
    m = obs.he_mask
    grains = np.stack([
        _safe_log(obs.he_age, m),
        _safe_log(obs.he_age_sd, m),
        _safe_log(obs.he_radius, m),
        _safe_log(obs.he_eU, m),
    ], axis=2)
    return {
        "aft": aft.astype(np.float32),
        "length_pdf": obs.length_pdf.astype(np.float32),
        "grains": grains.astype(np.float32),
        "grain_mask": m,
    }


def mix_splits(base: CorpusSplit, extra: CorpusSplit, latent: Latent, rng: np.random.Generator) -> CorpusSplit:
    """``base`` with ``extra``'s rows appended: a corpus drawn from a mixture of two priors.

    ``extra`` may come from a prior with other hyperparameters (the episodic
    one, Stage 13).  The latent's hyperparameters it lacks are drawn from
    their own hyperprior, independently of the path, so the joint is the
    mixture ``(1 - m) p_GP(h, T) + m p_extra(T) p_0(h)`` -- a proper prior
    whose path marginal is the mixture of the two, which is what the head is
    being asked to learn.  What the head then says about ``tau`` and
    ``rate_sd`` is a statement under that mixture.  Row ``index`` values
    from ``extra`` are offset by ``2**40`` so a row can still be traced to
    its file; ``source`` is 0 for ``base`` and 1 for ``extra``.
    """
    for name in ("t_grid", "length_bins"):
        if not np.array_equal(getattr(base, name), getattr(extra, name)):
            raise ValueError(f"the two corpora differ in {name}")
    if base.g_max != extra.g_max:
        raise ValueError(f"the two corpora carry {base.g_max} and {extra.g_max} grains")
    n = len(extra)
    hyper = dict(extra.hyper)
    for spec in latent.hyper:
        if spec.name not in hyper:
            u = rng.random(n)
            hyper[spec.name] = (np.exp(np.log(spec.lo) + u * (np.log(spec.hi) - np.log(spec.lo))) if spec.log_axis
                                else spec.lo + u * (spec.hi - spec.lo)).astype(base.hyper[spec.name].dtype)
    out = copy.copy(base)
    for name in ("T", "aft_age", "dpar", "length_pdf", "he_age", "he_radius", "he_eU", "U"):
        setattr(out, name, np.concatenate([getattr(base, name), getattr(extra, name)]))
    out.hyper = {k: np.concatenate([base.hyper[k], hyper[k]]) for k in base.hyper}
    out.index = np.concatenate([base.index, extra.index + 2 ** 40])
    out.source = np.concatenate([np.zeros(len(base), np.int8), np.ones(n, np.int8)])
    out.meta = {**base.meta, "mixed_with": str(extra.path), "n_mixed": int(n), "n_rows": len(out.index)}
    if base.z is not None:
        out.z = out._whiten(latent)
    return out


def empty_features(n: int, n_bins: int, g_max: int) -> dict[str, np.ndarray]:
    """The encoder's input for ``n`` samples that measured nothing.

    No sample in the training corpus is empty (:class:`EffortPrior` restores
    one modality when both are dropped), so a conditional head has never seen
    this input and says nothing reliable about it.  A head trained *only* on
    it is unconditional -- its encoder emits one constant embedding -- and is
    how Stage 13 puts a density on the prior in the heads' own coordinates
    (``TrainConfig.unconditional``).
    """
    return {
        "aft": np.zeros((n, len(AFT_INPUTS)), dtype=np.float32),
        "length_pdf": np.zeros((n, n_bins), dtype=np.float32),
        "grains": np.zeros((n, max(g_max, 1), len(GRAIN_INPUTS)), dtype=np.float32),
        "grain_mask": np.zeros((n, max(g_max, 1)), dtype=bool),
    }


class NoisyBatches(Dataset):
    """Indexed by a *block of rows*, returns one observed, featurised batch.

    Used with a ``DataLoader(batch_size=None)`` over a :class:`BatchSampler`, so
    the observation model draws a whole block at once -- NumPy's per-row cost
    is what made 4 096 rows take 41 ms -- and the loader's workers each hold
    their own generator, seeded from the worker seed PyTorch hands them, so
    two workers never draw the same noise and a fixed loader seed reproduces
    the run.

    The batch carries the truth as well as the data: ``z`` (if the corpus was
    whitened), ``T`` and the corpus ``index``, so a consumer can score, report
    in degrees, and trace a row back to the file.  The drawn ``effort`` rides
    along for stratifying diagnostics by what was measured.
    """

    def __init__(self, corpus: CorpusSplit, observation: ObservationModel,
                 effort: EffortPrior | None, seed: int = 0,
                 dequantise: WallDequantiser | None = None, inputs: bool = True,
                 p_empty: float = 0.0):
        self.corpus = corpus
        self.inputs = inputs
        if not 0.0 <= p_empty < 1.0:
            raise ValueError(f"p_empty must be in [0, 1), got {p_empty}")
        self.p_empty = p_empty
        self.observation = observation
        self.effort = effort
        self.dequantise = dequantise
        self.seed = int(seed)
        self._rng: np.random.Generator | None = None

    def __len__(self) -> int:
        return len(self.corpus)

    def _generator(self) -> np.random.Generator:
        if self._rng is None:
            # In a worker ``torch.initial_seed()`` is that worker's seed for
            # this epoch; in the main process it is the global one.
            self._rng = np.random.default_rng([self.seed, torch.initial_seed() % 2**63])
        return self._rng

    def __getitem__(self, rows: Any) -> dict[str, Tensor]:
        rows = np.asarray(rows)
        rng = self._generator()
        return self.observe(rows, rng)

    def observe(self, rows: np.ndarray, rng: np.random.Generator) -> dict[str, Tensor]:
        """One draw for ``rows`` with a given generator (the deterministic path)."""
        return self.observe_aliquot(rows, rng)[1]

    def observe_aliquot(self, rows: np.ndarray, rng: np.random.Generator
                        ) -> tuple[Aliquot | None, dict[str, Tensor]]:
        """The draw of :meth:`observe`, and the :class:`Aliquot` it was made from.

        Same generator stream, same batch: the aliquot is what the exact
        likelihood scores (Stage 8) and the batch is what the network reads,
        and they are one measurement.
        """
        c = self.corpus
        if self.inputs:
            effort = self.effort.draw(rows.size, rng, c.g_max) if self.effort is not None else None
            obs = self.observation.observe(c.truth(rows), rng, effort=effort)
            feats = features_from_aliquot(obs)
            if self.p_empty > 0:
                # This share of rows is shown to the head as having measured
                # nothing, so one network carries the prior beside the
                # posterior and ``q(z | empty)`` shares its density errors.
                gone = rng.random(rows.size) < self.p_empty
                blank = empty_features(int(gone.sum()), c.length_bins.size, c.g_max)
                for k, v in blank.items():
                    feats[k][gone] = v
        else:                                          # the prior: nothing observed
            effort, obs = None, None
            feats = empty_features(rows.size, c.length_bins.size, c.g_max)
        out = {k: torch.from_numpy(np.ascontiguousarray(v)) for k, v in feats.items()}
        out["T"] = torch.from_numpy(c.T[rows])
        # The present-day node as stored: an input for a head whose latent takes it.
        out["T0"] = torch.from_numpy(np.ascontiguousarray(c.T[rows, 0], dtype=np.float32))
        out["index"] = torch.from_numpy(c.index[rows].astype(np.int64))
        if c.z is not None:
            if self.dequantise is None:
                out["z"] = torch.from_numpy(c.z[rows])
            else:
                lo, hi = c.box
                T = self.dequantise.apply(c.T[rows], lo, hi, rng)
                if c.present_input:
                    # T_0 is given exactly; the floor is on the excesses measured from it.
                    T[:, 0] = c.T[rows, 0]
                out["z"] = c.whiten(T, rows)
        if effort is not None:
            out["effort"] = {k: torch.from_numpy(np.asarray(v)) for k, v in effort.items()}
        return obs, out


_WORKER_LIMITS = None


def _worker_init(_: int) -> None:
    """One thread per worker.

    Each worker inherits NumPy's BLAS pool sized for the whole machine, and
    eight workers each running a 200 x 200 matmul on 112 threads spend their
    time contending: measured 43 ms per batch against 8 ms with the pools
    pinned to one thread.  ``threadpoolctl`` reaches the BLAS that NumPy was
    linked against; the object is kept alive so the limit is.
    """
    global _WORKER_LIMITS
    torch.set_num_threads(1)
    from threadpoolctl import threadpool_limits
    _WORKER_LIMITS = threadpool_limits(1)


def make_loader(dataset: NoisyBatches, batch_size: int, *, shuffle: bool = True,
                num_workers: int = 8, seed: int = 0, drop_last: bool = True,
                persistent: bool = True) -> DataLoader:
    """A loader that yields one featurised batch per step, noise drawn in the workers."""
    n = len(dataset)
    if shuffle:
        g = torch.Generator().manual_seed(seed)
        sampler = BatchSampler(RandomSampler(range(n), generator=g), batch_size, drop_last)
    else:
        sampler = BatchSampler(SequentialSampler(range(n)), batch_size, drop_last)
    return DataLoader(dataset, batch_size=None, sampler=sampler, num_workers=num_workers,
                      pin_memory=torch.cuda.is_available(),
                      persistent_workers=persistent and num_workers > 0,
                      worker_init_fn=_worker_init,
                      generator=torch.Generator().manual_seed(seed))


def fixed_batches(dataset: NoisyBatches, batch_size: int, *, seed: int,
                  max_rows: int | None = None) -> list[dict[str, Tensor]]:
    """One deterministic observation of the first ``max_rows`` rows, in batches.

    For validation: the same noisy dataset every time it is scored, so a
    change in the number is a change in the model.
    """
    n = len(dataset) if max_rows is None else min(max_rows, len(dataset))
    rng = np.random.default_rng(seed)
    return [dataset.observe(np.arange(s, min(s + batch_size, n)), rng)
            for s in range(0, n, batch_size)]
