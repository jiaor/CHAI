"""The sample container.

A *sample*, in this project, is what one rock yields: one apatite
fission-track observation -- a pooled age, a set of measured confined lengths,
a Dpar -- together with a handful of single-grain (U-Th)/He ages, each with its
own radius and eU.  Stage 5 asks for that object to exist as a type, and asks
for it to be **permutation-invariant** in the helium grains, because the grains
arrive as a *set*: nothing in the geology distinguishes "grain 1" from
"grain 3", and a model that can tell them apart is reading an artifact of
alphabetical ordering in a spreadsheet.

Two consequences shape this module.

**The grain axis is masked, not padded silently.**  Real aliquots report two
grains or ten.  A fixed ``(n, G)`` array with a companion ``(n, G)`` boolean is
the representation every set encoder wants, and it is the one that makes
"how many grains" an explicit input rather than a number recoverable only by
looking for zeros.

**Grain order is randomised at construction.**  A consumer that is invariant
cannot be harmed by it, and a consumer that is *not* invariant is a bug that
the randomisation turns from silent into visible: results that wobble between
runs get investigated, results that are quietly wrong do not.
:meth:`Aliquot.permute_grains` is the same operation, exposed, for tests that
want to assert invariance directly.

The AFT side is deliberately *not* a set.  The pooled age is one number for the
aliquot by construction, and the lengths are already reduced to a histogram on
the oracle's own 200 bins -- carrying 100 individual lengths instead would be a
different object with no more information in it, since the likelihood only ever
sees the counts (:meth:`chai.data.observe.ObservationModel.log_prob`).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Any

import numpy as np

__all__ = ["AFT_FEATURES", "GRAIN_FEATURES", "Aliquot"]

#: Column order of :meth:`Aliquot.aft_features`.  ``mean_length`` is a function
#: of ``length_counts`` and is carried anyway: it costs one column and it is the
#: number every geologist reads first.
AFT_FEATURES = ("age", "age_sd", "n_spontaneous", "n_lengths", "mean_length", "dpar", "dpar_sd")

#: Column order of :meth:`Aliquot.grain_features`, one row per helium grain.
GRAIN_FEATURES = ("age", "age_sd", "radius", "eU")


@dataclass(frozen=True)
class Aliquot:
    """``n`` samples, each one AFT observation plus a set of AHe grains.

    Batched rather than single, because everything upstream is: the corpus is
    read in blocks, the observation model draws noise for a block at a time,
    and Goal 2 trains on batches.  A single sample is ``aliquot[i]``, which is
    an :class:`Aliquot` of length 1 rather than a different type.

    Attributes:
        aft_age: ``(n,)`` measured pooled fission-track age, Ma.
        aft_age_sd: ``(n,)`` one sigma on it, Ma -- **counting statistics plus
            the systematic calibration terms**, i.e. the number a likelihood
            should use.  :attr:`meta` carries the split.
        n_spontaneous: ``(n,)`` spontaneous tracks counted.  Carried because it
            is what the age's precision is made of, and because an age from
            three tracks and an age from three hundred are different data.
        length_counts: ``(n, n_bins)`` measured confined lengths, binned on
            :attr:`length_bins`.  Counts, not a density: the multinomial
            likelihood needs the sample size and a normalised histogram has
            thrown it away.
        length_bins: ``(n_bins,)`` bin centres in um, shared by every row.
        dpar: ``(n,)`` mean etch-pit diameter, um.
        dpar_sd: ``(n,)`` one sigma on that mean.
        he_age: ``(n, G)`` alpha-ejection-corrected single-grain ages, Ma.
        he_age_sd: ``(n, G)`` one sigma, analytical *and* excess dispersion.
        he_radius: ``(n, G)`` measured equivalent-sphere radius, um.
        he_eU: ``(n, G)`` measured effective uranium, ppm.
        he_mask: ``(n, G)`` True where a grain is present.  Masked entries are
            zero-filled, but nothing downstream may rely on that.
        meta: provenance and anything not part of the data vector -- the
            observation-model config, the analytical-only sigmas, the corpus
            rows the sample came from.  Not carried through
            :meth:`permute_grains`, which cannot know what in it is per-grain.
        aft_mask: ``(n,)`` True where the sample has a fission-track analysis
            at all.  ``None`` means every sample does, which is what every
            caller before Stage 11 assumed.  A sample with helium grains and
            no AFT is a real thing a laboratory hands over, and it is *not*
            the same as an AFT analysis that counted zero tracks -- that one
            carries an upper bound on the age, this one carries nothing -- so
            the absence has to be a flag rather than a zero.  Where it is
            False the AFT fields are zero-filled and ``length_counts`` is all
            zeros, and nothing downstream may read them.
    """

    aft_age: np.ndarray
    aft_age_sd: np.ndarray
    n_spontaneous: np.ndarray
    length_counts: np.ndarray
    length_bins: np.ndarray
    dpar: np.ndarray
    dpar_sd: np.ndarray
    he_age: np.ndarray
    he_age_sd: np.ndarray
    he_radius: np.ndarray
    he_eU: np.ndarray
    he_mask: np.ndarray
    meta: dict[str, Any] | None = None
    aft_mask: np.ndarray | None = None

    _ROW_FIELDS = ("aft_age", "aft_age_sd", "n_spontaneous", "length_counts", "dpar",
                   "dpar_sd", "he_age", "he_age_sd", "he_radius", "he_eU", "he_mask",
                   "aft_mask")

    def __post_init__(self) -> None:
        n = self.aft_age.shape[0]
        if self.aft_mask is None:
            object.__setattr__(self, "aft_mask", np.ones(n, dtype=bool))
        elif self.aft_mask.dtype != np.bool_:
            raise ValueError(f"aft_mask must be boolean, got {self.aft_mask.dtype}")
        for name in self._ROW_FIELDS:
            v = getattr(self, name)
            if v.shape[0] != n:
                raise ValueError(f"{name} has {v.shape[0]} rows, expected {n}")
        if self.length_counts.shape[1] != self.length_bins.size:
            raise ValueError(f"length_counts has {self.length_counts.shape[1]} bins against "
                             f"{self.length_bins.size} bin centres")
        g = self.he_age.shape[1]
        for name in ("he_age_sd", "he_radius", "he_eU", "he_mask"):
            if getattr(self, name).shape[1] != g:
                raise ValueError(f"{name} has {getattr(self, name).shape[1]} grains, expected {g}")
        if self.he_mask.dtype != np.bool_:
            raise ValueError(f"he_mask must be boolean, got {self.he_mask.dtype}")

    def __len__(self) -> int:
        return int(self.aft_age.shape[0])

    @property
    def n_grains(self) -> np.ndarray:
        """``(n,)`` how many helium grains each sample actually reports."""
        return self.he_mask.sum(axis=1)

    @property
    def has_age(self) -> np.ndarray:
        """``(n,)`` True where there is a fission-track *age*: an analysis, and tracks in it."""
        return self.aft_mask & (self.n_spontaneous > 0)

    @property
    def n_lengths(self) -> np.ndarray:
        """``(n,)`` confined lengths measured."""
        return self.length_counts.sum(axis=1)

    @property
    def mean_length(self) -> np.ndarray:
        """``(n,)`` measured mean c-axis-projected length, um; 0 where none were measured.

        Zero is not a length, and that is the point: it is only ever paired
        with ``n_lengths == 0``, which is the column a consumer must read to
        know the mean is absent.  Filling with NaN instead would poison every
        downstream aggregation to signal the same thing.
        """
        n = self.n_lengths
        total = self.length_counts @ self.length_bins
        return np.divide(total, n, out=np.zeros_like(total, dtype=float), where=n > 0)

    @property
    def length_pdf(self) -> np.ndarray:
        """``(n, n_bins)`` the measured histogram as bin probabilities."""
        n = self.n_lengths[:, None]
        return np.divide(self.length_counts, n, out=np.zeros(self.length_counts.shape),
                         where=n > 0)

    def __getitem__(self, key: int | slice | np.ndarray) -> Aliquot:
        idx = np.asarray([key]) if isinstance(key, (int, np.integer)) else key
        return replace(self, **{name: getattr(self, name)[idx] for name in self._ROW_FIELDS})

    def permute_grains(self, rng: np.random.Generator) -> Aliquot:
        """Independently shuffle each row's grain axis.

        The identity on any permutation-invariant consumer, which is what makes
        it a test: a Deep Sets encoder must return the same numbers before and
        after, and so must
        :meth:`chai.data.observe.ObservationModel.log_prob` when the
        prediction it is scored against is permuted the same way.

        :attr:`meta` is **not** permuted -- it is free-form, so this method
        cannot know which of its entries are per-grain.
        """
        order = np.argsort(rng.random(self.he_age.shape), axis=1)
        fields = ("he_age", "he_age_sd", "he_radius", "he_eU", "he_mask")
        return replace(self, **{f: np.take_along_axis(getattr(self, f), order, axis=1)
                                for f in fields})

    def aft_features(self) -> np.ndarray:
        """``(n, 7)`` the AFT half as a flat vector; columns are :data:`AFT_FEATURES`."""
        return np.stack([self.aft_age, self.aft_age_sd, self.n_spontaneous.astype(float),
                         self.n_lengths.astype(float), self.mean_length,
                         self.dpar, self.dpar_sd], axis=1)

    def grain_features(self) -> np.ndarray:
        """``(n, G, 4)`` the helium set; columns are :data:`GRAIN_FEATURES`.

        Masked rows are zero.  Pair with :attr:`he_mask` and pool with a
        symmetric function (sum, mean, max) to get a Deep Sets encoding; the
        mask must multiply the *per-grain embeddings*, not these features, or a
        bias term leaks the padding in.
        """
        f = np.stack([self.he_age, self.he_age_sd, self.he_radius, self.he_eU], axis=2)
        return f * self.he_mask[:, :, None]

    def to_torch(self, device: Any = "cpu", dtype: Any = None) -> dict[str, Any]:
        """The data vector as tensors, ready for a set-encoder.

        Returns ``{"aft", "aft_mask", "length_pdf", "n_lengths", "grains", "grain_mask"}``.
        Raw physical units throughout -- normalisation is a property of a
        trained model and belongs with it, not with the data.
        """
        import torch

        dtype = dtype or torch.float32
        def t(x, d=None):
            return torch.as_tensor(x, dtype=d or dtype, device=device)

        return {
            "aft": t(self.aft_features()),
            "aft_mask": t(self.aft_mask, torch.bool),
            "length_pdf": t(self.length_pdf),
            "n_lengths": t(self.n_lengths.astype(float)),
            "grains": t(self.grain_features()),
            "grain_mask": t(self.he_mask, torch.bool),
        }

    @classmethod
    def concat(cls, parts: list[Aliquot]) -> Aliquot:
        """Join batches.  Grain axes are zero-padded up to the widest."""
        if not parts:
            raise ValueError("nothing to concatenate")
        g = max(p.he_age.shape[1] for p in parts)
        bins = parts[0].length_bins
        if any(p.length_bins.size != bins.size for p in parts):
            raise ValueError("cannot join aliquots binned differently")

        def pad(v: np.ndarray) -> np.ndarray:
            return np.pad(v, ((0, 0), (0, g - v.shape[1])))

        out: dict[str, Any] = {}
        for name in cls._ROW_FIELDS:
            vals = [getattr(p, name) for p in parts]
            if name.startswith("he_"):
                vals = [pad(v) for v in vals]
            out[name] = np.concatenate(vals)
        return cls(length_bins=bins, meta=parts[0].meta, **out)

    def save(self, path: Any) -> None:
        """Write to ``.npz``.  ``meta`` is dropped -- it is not an array."""
        arrays = {name: getattr(self, name) for name in self._ROW_FIELDS}
        np.savez_compressed(path, length_bins=self.length_bins, **arrays)

    @classmethod
    def load(cls, path: Any) -> Aliquot:
        with np.load(path) as f:
            # ``aft_mask`` arrived with Stage 11; files saved before it have none.
            return cls(length_bins=f["length_bins"],
                       **{name: f[name] for name in cls._ROW_FIELDS if name in f})

    def summary(self) -> dict[str, float]:
        """Medians of the things one checks first when a batch looks wrong."""
        a = self.aft_mask
        h = self.he_mask
        def med(x):
            return float(np.median(x)) if x.size else float("nan")
        return {
            "n": float(len(self)),
            "aft_age": med(self.aft_age[a]),
            "aft_age_rel_sd": med((self.aft_age_sd / np.maximum(self.aft_age, 1e-9))[a]),
            "n_spontaneous": med(self.n_spontaneous[a]),
            "n_lengths": med(self.n_lengths[a]),
            "mean_length": med(self.mean_length[a]),
            "he_age": med(self.he_age[h]),
            "he_age_rel_sd": med((self.he_age_sd / np.maximum(self.he_age, 1e-9))[h]),
            "grains": float(np.median(self.n_grains)),
            "with_aft": float(a.mean()),
            "with_he": float((self.n_grains > 0).mean()),
        }

    def to_config(self) -> dict[str, Any]:
        return asdict(self)
