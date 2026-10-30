"""The thin API: measurements in, thermal histories out.

Everything a user needs to go from a table of measurements to a posterior over
thermal histories, and nothing a user does not::

    from chai import api

    family = api.load()               # the shipped family: a 100 Ma and a 250 Ma member, and the chooser
    ex = api.example_sample(family, "pulse")     # a synthetic sample with its true history
    post = family.infer(ex.sample, n=2000, T_present=ex.T_present)
    print(post.summary())             # which member answered, and the quantiles of T against time
    post.quantiles().shape            # (5, 96): 2.5, 16, 50, 84, 97.5% at each node of post.t
    post.plot(truth=ex.truth)

    # your own measurements, in a data table's units
    sample = api.Sample(
        aft=api.AFT(age=42.0, n_spontaneous=180, dpar=1.9,
                    lengths=[13.1, 12.4, 14.0, 13.6, 12.9, 13.8, 11.7, 13.3, 14.2, 12.6, 13.5, 14.4,
                             12.8, 13.9, 13.2, 11.5, 14.1, 13.0, 12.7, 13.7, 14.6, 12.2, 13.4, 13.8,
                             12.5, 14.3, 13.1, 11.9, 13.6, 14.0, 12.9, 13.3, 14.5, 12.3, 13.8, 13.0,
                             12.6, 14.2, 13.5, 11.8]),     # c-axis projected, µm
        grains=[api.Grain(age=31.0, radius=62, eU=40), api.Grain(age=36.5, radius=85, eU=22)],
    )
    post = family.infer(sample, T_present=(5, 15))

    # with geology the data do not see
    post = family.infer(sample, T_present=(5, 15), boxes=[api.Box(t=(55, 60), T=(0, 25), mode="within")])
    post.acceptance                   # the share of the member's draws that satisfied the boxes

The present-day surface temperature is an input of the shipped members, given as
a range ``T_present=(lo, hi)`` in °C inside 0-25 and at least 1 K wide; each path
draws its own value from it.  :meth:`Family.infer` answers with one member, the
more probable given the data (``post.choice`` says which, and how sure), on that
member's own grid: 96 nodes to 100 Ma, or 160 to 250 Ma.

What the model is conditioned on, and what it assumes
-----------------------------------------------------
The members were trained on one observation model (``assets/observation_v2.yaml``)
over a range of laboratory effort, and it reads the numbers that model
produces.  Two of them a laboratory reports differently, so the API computes
them the model's way and *warns* where the user's own figure disagrees,
rather than handing the network an input it never saw:

- a helium grain's one sigma is the model's 12.3% of its age (2.5% analytical,
  12% single-grain dispersion), whatever the table says;
- a fission-track age's one sigma is Poisson on ``n_spontaneous`` plus the
  model's 3.6% systematic (zeta, U).  Give ``n_spontaneous``, or give
  ``age_sd`` and the count is backed out of it.

Confined lengths are read **c-axis projected**, µm, as the Ketcham (2007)
forward model predicts them.  Give each track's angle to the c axis as well
(``AFT(lengths=..., angles=...)``) and the lengths are taken as measured and
projected here, by Ketcham et al. (2007a) with the Barbarand et al. (2003)
analyst 1 constants unless ``projection=`` names another set
(:mod:`chai.data.projection`); without angles they are taken as already
projected.  Raw lengths without
angles cannot be projected, and the head would read them as if they were:
too short, so too hot a history.  Inputs outside what the head was trained on -- more
helium grains than the corpus carried, radii, eU or Dpar outside the grain
prior, an age past the 100 Ma member's time span -- are accepted and listed in
:attr:`PathPosterior.warnings`, because an amortized posterior extrapolates
silently and the user should know when it is doing so.

Does the answer fit
-------------------
:func:`check_fit` (:mod:`chai.fit`): for each datum, where the observed value
falls among what the posterior's paths predict with the model's own noise
added.  Data past a level calibrated for one false alarm in twenty
are listed in :attr:`PathPosterior.warnings` with their
predicted interval, and the result is kept on :attr:`PathPosterior.fit`.  A
warning, not a gate: the posterior is returned either way.

Boxes
-----
:mod:`chai.constraints`: HeFTy-style boxes as rejection on the head's draws,
exact because a box only removes prior mass.  A box set the prior cannot
satisfy is refused (:class:`~chai.constraints.InfeasibleBoxes`) rather than
answered with zero draws; a satisfiable one comes back with its acceptance
rate, and a low one is a statement about the data as much as the box
(below about 1% the draws are too few to trust).  ``cooling_only=True``
is QTQt's monotonic option the same way: draws that reheat (by more than
:data:`~chai.constraints.REHEAT_TOL`, the head's noise floor) are rejected.

Input files
-----------
:mod:`chai.inputs`: a sample, its constraints and the run's settings in one
JSON file, the one the web page saves and loads.
"""
from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from chai.constraints import (
    PASSES,
    REHEAT_TOL,
    WITHIN,
    Box,
    InfeasibleBoxes,
    PriorEnvelope,
    box_weights,
    check_feasible,
    cooling_weights,
)
from chai.data.aliquot import Aliquot
from chai.data.observe import ObservationModel
from chai.data.projection import ProjectionModel, project_lengths, projection_model
from chai.eval.calibration import FUNCTIONALS, functional_values
from chai.fit import Datum, Fit
from chai.models.gaussian import AmortizedPosterior, draw_z
from chai.train.data import features_from_aliquot

__all__ = ["AFT", "EXAMPLES", "PASSES", "WITHIN", "Box", "Choice", "Datum", "Example", "Family", "Fit", "Grain",
           "InfeasibleBoxes", "Model", "PathPosterior", "Prediction", "Sample", "Window", "alpha_correct",
           "check_fit", "example_sample", "load", "load_family", "load_member", "predict", "sensitive_window",
           "simulate_sample"]

_REPO = Path(__file__).resolve().parents[2]
#: The prior whose family :func:`load` returns by default: the one key of :data:`FAMILY_ASSETS`.
DEFAULT_PRIOR = "v2.2"
#: What an installed package carries, so that :func:`load` needs no checkout: the
#: shipped head's weights and the two configs it was trained under (copies of
#: ``configs/``, kept identical by ``tests/test_packaging.py``).
ASSETS = Path(__file__).resolve().parent / "assets"
#: The shipped head a package without the family falls back to: the default prior's 100 Ma member.
DEFAULT_MODEL = ASSETS / "flow_v22_100.pt"
#: The oracle's length axis: 200 bins of 0.1 µm.
N_BINS, BIN_WIDTH = 200, 0.1
LENGTH_BINS = (np.arange(N_BINS) + 0.5) * BIN_WIDTH
_EU_TH_WEIGHT = 0.235
QUANTILES = (0.025, 0.16, 0.5, 0.84, 0.975)
#: Fewer confined lengths than this are noted: they constrain the history only loosely.
FEW_LENGTHS = 50


# ---------------------------------------------------------------------------
# The sample
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class AFT:
    """One apatite fission-track analysis.

    Args:
        age: pooled central age, Ma.
        n_spontaneous: spontaneous tracks counted.  Optional if ``age_sd`` is given.
        age_sd: one sigma, Ma, if the count is not.  Otherwise compared with the
            model's own and a disagreement is reported.
        lengths: confined track lengths, µm; empty for none.  **C-axis projected**
            unless ``angles`` is given.
        dpar: mean etch-pit diameter, µm.
        angles: each track's angle to the c axis, degrees, one per length.  With
            angles the lengths are the measured ones and are projected by
            :func:`chai.data.projection.project_lengths` before the head reads
            them; :attr:`projected_lengths` is the result.
        projection: the projection's constants, a name in
            :data:`chai.data.projection.PROJECTIONS` or a ``ProjectionModel``:
            ``"ketcham2007_b03"`` (the default), ``"ketcham2007_b03_analyst3"``,
            ``"ketcham2007_c99"`` or ``"donelick1999"``.  Used only with angles.
    """

    age: float
    n_spontaneous: int | None = None
    age_sd: float | None = None
    lengths: Sequence[float] = ()
    dpar: float = 2.0
    angles: Sequence[float] | None = None
    projection: str | ProjectionModel = "ketcham2007_b03"

    def __post_init__(self) -> None:
        if not self.age > 0:
            raise ValueError(f"AFT age must be positive, got {self.age}")
        if self.n_spontaneous is None and self.age_sd is None:
            raise ValueError("give n_spontaneous or age_sd: the head reads the age's precision")
        if self.n_spontaneous is not None and self.n_spontaneous < 0:
            raise ValueError(f"n_spontaneous must be >= 0, got {self.n_spontaneous}")
        if self.age_sd is not None and not self.age_sd > 0:
            raise ValueError(f"age_sd must be positive, got {self.age_sd}")
        object.__setattr__(self, "lengths", tuple(float(x) for x in self.lengths))
        if self.angles is not None:
            angles = tuple(float(x) for x in self.angles)
            if len(angles) != len(self.lengths):
                raise ValueError(f"{len(self.lengths)} lengths but {len(angles)} angles: give one angle per track")
            object.__setattr__(self, "angles", angles)
            object.__setattr__(self, "projection", projection_model(self.projection))
            # Validates the angles, and fails here rather than at inference.
            lc = project_lengths(self.lengths, angles, self.projection)
            object.__setattr__(self, "_projected", tuple(float(x) for x in lc))
        else:
            object.__setattr__(self, "_projected", self.lengths)

    @property
    def projected_lengths(self) -> tuple[float, ...]:
        """The c-axis projected lengths the head reads: :attr:`lengths` projected if angles were given."""
        return self._projected  # type: ignore[attr-defined]


#: Th/U assumed when an uncorrected helium age is corrected here: a table gives eU, not U and Th.
#: Across the prior's 0.2-10 it moves Ft by under 3%.
RAW_AGE_TH_U = 1.0


def alpha_correct(age: float, radius: float, eU: float, th_u: float = RAW_AGE_TH_U) -> tuple[float, float]:
    """An uncorrected helium age corrected for alpha ejection: ``(corrected age, Ft)``.

    Ft per parent is the sphere's, ``1 - 3S/4R + S^3/16R^3`` (Farley et al. 1996),
    with the stopping distances of Ketcham et al. (2011) that the forward model
    uses.  The corrected age solves the age equation with each parent's
    production scaled by its Ft, as the forward model's ``corrected_age`` does.
    The Ft returned is the production-weighted mean.
    """
    from chai.forward.torchimpl.rdaam import (
        AS_DIST,
        PPM_NMPG_TH232,
        PPM_NMPG_U235,
        PPM_NMPG_U238,
        TH232YR,
        U235YR,
        U238U235RATIO,
        U238YR,
        YIELD,
    )

    if not radius > max(AS_DIST[:3]) / 2:
        raise ValueError(f"grain radius {radius:g} µm is too small to correct for alpha ejection "
                         f"(at least {max(AS_DIST[:3]) / 2:g} µm)")
    U = eU / (1.0 + _EU_TH_WEIGHT * th_u)
    prod = np.array([U * (U238U235RATIO - 1) / U238U235RATIO * PPM_NMPG_U238 * YIELD[0],
                     U / U238U235RATIO * PPM_NMPG_U235 * YIELD[1], th_u * U * PPM_NMPG_TH232 * YIELD[2]])
    lam = np.array([U238YR, U235YR, TH232YR]) * 1e6                  # per Ma
    x = np.asarray(AS_DIST[:3]) / radius
    ft = 1.0 - 0.75 * x + x ** 3 / 16.0
    he = float((prod * np.expm1(lam * age)).sum())
    t = age / float((ft * prod * lam).sum() / (prod * lam).sum())
    for _ in range(50):
        f = float((ft * prod * np.expm1(lam * t)).sum()) - he
        step = f / float((ft * prod * lam * np.exp(lam * t)).sum())
        t -= step
        if abs(step) < 1e-10 * t:
            break
    return t, float((ft * prod * lam).sum() / (prod * lam).sum())


@dataclass(frozen=True)
class Grain:
    """One single-grain apatite (U-Th)/He age.

    Args:
        age: alpha-ejection-corrected age, Ma; or the uncorrected age with
            ``corrected=False``.
        radius: equivalent-sphere radius, µm.
        eU: effective uranium, ppm (U + 0.235 Th).
        age_sd: one sigma, Ma, as reported.  The head does not read it: it reads
            the model's 12.3%, and says so when this differs.
        corrected: ``False`` if ``age`` is uncorrected; it is then corrected
            here by :func:`alpha_correct` from the radius (Th/U taken as
            :data:`RAW_AGE_TH_U`), and :attr:`corrected_age` is the result.
    """

    age: float
    radius: float
    eU: float
    age_sd: float | None = None
    corrected: bool = True

    def __post_init__(self) -> None:
        for k in ("age", "radius", "eU"):
            if not getattr(self, k) > 0:
                raise ValueError(f"grain {k} must be positive, got {getattr(self, k)}")
        age, ft = (self.age, None) if self.corrected else alpha_correct(self.age, self.radius, self.eU)
        object.__setattr__(self, "_corrected", (age, ft))

    @property
    def corrected_age(self) -> float:
        """The alpha-ejection-corrected age the head reads, Ma."""
        return self._corrected[0]  # type: ignore[attr-defined]

    @property
    def ft(self) -> float | None:
        """The Ft applied here, or ``None`` for an age given corrected."""
        return self._corrected[1]  # type: ignore[attr-defined]


@dataclass(frozen=True)
class Sample:
    """What one rock yields: an optional AFT analysis and any number of helium grains."""

    aft: AFT | None = None
    grains: Sequence[Grain] = ()
    name: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "grains", tuple(self.grains))
        if self.aft is None and not self.grains:
            raise ValueError("a sample with no data has the prior for its posterior; give an AFT analysis or grains")

    @property
    def oldest_age(self) -> float:
        """The oldest age the sample records, Ma: past it the posterior is the prior's conditional."""
        ages = [g.corrected_age for g in self.grains] + ([self.aft.age] if self.aft is not None else [])
        return float(max(ages))

    @classmethod
    def from_aliquot(cls, obs: Aliquot, i: int = 0, name: str = "") -> Sample:
        """Row ``i`` of an :class:`Aliquot`, lengths expanded from their bin counts to bin centres."""
        one = obs[i]
        aft = None
        if bool(one.aft_mask[0]):
            counts = one.length_counts[0].astype(int)
            lengths = np.repeat(one.length_bins, counts)
            aft = AFT(age=float(one.aft_age[0]), n_spontaneous=int(one.n_spontaneous[0]),
                      age_sd=float(one.aft_age_sd[0]), lengths=lengths, dpar=float(one.dpar[0]))
        m = one.he_mask[0]
        grains = [Grain(age=float(a), radius=float(r), eU=float(e), age_sd=float(s))
                  for a, r, e, s in zip(one.he_age[0][m], one.he_radius[0][m], one.he_eU[0][m], one.he_age_sd[0][m])]
        return cls(aft, grains, name)


# ---------------------------------------------------------------------------
# The model
# ---------------------------------------------------------------------------

def _prior_config(model: AmortizedPosterior, corpus: Path | None) -> tuple[dict[str, Any], dict[str, Any], int]:
    """The history prior, the grain prior and the number of grains the corpus carried.

    From the corpus's own attributes where the corpus is on disk.  Otherwise the
    history prior the latent was fitted under (every latent records it), the grain
    prior of the shipped ``configs/prior_v22.yaml`` (``assets/prior_v22.yaml``;
    both members share it) and the 12 grains both members' corpora carry.
    """
    import json

    if corpus is not None and corpus.exists():
        import h5py
        with h5py.File(corpus, "r") as f:
            g_max = int(f.attrs["n_grains_total"]) if "n_grains_total" in f.attrs \
                else int(json.loads(f.attrs["spec"])["n_grains"])
            return json.loads(f.attrs["prior_config"]), json.loads(f.attrs["grain_config"]), g_max
    cfg = yaml.safe_load(_resolve("configs/prior_v22.yaml").read_text())
    history = model.latent.meta.get("prior_config") or cfg["history"]
    return dict(history), cfg["grain"], 12


class Model:
    """A trained head, with what it was trained under.

    Args:
        net: the :class:`AmortizedPosterior`.
        observation: the observation model it was trained on.
        prior: the history prior's config (walls, time span, ``rate_max``).
        grain_prior: the grain prior's ranges.
        g_max: helium grains per sample in the corpus.
        device: where it runs.
    """

    def __init__(self, net: AmortizedPosterior, observation: ObservationModel, prior: dict[str, Any],
                 grain_prior: dict[str, Any], g_max: int, device: Any = "cpu"):
        self.net = net.eval()
        self.observation = observation
        self.prior = prior
        self.grain_prior = grain_prior
        self.g_max = g_max
        self.device = torch.device(device)
        effort = net.meta.get("config", {}).get("effort", {})
        self.effort = effort
        L = net.latent
        self.t = L.t_grid.cpu().numpy()
        self.envelope = PriorEnvelope(self.t, L.box_lo.cpu().numpy(), L.box_hi.cpu().numpy(),
                                      float(prior["rate_max"]))

    @property
    def t_max(self) -> float:
        return float(self.t[-1])

    # -- inputs -------------------------------------------------------------

    def to_aliquot(self, sample: Sample) -> tuple[Aliquot, list[str]]:
        """The sample as the head reads it, and what about it the head was not trained on."""
        notes: list[str] = []
        obs_cfg = self.observation
        sys_var = obs_cfg.aft.systematic_rel_var
        he_rel = obs_cfg.he.rel_sd
        a = sample.aft
        if a is not None:
            if a.n_spontaneous is not None:
                ns = int(a.n_spontaneous)
            else:
                count_rel2 = (a.age_sd / a.age) ** 2 - sys_var
                if count_rel2 <= 0:
                    raise ValueError(f"AFT age_sd {a.age_sd:g} Ma is below the model's systematic floor of "
                                     f"{a.age * np.sqrt(sys_var):.2g} Ma; give n_spontaneous instead")
                ns = max(round(1.0 / count_rel2), 1)
                notes.append(f"AFT: your 1σ corresponds to {ns} counted tracks")
            age_sd = a.age * np.sqrt(1.0 / max(ns, 1) + sys_var)
            if a.age_sd is not None and a.n_spontaneous is not None and not 0.67 < a.age_sd / age_sd < 1.5:
                notes.append(f"AFT: your 1σ of {a.age_sd:.2g} Ma differs from the {age_sd:.2g} Ma that {ns} tracks "
                             f"imply; the model uses {age_sd:.2g} Ma")
            counts = np.zeros(N_BINS)
            if a.lengths:
                L = np.asarray(a.projected_lengths)
                if (L <= 0).any() or (L >= N_BINS * BIN_WIDTH).any():
                    raise ValueError(f"track lengths must be in (0, {N_BINS * BIN_WIDTH:g}) µm")
                np.add.at(counts, np.minimum((L / BIN_WIDTH).astype(int), N_BINS - 1), 1.0)
            n_len = len(a.lengths)
            if a.angles is not None and n_len:
                notes.append(f"AFT: lengths projected to the c axis, mean {np.mean(a.lengths):.2f} µm measured, "
                             f"{np.mean(L):.2f} µm projected")
            lo_len, hi_len = self.effort.get("n_lengths", (20, 150))
            if n_len and n_len < lo_len:
                notes.append(f"AFT: only {n_len} track lengths, fewer than the model was trained on ({lo_len})")
            elif n_len and n_len < FEW_LENGTHS:
                notes.append(f"AFT: only {n_len} track lengths, so they constrain the history loosely")
            elif n_len > hi_len:
                notes.append(f"AFT: {n_len} track lengths, more than the model was trained on ({hi_len})")
            d_lo, d_hi = self.grain_prior["dpar_range"]
            if not d_lo - 0.1 <= a.dpar <= d_hi + 0.1:
                notes.append(f"AFT: Dpar {a.dpar:g} µm is outside the model's range ({d_lo:g}–{d_hi:g} µm)")
            if a.age > self.t_max:
                notes.append(f"AFT: age {a.age:g} Ma is older than the {self.t_max:g} Myr time span")
            aft_fields = {"aft_age": [a.age], "aft_age_sd": [age_sd], "n_spontaneous": [ns],
                              "length_counts": counts[None], "dpar": [a.dpar], "dpar_sd": [obs_cfg.aft.dpar_sd]}
        else:
            aft_fields = {"aft_age": [0.0], "aft_age_sd": [0.0], "n_spontaneous": [0], "length_counts": np.zeros((1, N_BINS)),
                              "dpar": [0.0], "dpar_sd": [0.0]}
        G = max(len(sample.grains), 1)
        he = np.zeros((4, 1, G))
        mask = np.zeros((1, G), dtype=bool)
        r_lo, r_hi = self.grain_prior["radius_range"]
        u_lo, u_hi = self.grain_prior["U_range"]
        th_lo, th_hi = self.grain_prior["th_u_ratio_range"]
        eu_lo, eu_hi = u_lo * (1 + _EU_TH_WEIGHT * th_lo), u_hi * (1 + _EU_TH_WEIGHT * th_hi)
        odd: dict[str, list[tuple[int, float]]] = {}
        for k, g in enumerate(sample.grains):
            he[:, 0, k] = (g.corrected_age, g.corrected_age * he_rel, g.radius, g.eU)
            mask[0, k] = True
            if g.age_sd is not None and not 0.67 < g.age_sd / (g.age * he_rel) < 1.5:
                odd.setdefault("sd", []).append((k, g.age_sd / g.age))
            if not 0.85 * r_lo <= g.radius <= 1.15 * r_hi:
                odd.setdefault("radius", []).append((k, g.radius))
            if not 0.9 * eu_lo <= g.eU <= 1.1 * eu_hi:
                odd.setdefault("eU", []).append((k, g.eU))
            if g.corrected_age > self.t_max:
                odd.setdefault("age", []).append((k, g.corrected_age))
        what = {
            "sd": lambda v: f"1σ of {v}, far from the {he_rel:.1%} the model assumes",
            "radius": lambda v: f"radius {v} µm is outside the model's range ({r_lo:g}–{r_hi:g} µm)",
            "eU": lambda v: f"eU {v} ppm is outside the model's range ({eu_lo:.3g}–{eu_hi:.3g} ppm)",
            "age": lambda v: f"age {v} Ma is older than the {self.t_max:g} Myr time span",
        }
        for kind, hits in odd.items():
            vals = [v for _, v in hits]
            fmt = (lambda x: f"{x:.1%}") if kind == "sd" else (lambda x: f"{x:g}")
            v = fmt(vals[0]) if min(vals) == max(vals) else f"{fmt(min(vals))}–{fmt(max(vals))}"
            idx = [k + 1 for k, _ in hits]
            who = f"He grain {idx[0]}" if len(idx) == 1 else f"He grains {', '.join(map(str, idx))}" if len(idx) <= 4 \
                else f"{len(idx)} of {len(sample.grains)} He grains"
            notes.append(f"{who}: {what[kind](v)}")
        if len(sample.grains) > self.g_max:
            notes.append(f"{len(sample.grains)} He grains, more than the model was trained on ({self.g_max})")
        obs = Aliquot(
            **{k: np.asarray(v, dtype=np.int64 if k == "n_spontaneous" else float) for k, v in aft_fields.items()},
            length_bins=LENGTH_BINS, he_age=he[0], he_age_sd=he[1], he_radius=he[2], he_eU=he[3],
            he_mask=mask, aft_mask=np.array([a is not None]),
        )
        return obs, notes

    def batch(self, obs: Aliquot, T0: Any = None) -> dict[str, torch.Tensor]:
        """The encoder's input for an :class:`Aliquot` of any length, on the model's device.

        ``T0``: the present-day temperature per row, for a model that takes it.
        """
        b = {k: torch.as_tensor(np.ascontiguousarray(v), device=self.device)
             for k, v in features_from_aliquot(obs).items()}
        if T0 is not None:
            b["T0"] = torch.as_tensor(T0, dtype=torch.float32, device=self.device).reshape(-1)
        return b

    @property
    def min_draws(self) -> int:
        """Draws per pass of the head: :data:`MIN_FLOOR_DRAWS`, or two per corrected node if more, for a
        member that takes its floor back out; 1 for any other."""
        if not self.net.floor_sd:
            return 1
        return max(MIN_FLOOR_DRAWS, 2 * (len(self.t) - int(self.present_input)))

    @property
    def present_input(self) -> bool:
        """Whether the user supplies the present-day temperature (``T_present``) rather than the model drawing it."""
        return self.net.latent.present_input

    @property
    def T_present_range(self) -> tuple[float, float]:
        """The present-day temperatures the model was trained on, °C."""
        lo, hi = self.prior["T_present_range"]
        return float(lo), float(hi)

    def present_range(self, T_present: Sequence[float] | None) -> tuple[float, float] | None:
        """``T_present`` as a checked ``(lo, hi)``, or ``None`` for a model that draws it.

        Always a range, at least :data:`MIN_PRESENT_RANGE` wide, so the draws
        spread over it rather than sit on one value.  A model that takes the
        present day refuses to run without it, and refuses a range outside the
        one it was trained on rather than extrapolate.
        """
        if not self.present_input:
            if T_present is not None:
                raise ValueError("this model draws the present-day temperature itself; it takes no T_present")
            return None
        p_lo, p_hi = self.T_present_range
        if T_present is None or np.ndim(T_present) != 1 or len(T_present) != 2:
            raise ValueError(f"give the present-day (surface) temperature as a range: T_present=(lo, hi) in °C, "
                             f"inside {p_lo:g}-{p_hi:g} °C and at least {MIN_PRESENT_RANGE:g} K wide")
        lo, hi = map(float, T_present)
        if not p_lo <= lo < hi <= p_hi:
            raise ValueError(f"T_present ({lo:g}, {hi:g}) °C must be an ordered range inside "
                             f"{p_lo:g}-{p_hi:g} °C, the present-day temperatures the model was trained on")
        if hi - lo < MIN_PRESENT_RANGE:
            raise ValueError(f"T_present ({lo:g}, {hi:g}) °C is {hi - lo:g} K wide; the narrowest range is "
                             f"{MIN_PRESENT_RANGE:g} K")
        return lo, hi

    # -- inference ----------------------------------------------------------

    @torch.no_grad()
    def infer(self, sample: Sample | Aliquot, n: int = 2000, *, T_present: Sequence[float] | None = None,
              boxes: Sequence[Box] = (), cooling_only: bool = False,
              reheat_tol: float = REHEAT_TOL, seed: int | None = 0, max_draws: int = 1_000_000, chunk: int = 20_000,
              min_acceptance_warn: float = 0.01) -> PathPosterior:
        """``n`` thermal histories from the posterior given ``sample``, and the constraints if any.

        Args:
            sample: a :class:`Sample`, or a length-1 :class:`Aliquot` as the
                observation model makes them -- read as it is, since it is
                already in the model's terms (the studies use this path, so
                the head sees exactly what Stages 8 and 12 scored it on).
            n: accepted draws wanted.
            T_present: the present-day (surface) temperature, °C, as a range
                ``(lo, hi)`` at least :data:`MIN_PRESENT_RANGE` wide.  Required by a model that takes it
                (:attr:`present_input`), refused by one that draws
                it.  Each draw's present day is uniform on the range, so the
                answer is the posterior given "somewhere in this range" --
                exactly so while the data cannot tell temperatures inside it
                apart, which below ~30 °C they barely can.
            boxes: :class:`Box` constraints.  Refused with
                :class:`InfeasibleBoxes` if the prior cannot satisfy them.
            cooling_only: keep only the draws that never reheat by more than
                ``reheat_tol`` kelvin (:func:`chai.constraints.reheating`), as
                QTQt's monotonic option does.  Counts in the acceptance like a box.
            reheat_tol: kelvin; :data:`~chai.constraints.REHEAT_TOL` by default.
            seed: for the draws; ``None`` for fresh ones.
            max_draws: the most the head is asked for before giving up.  If
                fewer than ``n`` survive, the result holds what did and says so;
                if none do, :class:`InfeasibleBoxes` is raised with the bound
                on the acceptance rate.
            chunk: draws per pass.
            min_acceptance_warn: acceptance below this adds a warning.
        """
        t0 = time.perf_counter()
        boxes = tuple(boxes)
        check_feasible(boxes, self.envelope)
        constrained = bool(boxes) or cooling_only
        if cooling_only and not reheat_tol >= 0.0:
            raise ValueError(f"reheat_tol must be >= 0, got {reheat_tol}")
        if isinstance(sample, Aliquot):
            if len(sample) != 1:
                raise ValueError(f"infer takes one sample, got an Aliquot of {len(sample)}")
            obs, notes = sample, []
            oldest = Sample.from_aliquot(sample).oldest_age
        else:
            obs, notes = self.to_aliquot(sample)
            oldest = sample.oldest_age
        present = self.present_range(T_present)
        gen = torch.Generator(device=self.device)
        gen.manual_seed(seed) if seed is not None else gen.seed()
        feats = self.batch(obs)
        q = None if present else self.net(feats)
        kept_T, kept_h, drawn, n_kept = [], [], 0, 0
        while n_kept < n and drawn < max_draws:
            k = max(min(chunk, max_draws - drawn) if constrained else n, self.min_draws)
            if present is None:
                z, T0 = draw_z(q, k, gen)[:, 0], None
            else:
                # One present day per draw: the sample's inputs k times, each with its own T0.
                lo, hi = present
                T0 = lo + (hi - lo) * torch.rand(k, generator=gen, device=self.device, dtype=torch.float64)
                rows = {name: v.expand(k, *v.shape[1:]) for name, v in feats.items()}
                z = draw_z(self.net({**rows, "T0": T0.to(torch.float32)}), 1, gen)[0]
            # One query, k draws: (k, 1, dim), as the head's paths() takes them, so a member that
            # removes its floor does so across the k draws before the clamp.
            T, hyper = self.net.paths(z[:, None], {} if T0 is None else {"T0": T0[:, None]})
            T, hyper = T[:, 0], {name: v[:, 0] for name, v in hyper.items()}
            drawn += k
            if constrained:
                w = box_weights(boxes, self.t, T)
                if cooling_only:
                    w = w * cooling_weights(T, reheat_tol)
                if all(b.hard for b in boxes):
                    keep = w > 0.5
                else:
                    keep = torch.rand(w.shape, generator=gen, device=w.device, dtype=w.dtype) < w
                T = T[keep]
                hyper = {name: v[keep] for name, v in hyper.items()}
            kept_T.append(T.float().cpu())
            kept_h.append({name: v.float().cpu() for name, v in hyper.items()})
            n_kept += T.shape[0]
        if n_kept == 0:
            what = "the boxes" if not cooling_only else "the constraints" if boxes else "cooling only"
            raise InfeasibleBoxes(f"none of {drawn:,} draws satisfied {what} (acceptance < {1 / drawn:.1e}); "
                                  f"the data and {what} disagree beyond what the head can say")
        T = torch.cat(kept_T)[:n].numpy()
        hyper = {name: torch.cat([h[name] for h in kept_h])[:n].numpy() for name in kept_h[0]}
        acceptance = n_kept / drawn
        if constrained and n_kept < n:
            notes.append(f"only {n_kept} of {n} paths passed the constraints")
        if constrained and acceptance < min_acceptance_warn:
            notes.append("under 1% of paths passed the constraints: they barely agree with the data, "
                         "so the answer is less reliable")
        if present is not None:
            hyper["T_0"] = T[:, 0].copy()
        return PathPosterior(t=self.t.copy(), T=T, hyper=hyper, boxes=boxes, T_present=present,
                             reheat_tol=float(reheat_tol) if cooling_only else None, acceptance=acceptance, n_drawn=drawn,
                             oldest_age=oldest, warnings=notes, seconds=time.perf_counter() - t0,
                             T_max=float(self.prior.get("T_max", self.prior.get("T_top"))))


def _resolve(path: str | Path) -> Path:
    """A config path as a training run recorded it: as given, else from the checkout, else shipped."""
    path = Path(path)
    for candidate in (path, _REPO / path, ASSETS / path.name):
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"{path}: not found as given, under {_REPO}, or in {ASSETS}")


def load(run: str | Path | None = None, *, prior: str = DEFAULT_PRIOR, device: Any = "cpu") -> Model | Family:
    """The trained head in ``run`` (a directory holding ``model.pt``, or the file itself).

    With no argument, the family the package ships for ``prior`` (:class:`Family`:
    the 100 and 250 Ma members and the chooser between them), so an installed
    ``chai`` needs no checkout, corpus or training run.  ``prior`` names a key of
    :data:`FAMILY_ASSETS`; the package ships one, ``"v2.2"`` (the default): a
    pulse prior with cooling and heating rates to 100 K/Ma (~90 as resolved on
    the grid), whose members take the present-day temperature as a range
    (``T_present``).  Another family is loaded from its files with
    :func:`load_family`.  A package without the second member falls back to the
    one head it has (:data:`DEFAULT_MODEL`).
    """
    if run is None:
        if prior not in FAMILY_ASSETS:
            raise ValueError(f"prior must be one of {sorted(FAMILY_ASSETS)}, got {prior!r}")
        members, chooser = FAMILY_ASSETS[prior]
        if all(p.exists() for p in members.values()) and chooser.exists():
            return load_family(members, chooser, device=device)
        run = DEFAULT_MODEL if prior == DEFAULT_PRIOR else members[100]
    return load_member(run, device=device)


def load_member(run: str | Path, *, device: Any = "cpu") -> Model:
    """One member: the trained head in ``run`` (a run directory or a ``model.pt``)."""
    run = Path(run)
    path = run / "model.pt" if run.is_dir() else run
    net = AmortizedPosterior.load(path, map_location=device)
    cfg = net.meta.get("config", {})
    if net.meta.get("remove_floor"):
        # The shipped members take their training floor back out of their draws; a run directory
        # trained at the same floor without the flag draws raw.
        net.floor_sd = float(cfg["dequantise"]["noise_sd"])
    obs_path = _resolve(cfg.get("observation", "configs/observation_v2.yaml"))
    observation = ObservationModel.from_config(yaml.safe_load(obs_path.read_text()))
    corpus = Path(cfg["corpus"]) if "corpus" in cfg else None
    if corpus is not None and not corpus.is_absolute() and not corpus.exists():
        corpus = _REPO / corpus
    prior, grain, g_max = _prior_config(net, corpus)
    return Model(net, observation, prior, grain, g_max, device=device)


# ---------------------------------------------------------------------------
# The result
# ---------------------------------------------------------------------------

@dataclass
class PathPosterior:
    """Draws of the thermal history, and what went into them.

    Attributes:
        t: ``(n_grid,)`` Ma before present.
        T: ``(n, n_grid)`` °C, piecewise linear between nodes, clamped to the prior's walls.
        hyper: the GP prior's hyperparameters per draw (``tau``, ``rate_sd``, ``drift``).
        boxes: the constraints applied.
        reheat_tol: with ``cooling_only``, the reheating allowed, K; ``None`` without.
        acceptance: the share of the head's draws that satisfied the constraints (1 without any).
        n_drawn: draws asked of the head.
        oldest_age: the oldest age in the sample; older than this the posterior
            is the prior's conditional on the recent path, not data.
        warnings: what about the sample or the result the user should know.
        seconds: wall-clock.
        choice: from :meth:`Family.infer`, which member answered and why; ``None``
            from one member's :meth:`Model.infer`.
        fit: from :func:`check_fit`, whether the paths predict the data; ``None``
            until it is run (it runs the forward models, so :meth:`Model.infer`
            does not).
    """

    t: np.ndarray
    T: np.ndarray
    hyper: dict[str, np.ndarray]
    boxes: tuple[Box, ...] = ()
    reheat_tol: float | None = None
    T_present: tuple[float, float] | None = None
    acceptance: float = 1.0
    n_drawn: int = 0
    oldest_age: float = float("nan")
    warnings: list[str] = field(default_factory=list)
    seconds: float = 0.0
    T_max: float = 200.0
    choice: Choice | None = None
    fit: Fit | None = None

    @property
    def time_span(self) -> int:
        """The answering member's time span, Ma: how far back its histories go (100 or 250)."""
        return round(float(self.t[-1]))

    @property
    def cooling_only(self) -> bool:
        return self.reheat_tol is not None

    @property
    def constrained(self) -> bool:
        return bool(self.boxes) or self.cooling_only

    def __len__(self) -> int:
        return self.T.shape[0]

    def quantiles(self, qs: Sequence[float] = QUANTILES) -> np.ndarray:
        """``(len(qs), n_grid)`` pointwise quantiles, °C."""
        return np.quantile(self.T, qs, axis=0)

    def at(self, t: float) -> np.ndarray:
        """``(n,)`` the draws' temperature at time ``t``, interpolated."""
        k = int(np.clip(np.searchsorted(self.t, t), 1, self.t.size - 1))
        a = (t - self.t[k - 1]) / (self.t[k] - self.t[k - 1])
        return self.T[:, k - 1] * (1 - a) + self.T[:, k] * a

    def functionals(self) -> dict[str, np.ndarray]:
        """The per-draw summaries the calibration criterion scores (differences, window max and mean, largest rate)."""
        v = functional_values(torch.as_tensor(self.T), self.t).numpy()
        return {name: v[:, i] for i, name in enumerate(FUNCTIONALS)}

    def summary(self) -> str:
        q = self.quantiles((0.16, 0.5, 0.84))
        lines = [f"{len(self)} histories ({self.seconds * 1e3:.0f} ms)"
                 + (f", acceptance {self.acceptance:.2%} of {self.n_drawn:,}" if self.constrained else "")]
        if self.choice is not None:
            lines.append(f"  {self.choice.describe()}")
        for b in self.boxes:
            lines.append(f"  box: {b.describe()}")
        if self.cooling_only:
            lines.append(f"  cooling only: reheating at most {self.reheat_tol:g} K")
        for tt in (0.0, 10.0, 30.0, 60.0, 100.0, 150.0, 250.0):
            if tt > self.t[-1]:
                continue
            k = int(np.argmin(np.abs(self.t - tt)))
            flag = "  (past the oldest age: prior's conditional)" if self.t[k] > self.oldest_age else ""
            lines.append(f"  {self.t[k]:5.1f} Ma  {q[1, k]:6.1f} °C  [{q[0, k]:.1f}, {q[2, k]:.1f}]{flag}")
        lines += [f"  ! {w}" for w in self.warnings]
        return "\n".join(lines)

    def plot(self, ax: Any = None, *, truth: tuple[np.ndarray, np.ndarray] | None = None, n_paths: int = 40,
             title: str | None = None, window: Window | None = None) -> Any:
        """The envelope in the house style: 95% and 68% bands, the median, a few draws, the boxes, the time span.

        ``window``: a :func:`sensitive_window`; the temperatures outside it are hatched.
        """
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle

        from chai.viz import style

        style.use()
        if ax is None:
            _, ax = plt.subplots(figsize=(6.0, 3.6), dpi=110)
        q = self.quantiles()
        blue = style.SERIES[0]
        ax.fill_between(self.t, q[0], q[4], color=blue, alpha=0.14, linewidth=0, label="95%")
        ax.fill_between(self.t, q[1], q[3], color=blue, alpha=0.28, linewidth=0, label="68%")
        rng = np.random.default_rng(0)
        for i in rng.choice(len(self), min(n_paths, len(self)), replace=False):
            ax.plot(self.t, self.T[i], color=blue, linewidth=0.4, alpha=0.25)
        ax.plot(self.t, q[2], color=style.INK, linewidth=1.3, label="median")
        if truth is not None:
            ax.plot(truth[0], truth[1], color=style.SERIES[1], linewidth=1.6, linestyle="--", label="truth")
        for b in self.boxes:
            (t0, t1), (lo, hi) = b.t, b.T
            ax.add_patch(Rectangle((t0, lo), max(t1 - t0, 0.4), hi - lo, fill=False, edgecolor=style.SERIES[2],
                                   linewidth=1.4, linestyle="-" if b.mode == WITHIN else (0, (3, 2))))
        if np.isfinite(self.oldest_age) and self.oldest_age < self.t[-1]:
            ax.axvline(self.oldest_age, color=style.MUTED, linewidth=0.8, linestyle=":")
            ax.text(self.oldest_age, 4, " oldest age", color=style.MUTED, fontsize=7, ha="right", va="top")
        if window is not None:
            for lo, hi in ((0.0, window.lo), (window.hi, self.T_max)):
                ax.axhspan(lo, hi, facecolor="none", edgecolor=style.MUTED, hatch="///", linewidth=0, alpha=0.5)
        style.thermal_axes(ax, float(self.t[-1]), self.T_max)
        ax.legend(loc="lower left", fontsize=7, ncols=4)
        if title or self.constrained:
            ax.set_title(title or f"acceptance {self.acceptance:.1%}")
        elif self.choice is not None:
            ax.set_title(f"{self.time_span} Ma member, p = {self.choice.p:.2f}")
        return ax


# ---------------------------------------------------------------------------
# The family: two members and the chooser between them
# ---------------------------------------------------------------------------

#: What the package ships, by the name :func:`load` takes as ``prior``: the members by time span in Ma,
#: and the chooser.  ``"v2.2"``: ``results/flow_v22_100`` (12 couplings) and ``results/flow_v22_250`` (16)
#: at 200 000 steps (``model_200k.pt``) with ``meta["remove_floor"]``, and ``results/chooser_v22``.
#: A second family trained later goes here under its own name; the default stays :data:`DEFAULT_PRIOR`.
FAMILY_ASSETS = {
    "v2.2": ({100: ASSETS / "flow_v22_100.pt", 250: ASSETS / "flow_v22_250.pt"}, ASSETS / "chooser_v22.pt"),
}
#: The default family's members and chooser.
MEMBER_ASSETS, CHOOSER_ASSET = FAMILY_ASSETS[DEFAULT_PRIOR]
#: ``p(250 | sample)`` inside this band, on a sample older than :data:`OLD_SAMPLE_MA`,
#: is reported as a sample the two members cannot decide between.
UNDECIDED = (0.2, 0.8)
OLD_SAMPLE_MA = 50.0
#: Measured, not assumed: past 100 Ma the shipped 250 Ma member's central 68% covers the
#: truth 0.657-0.678 of the time, on its own prior's held-out rows.
LONG_COVERAGE_68 = (0.657, 0.678)


def _horizon(model: Model) -> int:
    return round(model.t_max)


@dataclass(frozen=True)
class Choice:
    """Which member answered a :meth:`Family.infer` query, and why.

    Attributes:
        member: the answering member's time span, Ma.
        p: that member's probability given the sample (and the boxes, if any)
            under ``prior_odds``.
        log_bayes_factor: the chooser's ``log p(x | long) - log p(x | short)``.
        prior_odds: the long member's prior odds used.
        reason: ``"data"`` (the more probable member), ``"asked"`` (``member=``
            named it), ``"boxes"`` (only it can satisfy the boxes).
    """

    member: int
    p: float
    log_bayes_factor: float
    prior_odds: float
    reason: str

    def describe(self) -> str:
        why = {"data": "the more probable", "asked": "as asked", "boxes": "the one the boxes allow"}[self.reason]
        return (f"{self.member} Ma member ({why}): p = {self.p:.3f}, log Bayes factor {self.log_bayes_factor:+.2f}, "
                f"prior odds {self.prior_odds:g}")


#: Present-day temperatures a range is read at by the chooser (:meth:`Family.log_bayes_factor`).
N_PRESENT_BF = 9

#: The narrowest present-day range :meth:`Model.infer` accepts, K: the user gives a range, never a point.
MIN_PRESENT_RANGE = 1.0
#: The fewest draws a member that takes its training floor back out (``meta["remove_floor"]``,
#: :func:`chai.models.gaussian.remove_floor`) is asked for at once: the correction reads the draws'
#: covariance, so it needs at least two per node, and its Marchenko--Pastur edge moves with the count.
#: Fewer paths asked for are drawn this many and subsampled.
MIN_FLOOR_DRAWS = 200


class Family:
    """The members and the chooser: ``infer`` answers with one member, the data's choice or yours.

    Args:
        members: :class:`Model` per time span; two, and the short one's grid must
            be the first nodes of the long one's (it is: ``piecewise:64@100,32@250``).
        chooser: a :class:`~chai.models.chooser.MemberChooser` whose members are
            those time spans, in the same order.
    """

    def __init__(self, members: dict[int, Model], chooser: Any, device: Any = "cpu"):
        if len(members) != 2:
            raise ValueError(f"a family is two members, got {sorted(members)}")
        self.members = dict(members)
        self.short, self.long = (self.members[h] for h in sorted(self.members))
        k = self.short.t.size
        if self.long.t.size <= k or not np.allclose(self.long.t[:k], self.short.t):
            raise ValueError("the short member's grid must be the first nodes of the long member's")
        if tuple(int(m) for m in chooser.members) != tuple(sorted(self.members)):
            raise ValueError(f"the chooser is for members {chooser.members}, the family has {sorted(self.members)}")
        takes = {m.present_input for m in self.members.values()} | {bool(getattr(chooser, "present_input", False))}
        if len(takes) > 1:
            raise ValueError("the members and the chooser must all take the present-day temperature, or none")
        self.device = torch.device(device)
        self.chooser = chooser.to(self.device).eval()

    def __getitem__(self, time_span: int) -> Model:
        return self.members[time_span]

    @property
    def t(self) -> np.ndarray:
        return self.long.t

    @property
    def t_max(self) -> float:
        return self.long.t_max

    @torch.no_grad()
    def log_bayes_factor(self, sample: Sample | Aliquot, T_present: Sequence[float] | None = None) -> float:
        """The chooser's ``log p(x | long) - log p(x | short)`` for one sample.

        A chooser given the present day reads ``T_present`` as the members do.
        Over a range it is the mean of its log Bayes factor on
        :data:`N_PRESENT_BF` evenly spaced temperatures: the exact factor
        between the range-averaged evidences needs each member's evidence on
        its own, which a classifier does not give, and the two agree while the
        data barely see the present day.
        """
        obs = sample if isinstance(sample, Aliquot) else self.long.to_aliquot(sample)[0]
        present = self.long.present_range(T_present)
        feats = self.long.batch(obs)
        if not self.chooser.present_input:
            return float(self.chooser(feats)[0])
        if present is None:
            raise ValueError("this family's chooser reads the present day; its members must too")
        T0 = torch.linspace(*present, N_PRESENT_BF, device=self.device)
        rows = {k: v.expand(len(T0), *v.shape[1:]) for k, v in feats.items()}
        return float(self.chooser({**rows, "T0": T0}).mean())

    def p_long(self, sample: Sample | Aliquot, prior_odds: float = 1.0,
               T_present: Sequence[float] | None = None) -> float:
        """``p(long member | sample)`` under ``prior_odds``."""
        if not prior_odds > 0:
            raise ValueError(f"prior_odds must be positive, got {prior_odds}")
        return float(1.0 / (1.0 + np.exp(-(self.log_bayes_factor(sample, T_present) + np.log(prior_odds)))))

    def infer(self, sample: Sample | Aliquot, n: int = 2000, *, member: str | int = "auto",
              T_present: Sequence[float] | None = None,
              prior_odds: float = 1.0, boxes: Sequence[Box] = (), cooling_only: bool = False, seed: int | None = 0,
              **kw: Any) -> PathPosterior:
        """The posterior given ``sample``, from one member: the more probable (``"auto"``) or the one named.

        Every draw comes from that one member, on its own grid, so a 250 Ma answer
        runs to 250 Ma and a 100 Ma answer to 100 Ma.  The members are not mixed
        (selection costs 1-3 points of coverage at the worst node against averaging).

        Args:
            member: ``"auto"``, or a time span in Ma (``100``, ``250``).
            T_present: as for :meth:`Model.infer`; it enters the chooser too.
            prior_odds: the long member's prior odds against the short one, for a
                user who knows the setting (a craton, an active orogen).  1 by default.
            boxes: as for :meth:`Model.infer`.  A box past the short member's
                time span can only be answered by the long member.  Otherwise, under
                ``"auto"``, both members are run and each one's acceptance rate
                enters the odds, since it is that member's probability of the
                boxes given the data; the more probable one answers.
            cooling_only: as for :meth:`Model.infer`, and it enters the odds as
                boxes do.
            kw: passed to the member's :meth:`Model.infer`.

        Returns:
            A :class:`PathPosterior` whose :attr:`~PathPosterior.choice` says
            which member answered and with what probability.
        """
        t0 = time.perf_counter()
        boxes = tuple(boxes)
        kw["cooling_only"] = cooling_only
        kw["T_present"] = T_present
        h_short, h_long = sorted(self.members)
        if not prior_odds > 0:
            raise ValueError(f"prior_odds must be positive, got {prior_odds}")
        log_bf = self.log_bayes_factor(sample, T_present)
        log_odds = log_bf + float(np.log(prior_odds))
        p = float(1.0 / (1.0 + np.exp(-log_odds)))

        def done(post: PathPosterior, h: int, p_h: float, reason: str, notes: Sequence[str] = ()) -> PathPosterior:
            post.warnings[:0] = notes
            post.choice = Choice(member=h, p=p_h, log_bayes_factor=log_bf, prior_odds=float(prior_odds), reason=reason)
            post.seconds = time.perf_counter() - t0
            return post

        if member != "auto":
            if member not in self.members:
                raise ValueError(f"member must be 'auto' or one of {sorted(self.members)}, got {member!r}")
            post = self.members[member].infer(sample, n, boxes=boxes, seed=seed, **kw)
            p_chosen = p if member == h_long else 1.0 - p
            notes = []
            if p_chosen < UNDECIDED[0]:
                other = h_short if member == h_long else h_long
                notes.append(f"you chose {member} Myr; the data favour {other} Myr")
            return done(post, int(member), p_chosen, "asked", notes)

        past = [b for b in boxes if b.t[1] > self.short.t_max + 1e-9]
        if past:
            note = f"a constraint is older than {h_short} Ma, so the {h_long} Myr model answers"
            return done(self.long.infer(sample, n, boxes=boxes, seed=seed, **kw), h_long, 1.0, "boxes", [note])
        if boxes or cooling_only:
            posts: dict[int, PathPosterior] = {}
            refused: dict[int, InfeasibleBoxes] = {}
            for h, m in self.members.items():
                try:
                    posts[h] = m.infer(sample, n, boxes=boxes, seed=seed, **kw)
                except InfeasibleBoxes as err:
                    refused[h] = err
            if not posts:
                # Neither can: the long member's refusal, which names what is wrong with the boxes.
                raise refused[h_long]
            if len(posts) == 1:
                (h, post), = posts.items()
                other = h_short if h == h_long else h_long
                note = f"the {other} Myr model can't meet the constraints, so the {h} Myr model answers"
                return done(post, h, 1.0, "boxes", [note])
            # p(member | x, boxes) is p(member | x) times that member's p(boxes | x), its acceptance rate.
            log_odds += float(np.log(posts[h_long].acceptance / posts[h_short].acceptance))
            p = float(1.0 / (1.0 + np.exp(-log_odds)))
        h = h_long if p >= 0.5 else h_short
        p_h = p if h == h_long else 1.0 - p
        post = posts[h] if boxes or cooling_only else self.members[h].infer(sample, n, boxes=boxes, seed=seed, **kw)
        notes = []
        if OLD_SAMPLE_MA < post.oldest_age and UNDECIDED[0] < p < UNDECIDED[1]:
            other = h_short if h == h_long else h_long
            notes.append(f"the data barely prefer {h} Myr over {other} Myr; the other may give a different answer")
        return done(post, h, p_h, "data", notes)


def load_family(members: dict[int, str | Path] | None = None, chooser: str | Path | None = None, *,
                prior: str = DEFAULT_PRIOR, device: Any = "cpu") -> Family:
    """The family from run directories or files: ``{100: run, 250: run}`` and a chooser run.

    With no arguments, what the package ships for ``prior`` (:data:`FAMILY_ASSETS`).
    """
    from chai.models.chooser import MemberChooser

    members = members or FAMILY_ASSETS[prior][0]
    chooser = Path(chooser) if chooser is not None else FAMILY_ASSETS[prior][1]
    models = {int(h): load_member(r, device=device) for h, r in members.items()}
    for h, m in models.items():
        if _horizon(m) != h:
            raise ValueError(f"member {h} has a {m.t_max:g} Ma time span")
    net = MemberChooser.load(chooser / "model.pt" if chooser.is_dir() else chooser, map_location=device)
    return Family(models, net, device=device)


# ---------------------------------------------------------------------------
# What the posterior predicts for the data
# ---------------------------------------------------------------------------

#: The forward models' resolution for :func:`predict`: within 0.06% of ``n_sub = 16`` on
#: posterior paths, at half the cost (the browser uses the same).
PREDICT_N_SUB, PREDICT_RDIM = 8, 256


@dataclass
class Prediction:
    """The data a posterior's paths predict: the forward models run on ``n`` of its draws.

    No measurement noise is added: this is the spread the history's uncertainty
    alone implies, to set beside the data and their own error bars.

    Attributes:
        paths: ``(n,)`` the rows of :attr:`PathPosterior.T` used (the first ``n``).
        aft_age: ``(n,)`` Ma, or ``None`` without an AFT analysis.
        length_pdf: ``(n, 200)`` c-axis projected length density per µm on
            :attr:`bins`, or ``None``.
        he_age: ``(n, n_grains)`` alpha-corrected helium ages, Ma.
        bins: the length bins' centres, µm.
    """

    paths: np.ndarray
    aft_age: np.ndarray | None
    length_pdf: np.ndarray | None
    he_age: np.ndarray
    bins: np.ndarray

    def quantiles(self, x: np.ndarray, qs: Sequence[float] = QUANTILES) -> np.ndarray:
        """Quantiles over the paths of one of the attributes, along its first axis."""
        return np.quantile(x, qs, axis=0)


def predict(post: PathPosterior, sample: Sample, n: int = 100, *, device: Any = "cpu") -> Prediction:
    """What ``n`` of the posterior's paths predict for the sample's data.

    The first ``n``: the draws are independent, so any ``n`` of them will do.
    ``device``: where the forward models run (a GPU is much faster for many paths).

    AFT: the Ketcham et al. (2007) age and projected length distribution at the
    sample's Dpar.  Helium: each grain's RDAAM corrected age at its radius and eU,
    with Th/U taken as :data:`RAW_AGE_TH_U` to split eU (a table gives eU only).
    """
    from chai.forward.torchimpl.ketcham import aft_forward
    from chai.forward.torchimpl.rdaam import rdaam_age

    rows = np.arange(min(n, len(post)))
    kw = {"dtype": torch.float64, "device": device}
    T = torch.as_tensor(post.T[rows], **kw)
    t = torch.as_tensor(np.broadcast_to(post.t, T.shape).copy(), **kw)
    B = len(rows)
    bins = torch.as_tensor(LENGTH_BINS, **kw)
    aft_age = length_pdf = None
    if sample.aft is not None:
        a = aft_forward(t, T, torch.full((B,), float(sample.aft.dpar), **kw), n_sub=PREDICT_N_SUB, bins=bins)
        aft_age, length_pdf = a["age"].cpu().numpy(), a["length_pdf"].cpu().numpy()
    he = np.zeros((B, len(sample.grains)))
    if sample.grains:
        # Every grain on every path in one call: grain-major rows, B per grain.
        G = len(sample.grains)
        U = torch.as_tensor([g.eU / (1.0 + _EU_TH_WEIGHT * RAW_AGE_TH_U) for g in sample.grains], **kw).repeat_interleave(B)
        radius = torch.as_tensor([float(g.radius) for g in sample.grains], **kw).repeat_interleave(B)
        age = rdaam_age(t.repeat(G, 1), T.repeat(G, 1), radius, U, U * RAW_AGE_TH_U, n_sub=PREDICT_N_SUB,
                        rdim=PREDICT_RDIM, use_checkpoint=False)["corrected_age"]
        he = age.reshape(G, B).T.cpu().numpy()
    return Prediction(paths=rows, aft_age=aft_age, length_pdf=length_pdf, he_age=he, bins=LENGTH_BINS.copy())


def check_fit(post: PathPosterior, sample: Sample, model: Model | Family, *, prediction: Prediction | None = None,
              n: int = 100) -> Fit:
    """Whether the posterior's paths predict the sample's data: a warning, not a gate (:mod:`chai.fit`).

    For each datum, the two-sided tail probability of the observed value under
    the paths' predictions with the model's observation noise added; a datum
    past the level (set so a sample the model fits warns one time in twenty)
    is listed with its predicted interval.  The result is
    stored on ``post.fit`` and its lines added to ``post.warnings``: the
    caution first, the data after the other notes.

    Args:
        post: from :meth:`Model.infer` or :meth:`Family.infer` on ``sample``.
        model: what answered (a :class:`Family` lends the member that did), for
            its observation model.
        prediction: :func:`predict`'s, if already run; otherwise run on ``n`` paths.
    """
    if isinstance(model, Family):
        model = model[post.time_span]
    from chai import fit as _fit

    pred = prediction if prediction is not None else predict(post, sample, n)
    f = _fit.check(sample, pred, model.observation)
    post.fit = f
    lines = [w for w in f.warnings if w != _fit.CAUTION]
    post.warnings[:0] = [_fit.CAUTION] if f.caution else []
    post.warnings.extend(lines)
    return f


#: The sensitive window's recipe: an isothermal hold of 10 Myr (the timescale of the textbook partial
#: annealing and retention zones), a chronometer "resolving" a temperature between keeping 90% and 10%
#: of the age it has when cold, on a 1 °C grid, with the forward models at ``n_sub = 32``.
WINDOW_HOLD_MA, WINDOW_RETAINED, WINDOW_N_SUB = 10.0, (0.9, 0.1), 32


@dataclass(frozen=True)
class Window:
    """The temperatures a sample's chronometers resolve, °C.

    Attributes:
        lo: the coldest temperature any of them resolves (its 90% point).
        hi: the hottest (its 10% point).  Colder than ``lo`` every chronometer keeps its
            whole age, so the data cannot tell 10 °C from ``lo``; hotter than ``hi`` every
            one is reset, so the data say only "at least this hot".
        systems: ``(lo, hi)`` per chronometer, ``"AFT"`` and ``"He 1"``, ``"He 2"``, ...
    """

    lo: float
    hi: float
    systems: dict[str, tuple[float, float]]


def sensitive_window(sample: Sample, hold: float = WINDOW_HOLD_MA) -> Window:
    """Where the sample's chronometers are sensitive: each one's partial zone for an isothermal hold.

    For each chronometer, the forward model's age after ``hold`` Ma at a constant temperature,
    as a fraction of its age at 0 °C; the zone runs from where that fraction falls to 0.9 to
    where it falls to 0.1.  AFT at the sample's Dpar; each helium grain at its radius and eU
    (Th/U as in :func:`predict`).  A guide to where the posterior is the data's and where it is
    mostly the prior's, not a calibration statement.
    """
    from chai.forward.torchimpl.ketcham import aft_forward
    from chai.forward.torchimpl.rdaam import rdaam_age

    Tg = np.arange(0.0, 201.0)
    B = Tg.size
    t = torch.tensor([[0.0, float(hold)]] * B, dtype=torch.float64)
    T = torch.as_tensor(np.stack([Tg, Tg], 1))
    full = lambda v: torch.full((B,), float(v), dtype=torch.float64)
    lo_lev, hi_lev = WINDOW_RETAINED

    def zone(age: np.ndarray) -> tuple[float, float]:
        f = age / age[0]
        out = []
        for lev in (lo_lev, hi_lev):
            k = int(np.argmax(f < lev))
            out.append(float(Tg[k - 1] + (f[k - 1] - lev) / (f[k - 1] - f[k]) * (Tg[k] - Tg[k - 1])))
        return out[0], out[1]

    systems: dict[str, tuple[float, float]] = {}
    if sample.aft is not None:
        a = aft_forward(t, T, full(sample.aft.dpar), n_sub=WINDOW_N_SUB, bins=torch.as_tensor(LENGTH_BINS))
        systems["AFT"] = zone(a["age"].numpy())
    for k, g in enumerate(sample.grains):
        U = g.eU / (1.0 + _EU_TH_WEIGHT * RAW_AGE_TH_U)
        he = rdaam_age(t, T, full(g.radius), full(U), full(U * RAW_AGE_TH_U), n_sub=WINDOW_N_SUB, rdim=PREDICT_RDIM,
                       use_checkpoint=False)["corrected_age"].numpy()
        systems[f"He {k + 1}"] = zone(he)
    return Window(lo=min(v[0] for v in systems.values()), hi=max(v[1] for v in systems.values()), systems=systems)


# ---------------------------------------------------------------------------
# A sample from a path of one's choosing
# ---------------------------------------------------------------------------

def simulate_sample(model: Model | Family, t: Sequence[float], T: Sequence[float], *, dpar: float = 2.0,
                    grains: Sequence[tuple[float, float, float]] = ((60.0, 30.0, 60.0),) * 5,
                    n_lengths: int = 100, n_grains_counted: int = 20, area_per_grain_um2: float = 2000.0,
                    seed: int = 0) -> tuple[Sample, tuple[np.ndarray, np.ndarray]]:
    """A measurement of a history you choose, through the same forward models and noise the head was trained on.

    Args:
        model: supplies the grid and the observation model; a :class:`Family`
            lends its long member's, whose grid holds the short one's.
        t, T: control points, Ma (``t[0] = 0``) and °C; interpolated onto the
            model's grid, so detail finer than its spacing (1.05 Ma; 2.34 Ma past 100 Ma on
            the 250 Ma member) is lost the way the prior loses it.
        dpar: µm.
        grains: ``(radius µm, U ppm, Th ppm)`` per helium grain; empty for none.
        n_lengths: confined lengths measured; 0 for none.
        n_grains_counted, area_per_grain_um2: the fission-track count's effort.
            ``n_grains_counted = 0`` means no fission-track analysis.
        seed: for the measurement noise.

    Returns:
        The :class:`Sample`, and the true path on the model's grid ``(t, T)``.
    """
    from chai.forward.torchimpl.ketcham import aft_forward, length_bin_centres
    from chai.forward.torchimpl.rdaam import rdaam_age

    if isinstance(model, Family):          # the long member's grid holds the short one's
        model = model.long
    tg = model.t
    Tg = np.interp(tg, np.asarray(t, float), np.asarray(T, float))
    tt = torch.as_tensor(tg[None], dtype=torch.float64)
    TT = torch.as_tensor(Tg[None], dtype=torch.float64)
    a = aft_forward(tt, TT, torch.tensor([dpar], dtype=torch.float64), n_sub=16,
                    bins=length_bin_centres(dtype=torch.float64))
    g = np.asarray(grains, float).reshape(-1, 3)
    G = max(len(g), 1)
    if len(g):
        he = rdaam_age(tt.expand(len(g), -1), TT.expand(len(g), -1),
                       *(torch.as_tensor(g[:, i]) for i in range(3)), n_sub=16, rdim=256, use_checkpoint=False)
        he_age = he["corrected_age"].numpy()[None]
    else:
        g, he_age = np.zeros((1, 3)), np.zeros((1, 1))
    truth = {
        "aft_age": a["age"].numpy(), "length_pdf": a["length_pdf"].numpy(), "dpar": np.array([dpar]),
        "U": np.array([g[:, 1].mean() if len(grains) else 30.0]),
        "he_age": he_age, "he_radius": g[None, :, 0], "he_eU": (g[:, 1] + _EU_TH_WEIGHT * g[:, 2])[None],
    }
    effort = {"n_grains_counted": np.array([max(n_grains_counted, 1)]), "area_per_grain_um2": np.array([area_per_grain_um2]),
              "n_lengths": np.array([n_lengths if n_grains_counted > 0 else 0]),
              "n_he_grains": np.array([len(grains)]), "aft_present": np.array([n_grains_counted > 0])}
    obs = model.observation.observe(truth, np.random.default_rng(seed), effort=effort)
    assert obs.he_age.shape[1] == G
    return Sample.from_aliquot(obs), (tg, Tg)


@dataclass
class Example:
    """A synthetic sample with the history that made it: what :func:`example_sample` returns.

    Attributes:
        sample: the measurements, as a laboratory would report them.
        T_present: a present-day temperature range around the true one, °C, for ``infer``.
        t, T: the true history on the member's grid, Ma and °C; ``post.plot(truth=(ex.t, ex.T))``.
        description: what the history is.
    """

    sample: Sample
    T_present: tuple[float, float]
    t: np.ndarray
    T: np.ndarray
    description: str

    @property
    def truth(self) -> tuple[np.ndarray, np.ndarray]:
        return self.t, self.T


#: The named histories of :func:`example_sample`: ``(control times Ma, °C, time span, description)``.
EXAMPLES = {
    "pulse": ((0.0, 15.0, 20.0, 25.0, 100.0), (12.0, 30.0, 45.0, 110.0, 125.0), 100,
              "held near 110-125 °C, exhumed by 65 K between 25 and 20 Ma, then slow cooling to the surface"),
    "old": ((0.0, 40.0, 70.0, 120.0, 180.0, 250.0), (15.0, 35.0, 85.0, 55.0, 95.0, 190.0), 250,
            "slow cooling from 190 °C, buried and reheated to 85 °C by 70 Ma, then exhumed"),
}


def example_sample(family: Family, name: str = "pulse", *, seed: int = 0) -> Example:
    """A synthetic sample to try the family on, with the true history to compare the answer against.

    ``"pulse"`` is a young exhumation the 100 Ma member answers; ``"old"`` a 250 Ma history with a burial
    and reheating, which the chooser hands to the 250 Ma member; ``"random"`` a history drawn from the 100 Ma
    member's own prior (``seed`` picks it).  Each is measured through the forward models and laboratory noise
    the members were trained on (:func:`simulate_sample`): one fission-track analysis with 100 confined
    lengths and Dpar 2.0 µm, and four helium grains.  ``seed`` also sets the measurement noise.

        family = api.load()
        ex = api.example_sample(family, "pulse")
        post = family.infer(ex.sample, T_present=ex.T_present)
        post.plot(truth=ex.truth)
    """
    grains = ((45.0, 15.0, 30.0), (60.0, 35.0, 60.0), (80.0, 60.0, 90.0), (100.0, 25.0, 50.0))
    rng = np.random.default_rng(seed)
    if name == "random":
        from chai.data.priors import prior_from_config

        m = family[min(family.members)]
        cfg = dict(m.prior)
        cfg.pop("T_top", None)
        kind = "PulsePrior" if "pulse_rate" in cfg else "ThermalHistoryPrior"
        prior = prior_from_config(cfg, kind)
        while True:
            T = prior.sample_on(1, rng, m.t)[0][0]
            s, (tg, Tg) = simulate_sample(m, m.t, T, dpar=2.0, grains=grains, seed=int(rng.integers(1 << 31)))
            if s.aft is not None and len(s.aft.lengths) >= 20 and min(g.age for g in s.grains) >= 1.0:
                break
        description = f"a history drawn from the {m.t_max:g} Ma member's prior (seed {seed})"
    elif name in EXAMPLES:
        t, T, span, description = EXAMPLES[name]
        m = family[span]
        s, (tg, Tg) = simulate_sample(m, t, T, dpar=2.0, grains=grains, seed=seed)
    else:
        raise ValueError(f"name must be one of {sorted(EXAMPLES) + ['random']}, got {name!r}")
    s = Sample(s.aft, s.grains, f"example: {name}")
    if m.present_input:
        lo, hi = m.T_present_range
        T0 = float(Tg[0])
        present = (round(max(lo, T0 - 2.5), 1), round(min(hi, T0 + 2.5), 1))
    else:
        present = None
    return Example(s, present, np.asarray(tg), np.asarray(Tg), description)
