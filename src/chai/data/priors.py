"""A generative prior over thermal histories.

The prototype's prior was a mixture of hand-tuned path *shapes* --
``{mono 0.4, pulse 0.2, heating 0.15, hold 0.15, random 0.1}`` applied to sorted
Latin-hypercube temperatures (the earlier prototype's generator).  It is not
written down anywhere as a probability model, its shapes are not disjoint, and
its mixture weights are free parameters that nothing sets.  Since a Goal 2
posterior is only ever as defensible as the prior it was amortized over
(PLAN.md Sec 5, highest-severity risk), that had to be replaced by something
stateable in one paragraph.

The model
---------
The prior is placed on the **cooling-rate history**, not on temperature.  That
is the quantity geologists actually reason about -- exhumation rate -- it is
where the physical bounds live, and integrating it gives a temperature history
that is smooth by construction rather than by smoothing.

With ``t`` the time before present in Ma (increasing into the past, PLAN.md
Sec 2), so that ``dT/dt > 0`` means the rock is cooling forward in geological
time:

.. code-block:: text

    tau     ~ LogUniform(tau_range)                 correlation length, Ma
    a       ~ LogUniform(rate_sd_range)             rate variability, K/Ma
    dT_drift~ Uniform(drift_total_range)            net cooling over the time span, K
    T_0     ~ Uniform(T_present_range)              present-day temperature, degC
    z(.)    ~ GP(0, 1), squared-exponential, length tau

    u(t)  = dT_drift / t_max + a * z(t)             the raw rate, K/Ma
    v(t)  = r_max * tanh(u(t) / r_max)              soft cap: |v| < r_max, exactly
    dT/dt = v(t) * gate(T, sign v)                  box:  T_min <= T <= T_max, exactly
    T(0)  = T_0

``gate`` is 1 unless the path is pressed against the wall it is heading for, in
which case it is 0: the rock sits at the surface for as long as the drive keeps
pushing it down, and starts heating on the step the drive reverses.  That is a
projection rather than a smooth barrier, and :meth:`ThermalHistoryPrior._flow`
records why the smooth version had to go -- it is absorbing.

What that buys, and what the plan asked for:

* **A rate limit that is a guarantee, not a tendency.**  ``|dT/dt| <= rate_max``
  holds pathwise for every draw, because ``tanh`` is bounded and the gate is 0
  or 1.  A Gaussian process placed directly on ``T`` -- the obvious
  alternative, and what ``scripts/node_count_study.py`` uses as its truth --
  bounds the rate only in distribution, so its tail admits arbitrarily fast
  excursions.
* **A characteristic timescale that is a parameter.**  ``tau`` is the
  correlation length of the rate, and PLAN.md Stage 1 requires the prior to
  report it, because ``tau`` read against ``docs/node-count-study.md`` Result 1
  is what sets ``n_grid``.  The realised distribution is measured in
  ``docs/prior.md``; the analytic knobs are in :meth:`ThermalHistoryPrior.describe`.
* **The prototype's path families as a consequence, not a construction.**  The
  ratio ``drift / a`` controls how much of the history is monotone cooling:
  large ratio gives the ``mono`` family, small ratio gives ``pulse`` and
  ``heating``, and the lower wall gives ``hold`` (a path that reaches the
  surface cannot keep cooling).  The monotone fraction is therefore a *reported
  number* -- a measured 32% over 20 000 draws, against the prototype's asserted
  0.4 -- rather than a tuned mixture weight.

Known limitation: the latent ``z`` is stationary, so a short ``tau`` means
*many* thermal episodes rather than one short one.  Geological histories are
intermittent -- a few discrete events, which may individually be brief -- and no
stationary process reproduces that.  ``tau_range`` is therefore floored well
above zero so the prior does not fill up with implausibly oscillatory histories;
the principled fix, if prior-predictive checks ever demand it, is a marked point
process (``K ~ Poisson`` episodes) in place of the GP, behind this same
interface.  See ``docs/prior.md``.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Any

import numpy as np

__all__ = ["PRIOR_TYPES", "EpisodicPrior", "GrainPrior", "HistoryBatch", "PulsePrior",
           "ThermalHistoryPrior", "integrate_rate", "prior_from_config"]

#: Cap on the working-array size used while evaluating the random-feature sum,
#: in elements.  Only affects peak memory and chunking, never the result.
_MAX_WORK = 1 << 24


def _loguniform(rng: np.random.Generator, lo: float, hi: float, n: int) -> np.ndarray:
    return np.exp(rng.uniform(np.log(lo), np.log(hi), size=n))


def _gp_on(t: np.ndarray, omega: np.ndarray, phase: np.ndarray) -> np.ndarray:
    """Evaluate the random-Fourier-feature GP draws at times ``t``.

    ``sqrt(2/M) * sum_j cos(omega_j t + phi_j)`` with ``omega ~ N(0, 1/tau^2)``
    and ``phi ~ U(0, 2pi)`` is a unit-variance stationary process whose
    covariance converges to the squared-exponential kernel with length ``tau``.
    Using features rather than a covariance factorisation means a draw is an
    ordinary function -- evaluable at any time, with no interpolation of its own.

    Args:
        t: ``(K,)`` times.
        omega: ``(n, M)`` frequencies.
        phase: ``(n, M)`` phases.

    Returns:
        ``(n, K)``.
    """
    n, m = omega.shape
    out = np.empty((n, t.size))
    step = max(1, _MAX_WORK // (n * m))
    for lo in range(0, t.size, step):
        chunk = t[lo : lo + step]
        out[:, lo : lo + step] = np.cos(
            omega[:, None, :] * chunk[None, :, None] + phase[:, None, :]
        ).sum(axis=2)
    return out * np.sqrt(2.0 / m)


@dataclass(frozen=True)
class HistoryBatch:
    """A batch of thermal histories, as solved on the prior's internal grid.

    The internal grid is a quadrature choice, not the model: histories are
    continuous functions and :meth:`at` evaluates them anywhere.  Because the
    solver knows ``dT/dt`` exactly at every node, that evaluation is a cubic
    Hermite interpolant -- fourth-order accurate and consistent with the ODE by
    construction, which is what lets a draw serve as the "continuous truth" in a
    grid-convergence study.

    Attributes:
        t: ``(N + 1,)`` internal times, uniform, ``t[0] == 0``, Ma before present.
        T: ``(n, N + 1)`` temperatures, degC.
        dTdt: ``(n, N + 1)`` cooling rate, K/Ma, positive when cooling forward in
            geological time.
        hyper: the per-draw latent parameters, each ``(n,)``.
    """

    t: np.ndarray
    T: np.ndarray
    dTdt: np.ndarray
    hyper: dict[str, np.ndarray]

    def __len__(self) -> int:
        return self.T.shape[0]

    def at(self, t_query: np.ndarray) -> np.ndarray:
        """Evaluate every history at ``t_query``; returns ``(n, len(t_query))``."""
        t_query = np.clip(np.asarray(t_query, dtype=float), self.t[0], self.t[-1])
        h = self.t[1] - self.t[0]
        k = np.clip((t_query / h).astype(int), 0, self.t.size - 2)
        s = (t_query - self.t[k]) / h
        s2, s3 = s * s, s * s * s
        y0, y1 = self.T[:, k], self.T[:, k + 1]
        d0, d1 = self.dTdt[:, k], self.dTdt[:, k + 1]
        return (
            (2 * s3 - 3 * s2 + 1) * y0
            + h * (s3 - 2 * s2 + s) * d0
            + (-2 * s3 + 3 * s2) * y1
            + h * (s3 - s2) * d1
        )

    def diagnostics(self, reheat_tol: float = 1.0) -> dict[str, np.ndarray]:
        """Per-history summaries, all ``(n,)``.

        These are the numbers PLAN.md Stage 1 asks the prior to report, and the
        ones that decide ``n_grid``: ``max_cooling_rate`` and
        ``max_heating_rate`` read against ``docs/node-count-study.md`` Result 1,
        ``total_variation`` against Result 2's cost rule.

        Args:
            reheat_tol: reheating below this amplitude (K) is not counted as an
                episode, so ``monotone`` means "no geologically real reheating"
                rather than "no negative rate anywhere", which the lower wall
                would make impossible to satisfy exactly.
        """
        # Largest rise in temperature going *forward* in geological time, i.e.
        # from an older (larger t) node to a younger one.
        suffix_min = np.minimum.accumulate(self.T[:, ::-1], axis=1)[:, ::-1]
        reheat = (self.T - suffix_min).max(axis=1)

        sign = np.sign(np.where(np.abs(self.dTdt) < 1e-3, 0.0, self.dTdt))
        nz = [s[s != 0] for s in sign]
        reversals = np.array([int((np.diff(s) != 0).sum()) if s.size else 0 for s in nz])

        return {
            "max_cooling_rate": self.dTdt.max(axis=1),
            "max_heating_rate": -self.dTdt.min(axis=1),
            "total_variation": np.abs(np.diff(self.T, axis=1)).sum(axis=1),
            "T_present": self.T[:, 0],
            "T_oldest": self.T[:, -1],
            "T_max": self.T.max(axis=1),
            "reheat_amplitude": reheat,
            "monotone": reheat < reheat_tol,
            "reversals": reversals,
            **self.hyper,
        }


def _gate(T: np.ndarray, v: np.ndarray, T_min: float, T_max: float) -> np.ndarray:
    """``dT/dt`` at temperature ``T`` under raw rate ``v``: the walls applied.

    The gate closes only against the wall the path is pressed into, so a path
    resting on a wall leaves it on the step the drive reverses.  Since the gate
    is 0 or 1, ``|flow| <= |v|`` and any rate limit passes through it untouched.

    This is a projection, and it has to be: *any* barrier that vanishes
    smoothly at the wall is absorbing.  Under ``dT/dt = v (T - T_min)/w`` the
    approach is exponential, so a path that spends 50 Ma against the floor
    arrives at ``T - T_min ~ 1e-13`` and then needs a comparable span to climb
    back out -- in a 3000-draw check, 30% of histories sat flat on the floor for
    the entire time span, including draws whose fluctuation amplitude was several
    hundred K.  That is an artifact of the barrier, not a statement about
    geology.
    """
    return v * np.where(v > 0.0, T < T_max, T > T_min)


def integrate_rate(u_half: np.ndarray, T0: np.ndarray, *, t_max: float, T_min: float,
                   T_max: float, rate_max: float, n_steps: int) -> HistoryBatch:
    """Cap a raw rate history, integrate it inside the box, and return the paths.

    This is the half of the prior that is *not* the model: given ``u(t)`` on the
    ``2 n_steps + 1`` half-grid the RK4 stages need, it applies the soft cap
    ``v = rate_max tanh(u / rate_max)``, gates ``v`` at the walls and integrates
    from ``T(0) = T0``.  Both priors in this module share it, so they differ in
    how their rate history is drawn and in nothing else -- which is exactly the
    comparison ``docs/prior-sensitivity.md`` needs, and would be worthless if
    the two also differed in their solver.

    The returned batch carries no hyperparameters; the caller adds its own.

    Args:
        u_half: ``(n, 2 n_steps + 1)`` raw rate, K/Ma, positive when cooling
            forward in geological time.
        T0: ``(n,)`` present-day temperature, degC.
    """
    N = n_steps
    h = t_max / N
    t = np.linspace(0.0, t_max, N + 1)
    v = rate_max * np.tanh(u_half / rate_max)

    T = np.empty((T0.size, N + 1))
    T[:, 0] = T0
    for k in range(N):
        va, vm, vb = v[:, 2 * k], v[:, 2 * k + 1], v[:, 2 * k + 2]
        k1 = _gate(T[:, k], va, T_min, T_max)
        k2 = _gate(T[:, k] + 0.5 * h * k1, vm, T_min, T_max)
        k3 = _gate(T[:, k] + 0.5 * h * k2, vm, T_min, T_max)
        k4 = _gate(T[:, k] + h * k3, vb, T_min, T_max)
        step = T[:, k] + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        # The gate cannot stop a stage that starts inside the box and ends
        # outside it, so the box is closed here.  Clipping only ever shortens a
        # step, which is why the rate limit survives it.
        T[:, k + 1] = np.clip(step, T_min, T_max)

    return HistoryBatch(t=t, T=T, dTdt=_gate(T, v[:, ::2], T_min, T_max), hyper={})


@dataclass(frozen=True)
class ThermalHistoryPrior:
    """The generative model documented at the top of this module.

    Every field is a knob of that model and nothing else is: there is no
    post-hoc shaping, resampling or rejection, so a draw's density is whatever
    the model says it is.

    Args:
        t_max: time span, Ma.  Histories run from the present to ``t_max`` ago.
        T_min: lower wall, degC.  The surface; a path cannot cool past it.
        T_max: upper wall, degC.  Above total annealing for apatite, so the wall
            is observationally inert -- it is there to keep the box finite and
            to cap the total temperature variation that drives oracle cost.
        T_present_range: present-day temperature, degC.
        drift_total_range: net cooling over the whole time span, K.  Drawn flat so
            that the temperature at the far end of the time span is close to flat
            across the box, which is the non-committal choice.
        rate_sd_range: standard deviation of the rate fluctuation, K/Ma,
            log-uniform.  The top of the range is well above any sustainable
            cooling rate on purpose: a large ``a`` against a moderate ``tau``
            drives the path into the walls, and a path that saturates both walls
            traverses the box rapidly and then rests -- which is what a discrete
            exhumation episode looks like.  The walls, not the hyperprior, are
            what make the fast draws episodic rather than oscillatory.
        tau_range: correlation length of the rate, Ma, log-uniform.  Floored
            well above zero on purpose -- see the stationarity note in the module
            docstring.
        rate_max: the hard rate limit, K/Ma.  ``|dT/dt| <= rate_max`` for every
            draw at every time.  ``tanh`` reaches it only asymptotically, so it
            also compresses rates somewhat below it (3% at ``0.3 * rate_max``).
        n_features: random Fourier features per draw.  The kernel error is
            ``O(1/sqrt(n_features))``.
        n_steps: internal RK4 steps over the time span.  The gate is not smooth,
            so convergence at wall contact is first-order rather than fourth:
            2048 steps hold the p95 per-path error to 0.05 K and the worst to
            0.17 K against a 16384-step reference, concentrated at the walls
            where the observables are least sensitive.
    """

    t_max: float = 100.0
    T_min: float = 0.0
    T_max: float = 200.0
    T_present_range: tuple[float, float] = (0.0, 25.0)
    drift_total_range: tuple[float, float] = (0.0, 200.0)
    rate_sd_range: tuple[float, float] = (0.5, 30.0)
    tau_range: tuple[float, float] = (8.0, 80.0)
    rate_max: float = 50.0
    n_features: int = 256
    n_steps: int = 2048

    def __post_init__(self) -> None:
        if self.t_max <= 0:
            raise ValueError(f"t_max must be positive, got {self.t_max}")
        if self.T_min >= self.T_max:
            raise ValueError(f"need T_min < T_max, got {self.T_min}, {self.T_max}")
        lo, hi = self.T_present_range
        if not self.T_min <= lo <= hi <= self.T_max:
            raise ValueError(f"T_present_range must sit inside the box, got {self.T_present_range}")
        for name in ("rate_sd_range", "tau_range"):
            lo, hi = getattr(self, name)
            if not 0 < lo <= hi:
                raise ValueError(f"{name} must be positive and ordered, got {(lo, hi)}")
        if self.rate_max <= 0:
            raise ValueError(f"rate_max must be positive, got {self.rate_max}")
        if self.n_steps < 1 or self.n_features < 1:
            raise ValueError("n_steps and n_features must be >= 1")

    @property
    def rate_limit(self) -> float:
        """The pathwise bound on ``|dT/dt|``, K/Ma.  Exact, not a quantile."""
        return self.rate_max

    @property
    def hyper_names(self) -> tuple[str, ...]:
        """The per-draw latents this prior reports, and a corpus stores.

        A corpus is keyed on its prior, and the Goal 2 network is conditioned on
        these (PLAN.md Sec 5), so the writer takes the names from the prior
        rather than hard-coding one prior's four.
        """
        return ("tau", "rate_sd", "drift", "T_0")

    def _flow(self, T: np.ndarray, v: np.ndarray) -> np.ndarray:
        """``dT/dt`` at temperature ``T`` under raw rate ``v``: this box's walls.

        A thin binding of :func:`_gate` to this prior's box, kept because the
        gate is the model decision the module docstring argues for and a reader
        looking for it should find it on the prior.
        """
        return _gate(T, v, self.T_min, self.T_max)

    def draw_latents(self, n: int, rng: np.random.Generator
                     ) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
        """Everything :meth:`sample` draws before it integrates, in the order it draws it.

        ``(hyper, omega (n, M), phase (n, M))``, ``hyper`` in :attr:`hyper_names`
        units.  Split out so a sampler working in the prior's own coordinates
        (:mod:`chai.eval.mcmc`) can recover a stored draw's latent from the
        corpus's provenance.
        """
        tau = _loguniform(rng, *self.tau_range, n)
        rate_sd = _loguniform(rng, *self.rate_sd_range, n)
        drift = rng.uniform(*self.drift_total_range, size=n) / self.t_max
        T0 = rng.uniform(*self.T_present_range, size=n)
        omega = rng.normal(0.0, 1.0, size=(n, self.n_features)) / tau[:, None]
        phase = rng.uniform(0.0, 2.0 * np.pi, size=(n, self.n_features))
        return {"tau": tau, "rate_sd": rate_sd, "drift": drift, "T_0": T0}, omega, phase

    def sample(self, n: int, rng: np.random.Generator) -> HistoryBatch:
        """Draw ``n`` thermal histories.

        Args:
            n: number of histories.
            rng: seeded generator.  Draws depend on it and on nothing else, so
                a seed plus this config reproduces a corpus exactly.
        """
        if n < 1:
            raise ValueError(f"n must be >= 1, got {n}")

        # RK4 needs the midpoints, so the rate is drawn on the half-grid.
        t_half = np.linspace(0.0, self.t_max, 2 * self.n_steps + 1)
        hyper, omega, phase = self.draw_latents(n, rng)
        tau, rate_sd, drift, T0 = (hyper[k] for k in ("tau", "rate_sd", "drift", "T_0"))
        u = drift[:, None] + rate_sd[:, None] * _gp_on(t_half, omega, phase)

        batch = integrate_rate(u, T0, t_max=self.t_max, T_min=self.T_min,
                               T_max=self.T_max, rate_max=self.rate_max,
                               n_steps=self.n_steps)
        return replace(batch, hyper={"tau": tau, "rate_sd": rate_sd,
                                     "drift": drift, "T_0": T0})

    def sample_on(
        self, n: int, rng: np.random.Generator, t_grid: np.ndarray
    ) -> tuple[np.ndarray, HistoryBatch]:
        """Draw ``n`` histories and evaluate them on ``t_grid``.

        This is the call the simulator makes: ``t_grid`` is the canonical fixed
        grid from :func:`chai.paths.fixed_grid`, and the returned ``(n, len)``
        array is the path vector the corpus stores.  The batch comes back too,
        so the continuous draw behind each path stays available for diagnostics.
        """
        batch = self.sample(n, rng)
        return batch.at(np.asarray(t_grid, dtype=float)), batch

    def to_config(self) -> dict[str, Any]:
        """Plain-data form, for provenance attributes and config files."""
        return asdict(self)

    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> ThermalHistoryPrior:
        ranges = {"T_present_range", "drift_total_range", "tau_range", "rate_sd_range"}
        return cls(**{k: tuple(v) if k in ranges else v for k, v in cfg.items()})

    def describe(self) -> str:
        """The generative model with this instance's numbers substituted in."""
        tau, sd = self.tau_range, self.rate_sd_range
        drift, T0 = self.drift_total_range, self.T_present_range
        r = self.rate_max
        lines = [
            f"Thermal-history prior over [0, {self.t_max:g}] Ma",
            f"  box       [{self.T_min:g}, {self.T_max:g}] degC",
            f"  tau       ~ LogUniform({tau[0]:g}, {tau[1]:g}) Ma",
            f"  a         ~ LogUniform({sd[0]:g}, {sd[1]:g}) K/Ma",
            f"  dT_drift  ~ Uniform({drift[0]:g}, {drift[1]:g}) K over the time span",
            f"  T_0       ~ Uniform({T0[0]:g}, {T0[1]:g}) degC",
            f"  z         ~ GP(0, 1), sq-exp, length tau, {self.n_features} features",
            f"  u(t)      = dT_drift/{self.t_max:g} + a z(t)",
            f"  dT/dt     = {r:g} tanh(u/{r:g}) * gate(T)",
            f"  rate limit {r:g} K/Ma, pathwise and exact",
            f"  solved with RK4 on {self.n_steps} steps",
        ]
        return "\n".join(lines)


@dataclass(frozen=True)
class EpisodicPrior:
    """The intermittent prior: a marked point process on the rate, not a GP.

    :class:`ThermalHistoryPrior`'s latent is a *stationary* Gaussian process, so
    a short correlation length means **many** thermal episodes rather than one
    short one.  Geological histories are not like that.  They are intermittent:
    a rock sits at depth doing nothing for tens of millions of years, is
    exhumed over a few, and sits again.  No stationary process reproduces that,
    which ``docs/prior.md`` logs as the shipped prior's known limitation and the
    one thing the prior-sensitivity study could not reach by reweighting -- a
    change of hyperprior cannot make a stationary process intermittent.

    This is that model, behind the same interface, and it exists to be *wrong
    in the right way*: drawn from here, scored against a pool drawn from there,
    it measures what an amortized posterior costs when the world does not obey
    the family it was trained on (``docs/prior-sensitivity.md``).

    .. code-block:: text

        K       ~ Poisson(episodes_per_horizon * window / t_max)
        s_k     ~ Uniform(-d_hi/2, t_max + d_hi/2)      episode centre, Ma
        d_k     ~ LogUniform(duration_range)            episode duration, Ma
        A_k     ~ LogUniform(amplitude_range)           peak rate, K/Ma
        e_k     = +1 with probability p_cooling, else -1
        dT_drift~ Uniform(drift_total_range)            background, K over the time span
        T_0     ~ Uniform(T_present_range)

        u(t)  = dT_drift / t_max + sum_k e_k A_k b(2 (t - s_k) / d_k)
        b(z)  = (1 + cos(pi z)) / 2   for |z| < 1,  else 0

    and from there the shipped prior's own machinery: the soft cap at
    ``rate_max``, the gate at the walls, the RK4 integration
    (:func:`integrate_rate`).  The two priors therefore differ in **how the rate
    history is drawn and in nothing else** -- same bound, same box, same solver.
    A comparison between them that also changed the solver would be worthless.

    Three deliberate choices:

    * **The bump is compact.**  ``b`` is exactly zero outside the episode, so
      between episodes the rate *is* the background, which is what
      "intermittent" means.  A Gaussian bump would leave a small rate
      everywhere and quietly reintroduce the thing being tested.  It is also
      C1 at its edges, so the path stays smooth and the fixed grid represents
      it no worse than it represents a GP draw.
    * **The centres are drawn from a window wider than the time span.**  The
      process lives on the line and the time span is a window onto it; drawing
      centres only inside ``[0, t_max]`` would thin the rate near both edges by
      half an episode, which is an artifact of the window and not a statement
      about geology.
    * **Cooling and heating are equally likely** (``p_cooling = 0.5``), with
      any net preference for cooling carried by ``dT_drift``, exactly as in the
      shipped prior.  A prior that made exhumation more likely than burial
      would be asserting the very thing thermochronology is used to find out.

    Args:
        t_max: time span, Ma.
        T_min: lower wall, degC.
        T_max: upper wall, degC.
        T_present_range: present-day temperature, degC.
        drift_total_range: background net cooling over the time span, K.
        episodes_per_horizon: mean number of episode centres inside the
            time span.  Two over 100 Ma is a rock with a couple of events in it.
        duration_range: episode duration, Ma, log-uniform.  The floor is above
            the grid spacing so an episode is always resolvable; the ceiling is
            well below the time span so an episode remains an event rather than a
            trend.
        amplitude_range: peak rate of an episode, K/Ma, log-uniform.  The top
            sits above ``rate_max`` on purpose: an episode that saturates the
            cap is a rock crossing the box as fast as the model allows, which
            is the end-member this prior exists to produce.
        p_cooling: probability that an episode cools rather than heats.
        rate_max: the hard pathwise rate limit, K/Ma.
        n_steps: internal RK4 steps over the time span.
    """

    t_max: float = 100.0
    T_min: float = 0.0
    T_max: float = 200.0
    T_present_range: tuple[float, float] = (0.0, 25.0)
    drift_total_range: tuple[float, float] = (0.0, 200.0)
    episodes_per_horizon: float = 2.0
    duration_range: tuple[float, float] = (2.0, 30.0)
    amplitude_range: tuple[float, float] = (1.0, 60.0)
    p_cooling: float = 0.5
    rate_max: float = 50.0
    n_steps: int = 2048

    def __post_init__(self) -> None:
        if self.t_max <= 0:
            raise ValueError(f"t_max must be positive, got {self.t_max}")
        if self.T_min >= self.T_max:
            raise ValueError(f"need T_min < T_max, got {self.T_min}, {self.T_max}")
        lo, hi = self.T_present_range
        if not self.T_min <= lo <= hi <= self.T_max:
            raise ValueError(f"T_present_range must sit inside the box, got {self.T_present_range}")
        for name in ("duration_range", "amplitude_range"):
            lo, hi = getattr(self, name)
            if not 0 < lo <= hi:
                raise ValueError(f"{name} must be positive and ordered, got {(lo, hi)}")
        if not 0.0 <= self.p_cooling <= 1.0:
            raise ValueError(f"p_cooling must be a probability, got {self.p_cooling}")
        if self.episodes_per_horizon < 0:
            raise ValueError(f"episodes_per_horizon must be >= 0, got "
                             f"{self.episodes_per_horizon}")
        if self.rate_max <= 0:
            raise ValueError(f"rate_max must be positive, got {self.rate_max}")
        if self.n_steps < 1:
            raise ValueError("n_steps must be >= 1")

    @property
    def rate_limit(self) -> float:
        """The pathwise bound on ``|dT/dt|``, K/Ma.  Exact, not a quantile."""
        return self.rate_max

    @property
    def hyper_names(self) -> tuple[str, ...]:
        """The per-draw latents this prior reports, and a corpus stores.

        Not the shipped prior's four: an episode count and an impulse are what
        describe an intermittent history, and ``tau`` has no meaning here.
        """
        return ("n_episodes", "episode_impulse", "peak_amplitude", "drift", "T_0")

    def sample(self, n: int, rng: np.random.Generator) -> HistoryBatch:
        """Draw ``n`` thermal histories.

        Args:
            n: number of histories.
            rng: seeded generator.  Draws depend on it and on nothing else.
        """
        if n < 1:
            raise ValueError(f"n must be >= 1, got {n}")

        t_half = np.linspace(0.0, self.t_max, 2 * self.n_steps + 1)
        pad = 0.5 * self.duration_range[1]
        window = self.t_max + 2 * pad

        drift = rng.uniform(*self.drift_total_range, size=n) / self.t_max
        T0 = rng.uniform(*self.T_present_range, size=n)
        counts = rng.poisson(self.episodes_per_horizon * window / self.t_max, size=n)
        k_max = int(counts.max())

        u = np.repeat(drift[:, None], t_half.size, axis=1)
        n_in_horizon = np.zeros(n)
        impulse = np.zeros(n)
        peak = np.zeros(n)
        for k in range(k_max):
            live = counts > k
            centre = rng.uniform(-pad, self.t_max + pad, size=n)
            duration = _loguniform(rng, *self.duration_range, n)
            amp = _loguniform(rng, *self.amplitude_range, n)
            amp *= np.where(rng.random(n) < self.p_cooling, 1.0, -1.0)
            amp = np.where(live, amp, 0.0)

            z = 2.0 * (t_half[None, :] - centre[:, None]) / duration[:, None]
            u += amp[:, None] * np.where(np.abs(z) < 1.0, 0.5 * (1.0 + np.cos(np.pi * z)), 0.0)

            # Bookkeeping, for the diagnostics: an episode counts if any of it
            # falls inside the time span, and its "impulse" is the temperature
            # change it would drive on its own -- the integral of the bump,
            # ``|A| d / 2``, before the cap and the walls get to it.
            seen = live & (centre + 0.5 * duration > 0.0) & (centre - 0.5 * duration < self.t_max)
            n_in_horizon += seen
            impulse += np.where(seen, np.abs(amp) * duration * 0.5, 0.0)
            peak = np.maximum(peak, np.where(seen, np.abs(amp), 0.0))

        batch = integrate_rate(u, T0, t_max=self.t_max, T_min=self.T_min,
                               T_max=self.T_max, rate_max=self.rate_max,
                               n_steps=self.n_steps)
        return replace(batch, hyper={"n_episodes": n_in_horizon, "episode_impulse": impulse,
                                     "peak_amplitude": peak, "drift": drift, "T_0": T0})

    def sample_on(self, n: int, rng: np.random.Generator,
                  t_grid: np.ndarray) -> tuple[np.ndarray, HistoryBatch]:
        """Draw ``n`` histories and evaluate them on ``t_grid``."""
        batch = self.sample(n, rng)
        return batch.at(np.asarray(t_grid, dtype=float)), batch

    def to_config(self) -> dict[str, Any]:
        """Plain-data form, for provenance attributes and config files."""
        return asdict(self)

    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> EpisodicPrior:
        ranges = {"T_present_range", "drift_total_range", "duration_range",
                  "amplitude_range"}
        return cls(**{k: tuple(v) if k in ranges else v for k, v in cfg.items()})

    def describe(self) -> str:
        """The generative model with this instance's numbers substituted in."""
        d, a = self.duration_range, self.amplitude_range
        drift, T0 = self.drift_total_range, self.T_present_range
        r = self.rate_max
        return "\n".join([
            f"Episodic thermal-history prior over [0, {self.t_max:g}] Ma",
            f"  box       [{self.T_min:g}, {self.T_max:g}] degC",
            f"  K         ~ Poisson({self.episodes_per_horizon:g} per time span)",
            f"  d_k       ~ LogUniform({d[0]:g}, {d[1]:g}) Ma",
            (f"  A_k       ~ LogUniform({a[0]:g}, {a[1]:g}) K/Ma, "
             f"cooling w.p. {self.p_cooling:g}"),
            f"  dT_drift  ~ Uniform({drift[0]:g}, {drift[1]:g}) K over the time span",
            f"  T_0       ~ Uniform({T0[0]:g}, {T0[1]:g}) degC",
            f"  u(t)      = dT_drift/{self.t_max:g} + sum_k e_k A_k b(2(t - s_k)/d_k)",
            f"  dT/dt     = {r:g} tanh(u/{r:g}) * gate(T)",
            f"  rate limit {r:g} K/Ma, pathwise and exact",
            f"  solved with RK4 on {self.n_steps} steps",
        ])


def _ramp(z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """A raised-cosine step and its derivative: 0 below ``z = -1/2``, 1 above ``+1/2``.

    ``R(z) = (1 + sin(pi z)) / 2`` in between, so ``R'`` peaks at ``pi / 2`` at the
    centre and vanishes at both ends: the step is C1, and a step of height ``D``
    over width ``w`` reaches the rate ``pi D / (2 w)`` exactly once.
    """
    inside = np.abs(z) < 0.5
    R = np.where(z >= 0.5, 1.0, np.where(inside, 0.5 * (1.0 + np.sin(np.pi * z)), 0.0))
    dR = np.where(inside, 0.5 * np.pi * np.cos(np.pi * z), 0.0)
    return R, dR


@dataclass(frozen=True)
class PulsePrior:
    """A pulse prior: a slow background plus a marked point process of episodes.

    Drawn on temperature directly -- no rate integration, no walls, no gate --
    so nothing piles up on an edge of the box and "only cools" is a value of a
    latent (``N = 0``) rather than a thin event (``docs/decisions/0003``).

    .. code-block:: text

        T_0    ~ U(T_present_range)                    present-day temperature
        dT     ~ U(0, T_max - T_0)                     net cooling over the time span
        K      ~ Poisson(pulse_rate t_max)             exhumation pulses over the time span
                 + Poisson(recent_rate recent_window)  ... and extra ones in the recent window
        beta   ~ U(background_range), or 1 if K = 0    share cooled at a steady rate
        s      ~ Dirichlet(alpha 1_K)                  each pulse's share of (1 - beta) dT
        w_k    ~ LogUniform(max(w_lo, w_cap), w_hi)    pulse width; w_cap from the rate cap
        c_k    ~ U(w_k/2, t_max - w_k/2)               pulse centre: every pulse completes
               ~ U(w_k/2, recent_window)               ... a recent pulse's, inside the window
        backbone(t) = T_0 + dT [beta t/t_max + (1 - beta) sum_k s_k R((t - c_k)/w_k)]

        N      ~ Poisson(episode_rate t_max)           reheating episodes
        A_n    ~ LogUniform(amplitude_range)           K; A_min is the cooling tolerance
        h_n    ~ LogUniform(max(h_lo, h_cap), h_hi)    heating ramp, and its own cooling ramp
        p_n    ~ LogUniform(plateau_range)             time held at the peak
        e_n    ~ U(0, t_max - length)                  youngest end: every episode completes
        T(t)   = backbone(t) + sum_n A_n trapezoid_n(t)

        reject if T > T_max anywhere, or |dT/dt| > rate_max anywhere,
               or N > 0 and the path never reheats by more than A_min

    ``R`` is the raised-cosine step of :func:`_ramp`; a trapezoid is two of them.
    A pulse of height ``D`` over width ``w`` peaks at ``pi D / (2 w)``, so ``w_cap
    = pi D / (2 (rate_max - background))`` puts the rate cap on each pulse by
    construction and rejection only sees overlaps.  The same holds for each
    ramp of an episode.

    Three things that are exact, not tendencies:

    * **No two nodes are equal.**  ``beta > 0`` whenever ``dT > 0``, so the
      backbone is strictly monotone and episodes only add; a flat hold is a
      slow one.  The target is a density on the grid.
    * **The backbone stays in ``[T_0, T_0 + dT]``.**  Every pulse lies wholly
      inside the time span, so ``F(0) = 0`` and ``F(t_max) = 1``.
    * **The recent pulses are the young end-member.**  Under a centre law that
      is uniform over the time span, "hot until recently, then out" is a product
      of small chances: without them 0.3% of 100 Ma draws had an AFT age under
      10 Ma, against the 10% a rate-integrated GP prior gives.  The pulse process
      therefore has a second intensity, ``recent_rate`` per Ma on
      ``[0, recent_window]`` only, the same on every time span.
    * **``N = 0`` is the cooling-only sub-prior.**  With ``N = 0`` the path
      never reheats; drawing ``N = 0`` directly (:meth:`cooling_only`) is the
      conditional of this prior on it, rejection included, since a draw with
      no episodes is rejected only by overlapping pulses in both.  And
      ``N = 0`` *is* the event ``reheat <= A_min``: an episode riding a
      backbone that cools faster than it heats leaves no visible reheating,
      and such a draw is rejected rather than counted as an episode.

    Args:
        t_max: time span, Ma.
        T_max: the support's upper bound, degC.  Not a wall: no mass sits on it.
        T_present_range: present-day temperature, degC.
        pulse_rate: exhumation pulses per Ma.
        background_range: share of the net cooling carried by the steady background.
        pulse_width_range: ``(w_lo, w_hi)``, Ma, log-uniform above the cap's own floor.
        pulse_concentration: ``alpha`` of the shares' Dirichlet; below 1 one
            pulse tends to carry most of the exhumation.
        recent_rate: extra pulses per Ma of the recent window, on top of
            ``pulse_rate``; 0 for none.
        recent_window: the recent window, Ma before present; a recent pulse
            is centred inside it (and completes inside the time span).
        episode_rate: reheating episodes per Ma.  The 19a configs put the same
            rate on both time spans; from 2026-09-27 the configs set it so that
            the *count* has mean 1 on each time span (RJ: N = 1 the most common,
            2 fewer, 3 rare, more very rare), i.e. ``1 / t_max``.
        amplitude_range: episode amplitude, K.  The floor is the cooling-only
            tolerance: an episode below it is not an episode.
        ramp_range: ``(h_lo, h_hi)``, each ramp of an episode, Ma.
        plateau_range: an episode's time at its peak, Ma.
        rate_max: the pathwise rate limit, K/Ma.
        n_steps: internal grid steps over the time span; paths are evaluated
            between them by cubic Hermite with the exact rate.
    """

    t_max: float = 100.0
    T_max: float = 200.0
    T_present_range: tuple[float, float] = (0.0, 25.0)
    pulse_rate: float = 0.01
    background_range: tuple[float, float] = (0.05, 0.5)
    pulse_width_range: tuple[float, float] = (4.0, 30.0)
    pulse_concentration: float = 0.5
    recent_rate: float = 0.0
    recent_window: float = 25.0
    episode_rate: float = 0.0114
    amplitude_range: tuple[float, float] = (5.0, 100.0)
    ramp_range: tuple[float, float] = (8.0, 25.0)
    plateau_range: tuple[float, float] = (1.0, 40.0)
    rate_max: float = 50.0
    n_steps: int = 2048

    def __post_init__(self) -> None:
        if self.t_max <= 0:
            raise ValueError(f"t_max must be positive, got {self.t_max}")
        lo, hi = self.T_present_range
        if not lo <= hi < self.T_max:
            raise ValueError(f"T_present_range must sit below T_max, got {self.T_present_range}")
        b0, b1 = self.background_range
        if not 0.0 < b0 <= b1 <= 1.0:
            raise ValueError(f"background_range must be in (0, 1], got {self.background_range}")
        for name in ("pulse_width_range", "amplitude_range", "ramp_range", "plateau_range"):
            lo, hi = getattr(self, name)
            if not 0 < lo <= hi:
                raise ValueError(f"{name} must be positive and ordered, got {(lo, hi)}")
        if self.pulse_concentration <= 0:
            raise ValueError(f"pulse_concentration must be positive, got {self.pulse_concentration}")
        if self.pulse_rate < 0 or self.episode_rate < 0 or self.recent_rate < 0:
            raise ValueError("pulse_rate, recent_rate and episode_rate must be >= 0")
        if not 0 < self.recent_window <= self.t_max:
            raise ValueError(f"recent_window must lie in (0, t_max], got {self.recent_window}")
        if 2 * self.ramp_range[1] + self.plateau_range[1] >= self.t_max:
            raise ValueError("the longest episode must fit inside the time span")
        if self.rate_max <= 0 or self.n_steps < 1:
            raise ValueError("rate_max must be positive and n_steps >= 1")

    @property
    def rate_limit(self) -> float:
        """The pathwise bound on ``|dT/dt|``, K/Ma: exact, by rejection."""
        return self.rate_max

    @property
    def T_min(self) -> float:
        """The lowest temperature a path can reach: the coolest present-day value."""
        return self.T_present_range[0]

    @property
    def hyper_names(self) -> tuple[str, ...]:
        return ("n_pulses", "n_episodes", "beta", "dT", "T_0")

    def cooling_only(self) -> PulsePrior:
        """The ``N = 0`` sub-prior: this prior conditioned on never reheating."""
        return replace(self, episode_rate=0.0)

    def _propose(self, n: int, rng: np.random.Generator, t: np.ndarray
                 ) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
        """``n`` unconditioned draws on ``t``: ``(T, dTdt, hyper)``."""
        tm, r = self.t_max, self.rate_max
        T0 = rng.uniform(*self.T_present_range, size=n)
        dT = rng.uniform(0.0, 1.0, size=n) * (self.T_max - T0)
        K = rng.poisson(self.pulse_rate * tm, size=n)
        # The recent pulses come first in each row's list; with recent_rate = 0
        # nothing is drawn, so the 19a configs' streams are untouched.
        K_recent = rng.poisson(self.recent_rate * self.recent_window, size=n) if self.recent_rate > 0 \
            else np.zeros(n, dtype=K.dtype)
        K = K + K_recent
        beta = np.where(K == 0, 1.0, rng.uniform(*self.background_range, size=n))
        bg = beta * dT / tm                                    # background rate, K/Ma
        T = T0[:, None] + bg[:, None] * t[None, :]
        dTdt = np.repeat(bg[:, None], t.size, axis=1)

        k_max = int(K.max()) if n else 0
        if k_max:
            g = rng.standard_gamma(self.pulse_concentration, size=(n, k_max))  # Dirichlet
            g = np.where(np.arange(k_max)[None, :] < K[:, None], g, 0.0)
            share = g / np.maximum(g.sum(axis=1, keepdims=True), 1e-300)
            w_lo, w_hi = self.pulse_width_range
            for k in range(k_max):
                live = K > k
                D = (1.0 - beta) * dT * share[:, k]
                w_cap = np.pi * D / (2.0 * (r - bg))
                lo = np.maximum(w_lo, w_cap)
                w = np.exp(rng.uniform(np.log(lo), np.log(np.maximum(lo, w_hi))))
                c = rng.uniform(0.5 * w, tm - 0.5 * w)
                if self.recent_rate > 0:
                    recent = K_recent > k
                    c_recent = rng.uniform(0.5 * w, np.maximum(0.5 * w, self.recent_window))
                    c = np.where(recent, np.minimum(c_recent, tm - 0.5 * w), c)
                R, dR = _ramp((t[None, :] - c[:, None]) / w[:, None])
                D = np.where(live, D, 0.0)
                T += D[:, None] * R
                dTdt += (D / w)[:, None] * dR

        N = rng.poisson(self.episode_rate * tm, size=n)
        n_max = int(N.max()) if n else 0
        h_lo, h_hi = self.ramp_range
        for j in range(n_max):
            live = N > j
            A = _loguniform(rng, *self.amplitude_range, n)
            ramps = []
            for cap in (r - bg, np.full_like(bg, r)):          # cooling ramp, heating ramp
                lo = np.maximum(h_lo, np.pi * A / (2.0 * cap))
                ramps.append(np.exp(rng.uniform(np.log(lo), np.log(np.maximum(lo, h_hi)))))
            h_cool, h_heat = ramps
            p = _loguniform(rng, *self.plateau_range, n)
            end = rng.uniform(0.0, 1.0, size=n) * (tm - h_cool - p - h_heat)
            # In time before present: rising over the cooling ramp from ``end``,
            # flat over the plateau, falling back over the heating ramp.
            R1, dR1 = _ramp((t[None, :] - (end + 0.5 * h_cool)[:, None]) / h_cool[:, None])
            on = end + h_cool + p + 0.5 * h_heat
            R2, dR2 = _ramp((t[None, :] - on[:, None]) / h_heat[:, None])
            A = np.where(live, A, 0.0)
            T += A[:, None] * (R1 - R2)
            dTdt += A[:, None] * (dR1 / h_cool[:, None] - dR2 / h_heat[:, None])

        hyper = {"n_pulses": K.astype(float), "n_episodes": N.astype(float), "beta": beta,
                 "dT": dT, "T_0": T0}
        return T, dTdt, hyper

    def sample_with_stats(self, n: int, rng: np.random.Generator
                          ) -> tuple[HistoryBatch, dict[str, int]]:
        """:meth:`sample`, plus how many proposals each rejection rule removed."""
        if n < 1:
            raise ValueError(f"n must be >= 1, got {n}")
        t = np.linspace(0.0, self.t_max, self.n_steps + 1)
        parts: list[tuple] = []
        stats = {"proposed": 0, "over_top": 0, "over_rate": 0, "unseen": 0}
        have = 0
        while have < n:
            m = max(64, int(1.1 * (n - have)))
            T, dTdt, hyper = self._propose(m, rng, t)
            hot = T.max(axis=1) > self.T_max
            fast = np.abs(dTdt).max(axis=1) > self.rate_max * (1.0 + 1e-9)
            suffix_min = np.minimum.accumulate(T[:, ::-1], axis=1)[:, ::-1]
            unseen = (hyper["n_episodes"] > 0) & (
                (T - suffix_min).max(axis=1) <= self.amplitude_range[0])
            ok = ~(hot | fast | unseen)
            stats["proposed"] += m
            stats["over_top"] += int(hot.sum())
            stats["over_rate"] += int((fast & ~hot).sum())
            stats["unseen"] += int((unseen & ~hot & ~fast).sum())
            idx = np.flatnonzero(ok)[: n - have]
            parts.append((T[idx], dTdt[idx], {k: v[idx] for k, v in hyper.items()}))
            have += idx.size
        T = np.concatenate([p[0] for p in parts])
        dTdt = np.concatenate([p[1] for p in parts])
        hyper = {k: np.concatenate([p[2][k] for p in parts]) for k in parts[0][2]}
        return HistoryBatch(t=t, T=T, dTdt=dTdt, hyper=hyper), stats

    def sample(self, n: int, rng: np.random.Generator) -> HistoryBatch:
        """Draw ``n`` thermal histories; rejected proposals are redrawn."""
        return self.sample_with_stats(n, rng)[0]

    def sample_on(self, n: int, rng: np.random.Generator,
                  t_grid: np.ndarray) -> tuple[np.ndarray, HistoryBatch]:
        """Draw ``n`` histories and evaluate them on ``t_grid``."""
        batch = self.sample(n, rng)
        return batch.at(np.asarray(t_grid, dtype=float)), batch

    def to_config(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> PulsePrior:
        ranges = {"T_present_range", "background_range", "pulse_width_range",
                  "amplitude_range", "ramp_range", "plateau_range"}
        cfg = dict(cfg)
        if "T_top" in cfg:  # the bound's name in corpora, checkpoints and configs before 2026-10-23
            cfg["T_max"] = cfg.pop("T_top")
        return cls(**{k: tuple(v) if k in ranges else v for k, v in cfg.items()})

    def describe(self) -> str:
        f = lambda r: f"{r[0]:g}, {r[1]:g}"
        return "\n".join([
            f"Pulse prior (v2) over [0, {self.t_max:g}] Ma, support below {self.T_max:g} degC",
            f"  T_0     ~ U({f(self.T_present_range)}) degC,  dT ~ U(0, {self.T_max:g} - T_0)",
            f"  K       ~ Poisson({self.pulse_rate:g}/Ma x {self.t_max:g}),"
            f"  beta ~ U({f(self.background_range)})",
            f"  w_k     ~ LogUniform(max({self.pulse_width_range[0]:g}, w_cap),"
            f" {self.pulse_width_range[1]:g}) Ma",
            f"  recent  + Poisson({self.recent_rate:g}/Ma x {self.recent_window:g}) pulses"
            f" centred within {self.recent_window:g} Ma",
            f"  N       ~ Poisson({self.episode_rate:g}/Ma x {self.t_max:g}),"
            f"  A ~ LogUniform({f(self.amplitude_range)}) K",
            f"  ramps   ~ LogUniform(max({self.ramp_range[0]:g}, h_cap), {self.ramp_range[1]:g}) Ma,"
            f"  plateau ~ LogUniform({f(self.plateau_range)}) Ma",
            f"  reject  T > {self.T_max:g} or |dT/dt| > {self.rate_max:g} K/Ma",
            f"  {self.n_steps} internal steps",
        ])


#: Every history prior in this module, by class name.  A corpus records which
#: one drew it and reconstructs it through :func:`prior_from_config`, so a file
#: stays regenerable from itself no matter which prior is in fashion.
PRIOR_TYPES: dict[str, type] = {}


def prior_from_config(cfg: dict[str, Any],
                      kind: str = "ThermalHistoryPrior") -> Any:
    """Rebuild a history prior from its stored config.

    ``kind`` defaults to the shipped prior, because corpora written before there
    was a second one do not record a class and there is only one they can be.
    """
    if kind not in PRIOR_TYPES:
        raise KeyError(f"unknown history prior {kind!r}; known: {sorted(PRIOR_TYPES)}")
    return PRIOR_TYPES[kind].from_config(cfg)


@dataclass(frozen=True)
class GrainPrior:
    """The other half of a simulated sample: what was measured, not when.

    Ranges span the apatite partial-retention and partial-annealing zones, so
    the corpus covers the ``(age, eU, radius, Dpar)`` space the Stage 1 audit
    checks for dead regions.  Concentrations are log-uniform because natural
    apatite U contents span two orders of magnitude and are not remotely flat
    on a linear scale; ``Th`` is drawn through a Th/U ratio for the same reason,
    and because the ratio is the quantity that is actually conserved between
    grains of one rock.

    Args:
        radius_range: equivalent spherical radius, microns.
        U_range: uranium, ppm, log-uniform.
        th_u_ratio_range: Th/U, log-uniform.
        dpar_range: etch-pit diameter, microns -- the AFT kinetic parameter.
    """

    radius_range: tuple[float, float] = (30.0, 110.0)
    U_range: tuple[float, float] = (1.0, 150.0)
    th_u_ratio_range: tuple[float, float] = (0.2, 10.0)
    dpar_range: tuple[float, float] = (1.2, 3.5)

    def sample(self, n: int, rng: np.random.Generator) -> dict[str, np.ndarray]:
        U = _loguniform(rng, *self.U_range, n)
        return {
            "radius": rng.uniform(*self.radius_range, size=n),
            "U": U,
            "Th": U * _loguniform(rng, *self.th_u_ratio_range, n),
            "dpar": rng.uniform(*self.dpar_range, size=n),
        }

    def to_config(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> GrainPrior:
        return cls(**{k: tuple(v) for k, v in cfg.items()})


PRIOR_TYPES.update({cls.__name__: cls for cls in (ThermalHistoryPrior, EpisodicPrior,
                                                   PulsePrior)})
