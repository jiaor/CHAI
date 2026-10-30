"""The thermal-history prior: its guarantees, and the ones it used to break."""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

from chai.data import priors as priors_mod
from chai.data.priors import EpisodicPrior, GrainPrior, ThermalHistoryPrior
from chai.paths import fixed_grid, validate

#: Cheap but still converged; the defaults are sized for corpus generation.
FAST = ThermalHistoryPrior(n_steps=256, n_features=64)


def draw(prior=FAST, n=64, seed=0):
    return prior.sample(n, np.random.default_rng(seed))


def test_paths_obey_the_canonical_time_convention():
    b = draw()
    grid = fixed_grid(64, FAST.t_max, "uniform").numpy()
    T = b.at(grid)
    assert b.t[0] == 0.0 and b.t[-1] == pytest.approx(FAST.t_max)
    validate(torch.from_numpy(np.broadcast_to(grid, T.shape).copy()), torch.from_numpy(T))


@pytest.mark.parametrize("rate_max", [5.0, 50.0])
def test_the_rate_limit_is_a_pathwise_guarantee(rate_max):
    """Not a quantile: no draw exceeds it anywhere, ever.

    Including under a fluctuation far larger than the cap, which is the case
    that separates this prior from a Gaussian process placed directly on T.
    """
    p = replace(FAST, rate_max=rate_max, rate_sd_range=(200.0, 400.0))
    b = draw(p, n=128)
    assert np.abs(b.dTdt).max() <= rate_max
    assert np.abs(np.diff(b.T, axis=1)).max() <= rate_max * (b.t[1] - b.t[0]) + 1e-9


def test_paths_stay_inside_the_box():
    p = replace(FAST, rate_sd_range=(50.0, 100.0), drift_total_range=(-400.0, 400.0))
    b = draw(p, n=128)
    assert b.T.min() >= p.T_min
    assert b.T.max() <= p.T_max


def test_a_wall_does_not_trap_the_path():
    """Regression: the first version damped the rate smoothly towards each wall.

    Any barrier that vanishes at the wall is absorbing -- the approach is
    exponential, so a path that rests on the floor for tens of Ma cannot climb
    off it again, and 30% of draws were flat at 0 degC for the whole horizon.
    The gate must hand back the full rate the moment the drive reverses.
    """
    T_floor = np.full(3, FAST.T_min)
    assert FAST._flow(T_floor, np.array([5.0, -5.0, 0.0])) == pytest.approx([5.0, 0.0, 0.0])
    T_ceil = np.full(2, FAST.T_max)
    assert FAST._flow(T_ceil, np.array([5.0, -5.0])) == pytest.approx([0.0, -5.0])

    b = draw(n=512)
    stuck = (b.T < FAST.T_min + 1.0).mean(axis=1) > 0.9
    assert stuck.mean() < 0.15, f"{stuck.mean():.0%} of draws never leave the floor"


def test_draws_are_reproducible_from_the_seed_alone():
    a, b = draw(seed=3), draw(seed=3)
    assert np.array_equal(a.T, b.T)
    assert not np.array_equal(a.T, draw(seed=4).T)


def test_the_solver_is_converged_at_the_default_step_count():
    coarse, fine = draw(replace(FAST, n_steps=256)), draw(replace(FAST, n_steps=4096))
    tq = np.linspace(0.0, FAST.t_max, 401)
    assert np.abs(coarse.at(tq) - fine.at(tq)).max() < 2.0


def test_interpolation_reproduces_the_solver_nodes():
    b = draw()
    assert b.at(b.t) == pytest.approx(b.T)


def test_interpolation_clamps_outside_the_horizon():
    b = draw()
    assert b.at(np.array([-10.0, FAST.t_max + 10.0])) == pytest.approx(
        np.stack([b.T[:, 0], b.T[:, -1]], axis=1)
    )


def test_sample_on_matches_evaluating_the_batch():
    grid = fixed_grid(32, FAST.t_max, "log").numpy()
    T, b = FAST.sample_on(16, np.random.default_rng(5), grid)
    assert T == pytest.approx(b.at(grid))


def test_the_feature_sum_is_unaffected_by_chunking(monkeypatch):
    rng = np.random.default_rng(3)
    omega, phase = rng.normal(size=(5, 16)), rng.uniform(0.0, 2 * np.pi, size=(5, 16))
    t = np.linspace(0.0, 100.0, 301)
    whole = priors_mod._gp_on(t, omega, phase)
    monkeypatch.setattr(priors_mod, "_MAX_WORK", 128)
    assert priors_mod._gp_on(t, omega, phase) == pytest.approx(whole)


def test_diagnostics_agree_with_the_paths_they_describe():
    b = draw(n=128)
    d = b.diagnostics()
    assert d["max_cooling_rate"] == pytest.approx(b.dTdt.max(axis=1))
    assert d["max_heating_rate"] == pytest.approx(-b.dTdt.min(axis=1))
    assert d["T_present"] == pytest.approx(b.T[:, 0])
    assert np.all(d["monotone"] == (d["reheat_amplitude"] < 1.0))
    # a monotone history is one whose temperature never falls towards the present
    for i in np.flatnonzero(d["monotone"]):
        assert b.T[i].max() - b.T[i, -1] < 1.0


def test_reheating_amplitude_finds_a_burial_episode():
    """A path hotter now than it was earlier has been reheated by that much."""
    from chai.data.priors import HistoryBatch

    t = np.linspace(0.0, 100.0, 5)
    T = np.array([[10.0, 60.0, 20.0, 80.0, 100.0]])  # 40 K rise towards the present
    d = HistoryBatch(t=t, T=T, dTdt=np.zeros_like(T), hyper={}).diagnostics()
    assert d["reheat_amplitude"] == pytest.approx([40.0])
    assert not d["monotone"][0]


def test_config_round_trips():
    p = replace(FAST, tau_range=(3.0, 30.0), rate_max=7.5)
    assert ThermalHistoryPrior.from_config(p.to_config()) == p
    g = GrainPrior(U_range=(2.0, 50.0))
    assert GrainPrior.from_config(g.to_config()) == g


def test_describe_states_the_numbers_it_was_configured_with():
    text = replace(FAST, rate_max=12.5).describe()
    assert "12.5 K/Ma, pathwise and exact" in text
    assert "LogUniform(8, 80) Ma" in text


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"t_max": 0.0}, "t_max must be positive"),
        ({"T_min": 300.0}, "need T_min < T_max"),
        ({"T_present_range": (0.0, 500.0)}, "must sit inside the box"),
        ({"tau_range": (0.0, 10.0)}, "tau_range must be positive"),
        ({"rate_sd_range": (10.0, 1.0)}, "rate_sd_range must be positive and ordered"),
        ({"rate_max": -1.0}, "rate_max must be positive"),
        ({"n_steps": 0}, "n_steps and n_features"),
    ],
)
def test_bad_configurations_are_rejected(kwargs, match):
    with pytest.raises(ValueError, match=match):
        replace(FAST, **kwargs)


def test_sample_rejects_an_empty_batch():
    with pytest.raises(ValueError, match="n must be >= 1"):
        draw(n=0)


def test_grain_prior_covers_its_ranges():
    g = GrainPrior()
    s = g.sample(4096, np.random.default_rng(0))
    assert g.radius_range[0] <= s["radius"].min() and s["radius"].max() <= g.radius_range[1]
    assert g.U_range[0] <= s["U"].min() and s["U"].max() <= g.U_range[1]
    assert g.dpar_range[0] <= s["dpar"].min() and s["dpar"].max() <= g.dpar_range[1]
    ratio = s["Th"] / s["U"]
    assert g.th_u_ratio_range[0] <= ratio.min() and ratio.max() <= g.th_u_ratio_range[1]
    # log-uniform, so the median sits at the geometric mean, not the arithmetic one
    assert np.median(s["U"]) == pytest.approx(np.sqrt(np.prod(g.U_range)), rel=0.1)


def test_a_gp_prior_config_round_trips():
    """A pinned GP-prior config (what ``ThermalHistoryPrior.to_config`` writes) must survive the
    defaults moving underneath it."""
    prior = ThermalHistoryPrior.from_config(ThermalHistoryPrior(rate_max=40.0).to_config())
    grain = GrainPrior.from_config(GrainPrior().to_config())
    assert prior.rate_limit == 40.0
    assert grain.sample(4, np.random.default_rng(0))["U"].shape == (4,)


# ---------------------------------------------------------------------------
# The episodic prior -- the intermittent alternative (docs/prior-sensitivity.md)
# ---------------------------------------------------------------------------

EPISODIC = EpisodicPrior(n_steps=256)


@pytest.mark.parametrize("rate_max", [5.0, 50.0])
def test_the_episodic_rate_limit_is_also_a_pathwise_guarantee(rate_max):
    """The two priors share the cap, the gate and the solver, so they share this.

    Under episodes an order of magnitude past the cap, which is the case that
    matters: the bound has to come from the model, not from the draw.
    """
    p = replace(EPISODIC, rate_max=rate_max, amplitude_range=(200.0, 400.0))
    b = p.sample(128, np.random.default_rng(0))
    assert np.abs(b.dTdt).max() <= rate_max
    assert b.T.min() >= p.T_min and b.T.max() <= p.T_max


def test_with_no_episodes_the_prior_is_its_background():
    """The degenerate case, and the tightest check on the shared integrator.

    With no episodes the rate is the constant ``dT_drift / t_max``, so every
    path is a straight line until it meets a wall -- predictable in closed form.
    The one subtlety is that the cap applies to the background too: ``tanh``
    reaches ``rate_max`` only asymptotically, so it compresses even a 1.5 K/Ma
    drift, by 0.05% here and by 3% at 0.3 ``rate_max``.

    The drift is capped below the value that would reach the ceiling inside the
    horizon, so that no path touches a wall: at wall contact the gate closes
    part-way through an RK4 stage and the solver drops to first order there,
    which ``docs/prior.md`` measures and which would turn a closed-form
    comparison into a comparison against the solver's own step size.  That the
    box holds anyway is asserted above.
    """
    p = replace(EPISODIC, episodes_per_horizon=0.0, drift_total_range=(0.0, 150.0))
    b = p.sample(64, np.random.default_rng(1))
    rate = p.rate_max * np.tanh(b.hyper["drift"] / p.rate_max)
    expect = np.clip(b.hyper["T_0"][:, None] + rate[:, None] * b.t[None, :],
                     p.T_min, p.T_max)
    assert b.T == pytest.approx(expect, abs=1e-9)
    assert b.dTdt == pytest.approx(np.where(b.T < p.T_max, rate[:, None], 0.0), abs=1e-9)
    assert (b.hyper["n_episodes"] == 0).all()


def test_between_episodes_the_rate_is_exactly_the_background():
    """What "intermittent" means, and what no stationary GP can do.

    With the background switched off, a path may move only inside its episodes,
    so the fraction of the horizon over which it moves at all is bounded by the
    total duration it drew.  A squared-exponential GP has no such bound: its
    draws are non-zero almost everywhere.
    """
    p = replace(EPISODIC, drift_total_range=(0.0, 0.0), T_present_range=(80.0, 80.0),
                amplitude_range=(1.0, 3.0), duration_range=(5.0, 10.0),
                episodes_per_horizon=1.0)
    b = p.sample(256, np.random.default_rng(2))

    moving = (np.abs(b.dTdt) > 1e-12).mean(axis=1)
    # Nothing drawn can cover more than every episode's full duration, and the
    # draw's own count is in the batch.
    ceiling = b.hyper["n_episodes"] * p.duration_range[1] / p.t_max
    assert np.all(moving <= ceiling + 2.0 / p.n_steps)
    assert (moving == 0.0).any(), "a draw with no episode must be perfectly flat"
    assert b.T[moving == 0.0] == pytest.approx(80.0)


def test_the_episode_count_is_the_process_it_says_it_is():
    """Poisson in the horizon, at the rate advertised -- not thinned at the edges.

    Centres are drawn from a window wider than the horizon on purpose: drawing
    them inside it would thin the rate near both ends by half an episode, which
    is an artifact of the window rather than anything geological.
    """
    p = replace(EPISODIC, episodes_per_horizon=3.0, duration_range=(1.0, 1.0))
    k = p.sample(4000, np.random.default_rng(3)).hyper["n_episodes"]
    assert k.mean() == pytest.approx(3.0, rel=0.06)
    assert k.var() == pytest.approx(3.0, rel=0.15)        # Poisson: var = mean


def test_the_two_priors_agree_on_everything_but_the_rate():
    """Same interface, same grid, same guarantees -- so a comparison is fair."""
    gp = draw(FAST, n=16, seed=5)
    ep = EPISODIC.sample(16, np.random.default_rng(5))
    assert ep.t == pytest.approx(gp.t if FAST.n_steps == EPISODIC.n_steps else ep.t)
    assert EPISODIC.rate_limit == FAST.rate_limit
    grid = fixed_grid(64, EPISODIC.t_max, "uniform").numpy()
    on_grid, batch = EPISODIC.sample_on(8, np.random.default_rng(6), grid)
    assert on_grid.shape == (8, 64)
    assert on_grid == pytest.approx(batch.at(grid))
    validate(torch.from_numpy(np.broadcast_to(grid, on_grid.shape).copy()),
             torch.from_numpy(on_grid))


def test_the_episodic_config_round_trips():
    cfg = EPISODIC.to_config()
    assert EpisodicPrior.from_config(cfg) == EPISODIC


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"duration_range": (0.0, 5.0)}, "duration_range must be positive"),
        ({"amplitude_range": (9.0, 1.0)}, "amplitude_range must be positive and ordered"),
        ({"p_cooling": 1.5}, "p_cooling must be a probability"),
        ({"episodes_per_horizon": -1.0}, "episodes_per_horizon must be >= 0"),
    ],
)
def test_bad_episodic_configurations_are_rejected(kwargs, match):
    with pytest.raises(ValueError, match=match):
        replace(EPISODIC, **kwargs)


# ---------------------------------------------------------------------------
# Prior v2 -- background plus pulses plus episodes (PLAN.md Stage 19)
# ---------------------------------------------------------------------------

from chai.data.priors import PulsePrior, prior_from_config  # noqa: E402

PULSE = PulsePrior(n_steps=512)


def _pulse_draw(prior=PULSE, n=600, seed=0):
    return prior.sample_with_stats(n, np.random.default_rng(seed))


def test_pulse_prior_has_no_atoms():
    """No two nodes equal and nothing on a wall: the target is a density."""
    b, _ = _pulse_draw(PULSE.cooling_only())
    T = b.at(fixed_grid(64, 100.0, spacing="uniform"))
    assert (np.diff(T, axis=1) > 0).all()        # older is hotter, strictly, with N = 0 ...
    b2, _ = _pulse_draw(replace(PULSE, episode_rate=0.05), seed=1)
    T2 = b2.at(fixed_grid(64, 100.0, spacing="uniform"))
    assert (np.diff(T2, axis=1) != 0).all()      # ... and never flat with episodes on top


def test_pulse_prior_stays_in_its_support_and_under_the_cap():
    b, st = _pulse_draw(replace(PULSE, episode_rate=0.03))
    assert b.T.max() <= PULSE.T_max
    assert np.abs(b.dTdt).max() <= PULSE.rate_max * (1 + 1e-9)
    assert (b.T[:, 0] == b.hyper["T_0"]).all()
    assert (b.T >= b.hyper["T_0"][:, None] - 1e-9).all()     # never colder than today
    assert st["proposed"] >= len(b)


def test_pulse_prior_rate_is_the_derivative_of_the_path():
    b, _ = _pulse_draw(replace(PULSE, episode_rate=0.03), n=50)
    h = b.t[1] - b.t[0]
    fd = np.gradient(b.T, h, axis=1)
    assert np.abs(fd - b.dTdt)[:, 2:-2].max() < 0.05 * PULSE.rate_max


def test_no_episodes_is_exactly_only_cools():
    """With the invisible-episode rule, N = 0 and 'reheats <= A_min' are one event."""
    b, _ = _pulse_draw(replace(PULSE, episode_rate=0.02), n=1000, seed=2)
    reheat = b.diagnostics()["reheat_amplitude"]
    none = b.hyper["n_episodes"] == 0
    assert 0 < none.mean() < 1
    assert (reheat[none] < 1e-9).all()
    assert (reheat[~none] > PULSE.amplitude_range[0]).all()


def test_the_cooling_only_prior_draws_no_episodes():
    b, st = _pulse_draw(PULSE.cooling_only())
    assert (b.hyper["n_episodes"] == 0).all()
    assert st["unseen"] == 0


def test_pulse_prior_is_reproducible_and_round_trips():
    a = PULSE.sample(20, np.random.default_rng(5)).T
    assert np.array_equal(a, PULSE.sample(20, np.random.default_rng(5)).T)
    again = prior_from_config(PULSE.to_config(), "PulsePrior")
    assert again == PULSE


def test_the_shipped_prior_configs_load():
    """configs/prior_v22*.yaml pin what the shipped corpora were drawn under: a mean count of one pulse
    and one episode per time span, the recent pulses, and rates to 100 K/Ma."""
    import yaml

    for name, t_max in (("prior_v22.yaml", 100.0), ("prior_v22_250.yaml", 250.0)):
        cfg = yaml.safe_load(Path(__file__).parents[1].joinpath("configs", name).read_text())
        prior = prior_from_config(cfg["history"], cfg["kind"])
        assert prior.t_max == t_max and prior.rate_max == 100.0
        assert prior.pulse_rate * t_max == pytest.approx(1.0)
        assert prior.episode_rate * t_max == pytest.approx(1.0)
        assert prior.recent_rate > 0 and prior.recent_window == 25.0
        assert GrainPrior.from_config(cfg["grain"]).sample(2, np.random.default_rng(0))["U"].shape == (2,)


@pytest.mark.parametrize(
    "kwargs, match",
    [({"background_range": (0.0, 0.5)}, "background_range"),
     ({"ramp_range": (8.0, 45.0)}, "fit inside"),
     ({"pulse_concentration": 0.0}, "pulse_concentration")],
)
def test_bad_v2_configurations_are_rejected(kwargs, match):
    with pytest.raises(ValueError, match=match):
        replace(PULSE, **kwargs)


def test_recent_pulses_are_centred_inside_the_window():
    """With pulse_rate = 0 every pulse is recent, so past the window plus half the
    widest pulse the path is the background alone (2026-09-27)."""
    p = replace(PULSE, pulse_rate=0.0, recent_rate=0.2, recent_window=25.0, episode_rate=0.0)
    b, _ = _pulse_draw(p, n=400, seed=3)
    K, beta, dT, T0 = (b.hyper[k] for k in ("n_pulses", "beta", "dT", "T_0"))
    assert (K > 0).mean() > 0.9
    old = b.t >= 25.0 + 0.5 * p.pulse_width_range[1]
    expect = T0[:, None] + dT[:, None] * (beta[:, None] * b.t[None, old] / p.t_max + (1.0 - beta[:, None]))
    np.testing.assert_allclose(b.T[:, old], expect, atol=1e-6)
    # ... and inside the window the pulses are still there.
    assert (b.T[:, 0] < b.T[:, np.searchsorted(b.t, 25.0)] - 5.0).mean() > 0.5


def test_no_recent_rate_draws_exactly_the_19a_prior():
    a = replace(PULSE, recent_rate=0.0, recent_window=10.0).sample(30, np.random.default_rng(7)).T
    assert np.array_equal(a, PULSE.sample(30, np.random.default_rng(7)).T)


@pytest.mark.parametrize("kwargs", [{"recent_rate": -1.0}, {"recent_window": 0.0}, {"recent_window": 500.0}])
def test_bad_recent_pulse_settings_are_rejected(kwargs):
    with pytest.raises(ValueError, match="recent"):
        replace(PULSE, **kwargs)
