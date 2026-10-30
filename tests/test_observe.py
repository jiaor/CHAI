"""The observation model: the physics of the noise, and that it can be scored."""
from __future__ import annotations

import numpy as np
import pytest
import torch
from scipy.stats import lognorm, multinomial

from chai.data.observe import (
    AFTNoise,
    AHeNoise,
    ObservationModel,
    age_from_density,
    length_w1_sigma,
    mean_length_sigma,
    smear_matrix,
    spontaneous_density,
    track_density_per_ma_ppm,
)

BINS = (np.arange(200) + 0.5) * 0.1


def bell(centre=14.5, width=1.4) -> np.ndarray:
    """A plausible track-length distribution, as bin probabilities."""
    p = np.exp(-0.5 * ((BINS - centre) / width) ** 2)
    return p / p.sum()


def truth_for(n, age=50.0, U=20.0, he=30.0, g=3, pdf=None) -> dict[str, np.ndarray]:
    return {
        "aft_age": np.full(n, age),
        "length_pdf": np.tile(bell() if pdf is None else pdf, (n, 1)),
        "dpar": np.full(n, 1.9),
        "U": np.full(n, U),
        "he_age": np.full((n, g), he),
        "he_radius": np.full((n, g), 60.0),
        "he_eU": np.full((n, g), 30.0),
    }


# ---------------------------------------------------------------------------
# Counting statistics
# ---------------------------------------------------------------------------

def test_the_track_density_constant_reproduces_its_own_derivation():
    """Recomputed from the physical constants, not from the code path."""
    n238 = 3.20e-6 / 238.029 * 6.02214076e23 * 0.992742
    expect = n238 * 8.46e-17 * 1e6 * (16.0e-4 / 2) * 0.93
    assert track_density_per_ma_ppm() == pytest.approx(expect, rel=1e-12)


def test_track_densities_land_where_apatite_track_densities_are_found():
    """A sanity band, not a tautology: 1e5-1e6 /cm2 is what analysts count."""
    rho = spontaneous_density(np.array([20.0, 50.0, 100.0]), np.array([10.0, 20.0, 40.0]))
    assert np.all((rho > 1e4) & (rho < 3e6))


def test_density_and_age_are_inverses_including_the_exponential():
    age = np.array([0.5, 5.0, 50.0, 300.0, 1000.0])
    assert age_from_density(spontaneous_density(age, 25.0), 25.0) == pytest.approx(age)


def test_the_track_count_is_poisson_in_the_expected_number():
    m = ObservationModel()
    lam = float(m.aft.expected_counts(50.0, 20.0))
    obs = m.observe(truth_for(20_000), np.random.default_rng(0))
    ns = obs.n_spontaneous
    assert ns.mean() == pytest.approx(lam, rel=0.02)
    assert ns.var() == pytest.approx(lam, rel=0.05)


def test_precision_depends_on_the_age_and_the_uranium_not_on_a_flat_percentage():
    """The whole reason this module replaces the audit's stand-in tolerance."""
    m = ObservationModel()
    young_poor = float(m.aft.age_rel_sd(5.0, 3.0))
    old_rich = float(m.aft.age_rel_sd(100.0, 80.0))
    assert young_poor > 0.4                     # ~3 tracks: barely an age at all
    assert old_rich < 0.05                      # systematics-limited
    assert young_poor / old_rich > 10


def test_the_reported_sigma_matches_the_scatter_it_predicts():
    m = ObservationModel()
    obs = m.observe(truth_for(20_000, age=50.0, U=20.0), np.random.default_rng(1))
    realised = np.std(np.log(obs.aft_age[obs.n_spontaneous > 0]))
    forecast = float(m.aft.age_rel_sd(50.0, 20.0))
    assert realised == pytest.approx(forecast, rel=0.05)
    reported = np.median(obs.aft_age_sd / obs.aft_age)
    assert reported == pytest.approx(forecast, rel=0.05)


def test_a_bigger_counted_area_buys_precision_down_to_the_systematic_floor():
    small = AFTNoise(area_per_grain_um2=200.0)
    huge = AFTNoise(area_per_grain_um2=2.0e7)
    assert small.age_rel_sd(50.0, 20.0) > 3 * huge.age_rel_sd(50.0, 20.0)
    assert float(huge.age_rel_sd(50.0, 20.0)) == pytest.approx(
        np.sqrt(small.systematic_rel_var), rel=1e-3)


# ---------------------------------------------------------------------------
# Track lengths
# ---------------------------------------------------------------------------

def test_the_smear_conserves_probability_and_vanishes_at_zero_width():
    M = smear_matrix(BINS, 0.15)
    assert M.sum(axis=1) == pytest.approx(np.ones(BINS.size))
    assert np.array_equal(smear_matrix(BINS, 0.0), np.eye(BINS.size))


def test_smearing_widens_the_distribution_without_moving_its_mean():
    m = ObservationModel()
    p = bell()
    q = m.smeared(p)[0]
    assert q @ BINS == pytest.approx(p @ BINS, abs=1e-3)
    def var(x):
        return x @ BINS ** 2 - (x @ BINS) ** 2

    assert var(q) == pytest.approx(var(p) + 0.15 ** 2, rel=0.02)


def test_measured_lengths_are_multinomial_over_the_smeared_distribution():
    m = ObservationModel()
    obs = m.observe(truth_for(4000), np.random.default_rng(2))
    assert np.all(obs.n_lengths == m.aft.n_lengths)
    empirical = obs.length_counts.sum(axis=0) / obs.length_counts.sum()
    assert np.abs(empirical - m.smeared(bell())[0]).max() < 5e-3


def test_a_totally_annealed_history_yields_no_lengths_rather_than_a_crash():
    m = ObservationModel()
    truth = truth_for(4, pdf=np.zeros(200))
    obs = m.observe(truth, np.random.default_rng(3))
    assert np.all(obs.n_lengths == 0)
    assert np.all(obs.mean_length == 0.0)


def test_length_w1_sigma_predicts_the_transport_distance_it_will_see():
    """The derived replacement for the audit's 0.15 um stand-in, checked by sampling."""
    rng = np.random.default_rng(4)
    p = bell()
    n = 100
    F = np.cumsum(p)
    draws = rng.multinomial(n, p, size=4000)
    w1 = np.abs(np.cumsum(draws / n, axis=1) - F).sum(axis=1) * 0.1
    assert w1.mean() == pytest.approx(float(length_w1_sigma(p, n, 0.1)[0]), rel=0.05)


def test_mean_length_sigma_predicts_the_scatter_of_the_measured_mean():
    m = ObservationModel()
    obs = m.observe(truth_for(4000), np.random.default_rng(5))
    predicted = float(mean_length_sigma(m.smeared(bell()), m.aft.n_lengths, BINS)[0])
    assert obs.mean_length.std() == pytest.approx(predicted, rel=0.05)


def test_the_length_yield_model_bites_only_on_sparse_samples_and_only_if_asked():
    rich, poor = truth_for(2000, age=80.0, U=60.0), truth_for(2000, age=3.0, U=2.0)
    on = ObservationModel(aft=AFTNoise(length_yield_ref=1e6))
    assert np.all(ObservationModel().observe(poor, np.random.default_rng(6)).n_lengths == 100)
    assert on.observe(rich, np.random.default_rng(6)).n_lengths.mean() > 95
    assert on.observe(poor, np.random.default_rng(6)).n_lengths.mean() < 1


# ---------------------------------------------------------------------------
# Helium
# ---------------------------------------------------------------------------

def test_single_grain_dispersion_has_the_width_it_reports():
    m = ObservationModel()
    obs = m.observe(truth_for(20_000, he=30.0), np.random.default_rng(7))
    assert np.std(np.log(obs.he_age / 30.0)) == pytest.approx(m.he.rel_sd, rel=0.03)
    assert np.median(obs.he_age) == pytest.approx(30.0, rel=0.01)
    assert np.all(obs.he_age_sd / obs.he_age == pytest.approx(m.he.rel_sd))


def test_outliers_fatten_the_tails_and_only_when_switched_on():
    plain = ObservationModel()
    rough = ObservationModel(he=AHeNoise(outlier_fraction=0.1))
    t = truth_for(20_000, he=30.0)
    a = plain.observe(t, np.random.default_rng(8)).he_age
    b = rough.observe(t, np.random.default_rng(8)).he_age
    assert plain.he.outlier_fraction == 0.0, "outliers must be off by default"
    def far(x):
        return np.mean(np.abs(np.log(x / 30.0)) > 0.3)

    assert far(a) < 1e-4 < far(b)
    assert np.median(b) == pytest.approx(30.0, rel=0.02)     # still centred


def test_the_dated_subset_respects_the_requested_range_and_is_masked_not_deleted():
    m = ObservationModel(he=AHeNoise(n_grains_range=(1, 3)))
    obs = m.observe(truth_for(2000, g=4), np.random.default_rng(9))
    assert set(np.unique(obs.n_grains)) == {1, 2, 3}
    assert np.all(obs.he_age[~obs.he_mask] == 0.0)
    assert np.all(obs.he_age[obs.he_mask] > 0.0)


def test_grain_properties_are_perturbed_because_a_laboratory_reports_measurements():
    m = ObservationModel()
    obs = m.observe(truth_for(4000), np.random.default_rng(10))
    assert np.std(np.log(obs.he_radius / 60.0)) == pytest.approx(m.he.radius_rel_sd, rel=0.05)
    assert np.std(np.log(obs.he_eU / 30.0)) == pytest.approx(m.he.eU_rel_sd, rel=0.05)
    assert np.std(obs.dpar - 1.9) == pytest.approx(m.aft.dpar_sd, rel=0.05)


# ---------------------------------------------------------------------------
# Reproducibility and config
# ---------------------------------------------------------------------------

def test_the_same_seed_gives_the_same_measurement():
    m = ObservationModel()
    t = truth_for(50)
    a = m.observe(t, np.random.default_rng(11))
    b = m.observe(t, np.random.default_rng(11))
    assert np.array_equal(a.aft_age, b.aft_age)
    assert np.array_equal(a.length_counts, b.length_counts)
    assert np.array_equal(a.he_age, b.he_age)


def test_config_round_trips_through_plain_data():
    m = ObservationModel(aft=AFTNoise(n_lengths=57), he=AHeNoise(n_grains_range=(2, 5)))
    back = ObservationModel.from_config(m.to_config())
    assert back.to_config() == m.to_config()
    assert back.he.n_grains_range == (2, 5)


def test_replace_keeps_everything_it_was_not_asked_to_change():
    m = ObservationModel().replace(aft={"n_lengths": 30})
    assert m.aft.n_lengths == 30
    assert m.aft.area_per_grain_um2 == AFTNoise().area_per_grain_um2
    assert m.he == AHeNoise()


# ---------------------------------------------------------------------------
# The likelihood
# ---------------------------------------------------------------------------

def _pred(truth: dict[str, np.ndarray], scale: float = 1.0) -> dict[str, torch.Tensor]:
    return {
        "aft_age": torch.as_tensor(truth["aft_age"] * scale),
        "length_pdf": torch.as_tensor(truth["length_pdf"]),
        "he_age": torch.as_tensor(truth["he_age"] * scale),
    }


def test_each_term_equals_the_textbook_density_it_claims_to_be():
    m = ObservationModel()
    t = truth_for(16)
    obs = m.observe(t, np.random.default_rng(12))
    _, comp = m.log_prob(obs, _pred(t), components=True)

    s = obs.aft_age_sd / obs.aft_age
    assert comp["aft_age"].numpy() == pytest.approx(
        lognorm.logpdf(obs.aft_age, s, scale=t["aft_age"]), rel=1e-10)

    p = m.smeared(t["length_pdf"])
    expect = np.array([multinomial.logpmf(c, int(n), q)
                       for c, n, q in zip(obs.length_counts, obs.n_lengths, p)])
    assert comp["lengths"].numpy() == pytest.approx(expect, rel=1e-9)

    s_he = obs.he_age_sd / obs.he_age
    grain = lognorm.logpdf(obs.he_age, s_he, scale=t["he_age"])
    assert comp["he"].numpy() == pytest.approx(grain.sum(axis=1), rel=1e-10)


def test_the_components_add_up_to_the_total():
    m = ObservationModel()
    t = truth_for(8)
    obs = m.observe(t, np.random.default_rng(13))
    total, comp = m.log_prob(obs, _pred(t), components=True)
    assert total.numpy() == pytest.approx(sum(comp.values()).numpy())


@pytest.mark.parametrize("outlier_fraction", [0.0, 0.15])
def test_the_helium_term_is_a_normalised_density_in_the_observed_age(outlier_fraction):
    """Integrated over the observed age by quadrature, mixture included."""
    m = ObservationModel(he=AHeNoise(outlier_fraction=outlier_fraction))
    grid = np.exp(np.linspace(np.log(1e-3), np.log(1e6), 200_001))
    d = torch.as_tensor(np.log(grid) - np.log(30.0))
    s = torch.as_tensor(np.full_like(grid, m.he.rel_sd))
    dens = np.exp(m._lognormal_mixture(d, s).numpy()) / grid
    assert np.trapezoid(dens, grid) == pytest.approx(1.0, rel=1e-4)


def test_the_likelihood_does_not_care_which_grain_is_listed_first():
    m = ObservationModel()
    t = truth_for(6, g=4)
    t["he_age"] = np.tile(np.array([10.0, 25.0, 40.0, 70.0]), (6, 1))
    obs = m.observe(t, np.random.default_rng(15))
    pred = _pred(t)
    base = m.log_prob(obs, pred)

    # Permuting the data alone must change the score -- these grains really do
    # differ -- while permuting the data and the prediction together must not.
    assert not np.allclose(m.log_prob(obs.permute_grains(np.random.default_rng(17)),
                                      pred).numpy(), base.numpy())

    idx = np.argsort(np.random.default_rng(16).random((6, 4)), axis=1)
    grain_fields = ("he_age", "he_age_sd", "he_radius", "he_eU", "he_mask")
    both = type(obs)(**{**vars(obs),
                        **{f: np.take_along_axis(getattr(obs, f), idx, axis=1)
                           for f in grain_fields}})
    pred2 = dict(pred, he_age=torch.as_tensor(np.take_along_axis(t["he_age"], idx, axis=1)))
    assert m.log_prob(both, pred2).numpy() == pytest.approx(base.numpy())


def test_masked_grains_contribute_nothing():
    m = ObservationModel(he=AHeNoise(n_grains_range=(1, 2)))
    t = truth_for(50, g=4)
    obs = m.observe(t, np.random.default_rng(18))
    lp = m.log_prob(obs, _pred(t))
    assert np.all(np.isfinite(lp.numpy()))
    # Rows with fewer grains have fewer terms, so a two-grain row scores lower
    # than a one-grain row only through its extra (negative) term.
    assert np.all(lp.numpy()[obs.n_grains == 2] < 0)


def test_the_likelihood_is_maximised_at_the_truth():
    m = ObservationModel()
    t = truth_for(4000)
    obs = m.observe(t, np.random.default_rng(19))
    at_truth = m.log_prob(obs, _pred(t)).mean()
    for scale in (0.9, 1.1):
        assert m.log_prob(obs, _pred(t, scale)).mean() < at_truth


def test_a_sample_with_no_tracks_carries_no_age_information():
    m = ObservationModel()
    t = truth_for(64, age=0.3, U=0.5)          # ~0.03 expected tracks
    obs = m.observe(t, np.random.default_rng(20))
    assert np.any(obs.n_spontaneous == 0)
    _, comp = m.log_prob(obs, _pred(t), components=True)
    assert np.all(comp["aft_age"].numpy()[obs.n_spontaneous == 0] == 0.0)


def test_the_likelihood_hands_back_gradients_to_the_prediction():
    """PLAN.md R2: the point of a torch likelihood is d(log L)/dT through it."""
    m = ObservationModel()
    t = truth_for(16)
    obs = m.observe(t, np.random.default_rng(21))
    pred = _pred(t)
    for v in pred.values():
        v.requires_grad_(True)
    m.log_prob(obs, pred).sum().backward()
    for name, v in pred.items():
        assert torch.isfinite(v.grad).all() and v.grad.abs().sum() > 0, name


def test_absent_grains_and_untracked_samples_leave_the_gradient_finite():
    """In float32, where a clamped zero divides to infinity rather than 1e38.

    Regression: masking the term after computing it left ``-inf * 0 = nan`` in
    the backward pass for every absent grain and every sample with no counted
    tracks.  Both are substituted out before the division now.
    """
    m = ObservationModel(he=AHeNoise(n_grains_range=(1, 2)))
    t = truth_for(64, age=0.3, U=0.5, g=4)       # ~0.03 expected tracks
    obs = m.observe(t, np.random.default_rng(23))
    assert np.any(obs.n_spontaneous == 0) and np.any(~obs.he_mask)

    pred = {k: v.float().requires_grad_(True) for k, v in _pred(t).items()}
    lp = m.log_prob(obs, pred)
    assert torch.isfinite(lp).all()
    lp.sum().backward()
    for name, v in pred.items():
        assert torch.isfinite(v.grad).all(), name


# ---------------------------------------------------------------------------
# The Stage 5 / Goal 2 contract
# ---------------------------------------------------------------------------

def test_the_likelihood_closes_the_loop_from_a_thermal_history_to_a_gradient():
    """t-T path -> Stage 2/3 ports -> data -> log L -> d(log L)/dT, in one graph.

    This is the whole point of a torch likelihood (PLAN.md R2), and it is the
    one thing no unit test of either half can check on its own.
    """
    from chai.forward.torchimpl.ketcham import aft_forward
    from chai.forward.torchimpl.rdaam import rdaam_age

    t = torch.tensor([[0.0, 20.0, 60.0, 100.0]], dtype=torch.float64).repeat(2, 1)
    T = torch.tensor([[12.0, 55.0, 95.0, 120.0],
                      [18.0, 40.0, 70.0, 110.0]], dtype=torch.float64).requires_grad_(True)
    dpar = torch.full((2,), 1.9, dtype=torch.float64)

    aft = aft_forward(t, T, dpar, n_sub=8)
    grains = [rdaam_age(t, T, torch.full((2,), r, dtype=torch.float64),
                        torch.full((2,), 30.0, dtype=torch.float64),
                        torch.full((2,), 60.0, dtype=torch.float64),
                        n_sub=4, rdim=64)["corrected_age"] for r in (45.0, 90.0)]
    pred = {"aft_age": aft["age"], "length_pdf": aft["length_pdf"],
            "he_age": torch.stack(grains, dim=1)}

    m = ObservationModel()
    truth = {k: v.detach().numpy() for k, v in pred.items()}
    truth |= {"dpar": dpar.numpy(), "U": np.full(2, 30.0),
              "he_radius": np.tile([45.0, 90.0], (2, 1)), "he_eU": np.full((2, 2), 44.1)}
    obs = m.observe(truth, np.random.default_rng(22))

    lp = m.log_prob(obs, pred)
    assert torch.isfinite(lp).all()
    lp.sum().backward()
    assert torch.isfinite(T.grad).all()
    assert (T.grad.abs() > 0).any()
