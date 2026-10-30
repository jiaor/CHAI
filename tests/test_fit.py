"""The misfit check (chai.fit, PLAN.md Stage 18a): its tail probabilities on predictions whose answer is known,
its intervals, its warnings, and the two samples built to misfit."""
from __future__ import annotations

import dataclasses

import numpy as np
import pytest
from scipy.stats import norm, poisson

from chai import api, fit
from chai.data.observe import ObservationModel

OBS = ObservationModel()
BINS = api.LENGTH_BINS


def _pred(n=50, aft=None, he=(), pdf_at=None):
    pdf = None
    if pdf_at is not None:
        pdf = np.zeros((n, BINS.size))
        pdf[:, int(pdf_at / api.BIN_WIDTH)] = 1 / api.BIN_WIDTH
    return api.Prediction(paths=np.arange(n), aft_age=None if aft is None else np.full(n, float(aft)), length_pdf=pdf,
                          he_age=np.tile(np.asarray(he, float), (n, 1)).reshape(n, len(he)), bins=BINS.copy())


def test_level():
    assert fit.datum_level(1, 0.05) == pytest.approx(0.05)
    assert 1 - (1 - fit.datum_level(7, 0.05, 1.0)) ** 7 == pytest.approx(0.05)       # Šidák at beta = 1
    assert 1 - (1 - fit.datum_level(4, 0.3, 1.5)) ** 8 == pytest.approx(0.3)
    assert fit.datum_level(3) > fit.datum_level(12)


@pytest.mark.parametrize("z", [0.0, 1.0, -2.5, 3.2])
def test_helium_tail_is_the_lognormal(z):
    rel = OBS.he.rel_sd
    s = api.Sample(grains=[api.Grain(age=30.0 * np.exp(z * rel), radius=60, eU=30)])
    d = fit.check(s, _pred(he=[30.0]), OBS).data[0]
    assert d.F == pytest.approx(norm.cdf(z), abs=1e-12)
    assert d.p == pytest.approx(2 * norm.sf(abs(z)), abs=1e-12)


def test_aft_tail_is_the_poisson_without_systematics():
    obs = OBS.replace(aft={"zeta_rel_sd": 0.0, "u_rel_sd": 0.0})
    ns, age = 40, 30.0
    # A path predicting half the observed age expects about half the count.
    s = api.Sample(aft=api.AFT(age=age, n_spontaneous=ns))
    d = fit.check(s, _pred(aft=15.0), obs).data[0]
    lam = ns * np.expm1(fit.LAMBDA_D_MA * 15.0) / np.expm1(fit.LAMBDA_D_MA * age)
    mid_p = poisson.cdf(ns - 1, lam) + 0.5 * poisson.pmf(ns, lam)
    assert d.F == pytest.approx(mid_p, rel=0.05)
    assert d.p < 1e-3


def test_aft_at_the_prediction_is_central():
    s = api.Sample(aft=api.AFT(age=50.0, n_spontaneous=400))
    d = fit.check(s, _pred(aft=50.0), OBS).data[0]
    assert 0.45 < d.F < 0.55 and not d.outside


def test_mean_length_and_the_minimum_count():
    L = [14.0 + 0.05] * 30
    s = api.Sample(aft=api.AFT(age=50.0, n_spontaneous=400, lengths=L))
    f = fit.check(s, _pred(aft=50.0, pdf_at=14.05), OBS)
    assert [d.label for d in f.data] == ["AFT", "Mean length"]
    assert f.data[1].p == pytest.approx(1.0, abs=1e-6)
    few = api.Sample(aft=api.AFT(age=50.0, n_spontaneous=400, lengths=L[:fit.FIT_MIN_LENGTHS - 1]))
    assert [d.label for d in fit.check(few, _pred(aft=50.0, pdf_at=14.05), OBS).data] == ["AFT"]


def test_interval_holds_the_level():
    s = api.Sample(grains=[api.Grain(age=30.0, radius=60, eU=30), api.Grain(age=31.0, radius=60, eU=30)])
    pred = _pred(he=[29.0, 33.0])
    f = fit.check(s, pred, OBS)
    for d, mu in zip(f.data, (29.0, 33.0)):
        z = norm.isf(f.level / 2)
        assert d.interval[0] == pytest.approx(mu * np.exp(-z * OBS.he.rel_sd), rel=1e-9)
        assert d.interval[1] == pytest.approx(mu * np.exp(z * OBS.he.rel_sd), rel=1e-9)


def test_warnings_and_caution():
    grains = [api.Grain(age=a, radius=60, eU=30) for a in (30.0, 30.0, 45.0)]
    f = fit.check(api.Sample(grains=grains), _pred(he=[30.0, 30.0, 30.0]), OBS)
    assert [d.outside for d in f.data] == [False, False, True]
    assert not f.caution
    assert f.warnings == [f"He 3 (45.0 Ma) is outside what the histories predict, {100 * (1 - f.level):.3g}%: "
                          f"{f.data[2].interval[0]:.1f}–{f.data[2].interval[1]:.1f} Ma"]
    # The AFT age alone is enough for the caution.
    s = api.Sample(aft=api.AFT(age=90.0, n_spontaneous=400), grains=grains[:2])
    f = fit.check(s, _pred(aft=45.0, he=[30.0, 30.0]), OBS)
    assert f.caution and f.warnings[0] == fit.CAUTION and f.warnings[1].startswith("AFT (90.0 Ma)")


@pytest.fixture(scope="module")
def family():
    import torch

    # The forward models run on the CPU here; under xdist, every worker taking all cores thrashes.
    torch.set_num_threads(min(4, torch.get_num_threads()))
    from chai import api

    return api.load()


@pytest.fixture(scope="module")
def cases(family):
    from chai.export import fixture_samples, sample_from_dict

    return {c["name"]: (sample_from_dict(c["sample"]), c["T_present"]) for c in fixture_samples(family) if c.get("sample")}


@pytest.mark.parametrize("name", ["misfit_inverted_he", "misfit_old_aft"])
def test_samples_built_to_misfit_warn(family, cases, name):
    s, T_present = cases[name]
    post = family.infer(s, 500, T_present=T_present)
    f = api.check_fit(post, s, family)
    assert f.misfit, [(d.label, d.p) for d in f.data]
    assert post.fit is f
    assert all(w in post.warnings for w in f.warnings)


def test_a_simulated_sample_fits(family, cases):
    s, T_present = cases["slow_cooling"]
    post = family.infer(s, 500, T_present=T_present)
    f = api.check_fit(post, s, family)
    assert not f.misfit, [(d.label, d.p) for d in f.data]
