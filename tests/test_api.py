"""The thin API: a sample in the user's terms reaches the head as the training pipeline would hand it over,
and boxes are exact rejection on the head's own draws."""
from __future__ import annotations

from dataclasses import asdict

import numpy as np
import pytest
import torch

from chai import api
from chai.constraints import WITHIN, Box, InfeasibleBoxes, box_weights
from chai.data.observe import ObservationModel
from chai.data.priors import GrainPrior
from chai.models.encoder import EncoderConfig
from chai.models.flow import FlowConfig
from chai.models.gaussian import AmortizedPosterior, draw_z
from chai.train.data import (
    AFT_FLAGS,
    AFT_INPUTS,
    GRAIN_INPUTS,
    CorpusSplit,
    EffortPrior,
    NoisyBatches,
    features_from_aliquot,
)

torch.set_default_dtype(torch.float32)


@pytest.fixture(scope="module")
def model(tiny_corpus, tiny_latent) -> api.Model:
    torch.manual_seed(0)
    net = AmortizedPosterior(tiny_latent, n_aft=len(AFT_INPUTS), n_bins=200, n_grain=len(GRAIN_INPUTS),
                             aft_flags=AFT_FLAGS, encoder=EncoderConfig(d_embed=32, d_hidden=32, d_grain=16, d_hist=8),
                             head=FlowConfig(n_layers=2, n_bins=4, d_hidden=16, n_hidden=1),
                             meta={"config": {"effort": asdict(EffortPrior())}})
    import json

    import h5py
    with h5py.File(tiny_corpus, "r") as f:
        prior = json.loads(f.attrs["prior_config"])
    g = GrainPrior()
    grain = {"radius_range": g.radius_range, "U_range": g.U_range, "th_u_ratio_range": g.th_u_ratio_range,
             "dpar_range": g.dpar_range}
    return api.Model(net, ObservationModel(), prior, grain, g_max=5)


@pytest.fixture(scope="module")
def observed(tiny_corpus, tiny_latent):
    c = CorpusSplit(tiny_corpus, "val", latent=tiny_latent, max_rows=256)
    ds = NoisyBatches(c, ObservationModel(), EffortPrior(p_no_aft=0.25, p_no_he=0.25), seed=0)
    obs, _ = ds.observe_aliquot(np.arange(256), np.random.default_rng(3))
    return obs


def test_a_sample_reaches_the_head_as_the_training_pipeline_would_hand_it_over(model, observed):
    for i in range(40):
        one = observed[i]
        want = features_from_aliquot(one)
        got = features_from_aliquot(model.to_aliquot(api.Sample.from_aliquot(one))[0])
        np.testing.assert_allclose(got["length_pdf"], want["length_pdf"], atol=1e-6)
        np.testing.assert_array_equal(got["grain_mask"][:, :want["grain_mask"][0].sum()],
                                      want["grain_mask"][:, want["grain_mask"][0]])
        # grains in the order reported, padding dropped
        m = want["grain_mask"][0]
        np.testing.assert_allclose(got["grains"][0][:m.sum()], want["grains"][0][m], atol=1e-5)
        # every AFT column exact but the age sd, which the API recomputes from the count
        cols = [k for k, name in enumerate(AFT_INPUTS) if name != "log_age_sd"]
        np.testing.assert_allclose(got["aft"][:, cols], want["aft"][:, cols], atol=1e-5)
        sd = AFT_INPUTS.index("log_age_sd")
        assert np.abs(got["aft"][:, sd] - want["aft"][:, sd]).max() < 0.02    # within 2%


def test_warnings_name_what_the_head_was_not_trained_on(model):
    s = api.Sample(api.AFT(age=150.0, n_spontaneous=100, age_sd=40.0, lengths=[12.0] * 5, dpar=5.0),
                   [api.Grain(age=30, radius=200, eU=900, age_sd=0.3)] * 7)
    _, notes = model.to_aliquot(s)
    text = "\n".join(notes)
    for what in ("time span", "Dpar", "track lengths", "radius", "eU", "1σ of", "differs from", "7 He grains"):
        assert what in text, what
    clean = api.Sample(api.AFT(age=40.0, n_spontaneous=200, lengths=[13.0] * 80, dpar=2.0),
                       [api.Grain(age=30, radius=60, eU=40)] * 3)
    assert model.to_aliquot(clean)[1] == []


def test_lengths_with_angles_reach_the_head_projected(model):
    rng = np.random.default_rng(3)
    raw, phi = rng.uniform(10.0, 15.0, 80), rng.uniform(0.0, 90.0, 80)
    with_angles = api.Sample(api.AFT(age=40.0, n_spontaneous=200, lengths=raw, angles=phi, dpar=2.0))
    projected = api.Sample(api.AFT(age=40.0, n_spontaneous=200, lengths=with_angles.aft.projected_lengths, dpar=2.0))
    (a, notes_a), (b, notes_b) = model.to_aliquot(with_angles), model.to_aliquot(projected)
    np.testing.assert_array_equal(a.length_counts, b.length_counts)
    assert a.mean_length[0] > np.mean(raw)
    assert any("projected to the c axis" in n for n in notes_a) and notes_b == []


def test_the_count_is_backed_out_of_an_age_sd(model):
    ns = 150
    sd = 40.0 * np.sqrt(1 / ns + model.observation.aft.systematic_rel_var)
    obs, notes = model.to_aliquot(api.Sample(api.AFT(age=40.0, age_sd=sd)))
    assert int(obs.n_spontaneous[0]) == ns and "corresponds to" in notes[0]
    with pytest.raises(ValueError, match="systematic"):
        model.to_aliquot(api.Sample(api.AFT(age=40.0, age_sd=0.5)))


def test_samples_must_carry_data():
    with pytest.raises(ValueError):
        api.Sample()
    with pytest.raises(ValueError):
        api.AFT(age=10.0)


def test_boxes_are_exact_rejection_on_the_head_s_own_draws(model, observed):
    one = observed[0]
    box = Box((5.0, 15.0), (30.0, 90.0))
    free = model.infer(one, n=20_000, seed=7)
    frac = box_weights([box], free.t, torch.as_tensor(free.T)).mean().item()
    assert 0.02 < frac < 0.98
    post = model.infer(one, n=2000, boxes=[box], seed=7, chunk=5000)
    assert len(post) == 2000
    assert box_weights([box], post.t, torch.as_tensor(post.T)).min().item() == 1.0
    # the acceptance is the head's own probability of the box, to binomial error
    assert abs(post.acceptance - frac) < 4 * np.sqrt(frac * (1 - frac) / post.n_drawn) + 4 * np.sqrt(frac * (1 - frac) / 20_000)
    # the accepted draws are the free draws that satisfy the box: same seed, same stream
    gen = torch.Generator().manual_seed(7)
    with torch.no_grad():
        q = model.net(model.batch(one))
        T, _ = model.net.latent.inverse(draw_z(q, 5000, gen)[:, 0], clamp=True)
    keep = box_weights([box], post.t, T) > 0.5
    np.testing.assert_allclose(post.T[:int(keep.sum())], T[keep].numpy()[:2000], atol=1e-6)


def test_a_member_that_removes_its_floor_draws_enough_for_the_correction(model, observed):
    """Stage 19e: remove_floor reads the draws' covariance, so a flagged member draws at least
    min_draws per pass however few paths are asked for, and subsamples."""
    from chai.models.gaussian import remove_floor

    one = observed[0]
    assert model.min_draws == 1
    model.net.floor_sd = 1.0
    try:
        k = model.min_draws
        assert k == max(api.MIN_FLOOR_DRAWS, 2 * len(model.t))
        post = model.infer(one, n=10, seed=3)
        assert len(post) == 10 and post.n_drawn == k
        # The paths are the corrected draws: the same stream, corrected across all k, then the first 10.
        gen = torch.Generator().manual_seed(3)
        with torch.no_grad():
            q = model.net(model.batch(one))
            T, _ = model.net.latent.inverse(draw_z(q, k, gen), clamp=False)
        L = model.net.latent
        T = torch.maximum(torch.minimum(remove_floor(T, 1.0), L.box_hi.to(T)), L.box_lo.to(T))[:, 0]
        np.testing.assert_allclose(post.T, T[:10].numpy(), atol=1e-4)
        box = Box((5.0, 15.0), (30.0, 90.0))
        assert model.infer(one, n=50, boxes=[box], seed=3, chunk=64).n_drawn % k == 0
    finally:
        model.net.floor_sd = 0.0


def test_a_soft_box_keeps_draws_in_proportion_to_their_weight(model, observed):
    one = observed[1]
    free = model.infer(one, n=40_000, seed=1)
    box = Box((20.0, 20.0), (60.0, 80.0), soft_sd=15.0)
    w = box_weights([box], free.t, torch.as_tensor(free.T)).numpy()
    post = model.infer(one, n=20_000, boxes=[box], seed=2)
    assert abs(post.acceptance - w.mean()) < 0.02
    # the soft-constrained temperature at 20 Ma is the free one reweighted
    x_free, x_post = free.at(20.0), post.at(20.0)
    np.testing.assert_allclose(x_post.mean(), (w * x_free).sum() / w.sum(), atol=1.0)


def test_infeasible_boxes_are_refused_before_any_draw_and_hopeless_ones_after(model, observed):
    with pytest.raises(InfeasibleBoxes, match="rate limit"):
        model.infer(observed[0], boxes=[Box((0.5, 0.5), (190.0, 200.0))])
    # feasible for the prior, but nothing the head draws gets there
    with pytest.raises(InfeasibleBoxes, match="none of"):
        model.infer(observed[0], boxes=[Box((10, 90), (199.5, 200.0), WITHIN)], max_draws=4000, chunk=2000)


def test_cooling_only_rejects_the_draws_that_reheat(model, observed):
    from chai.constraints import reheating

    one = observed[0]
    free = model.infer(one, n=20_000, seed=7)
    heat = reheating(torch.as_tensor(free.T)).numpy()
    tol = float(np.quantile(heat, 0.4))                            # an untrained head reheats a lot: its own scale
    post = model.infer(one, n=1000, cooling_only=True, reheat_tol=tol, seed=7, chunk=5000)
    assert len(post) == 1000 and post.cooling_only and post.reheat_tol == tol
    assert reheating(torch.as_tensor(post.T)).max().item() <= tol + 1e-6
    frac = (heat <= tol).mean()
    assert abs(post.acceptance - frac) < 4 * np.sqrt(frac * (1 - frac) / post.n_drawn) + 0.02
    assert "cooling only: reheating at most" in post.summary() and "acceptance" in post.summary()
    # with a box, the product of the two indicators
    box = Box((5.0, 15.0), (30.0, 90.0))
    both = model.infer(one, n=500, boxes=[box], cooling_only=True, reheat_tol=tol, seed=7, chunk=5000)
    assert box_weights([box], both.t, torch.as_tensor(both.T)).min().item() == 1.0
    assert reheating(torch.as_tensor(both.T)).max().item() <= tol + 1e-6
    assert not model.infer(one, n=10, seed=7).cooling_only
    with pytest.raises(ValueError, match="reheat_tol"):
        model.infer(one, n=10, cooling_only=True, reheat_tol=-1.0)


def test_the_result_reports_what_it_is(model, observed):
    post = model.infer(observed[2], n=500, boxes=[Box((1, 1), (0, 200))])
    assert post.acceptance == 1.0 and post.n_drawn >= 500
    assert post.quantiles().shape == (5, post.t.size)
    assert set(post.functionals()) and {"tau", "rate_sd", "drift"} <= set(post.hyper)
    assert "acceptance" in post.summary()
    post.plot()


# ---------------------------------------------------------------------------
# The family (PLAN.md Stage 16g)
# ---------------------------------------------------------------------------

def _tiny_model(corpus: str, latent, g_max: int = 5) -> api.Model:
    import json

    import h5py
    torch.manual_seed(1)
    net = AmortizedPosterior(latent, n_aft=len(AFT_INPUTS), n_bins=200, n_grain=len(GRAIN_INPUTS),
                             aft_flags=AFT_FLAGS, encoder=EncoderConfig(d_embed=32, d_hidden=32, d_grain=16, d_hist=8),
                             head=FlowConfig(n_layers=2, n_bins=4, d_hidden=16, n_hidden=1),
                             meta={"config": {"effort": asdict(EffortPrior())}})
    with h5py.File(corpus, "r") as f:
        prior = json.loads(f.attrs["prior_config"])
    g = GrainPrior()
    grain = {"radius_range": g.radius_range, "U_range": g.U_range, "th_u_ratio_range": g.th_u_ratio_range,
             "dpar_range": g.dpar_range}
    return api.Model(net, ObservationModel(), prior, grain, g_max=g_max)


@pytest.fixture(scope="module")
def family(model, tiny_corpus_long) -> api.Family:
    from chai.models.chooser import MemberChooser
    from chai.models.latent import Latent

    long = _tiny_model(tiny_corpus_long, Latent.fit(tiny_corpus_long))
    chooser = MemberChooser(len(AFT_INPUTS), 200, len(GRAIN_INPUTS), aft_flags=AFT_FLAGS,
                            encoder=EncoderConfig(d_embed=16, d_hidden=16, d_grain=8, d_hist=8), d_hidden=16)
    return api.Family({100: model, 250: long}, chooser)


def _bias(family: api.Family, log_bf: float) -> None:
    """Pin the chooser's answer: an untrained one says ~0 for every sample."""
    with torch.no_grad():
        family.chooser.head[-1].weight.zero_()
        family.chooser.head[-1].bias.fill_(log_bf)


def test_the_family_answers_with_one_member_on_its_own_grid(family, observed):
    _bias(family, np.log(3.0))                                     # p(250 | x) = 0.75
    post = family.infer(observed[0], n=400, seed=3)
    assert isinstance(post, api.PathPosterior)
    assert post.choice.member == 250 and post.choice.reason == "data"
    assert post.choice.p == pytest.approx(0.75, abs=1e-6)
    assert post.T.shape == (400, 24) and post.quantiles().shape == (5, 24) and post.time_span == post.t[-1].round()
    # every draw is the member's own: nothing mixed in
    np.testing.assert_array_equal(post.T, family[250].infer(observed[0], n=400, seed=3).T)
    assert "250 Ma member (the more probable): p = 0.750" in post.summary()
    post.plot()
    _bias(family, -np.log(3.0))
    short = family.infer(observed[0], n=400, seed=3)
    assert short.choice.member == 100 and short.choice.p == pytest.approx(0.75, abs=1e-6)
    assert short.T.shape == (400, 16)


def test_prior_odds_move_the_choice_and_a_named_member_is_that_member(family, observed):
    _bias(family, 0.0)
    assert family.infer(observed[0], n=200, prior_odds=1e6).choice.member == 250
    assert family.infer(observed[0], n=200, prior_odds=1e-6).choice.member == 100
    with pytest.raises(ValueError, match="prior_odds"):
        family.infer(observed[0], prior_odds=0.0)
    one = family.infer(observed[0], n=100, member=100, seed=5)
    assert isinstance(one, api.PathPosterior) and one.choice.reason == "asked"
    np.testing.assert_array_equal(one.T, family[100].infer(observed[0], n=100, seed=5).T)
    _bias(family, 6.0)
    assert "you chose 100 Myr; the data favour 250 Myr" in family.infer(observed[0], n=50, member=100).warnings[0]


def test_an_undecided_old_sample_says_so(family):
    _bias(family, 0.2)                                             # p(250 | x) = 0.55
    s = api.Sample(aft=api.AFT(age=80.0, n_spontaneous=300, lengths=[13.0] * 40), grains=[api.Grain(70.0, 60, 30)])
    post = family.infer(s, n=50)
    assert post.choice.member == 250
    assert post.warnings[0].startswith("the data barely prefer 250 Myr over 100 Myr")


def test_cooling_only_enters_the_family_s_odds_as_boxes_do(family, observed):
    _bias(family, 0.0)                                             # p(250 | x) = 1/2 before the constraint
    kw = {"cooling_only": True, "reheat_tol": 60.0, "seed": 4}
    post = family.infer(observed[0], n=300, **kw)
    acc = {h: family[h].infer(observed[0], n=300, **kw).acceptance for h in (100, 250)}
    p250 = acc[250] / (acc[100] + acc[250])
    assert post.cooling_only and post.choice.reason == "data"
    assert post.choice.member == (250 if p250 >= 0.5 else 100)
    assert post.choice.p == pytest.approx(max(p250, 1 - p250), abs=1e-9)


def test_an_input_file_runs_as_the_family_would(family, observed, tmp_path):
    from chai import inputs

    _bias(family, 0.0)
    sample = api.Sample.from_aliquot(observed[0])
    inp = inputs.Input(sample, boxes=[Box((5.0, 15.0), (0.0, 200.0))], member=100, n=50, seed=9)
    post = inputs.load(inp.save(tmp_path / "x.json")).infer(family)
    ref = family.infer(sample, 50, member=100, boxes=inp.boxes, seed=9)
    np.testing.assert_array_equal(post.T, ref.T)
    np.testing.assert_array_equal(inp.infer(family[100]).T, family[100].infer(sample, 50, boxes=inp.boxes, seed=9).T)


def test_a_box_past_the_short_horizon_routes_to_the_long_member(family, observed):
    _bias(family, -5.0)                                            # the data prefer the short member
    box = Box((150.0, 200.0), (0.0, 200.0))
    post = family.infer(observed[0], n=100, boxes=[box])
    assert post.choice.member == 250 and post.choice.reason == "boxes" and len(post) == 100
    assert "so the 250 Myr model answers" in post.warnings[0]
    with pytest.raises(InfeasibleBoxes, match="outside the prior"):
        family.infer(observed[0], n=100, boxes=[box], member=100)


def test_boxes_neither_member_can_satisfy_are_refused(family, observed):
    # A present-day box above the 25 °C wall: both members refuse, and so does the family.
    with pytest.raises(InfeasibleBoxes, match="exclude the range"):
        family.infer(observed[0], n=100, boxes=[Box((0.0, 0.0), (40.0, 60.0), WITHIN)])


def test_the_chooser_must_be_for_the_family_s_members(model, family):
    from chai.models.chooser import MemberChooser

    wrong = MemberChooser(len(AFT_INPUTS), 200, len(GRAIN_INPUTS), members=(100, 300))
    with pytest.raises(ValueError, match="chooser is for"):
        api.Family({100: model, 250: family[250]}, wrong)
    with pytest.raises(ValueError, match="two members"):
        api.Family({100: model}, family.chooser)


def test_uncorrected_helium_ages_are_corrected_here():
    from chai.forward.torchimpl.rdaam import rdaam_age

    # The forward model's own raw and corrected ages for one path, Th/U = 1: alpha_correct maps one to the other.
    t = torch.tensor([[0.0, 40.0, 80.0]], dtype=torch.float64)
    T = torch.tensor([[10.0, 60.0, 120.0]], dtype=torch.float64)
    for r in (40.0, 70.0, 110.0):
        he = rdaam_age(t, T, torch.tensor([r], dtype=torch.float64), torch.tensor([20.0], dtype=torch.float64),
                       torch.tensor([20.0], dtype=torch.float64), n_sub=8, rdim=512, use_checkpoint=False)
        raw, corr = float(he["age"][0]), float(he["corrected_age"][0])
        mine, ft = api.alpha_correct(raw, r, 20.0 * (1 + 0.235))
        assert mine == pytest.approx(corr, rel=2e-3) and 0.5 < ft < 1.0
    g = api.Grain(age=30.0, radius=60.0, eU=30.0, corrected=False)
    assert g.corrected_age > 30.0 and g.ft == pytest.approx(30.0 / g.corrected_age, rel=0.01)
    assert api.Grain(age=30.0, radius=60.0, eU=30.0).corrected_age == 30.0
    with pytest.raises(ValueError, match="too small to correct"):
        api.Grain(age=30.0, radius=9.0, eU=30.0, corrected=False)


def test_predict_runs_the_forward_models_on_the_first_paths():
    from chai.forward.torchimpl.ketcham import aft_forward
    from chai.forward.torchimpl.rdaam import rdaam_age

    t = np.linspace(0.0, 100.0, 9)
    T = np.stack([np.linspace(10.0, 120.0, 9), np.linspace(20.0, 90.0, 9), np.linspace(5.0, 150.0, 9)])
    post = api.PathPosterior(t=t, T=T, hyper={})
    s = api.Sample(aft=api.AFT(age=40.0, n_spontaneous=200, dpar=2.2), grains=[api.Grain(30.0, 60.0, 37.05)])
    pred = api.predict(post, s, n=2)
    assert pred.paths.tolist() == [0, 1] and pred.length_pdf.shape == (2, 200) and pred.he_age.shape == (2, 1)
    tt = torch.as_tensor(np.stack([t, t]))
    TT = torch.as_tensor(T[:2])
    a = aft_forward(tt, TT, torch.full((2,), 2.2, dtype=torch.float64), n_sub=api.PREDICT_N_SUB,
                    bins=torch.as_tensor(api.LENGTH_BINS))
    np.testing.assert_allclose(pred.aft_age, a["age"].numpy())
    one = lambda v: torch.full((2,), v, dtype=torch.float64)  # noqa: E731
    he = rdaam_age(tt, TT, one(60.0), one(30.0), one(30.0), n_sub=api.PREDICT_N_SUB, rdim=api.PREDICT_RDIM,
                   use_checkpoint=False)["corrected_age"]
    np.testing.assert_allclose(pred.he_age[:, 0], he.numpy())
    assert pred.quantiles(pred.aft_age).shape == (5,)
    assert api.predict(post, api.Sample(grains=[api.Grain(30.0, 60.0, 30.0)])).aft_age is None


def test_the_sensitive_window_is_each_chronometer_s_partial_zone():
    s = api.Sample(aft=api.AFT(age=40.0, n_spontaneous=200, dpar=2.0),
                   grains=[api.Grain(30.0, 40.0, 10.0), api.Grain(30.0, 100.0, 150.0)])
    w = api.sensitive_window(s)
    lo_a, hi_a = w.systems["AFT"]
    assert 45 < lo_a < 55 and 105 < hi_a < 120                      # the partial annealing zone, 10 Myr
    assert w.systems["He 1"][0] < w.systems["He 2"][0]               # a small low-eU grain is the more sensitive
    assert w.lo == min(v[0] for v in w.systems.values()) and w.hi == hi_a
    post = api.PathPosterior(t=np.linspace(0, 100, 9), T=np.full((3, 9), 50.0), hyper={})
    post.plot(window=w)
