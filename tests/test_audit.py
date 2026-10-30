"""The corpus audit: what it measures, and whether the measurements mean anything."""
from __future__ import annotations

import h5py
import numpy as np
import pytest
import torch

from chai.data.priors import GrainPrior, ThermalHistoryPrior
from chai.data.simulate import CorpusSpec, simulate
from chai.eval.audit import (
    DT_INVISIBLE,
    Corpus,
    Grain,
    Tolerance,
    conditional_spread,
    coverage,
    forward_pool,
    marginals,
    temperature_jacobian,
)
from chai.forward.torchimpl.ketcham import aft_forward
from chai.forward.torchimpl.rdaam import rdaam_age

SPEC = CorpusSpec(n_histories=60, n_grains=2, block_size=20, seed=5)
PRIOR = ThermalHistoryPrior(n_steps=128, n_features=32)


@pytest.fixture(scope="module")
def corpus_path(tmp_path_factory):
    path = tmp_path_factory.mktemp("audit") / "c.h5"
    simulate(path, SPEC, PRIOR, GrainPrior(), workers=2, progress=False)
    return path


# ---------------------------------------------------------------------------
# Reading
# ---------------------------------------------------------------------------

def test_splits_partition_the_corpus_exactly(corpus_path):
    with Corpus(str(corpus_path)) as c:
        parts = [c.split_index(s) for s in ("train", "val", "test")]
        assert sum(p.size for p in parts) == len(c)
        assert np.array_equal(np.sort(np.concatenate(parts)), np.arange(len(c)))


def test_an_unfinished_corpus_is_refused(corpus_path, tmp_path):
    """A half-written corpus has zeros where blocks did not run, and averaging
    those in would quietly bias every number in the audit."""
    broken = tmp_path / "broken.h5"
    broken.write_bytes(corpus_path.read_bytes())
    with h5py.File(broken, "r+") as f:
        f["block_done"][0] = False
    with pytest.raises(ValueError, match="unfinished"):
        Corpus(str(broken))


def test_marginals_count_grains_for_per_grain_columns(corpus_path):
    with Corpus(str(corpus_path)) as c:
        m = marginals(c, np.arange(len(c)))
        assert m["aft/age"]["n"] == len(c)
        assert m["he/corrected_age"]["n"] == len(c) * SPEC.n_grains
        q = m["aft/age"]["quantiles"]
        assert q[0.0] <= q[0.5] <= q[1.0]


# ---------------------------------------------------------------------------
# Coverage
# ---------------------------------------------------------------------------

def test_coverage_does_not_call_the_corners_of_a_diagonal_band_dead():
    """The whole point of the interior test.  A band along y = x leaves both
    off-diagonal corners empty, and calling those "dead regions" would report a
    problem in every corpus of two correlated variables."""
    rng = np.random.default_rng(0)
    x = rng.uniform(0, 1, 200_000)
    y = x + rng.normal(0, 0.05, x.size)
    cov = coverage(x, y, bins=20, min_count=30)

    assert cov["n_interior"] < cov["counts"].size / 2      # corners excluded
    assert cov["n_thin"] == 0                              # the band itself is dense
    assert not cov["interior"][0, -1] and not cov["interior"][-1, 0]


def test_the_thin_reading_is_not_just_the_edge_of_the_distribution():
    """A clean Gaussian has no holes, but its tail cells are always its
    sparsest.  A reachability rule that ran to the last occupied cell would
    flag those and so would flag every dataset ever passed to it."""
    rng = np.random.default_rng(2)
    x = rng.uniform(0, 1, 200_000)
    y = rng.normal(0.5, 0.12, x.size)
    assert coverage(x, y, bins=20, min_count=30)["n_thin"] == 0


def test_coverage_finds_a_hole_punched_inside_the_support():
    rng = np.random.default_rng(1)
    x = rng.uniform(0, 1, 100_000)
    y = rng.uniform(0, 1, x.size)
    keep = ~((np.abs(x - 0.5) < 0.06) & (np.abs(y - 0.5) < 0.06))
    cov = coverage(x[keep], y[keep], bins=20, min_count=30)

    assert cov["n_thin"] > 0
    cx, cy = np.nonzero(cov["thin"])
    assert np.all(np.abs(cov["x_edges"][cx] - 0.5) < 0.12)
    assert np.all(np.abs(cov["y_edges"][cy] - 0.5) < 0.12)


# ---------------------------------------------------------------------------
# Degeneracy
# ---------------------------------------------------------------------------

def _synthetic_pool(n=4000, n_grid=16, seed=0):
    """A pool whose observables are known functions of known nodes.

    ``aft_age`` depends on node 3 alone and ``he_age`` on node 11 alone, so the
    estimator has a right answer to be checked against: conditioning must
    collapse those nodes and leave every other one at the prior.
    """
    rng = np.random.default_rng(seed)
    T = rng.normal(80.0, 20.0, (n, n_grid))
    pdf = np.full((n, 8), 1.0 / 8 / 0.1)          # uninformative, constant
    return T, {
        "aft_age": 50.0 + 0.2 * T[:, 3],
        "mean_length": np.full(n, 15.0),           # constant: says nothing
        "he_age": 30.0 + 0.2 * T[:, 11],
        "length_pdf": pdf,
    }


#: Tight enough that the kernel is far narrower than the prior, so a node the
#: observable determines must collapse to near zero rather than merely shrink.
#: The default tolerances are realistic, not sharp: 5% of a 66 Ma age is 16 K
#: against a 20 K prior sd, which legitimately leaves a ratio near 0.64.
SHARP = Tolerance(aft_age_rel=1e-3, mean_length_um=1e-3, he_age_rel=1e-3)


def test_conditioning_collapses_the_node_the_observable_depends_on():
    T, pool = _synthetic_pool()
    ladder = (("age", ("aft_age",)),)
    res = conditional_spread(T, pool, np.arange(40), SHARP, 0.1,
                             ladder=ladder, device="cpu")
    ratio = res["rungs"][0]["ratio"]

    assert ratio[3] < 0.1, "node 3 drives the age and must be pinned by it"
    others = np.delete(ratio, 3)
    assert others.min() > 0.8, "no other node is involved and none may collapse"


def test_an_uninformative_observable_shrinks_nothing():
    T, pool = _synthetic_pool()
    ladder = (("length", ("mean_length", "length_pdf")),)
    res = conditional_spread(T, pool, np.arange(20), Tolerance(), 0.1,
                             ladder=ladder, device="cpu")
    assert res["rungs"][0]["ratio"].min() > 0.95


def test_adding_a_rung_tightens_the_node_it_informs_and_costs_sample_size():
    """Each rung adds an observable, so it can only remove weight.

    In expectation the spread cannot grow, but the *estimate* is a weighted sd
    at finite effective sample size, so an uninformed node's ratio wanders
    either side of 1 and asserting exact monotonicity would be asserting the
    absence of Monte Carlo noise.  What must hold is that the informed node
    collapses and the ESS falls.
    """
    T, pool = _synthetic_pool()
    ladder = (("a", ("aft_age",)), ("a+he", ("aft_age", "he_age")))
    res = conditional_spread(T, pool, np.arange(40), SHARP, 0.1,
                             ladder=ladder, device="cpu")
    first, second = res["rungs"]

    assert np.median(second["ess"]) < np.median(first["ess"])
    assert first["ratio"][11] > 0.8 and second["ratio"][11] < 0.1
    assert second["ratio"][3] < 0.1                       # still held by the age
    assert np.all(second["ratio"] <= first["ratio"] + 0.1)  # nothing blows up


def test_a_query_always_matches_itself_so_weights_never_vanish():
    T, pool = _synthetic_pool(n=500)
    res = conditional_spread(T, pool, np.arange(10), Tolerance(aft_age_rel=1e-6),
                             0.1, ladder=(("age", ("aft_age",)),), device="cpu")
    assert np.all(res["rungs"][0]["ess"] >= 1.0)
    assert np.all(np.isfinite(res["rungs"][0]["ratio"]))


# ---------------------------------------------------------------------------
# Stage 5: measured tolerances and multi-grain aliquots
# ---------------------------------------------------------------------------

def test_the_length_distance_is_a_transport_distance_in_microns():
    """Regression: the bin width was missing, so this ran ten times too tight.

    Two length distributions a known 0.5 um apart, scored against a 0.5 um
    tolerance, must give z = 1 -- and must do so whether the pdf arrives as a
    density (what the torch port returns) or as bin probabilities (what the
    corpus stores).  Both conventions are in circulation in this project, and
    the disagreement with :func:`chai.eval.study.w1_length`, which had the
    width, is what surfaced the bug.
    """
    from chai.eval.audit import _mismatch, _pool_tensors
    from chai.eval.study import w1_length

    bins = (np.arange(200) + 0.5) * 0.1
    def bell(c):
        p = np.exp(-0.5 * ((bins - c) / 1.2) ** 2)
        return p / p.sum()

    p, q = bell(14.0), bell(14.5)
    assert w1_length(p, q) == pytest.approx(0.5, abs=1e-3)

    for scale in (1.0, 1 / 0.1):                      # probabilities, then density
        pool = {"aft_age": np.array([10.0, 10.0]), "length_pdf": np.stack([p, q]) * scale}
        cols = _pool_tensors(pool, 0.1, torch.device("cpu"))
        z2 = _mismatch(cols, 0, ("length_pdf",), Tolerance(length_w1_um=0.5), 0.1)
        assert float(z2[1]) ** 0.5 == pytest.approx(1.0, abs=1e-3)


def test_a_per_row_tolerance_is_read_at_the_query_row():
    """Stage 5 passes a measured sigma per sample; it must index, not broadcast."""
    T, pool = _synthetic_pool()
    ladder = (("age", ("aft_age",)),)
    per_row = np.full(T.shape[0], 1.0)
    per_row[7] = 1e-3
    array_tol = Tolerance(aft_age_rel=per_row)

    sharp = conditional_spread(T, pool, np.array([7]), array_tol, 0.1,
                               ladder=ladder, device="cpu")["rungs"][0]["ratio"]
    same = conditional_spread(T, pool, np.array([7]), Tolerance(aft_age_rel=1e-3), 0.1,
                              ladder=ladder, device="cpu")["rungs"][0]["ratio"]
    loose = conditional_spread(T, pool, np.array([9]), array_tol, 0.1,
                               ladder=ladder, device="cpu")["rungs"][0]["ratio"]
    assert sharp == pytest.approx(same)
    assert sharp[3] < 0.1 < loose[3]


def test_repeating_a_grain_is_the_same_as_measuring_it_more_precisely():
    """The control rung in the Stage 5 ladder: three replicates are sigma/sqrt(3)."""
    T, pool = _synthetic_pool()
    pool = {**pool, "he_age_1": pool["he_age"]}
    three = conditional_spread(T, pool, np.arange(20), Tolerance(he_age_rel=0.02), 0.1,
                               ladder=(("x3", ("aft_age", "he_age_1", "he_age_1",
                                               "he_age_1")),), device="cpu")
    once = conditional_spread(T, pool, np.arange(20),
                              Tolerance(he_age_rel=0.02 / np.sqrt(3)), 0.1,
                              ladder=(("x1", ("aft_age", "he_age_1")),), device="cpu")
    assert three["rungs"][0]["ratio"] == pytest.approx(once["rungs"][0]["ratio"])


def test_every_grain_of_a_multi_grain_aliquot_is_conditioned_on():
    T, pool = _synthetic_pool()
    pool = {**pool, "he_age_0": pool["he_age"], "he_age_1": 30.0 + 0.2 * T[:, 5]}
    res = conditional_spread(T, pool, np.arange(40), SHARP, 0.1,
                             ladder=(("two grains", ("he_age_0", "he_age_1")),),
                             device="cpu")
    ratio = res["rungs"][0]["ratio"]
    assert ratio[11] < 0.01 and ratio[5] < 0.01
    # Two sharp observables leave an ESS of a few, and a weighted sd over a few
    # members is small everywhere; what must hold is the separation, not the
    # level -- the two nodes the grains see are pinned an order below the rest.
    assert np.delete(ratio, [5, 11]).min() > 10 * max(ratio[5], ratio[11])


def test_an_observable_with_no_tolerance_is_refused_rather_than_guessed():
    T, pool = _synthetic_pool()
    with pytest.raises(KeyError, match="mystery"):
        conditional_spread(T, {**pool, "mystery": pool["aft_age"]}, np.arange(2),
                           Tolerance(), 0.1, ladder=(("?", ("mystery",)),), device="cpu")


# ---------------------------------------------------------------------------
# Forward pool and sensitivity
# ---------------------------------------------------------------------------

GRAIN = Grain("test", radius=55.0, U=20.0, Th=40.0, dpar=1.9)


def _short_paths(n=3, n_grid=6):
    t = np.linspace(0.0, 40.0, n_grid)
    rng = np.random.default_rng(3)
    T = np.sort(rng.uniform(10.0, 140.0, (n, n_grid)), axis=1)
    return t, T


def test_forward_pool_matches_calling_the_ports_directly():
    t, T = _short_paths()
    got = forward_pool(T, t, GRAIN, device="cpu", n_sub_aft=8, n_sub_he=2,
                       rdim=32, batch=2)

    tt = torch.as_tensor(t, dtype=torch.float32).expand(T.shape[0], -1)
    TT = torch.as_tensor(T, dtype=torch.float32)
    def one(v):
        return torch.full((T.shape[0],), v, dtype=torch.float32)

    with torch.no_grad():
        aft = aft_forward(tt, TT, one(GRAIN.dpar), n_sub=8)
        he = rdaam_age(tt, TT, one(GRAIN.radius), one(GRAIN.U), one(GRAIN.Th),
                       n_sub=2, rdim=32, use_checkpoint=False)
    assert got["aft_age"] == pytest.approx(aft["age"].numpy(), rel=1e-6)
    assert got["he_age"] == pytest.approx(he["corrected_age"].numpy(), rel=1e-6)
    assert got["length_pdf"].shape == (T.shape[0], 200)


def test_the_sensitivity_jacobian_matches_central_differences():
    t, T = _short_paths(n=2)
    tol = Tolerance()
    jac = temperature_jacobian(T, t, GRAIN, tol, device="cpu", n_sub_aft=8,
                               n_sub_he=2, rdim=32, batch=2)

    h, node = 0.05, 3
    Tp, Tm = T.copy(), T.copy()
    Tp[:, node] += h
    Tm[:, node] -= h
    kw = {"device": "cpu", "n_sub_aft": 8, "n_sub_he": 2, "rdim": 32, "batch": 2}
    up = forward_pool(Tp, t, GRAIN, **kw)
    dn = forward_pool(Tm, t, GRAIN, **kw)

    # The stored value must be the observable the gradient was taken of, or a
    # caller forming sigma from it would scale the wrong quantity.  The
    # Jacobian runs in float64 and forward_pool in float32, which is the whole
    # of the gap allowed for here.
    base = forward_pool(T, t, GRAIN, **kw)
    for k in ("aft_age", "mean_length", "he_age"):
        assert jac[f"{k}_value"] == pytest.approx(base[k], rel=1e-4)

    for key, obs in (("aft_age", "aft_age"), ("mean_length", "mean_length")):
        fd = (up[obs] - dn[obs]) / (2 * h)
        # The reference is float32 (forward_pool is, deliberately -- Stage 2-3
        # measured the penalty as negligible), so 1e-2 is the floor here.
        assert jac[f"{key}_jacobian"][:, node] == pytest.approx(fd, rel=1e-2, abs=1e-6)


def test_an_unreachable_node_is_reported_as_invisible_not_as_infinity():
    """``sigma / |dObs/dT|`` runs to infinity wherever the gradient underflows,
    which is the right answer and an unplottable one."""
    t, T = _short_paths(n=2)
    jac = temperature_jacobian(T, t, GRAIN, Tolerance(), device="cpu", n_sub_aft=8,
                               n_sub_he=2, rdim=32, batch=2)
    for k in ("aft_age", "mean_length", "he_age"):
        dT = jac[f"{k}_dT"]
        assert np.all(np.isfinite(dT))
        assert dT.max() <= DT_INVISIBLE

    # Node 0 is the present day, where every path starts and no track has yet
    # had time to anneal: the AFT age cannot depend on it.
    assert jac["aft_age_dT"][:, 0].min() > jac["aft_age_dT"][:, 1:].min()
