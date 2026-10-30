"""The Stage 11 data pipeline: effort drawn per sample, dropout, and a loader that redraws."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from chai.data.aliquot import Aliquot
from chai.data.observe import EFFORT_KEYS, ObservationModel
from chai.eval.calibration import MODALITIES, modality_of
from chai.train.data import (
    AFT_FLAGS,
    AFT_INPUTS,
    GRAIN_INPUTS,
    CorpusSplit,
    EffortPrior,
    NoisyBatches,
    features_from_aliquot,
    fixed_batches,
    make_loader,
)

BINS = (np.arange(200) + 0.5) * 0.1


def bell(centre=14.5, width=1.4) -> np.ndarray:
    p = np.exp(-0.5 * ((BINS - centre) / width) ** 2)
    return p / p.sum()


def truth_for(n, g=4, seed=0) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    return {
        "aft_age": rng.uniform(5.0, 90.0, n),
        "length_pdf": np.tile(bell(), (n, 1)),
        "dpar": rng.uniform(1.5, 2.5, n),
        "U": rng.uniform(5.0, 60.0, n),
        "he_age": rng.uniform(5.0, 60.0, (n, g)),
        "he_radius": rng.uniform(40.0, 90.0, (n, g)),
        "he_eU": rng.uniform(10.0, 80.0, (n, g)),
    }


# ---------------------------------------------------------------------------
# The effort prior
# ---------------------------------------------------------------------------

def test_effort_draws_lie_in_their_ranges_and_never_leave_a_sample_empty():
    ep = EffortPrior(p_no_aft=0.5, p_no_he=0.5, p_no_lengths=0.3)
    e = ep.draw(20_000, np.random.default_rng(0), g_max=12)
    assert set(e) <= set(EFFORT_KEYS)
    assert e["n_grains_counted"].min() >= 10 and e["n_grains_counted"].max() <= 40
    assert 1000.0 <= e["area_per_grain_um2"].min() and e["area_per_grain_um2"].max() <= 6000.0
    have_len = e["n_lengths"] > 0
    assert e["n_lengths"][have_len].min() >= 20 and e["n_lengths"].max() <= 150
    assert e["n_he_grains"].max() <= 12
    # No lengths without an AFT analysis, and never nothing at all.
    assert not (e["n_lengths"][~e["aft_present"]] > 0).any()
    assert (e["aft_present"] | (e["n_he_grains"] > 0)).all()
    # Both marginals still take their share of absences.
    assert 0.25 < (~e["aft_present"]).mean() < 0.45
    assert 0.25 < (e["n_he_grains"] == 0).mean() < 0.45


def test_effort_reads_g_max_from_the_corpus_unless_told_otherwise():
    e = EffortPrior().draw(5_000, np.random.default_rng(1), g_max=5)
    assert e["n_he_grains"].max() == 5
    e = EffortPrior(n_he_grains=(2, 3)).draw(5_000, np.random.default_rng(1), g_max=12)
    assert set(np.unique(e["n_he_grains"][e["n_he_grains"] > 0])) == {2, 3}


def test_effort_config_round_trips_and_rejects_nonsense():
    ep = EffortPrior(n_lengths=(30, 90), p_no_aft=0.1)
    assert EffortPrior.from_config(ep.to_config()) == ep
    with pytest.raises(ValueError):
        EffortPrior(n_grains_counted=(40, 10))
    with pytest.raises(ValueError):
        EffortPrior(p_no_aft=1.0, p_no_he=1.0)
    assert "per sample" in ep.describe()


# ---------------------------------------------------------------------------
# observe() with per-sample effort
# ---------------------------------------------------------------------------

def test_the_default_effort_is_the_scalar_model_bit_for_bit():
    """Passing the dataclass fields as arrays must not change a single draw."""
    m = ObservationModel()
    truth = truth_for(500)
    a = m.observe(truth, np.random.default_rng(3))
    n = len(truth["aft_age"])
    b = m.observe(truth, np.random.default_rng(3), effort={
        "n_grains_counted": np.full(n, m.aft.n_grains_counted),
        "area_per_grain_um2": np.full(n, m.aft.area_per_grain_um2),
        "n_lengths": np.full(n, m.aft.n_lengths),
        "aft_present": np.ones(n, dtype=bool),
    })
    for name in Aliquot._ROW_FIELDS:
        np.testing.assert_array_equal(getattr(a, name), getattr(b, name), err_msg=name)


def test_per_sample_effort_is_obeyed_row_by_row():
    m = ObservationModel()
    n = 2_000
    truth = truth_for(n, g=6)
    rng = np.random.default_rng(0)
    effort = {
        "n_grains_counted": rng.integers(10, 41, n),
        "area_per_grain_um2": rng.uniform(1000, 6000, n),
        "n_lengths": rng.integers(0, 151, n),
        "n_he_grains": rng.integers(0, 7, n),
        "aft_present": rng.random(n) < 0.8,
    }
    obs = m.observe(truth, rng, effort=effort)
    present = effort["aft_present"]
    np.testing.assert_array_equal(obs.aft_mask, present)
    np.testing.assert_array_equal(obs.n_lengths, np.where(present, effort["n_lengths"], 0))
    np.testing.assert_array_equal(obs.n_grains, effort["n_he_grains"])
    # Absent AFT: every AFT field zero, and the mask says why.
    for name in ("aft_age", "aft_age_sd", "n_spontaneous", "dpar", "dpar_sd"):
        assert not getattr(obs, name)[~present].any(), name
    assert not obs.length_counts[~present].any()
    # The expected count scales with the counted area, row by row.
    area = effort["n_grains_counted"] * effort["area_per_grain_um2"] * 1e-8
    lam = obs.meta["expected_counts"]
    ratio = lam / area
    assert np.allclose(ratio, m.aft.expected_counts(truth["aft_age"], truth["U"]) / m.aft.area_cm2)
    # And the realised count tracks it: more area, more tracks.
    big = area > np.median(area)
    assert obs.n_spontaneous[present & big].mean() > obs.n_spontaneous[present & ~big].mean()


def test_effort_rejects_unknown_keys_and_wrong_shapes():
    m = ObservationModel()
    truth = truth_for(10)
    with pytest.raises(ValueError, match="unknown keys"):
        m.observe(truth, np.random.default_rng(0), effort={"n_dpar": np.full(10, 5)})
    with pytest.raises(ValueError, match="must be"):
        m.observe(truth, np.random.default_rng(0), effort={"n_lengths": np.full(7, 50)})
    with pytest.raises(ValueError, match=">= 1 counted grain"):
        m.observe(truth, np.random.default_rng(0), effort={"n_grains_counted": np.zeros(10)})


def test_log_prob_scores_nothing_from_the_aft_side_of_a_sample_without_one():
    m = ObservationModel()
    n = 50
    truth = truth_for(n, g=3)
    present = np.arange(n) % 2 == 0
    obs = m.observe(truth, np.random.default_rng(0), effort={"aft_present": present})
    pred = {"aft_age": torch.as_tensor(truth["aft_age"]),
            "length_pdf": torch.as_tensor(truth["length_pdf"]),
            "he_age": torch.as_tensor(truth["he_age"])}
    total, parts = m.log_prob(obs, pred, components=True)
    assert torch.all(parts["aft_age"][~present] == 0)
    assert torch.all(parts["lengths"][~present] == 0)
    assert torch.all(parts["aft_age"][present] != 0)
    assert torch.allclose(total[~present], parts["he"][~present])
    assert torch.isfinite(total).all()


def test_aliquot_carries_aft_mask_through_indexing_concat_and_files(tmp_path):
    m = ObservationModel()
    obs = m.observe(truth_for(20), np.random.default_rng(0),
                    effort={"aft_present": np.arange(20) < 15})
    assert obs.aft_mask.sum() == 15
    assert obs[np.arange(15, 20)].aft_mask.sum() == 0
    both = Aliquot.concat([obs, obs])
    assert both.aft_mask.sum() == 30
    obs.save(tmp_path / "a.npz")
    back = Aliquot.load(tmp_path / "a.npz")
    np.testing.assert_array_equal(back.aft_mask, obs.aft_mask)
    assert "with_aft" in obs.summary() and obs.summary()["with_aft"] == 0.75


# ---------------------------------------------------------------------------
# The split in RAM, and the features
# ---------------------------------------------------------------------------

def test_corpus_split_reads_its_shape_from_the_file(tiny_corpus, tiny_latent):
    c = CorpusSplit(tiny_corpus, "train", latent=tiny_latent)
    assert c.n_grid == 16 and c.g_max == 5 and c.t_max == 100.0
    assert c.T.shape == (len(c), 16) and c.he_age.shape == (len(c), 5)
    assert c.length_pdf.dtype == np.float16, "kept as stored; converted per batch"
    assert c.z.shape == (len(c), tiny_latent.dim)
    # z is whitened on the split it was fitted on.
    assert np.abs(c.z.mean(0)).max() < 0.05
    assert np.allclose(c.z.std(0), 1.0, atol=0.05)
    val = CorpusSplit(tiny_corpus, "val", max_rows=100)
    assert len(val) == 100 and val.z is None
    with pytest.raises(ValueError, match="not in"):
        CorpusSplit(tiny_corpus, "holdout")


def test_features_are_finite_and_flagged_under_every_dropout_pattern(tiny_corpus):
    c = CorpusSplit(tiny_corpus, "train", max_rows=3_000)
    ep = EffortPrior(p_no_aft=0.4, p_no_he=0.4, p_no_lengths=0.4)
    rng = np.random.default_rng(0)
    obs = ObservationModel().observe(c.truth(np.arange(len(c))), rng, effort=ep.draw(len(c), rng, c.g_max))
    f = features_from_aliquot(obs)
    assert f["aft"].shape == (len(c), len(AFT_INPUTS))
    assert f["grains"].shape == (len(c), c.g_max, len(GRAIN_INPUTS))
    for k in ("aft", "length_pdf", "grains"):
        assert np.isfinite(f[k]).all(), k
    has_aft = f["aft"][:, AFT_INPUTS.index("has_aft")] > 0.5
    assert (~has_aft).any() and has_aft.any()
    assert not f["aft"][~has_aft].any(), "no AFT: the whole row is zero, flags included"
    assert not f["length_pdf"][~has_aft].any()
    no_len = f["aft"][:, AFT_INPUTS.index("has_lengths")] < 0.5
    assert not f["length_pdf"][no_len].any()
    assert np.allclose(f["length_pdf"][~no_len].sum(1), 1.0, atol=1e-5)
    assert not f["grains"][~f["grain_mask"]].any(), "masked grains are zero"
    assert all(AFT_INPUTS[i].startswith("has_") for i in AFT_FLAGS)
    # The modality classes partition the batch.
    mod = modality_of(obs.meta["effort"])
    assert set(np.unique(mod)) == set(range(len(MODALITIES)))


def test_the_loader_redraws_noise_and_fixed_batches_do_not(tiny_corpus, tiny_latent):
    c = CorpusSplit(tiny_corpus, "train", latent=tiny_latent, max_rows=2_048)
    ds = NoisyBatches(c, ObservationModel(), EffortPrior(), seed=0)
    loader = make_loader(ds, 512, shuffle=False, num_workers=0, drop_last=False)
    e1 = list(loader)
    e2 = list(loader)
    assert len(e1) == 4 and e1[0]["aft"].shape == (512, len(AFT_INPUTS))
    assert e1[0]["z"].shape == (512, tiny_latent.dim) and e1[0]["T"].shape == (512, 16)
    torch.testing.assert_close(e1[0]["index"], e2[0]["index"])
    assert not torch.equal(e1[0]["aft"], e2[0]["aft"]), "same rows, fresh noise"
    a = fixed_batches(ds, 512, seed=7, max_rows=1_000)
    b = fixed_batches(ds, 512, seed=7, max_rows=1_000)
    assert len(a) == 2 and a[1]["aft"].shape[0] == 488
    for k in ("aft", "grains", "length_pdf"):
        torch.testing.assert_close(a[0][k], b[0][k])


def test_worker_processes_draw_different_noise(tiny_corpus):
    c = CorpusSplit(tiny_corpus, "train", max_rows=1_024)
    ds = NoisyBatches(c, ObservationModel(), EffortPrior(), seed=0)
    loader = make_loader(ds, 256, shuffle=False, num_workers=2, drop_last=False, persistent=False)
    batches = list(loader)
    assert len(batches) == 4
    # Consecutive batches come from alternating workers; both must have drawn.
    assert not torch.equal(batches[0]["aft"][:, 0], batches[1]["aft"][:, 0])


def test_an_unconditional_dataset_observes_nothing_and_keeps_the_truth(tiny_corpus, tiny_latent):
    """Stage 13's prior density: every sample empty, the dequantised ``z`` exactly as a conditional run gets it."""
    from chai.models.gaussian import AmortizedPosterior
    from chai.train.data import WallDequantiser, empty_features

    c = CorpusSplit(tiny_corpus, "train", latent=tiny_latent, max_rows=1_024)
    deq = WallDequantiser(width=10.0, jitter=1.0, noise_sd=0.5)
    cond = NoisyBatches(c, ObservationModel(), EffortPrior(), seed=3, dequantise=deq)
    empty = NoisyBatches(c, ObservationModel(), EffortPrior(), seed=3, dequantise=deq, inputs=False)
    rows = np.arange(256)
    b = empty.observe(rows, np.random.default_rng(0))
    ref = empty_features(256, c.length_bins.size, c.g_max)
    for k, v in ref.items():
        assert b[k].shape == v.shape and not b[k].any(), k
    assert "effort" not in b
    torch.testing.assert_close(b["T"], torch.from_numpy(c.T[rows]))
    assert b["z"].shape == (256, tiny_latent.dim) and torch.isfinite(b["z"]).all()
    # Same rows, same dequantisation law: the targets match the conditional run's in distribution.
    zc = cond.observe(rows, np.random.default_rng(0))["z"]
    assert abs(float(b["z"].std() - zc.std())) < 0.1
    # A head fed only empty samples sees one input: its encoder emits one embedding.
    net = AmortizedPosterior(tiny_latent, n_aft=len(AFT_INPUTS), n_bins=c.length_bins.size,
                             n_grain=len(GRAIN_INPUTS), aft_flags=AFT_FLAGS)
    e = net.encoder(b)
    torch.testing.assert_close(e, e[:1].expand_as(e))


def test_an_episodic_corpus_reads_but_does_not_whiten(tiny_corpus, tmp_path, tiny_latent):
    import shutil

    import h5py
    path = tmp_path / "epi.h5"
    shutil.copy(tiny_corpus, path)
    with h5py.File(path, "r+") as f:
        del f["hyper/tau"]
    c = CorpusSplit(path, "val", max_rows=10)
    assert len(c) == 10 and "tau" not in c.hyper
    with pytest.raises(ValueError, match="cannot be whitened"):
        CorpusSplit(path, "val", latent=tiny_latent, max_rows=10)


def test_the_unconditional_switch_round_trips():
    from chai.train.loop import TrainConfig
    cfg = TrainConfig.from_config({"unconditional": True, "head": {"kind": "flow"}})
    assert cfg.unconditional and TrainConfig.from_config(cfg.to_config()).unconditional
    assert not TrainConfig().unconditional


def test_a_share_of_rows_can_be_shown_empty_with_their_truth_kept(tiny_corpus, tiny_latent):
    c = CorpusSplit(tiny_corpus, "train", latent=tiny_latent, max_rows=2_048)
    ds = NoisyBatches(c, ObservationModel(), EffortPrior(), seed=0, p_empty=0.25)
    b = ds.observe(np.arange(2_048), np.random.default_rng(1))
    empty = ~b["grain_mask"].any(1) & (b["aft"].abs().sum(1) == 0) & (b["length_pdf"].sum(1) == 0)
    assert 0.2 < float(empty.float().mean()) < 0.3
    assert torch.isfinite(b["z"]).all() and b["z"].shape[0] == 2_048
    with pytest.raises(ValueError, match="p_empty"):
        NoisyBatches(c, ObservationModel(), EffortPrior(), p_empty=1.0)


def test_a_second_prior_mixes_in_with_its_missing_hyperparameters_drawn(tiny_corpus, tmp_path, tiny_latent):
    import shutil

    import h5py

    from chai.train.data import mix_splits
    path = tmp_path / "epi.h5"
    shutil.copy(tiny_corpus, path)
    with h5py.File(path, "r+") as f:
        del f["hyper/tau"]
    base = CorpusSplit(tiny_corpus, "train", latent=tiny_latent, max_rows=1_000)
    extra = CorpusSplit(path, "train", max_rows=300)
    mixed = mix_splits(base, extra, tiny_latent, np.random.default_rng(0))
    assert len(mixed) == 1_300 and (mixed.source == 1).sum() == 300
    np.testing.assert_array_equal(mixed.T[1_000:], extra.T)
    np.testing.assert_array_equal(mixed.hyper["rate_sd"][1_000:], extra.hyper["rate_sd"])
    spec = next(s for s in tiny_latent.hyper if s.name == "tau")
    tau = mixed.hyper["tau"][1_000:]
    assert (tau >= spec.lo).all() and (tau <= spec.hi).all() and tau.std() > 0
    assert mixed.z.shape == (1_300, tiny_latent.dim) and np.isfinite(mixed.z).all()
    np.testing.assert_allclose(mixed.z[:1_000], base.z, atol=1e-5)
    assert len(base) == 1_000, "the base split is not modified"
    ds = NoisyBatches(mixed, ObservationModel(), EffortPrior(), seed=0)
    b = ds.observe(np.arange(990, 1_010), np.random.default_rng(0))
    assert torch.isfinite(b["z"]).all() and (b["index"][10:] >= 2 ** 40).all()
