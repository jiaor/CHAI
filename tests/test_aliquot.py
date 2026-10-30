"""The sample container: what it derives, and the invariance it exists to protect."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from chai.data.aliquot import AFT_FEATURES, GRAIN_FEATURES, Aliquot

BINS = (np.arange(8) + 0.5) * 2.0


def make(n=5, g=3, seed=0) -> Aliquot:
    rng = np.random.default_rng(seed)
    counts = rng.multinomial(100, np.full(BINS.size, 1 / BINS.size), size=n)
    mask = np.ones((n, g), dtype=bool)
    return Aliquot(
        aft_age=rng.uniform(10, 100, n), aft_age_sd=rng.uniform(1, 5, n),
        n_spontaneous=rng.integers(20, 500, n), length_counts=counts, length_bins=BINS,
        dpar=rng.uniform(1.5, 2.5, n), dpar_sd=np.full(n, 0.05),
        he_age=rng.uniform(5, 80, (n, g)), he_age_sd=rng.uniform(0.5, 5, (n, g)),
        he_radius=rng.uniform(40, 90, (n, g)), he_eU=rng.uniform(5, 100, (n, g)),
        he_mask=mask,
    )


def test_mismatched_rows_are_refused():
    a = make()
    with pytest.raises(ValueError, match="rows"):
        Aliquot(**{**vars(a), "dpar": a.dpar[:-1]})


def test_a_mask_that_is_not_boolean_is_refused():
    """Integers would work everywhere until someone multiplied by 2."""
    a = make()
    with pytest.raises(ValueError, match="boolean"):
        Aliquot(**{**vars(a), "he_mask": a.he_mask.astype(int)})


def test_derived_length_summaries_agree_with_the_counts():
    a = make()
    assert np.all(a.n_lengths == 100)
    assert a.mean_length == pytest.approx(a.length_counts @ BINS / 100)
    assert a.length_pdf.sum(axis=1) == pytest.approx(np.ones(len(a)))


def test_a_sample_with_no_measured_lengths_reports_zero_not_nan():
    """Zero is only ever paired with n_lengths == 0, which is the column to read."""
    a = make()
    b = Aliquot(**{**vars(a), "length_counts": np.zeros_like(a.length_counts)})
    assert np.all(b.mean_length == 0.0)
    assert np.all(np.isfinite(b.length_pdf))


def test_indexing_one_row_gives_an_aliquot_of_one():
    a = make()
    one = a[2]
    assert len(one) == 1 and one.aft_age[0] == a.aft_age[2]
    assert one.he_age.shape == (1, a.he_age.shape[1])


def test_permuting_grains_preserves_each_row_as_a_set():
    a = make(n=6, g=4)
    b = a.permute_grains(np.random.default_rng(3))
    for i in range(len(a)):
        assert sorted(a.he_age[i]) == pytest.approx(sorted(b.he_age[i]))
    # and it must be a real permutation, not the identity
    assert not np.allclose(a.he_age, b.he_age)
    # the radius must travel with its own age, or the grains have been scrambled
    order = [list(a.he_age[0]).index(v) for v in b.he_age[0]]
    assert b.he_radius[0] == pytest.approx(a.he_radius[0][order])


def test_a_symmetric_pooling_of_the_features_is_permutation_invariant():
    """The property every Deep Sets consumer relies on, asserted directly."""
    a = make(n=4, g=5)
    b = a.permute_grains(np.random.default_rng(11))
    assert a.grain_features().sum(axis=1) == pytest.approx(b.grain_features().sum(axis=1))
    assert a.grain_features().max(axis=1) == pytest.approx(b.grain_features().max(axis=1))


def test_masked_grains_are_zero_in_the_features():
    a = make(n=3, g=4)
    mask = a.he_mask.copy()
    mask[:, 2:] = False
    b = Aliquot(**{**vars(a), "he_mask": mask})
    f = b.grain_features()
    assert np.all(f[:, 2:, :] == 0.0)
    assert np.all(b.n_grains == 2)


def test_concat_pads_the_grain_axis_and_keeps_the_mask_honest():
    a, b = make(n=3, g=2, seed=1), make(n=4, g=5, seed=2)
    c = Aliquot.concat([a, b])
    assert len(c) == 7 and c.he_age.shape[1] == 5
    assert np.all(c.n_grains[:3] == 2) and np.all(c.n_grains[3:] == 5)
    assert np.all(c.he_age[:3, 2:] == 0.0)


def test_round_trip_through_npz(tmp_path):
    a = make(n=4, g=3)
    a.save(tmp_path / "a.npz")
    b = Aliquot.load(tmp_path / "a.npz")
    for name in Aliquot._ROW_FIELDS:
        assert np.array_equal(getattr(a, name), getattr(b, name))


def test_the_torch_view_has_the_documented_columns():
    a = make(n=4, g=3)
    t = a.to_torch()
    assert t["aft"].shape == (4, len(AFT_FEATURES))
    assert t["grains"].shape == (4, 3, len(GRAIN_FEATURES))
    assert t["grain_mask"].dtype == torch.bool
    assert t["aft"][:, 0].numpy() == pytest.approx(a.aft_age, rel=1e-6)
    assert t["grains"][:, :, 3].numpy() == pytest.approx(a.he_eU, rel=1e-6)
