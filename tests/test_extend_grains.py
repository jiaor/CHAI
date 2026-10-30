"""Appending grains to a finished corpus: what must move, and what must not."""
from __future__ import annotations

import json
import shutil

import h5py
import numpy as np
import pytest

from chai.data.priors import GrainPrior, ThermalHistoryPrior
from chai.data.simulate import (
    CorpusSpec,
    extend_grains,
    reproduce_extra_grains,
    simulate,
)

SPEC = CorpusSpec(n_histories=40, n_grains=2, block_size=10, seed=5)
PRIOR = ThermalHistoryPrior(n_steps=128, n_features=32)
N_NEW = 3

#: Every column that an append has no business touching.
UNTOUCHED = ("T", "split", "block", "aft/age", "aft/oldest_age", "aft/mean_length",
             "aft/dpar", "aft/length_pdf", "hyper/tau", "hyper/rate_sd",
             "hyper/drift", "hyper/T_0", "diag/total_variation", "t_grid")


@pytest.fixture(scope="module")
def base(tmp_path_factory):
    path = tmp_path_factory.mktemp("base") / "c.h5"
    simulate(path, SPEC, PRIOR, GrainPrior(), workers=2, progress=False)
    return path


@pytest.fixture(scope="module")
def extended(base, tmp_path_factory):
    path = tmp_path_factory.mktemp("ext") / "c12.h5"
    extend_grains(base, path, N_NEW, workers=2, progress=False)
    return path


def test_the_grain_axis_widens_and_nothing_else_moves(base, extended):
    with h5py.File(base) as a, h5py.File(extended) as b:
        for name in UNTOUCHED:
            np.testing.assert_array_equal(a[name][:], b[name][:], err_msg=name)
        for name in ("age", "corrected_age", "radius", "U", "Th", "eU"):
            assert b[f"he/{name}"].shape == (SPEC.n_histories, SPEC.n_grains + N_NEW)
            # The originals keep their columns *and* their positions.
            np.testing.assert_array_equal(a[f"he/{name}"][:],
                                          b[f"he/{name}"][:, :SPEC.n_grains])
        assert b.attrs["n_grains_total"] == SPEC.n_grains + N_NEW


def test_the_source_is_not_modified(base, extended):
    with h5py.File(base) as a:
        assert json.loads(a.attrs["spec"])["n_grains"] == SPEC.n_grains
        assert a["he/age"].shape == (SPEC.n_histories, SPEC.n_grains)
        assert "extension" not in a.attrs


def test_appended_ages_are_physical(extended):
    with h5py.File(extended) as f:
        new = f["he/age"][:, SPEC.n_grains:]
        assert np.isfinite(new).all() and (new > 0).all()
        r = f["he/radius"][:, SPEC.n_grains:]
        assert ((r >= 30.0) & (r <= 110.0)).all()


def test_the_extension_draws_its_own_grains_rather_than_copying(extended):
    """A lineage collision would show up as appended grains equal to the originals."""
    with h5py.File(extended) as f:
        old, new = f["he/radius"][:, :SPEC.n_grains], f["he/radius"][:, SPEC.n_grains:]
        for i in range(SPEC.n_grains):
            for j in range(N_NEW):
                assert not np.allclose(old[:, i], new[:, j])


def test_appended_grains_can_be_redrawn_from_the_file_alone(extended):
    with h5py.File(extended) as f:
        for index in (0, 7, SPEC.n_histories - 1):
            got = reproduce_extra_grains(extended, index)
            for name in ("radius", "U", "Th"):
                np.testing.assert_allclose(
                    got[name], f[f"he/{name}"][index, SPEC.n_grains:], rtol=1e-6)


def test_validity_is_the_conjunction_and_the_base_flag_survives(base, extended):
    with h5py.File(base) as a, h5py.File(extended) as b:
        np.testing.assert_array_equal(a["valid"][:], b["valid_base"][:])
        new = b["he/age"][:, SPEC.n_grains:]
        expect = b["valid_base"][:] & np.isfinite(new).all(1) & (new > 0).all(1)
        np.testing.assert_array_equal(b["valid"][:], expect)


def test_resuming_reproduces_an_uninterrupted_run(base, extended, tmp_path):
    """Half-finish an extension, then finish it, and get the original back.

    Same idiom as the base corpus's resume test: reopen a complete file, clear
    two grain blocks and scribble over their appended columns, which is what an
    interrupted run leaves behind minus the scribbles.
    """
    partial = tmp_path / "partial.h5"
    shutil.copy(extended, partial)
    k0 = SPEC.n_grains
    with h5py.File(partial, "r+") as f:
        for b in (0, 2):
            sl = SPEC.block_slice(b)
            f["grain_block_done"][b] = False
            f["he/age"][sl, k0:] = -999.0
            f["he/radius"][sl, k0:] = -999.0
            f["valid"][sl] = False

    extend_grains(base, partial, N_NEW, workers=2, progress=False)

    with h5py.File(partial) as a, h5py.File(extended) as b:
        for name in ("age", "corrected_age", "radius", "U", "Th", "eU"):
            np.testing.assert_array_equal(a[f"he/{name}"][:], b[f"he/{name}"][:], name)
        np.testing.assert_array_equal(a["valid"][:], b["valid"][:])


def test_a_finished_extension_is_not_recomputed(base, extended, tmp_path):
    done = tmp_path / "done.h5"
    shutil.copy(extended, done)
    with h5py.File(done, "r+") as f:
        f["he/age"][0, SPEC.n_grains] = 12345.0  # overwritten if the block reran
    extend_grains(base, done, N_NEW, workers=2, progress=False)
    with h5py.File(done) as f:
        assert f["he/age"][0, SPEC.n_grains] == 12345.0


def test_an_extension_refuses_to_change_its_own_shape(base, extended):
    with pytest.raises(ValueError, match="n_new"):
        extend_grains(base, extended, N_NEW + 1, workers=1, progress=False)


def test_an_unfinished_corpus_cannot_be_extended(tmp_path):
    path = tmp_path / "unfinished.h5"
    simulate(path, SPEC, PRIOR, GrainPrior(), workers=1, progress=False)
    with h5py.File(path, "r+") as f:
        f["block_done"][0] = False
    with pytest.raises(ValueError, match="unfinished"):
        extend_grains(path, tmp_path / "out.h5", N_NEW, workers=1, progress=False)
