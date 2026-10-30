"""The corpus: what it stores, and the guarantees it makes about how."""
from __future__ import annotations

import json
import shutil
from dataclasses import replace
from pathlib import Path

import h5py
import numpy as np
import pytest

from chai.data.priors import GrainPrior, ThermalHistoryPrior
from chai.data.simulate import CorpusSpec, provenance, reproduce, simulate

#: Small enough to run the oracles in a test, large enough to have >1 block.
SPEC = CorpusSpec(n_histories=60, n_grains=2, block_size=20, seed=5)
#: The prior is the dominant cost of a corpus, so tests use a cheap one.
PRIOR = ThermalHistoryPrior(n_steps=128, n_features=32)


@pytest.fixture(scope="module")
def corpus(tmp_path_factory):
    path = tmp_path_factory.mktemp("corpus") / "c.h5"
    simulate(path, SPEC, PRIOR, GrainPrior(), workers=2, progress=False)
    return path


def test_the_corpus_has_every_column_a_stage_5_sample_needs(corpus):
    with h5py.File(corpus) as f:
        n, k = SPEC.n_histories, SPEC.n_grains
        assert f["T"].shape == (n, SPEC.n_grid)
        assert f["aft/length_pdf"].shape == (n, 200)
        assert f["aft/length_pdf"].dtype == np.float16
        for name in ("age", "oldest_age", "mean_length", "dpar"):
            assert f[f"aft/{name}"].shape == (n,)
        for name in ("age", "corrected_age", "radius", "U", "Th", "eU"):
            assert f[f"he/{name}"].shape == (n, k), name
        for name in ("tau", "rate_sd", "drift", "T_0"):
            assert f[f"hyper/{name}"].shape == (n,)
        assert f["t_grid"].shape == (SPEC.n_grid,)
        assert f["valid"][:].all()


def test_ages_are_physical_and_length_pdfs_are_normalised(corpus):
    with h5py.File(corpus) as f:
        assert (f["aft/age"][:] > 0).all()
        assert (f["aft/age"][:] <= 1.1 * SPEC.t_max).all()
        assert (f["he/corrected_age"][:] >= f["he/age"][:]).all()  # Ft correction ages up
        assert f["aft/length_pdf"][:].sum(axis=1) == pytest.approx(1.0, abs=2e-3)
        assert f["he/eU"][:] == pytest.approx(f["he/U"][:] + 0.235 * f["he/Th"][:], rel=1e-5)


def test_every_history_is_distinct(corpus):
    """Regression: the block seeds were spawned but only their *entropy* was sent
    to the workers.  Spawned SeedSequences share their parent's entropy and differ
    only in ``spawn_key``, so every block drew the same histories -- a corpus of
    one block repeated, which nothing downstream would have noticed."""
    with h5py.File(corpus) as f:
        T = f["T"][:]
    assert len(np.unique(T.round(5), axis=0)) == SPEC.n_histories


def test_a_sample_can_be_redrawn_from_the_file_alone(corpus):
    with h5py.File(corpus) as f:
        stored = {"T": f["T"][31], **{k: f[f"hyper/{k}"][31] for k in
                                      ("tau", "rate_sd", "drift", "T_0")}}
        grain = {k: f[f"he/{k}"][31] for k in ("radius", "U", "Th")}
    got = reproduce(corpus, 31)
    assert got["T"] == pytest.approx(stored["T"], abs=1e-5)
    for k in ("tau", "rate_sd", "drift", "T_0"):
        assert got[f"hyper_{k}"] == pytest.approx(stored[k], rel=1e-5)
    for k, v in grain.items():
        assert got[f"grain_{k}"] == pytest.approx(v, rel=1e-5)


def test_resuming_reproduces_an_uninterrupted_run(corpus, tmp_path):
    """Half-finish a corpus, then finish it, and get the original back.

    Simulated by reopening a complete file, clearing two blocks and scribbling
    over their rows -- which is what an interrupted run leaves behind, minus the
    scribbles.
    """
    partial = tmp_path / "partial.h5"
    shutil.copy(corpus, partial)
    with h5py.File(partial, "r+") as f:
        for b in (0, 2):
            sl = SPEC.block_slice(b)
            f["block_done"][b] = False
            f["T"][sl] = -999.0
            f["aft/age"][sl] = -999.0
            f["he/corrected_age"][sl] = -999.0

    simulate(partial, SPEC, PRIOR, GrainPrior(), workers=2, progress=False)

    with h5py.File(corpus) as a, h5py.File(partial) as b:
        for name in ("T", "aft/age", "aft/length_pdf", "he/corrected_age", "hyper/tau",
                     "diag/max_cooling_rate", "split", "valid"):
            assert np.array_equal(a[name][:], b[name][:]), name


def test_a_finished_corpus_is_not_recomputed(corpus, tmp_path):
    done = tmp_path / "done.h5"
    shutil.copy(corpus, done)
    with h5py.File(done, "r+") as f:
        f["T"][0] = 12345.0  # would be overwritten if the block were rerun
    simulate(done, SPEC, PRIOR, GrainPrior(), workers=2, progress=False)
    with h5py.File(done) as f:
        assert f["T"][0][0] == 12345.0


def test_splits_are_shuffled_disjoint_and_fixed_by_the_seed(corpus):
    """PLAN.md Sec 6.2: the prototype's split was contiguous over a corpus built
    in blocks of increasing path length, so path lengths 8-9 appeared only in
    validation.  A shuffled split written at creation cannot do that."""
    with h5py.File(corpus) as f:
        split, block = f["split"][:], f["block"][:]
    assert np.bincount(split, minlength=3).tolist() == [48, 6, 6]
    # every block contributes to more than one split: the split is not blockwise
    assert any(len(set(split[block == b])) > 1 for b in range(SPEC.n_blocks))
    # and it is a function of the seed, not of when the file was written
    with h5py.File(corpus) as f:
        again = CorpusSpec(**{k: tuple(v) if k == "split_fractions" else v
                              for k, v in json.loads(f.attrs["spec"]).items()})
    assert again == SPEC


def test_the_file_says_where_it_came_from(corpus):
    with h5py.File(corpus) as f:
        assert json.loads(f.attrs["prior_config"])["rate_max"] == PRIOR.rate_max
        assert json.loads(f.attrs["grain_config"])["U_range"] == list(GrainPrior().U_range)
        assert f.attrs["aft_model"] == "KETCHAM_2007"
        assert len(f.attrs["git_sha"]) >= 7
        assert set(json.loads(f.attrs["oracles"])) >= {"libketcham.so", "librdaam.so"}


def test_a_corpus_refuses_to_be_extended_by_a_different_spec(corpus):
    with pytest.raises(ValueError, match="different CorpusSpec"):
        simulate(corpus, replace(SPEC, n_histories=80), PRIOR, GrainPrior(), progress=False)


def test_the_time_span_must_match_the_prior(tmp_path):
    with pytest.raises(ValueError, match="time span"):
        simulate(tmp_path / "x.h5", replace(SPEC, t_max=50.0), PRIOR, progress=False)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"n_histories": 0}, "n_histories must be >= 1"),
        ({"n_grains": 0}, "n_grains must be >= 1"),
        ({"block_size": 0}, "block_size must be >= 1"),
        ({"split_fractions": (0.5, 0.2, 0.2)}, "must sum to 1"),
        ({"split_fractions": (1.2, -0.1, -0.1)}, "non-negative"),
    ],
)
def test_bad_specs_are_rejected(kwargs, match):
    with pytest.raises(ValueError, match=match):
        replace(SPEC, **kwargs)


def test_provenance_reports_the_built_oracles():
    if not (Path(__file__).resolve().parents[1] / "native").exists():
        pytest.skip("the reference sources are not in this tree (docs/release.md §12)")
    p = provenance()
    # Mad_He is built only where its source is (a cross-check, not in the sdist or the public tree).
    expected = {"libketcham.so", "librdaam.so"}
    if (Path(__file__).resolve().parents[1] / "native" / "madhe" / "Mad_He.f90").exists():
        expected.add("libmadhe.so")
    assert set(p["oracles"]) >= expected
    assert all(len(v) == 64 for v in p["oracles"].values())


def test_a_corpus_can_be_drawn_from_any_prior_that_states_its_latents(tmp_path):
    """The writer asks the prior for its hyperparameter names, and stores those.

    The episodic prior reports an episode count where the shipped one reports a
    correlation length, so a writer that hard-codes ``tau`` cannot build the
    mixed corpus ``docs/prior-sensitivity.md`` finding 8 argues for.  Everything
    else about the file is unchanged, which is what makes the two corpora
    comparable.
    """
    from chai.data.priors import EpisodicPrior

    prior = EpisodicPrior(n_steps=128)
    path = tmp_path / "episodic.h5"
    simulate(path, replace(SPEC, n_histories=20, block_size=10), prior, GrainPrior(),
             workers=2, progress=False)

    with h5py.File(path) as f:
        assert list(f.attrs["hyper_names"]) == list(prior.hyper_names)
        for name in prior.hyper_names:
            assert f[f"hyper/{name}"].shape == (20,)
        assert "hyper/tau" not in f
        assert f["valid"][:].all()
        assert json.loads(f.attrs["prior_config"])["episodes_per_horizon"] == 2.0
        # Episodes are counts, and they survive the float32 column.
        k = f["hyper/n_episodes"][:]
        assert np.all(k == np.round(k)) and k.max() >= 1
