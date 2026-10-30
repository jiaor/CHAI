"""Shared fixtures: a tiny corpus file with the real layout and made-up physics."""
from __future__ import annotations

import json
import re

import h5py
import numpy as np
import pytest

from chai.data.priors import GrainPrior, ThermalHistoryPrior
from chai.models.latent import Latent


def _skip_if_a_reference_library_is_missing(outcome):
    """The reference sources (``KETCH.c``, ``RDAAM.cpp``, ``Mad_He.f90``) are third-party code and are
    left out of the sdist and the public tree (docs/release.md §12), so a tree without ``native/``
    has no ``lib*.so``.  The loader raises ``FileNotFoundError`` naming the library on first use; a
    test (or a fixture) that hits one then skips instead of failing."""
    exc = outcome.excinfo
    if exc is not None and exc[0] is FileNotFoundError:
        m = re.search(r"lib(ketcham|rdaam|madhe)\.so", str(exc[1]))
        if m:
            outcome.force_exception(pytest.skip.Exception(
                f"{m.group(0)} is not built here (the reference implementations are not redistributed; "
                "see docs/release.md §12)"))


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_setup(item):
    _skip_if_a_reference_library_is_missing((yield))


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_call(item):
    _skip_if_a_reference_library_is_missing((yield))


TINY_PRIOR = ThermalHistoryPrior(n_steps=256, n_features=64)
TINY_N_GRID = 16
TINY_G = 5
TINY_ROWS = 12_000
TINY_BINS = (np.arange(200) + 0.5) * 0.1


def _bell(centre: np.ndarray, width: float = 1.4) -> np.ndarray:
    p = np.exp(-0.5 * ((TINY_BINS[None, :] - centre[:, None]) / width) ** 2)
    return p / p.sum(axis=1, keepdims=True)


def _write_tiny(path, prior: ThermalHistoryPrior, t_grid: np.ndarray, seed: int, spacing: str = "uniform") -> str:
    """A corpus on ``t_grid`` with 5 grains, observables that are smooth
    functions of the path so that a head has something to learn, and every
    attribute :class:`chai.train.data.CorpusSplit` reads.  Not physics."""
    rng = np.random.default_rng(seed)
    T, batch = prior.sample_on(TINY_ROWS, rng, t_grid)
    grains = GrainPrior().sample(TINY_ROWS * TINY_G, rng)
    U = grains["U"].reshape(TINY_ROWS, TINY_G)
    eU = U + 0.235 * grains["Th"].reshape(TINY_ROWS, TINY_G)
    radius = grains["radius"].reshape(TINY_ROWS, TINY_G)
    dpar = grains["dpar"].reshape(TINY_ROWS, TINY_G)[:, 0]

    recent = T[:, t_grid <= 50.0].mean(axis=1)
    aft_age = np.clip(prior.t_max * (1.0 - recent / 200.0) ** 2 + 1.0, 1.0, None)
    he_age = aft_age[:, None] * 0.5 * (eU / 50.0) ** 0.15 * (radius / 60.0) ** 0.1
    length_pdf = _bell(16.0 - 3.0 * recent / 200.0)

    split = np.zeros(TINY_ROWS, dtype=np.uint8)
    split[int(0.8 * TINY_ROWS):int(0.9 * TINY_ROWS)] = 1
    split[int(0.9 * TINY_ROWS):] = 2
    with h5py.File(path, "w") as f:
        f["T"] = T.astype(np.float32)
        for k, v in batch.hyper.items():
            f[f"hyper/{k}"] = v.astype(np.float32)
        f["aft/age"] = aft_age.astype(np.float32)
        f["aft/dpar"] = dpar.astype(np.float32)
        f["aft/length_pdf"] = length_pdf.astype(np.float16)
        f["he/corrected_age"] = he_age.astype(np.float32)
        f["he/radius"] = radius.astype(np.float32)
        f["he/eU"] = eU.astype(np.float32)
        f["he/U"] = U.astype(np.float32)
        f["split"] = split
        f["t_grid"] = t_grid
        f["length_bins"] = TINY_BINS
        f.attrs["split_names"] = ["train", "val", "test"]
        f.attrs["prior_config"] = json.dumps(prior.to_config())
        f.attrs["spec"] = json.dumps({"n_histories": TINY_ROWS, "n_grains": TINY_G,
                                      "n_grid": int(t_grid.size), "t_max": prior.t_max,
                                      "spacing": spacing})
        f.attrs["n_grains_total"] = TINY_G
        f.attrs["git_sha"] = "test"
    return str(path)


@pytest.fixture(scope="session")
def tiny_corpus(tmp_path_factory) -> str:
    """The 100 Ma tiny corpus on a uniform 16-node grid."""
    t_grid = np.linspace(0.0, TINY_PRIOR.t_max, TINY_N_GRID)
    return _write_tiny(tmp_path_factory.mktemp("corpus") / "tiny.h5", TINY_PRIOR, t_grid, seed=0)


#: The tiny family's second member: 250 Ma on a piecewise grid, the 100 Ma
#: member's 16 nodes then 8 more, with the net cooling kept 0-200 K over the
#: horizon as Stage 16 decided.
TINY_PRIOR_LONG = ThermalHistoryPrior(t_max=250.0, n_steps=640, n_features=64)
TINY_SPACING_LONG = "piecewise:16@100,8@250"


@pytest.fixture(scope="session")
def tiny_corpus_long(tmp_path_factory) -> str:
    """The 250 Ma tiny corpus, 24 nodes, same grains and observable map."""
    from chai.paths import fixed_grid

    t_grid = fixed_grid(24, 250.0, TINY_SPACING_LONG).numpy()
    return _write_tiny(tmp_path_factory.mktemp("corpus") / "tiny_long.h5", TINY_PRIOR_LONG, t_grid,
                       seed=1, spacing=TINY_SPACING_LONG)


@pytest.fixture(scope="session")
def tiny_latent(tiny_corpus) -> Latent:
    return Latent.fit(tiny_corpus)

