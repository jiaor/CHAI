"""What an installed ``chai`` carries, and what it can do without the checkout.

``make wheel-check`` is the end-to-end test (a built wheel in a fresh venv); these
are the parts of it cheap enough to run with the suite.
"""
from __future__ import annotations

import json
import subprocess
import sys
import tomllib
from pathlib import Path

import numpy as np
import pytest
import torch

import chai
from chai import api

REPO = Path(__file__).resolve().parents[1]
META = tomllib.loads((REPO / "pyproject.toml").read_text())
SHIPPED_RUN = REPO / "results" / "flow_v22_100"        # its 200 000-step weights ship
SHIPPED_FILE = SHIPPED_RUN / "model_200k.pt"


def test_version_matches_pyproject():
    assert chai.__version__ == META["project"]["version"]


def test_package_data_declares_the_assets():
    """The wheel carries the shipped family and the three configs it was trained under, and nothing
    else (PyPI's file limit is 100 MB)."""
    globs = META["tool"]["setuptools"]["package-data"]["chai"]
    shipped = {p.name for p in api.ASSETS.iterdir()}
    covered = {p.name for g in globs for p in (REPO / "src" / "chai").glob(g)}
    assert shipped == covered == {"flow_v22_100.pt", "flow_v22_250.pt", "chooser_v22.pt",
                                  "observation_v2.yaml", "prior_v22.yaml", "prior_v22_250.yaml"}
    assert set(api.FAMILY_ASSETS) == {api.DEFAULT_PRIOR} == {"v2.2"}
    assert sum(p.stat().st_size for p in api.ASSETS.iterdir()) < 100_000_000   # PyPI's per-file limit


def test_the_pypi_readme_and_sdist_leave_the_checkout_behind():
    """The readme PyPI shows has no link into the checkout, and the sdist carries none of the
    reference sources (third-party code under their authors' terms; RJ, 2026-10-29)."""
    readme = (REPO / META["project"]["readme"]).read_text()
    assert "](docs/" not in readme and "](PLAN.md" not in readme and "](notebooks/" not in readme
    manifest = (REPO / "MANIFEST.in").read_text()
    assert "prune native" in manifest and "graft native" not in manifest
    assert "graft configs" not in manifest            # the sdist carries the shipped family's configs only
    for name in ("prior_v22.yaml", "prior_v22_250.yaml", "observation_v2.yaml",
                 "train_flow_v22_100.yaml", "train_flow_v22_250.yaml", "chooser_v22.yaml"):
        assert f"configs/{name}" in manifest


def test_a_built_sdist_carries_the_shipped_configs_alone():
    """The sdist `make wheel` built (skipped where none is): six configs and nothing under `native/`.
    Guards against setuptools seeding the manifest from a stale egg-info/SOURCES.txt."""
    import tarfile
    sdist = REPO / "dist" / f"chai_thermo-{chai.__version__}.tar.gz"
    if not sdist.exists():
        pytest.skip("no sdist built")
    with tarfile.open(sdist) as tf:
        names = [n.split("/", 1)[1] for n in tf.getnames() if "/" in n]
    configs = sorted(n for n in names if n.startswith("configs/") and n.endswith(".yaml"))
    assert configs == ["configs/chooser_v22.yaml", "configs/observation_v2.yaml", "configs/prior_v22.yaml",
                       "configs/prior_v22_250.yaml", "configs/train_flow_v22_100.yaml", "configs/train_flow_v22_250.yaml"]
    assert not any(n.startswith("native/") for n in names)
    assert META["project"]["urls"] and META["project"]["authors"]


@pytest.mark.parametrize("name", ["observation_v2.yaml", "prior_v22.yaml", "prior_v22_250.yaml"])
def test_shipped_configs_are_the_repository_configs(name):
    assert (api.ASSETS / name).read_bytes() == (REPO / "configs" / name).read_bytes()


#: The shipped members are their runs' 200k snapshots with ``meta["remove_floor"]`` added, so they are
#: compared tensor for tensor; the chooser is byte for byte.
_V22 = api.FAMILY_ASSETS["v2.2"]


@pytest.mark.parametrize("horizon", [100, 250])
def test_the_v22_members_are_their_200k_snapshots_with_the_floor_flag(horizon):
    import torch
    source = REPO / "results" / f"flow_v22_{horizon}" / "model_200k.pt"
    if not source.exists():
        pytest.skip("training run not on disk")
    a = torch.load(_V22[0][horizon], map_location="cpu", weights_only=False)
    b = torch.load(source, map_location="cpu", weights_only=False)
    assert a["meta"].pop("remove_floor") is True
    assert a["meta"] == b["meta"] and a["state_dict"].keys() == b["state_dict"].keys()
    assert all(torch.equal(a["state_dict"][k], b["state_dict"][k]) for k in b["state_dict"])


def test_the_v22_chooser_is_its_run():
    source = REPO / "results" / "chooser_v22" / "model.pt"
    if not source.exists():
        pytest.skip("training run not on disk")
    assert _V22[1].read_bytes() == source.read_bytes()


def test_only_the_shipped_family_is_named():
    """``prior=`` selects among ``api.FAMILY_ASSETS``, which holds the one shipped family; any other
    name is refused by name, and another family is loaded from its files with ``load_family``."""
    assert set(api.FAMILY_ASSETS) == {api.DEFAULT_PRIOR}
    assert api.load(prior=api.DEFAULT_PRIOR)[100].present_input
    with pytest.raises(ValueError, match="prior must be one of"):
        api.load(prior="v1")


_NO_ORACLES = """
import importlib.abc, json, sys
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path, target=None):
        if name.startswith("chai.forward.native"):
            raise ImportError(f"blocked: {name}")
sys.meta_path.insert(0, Block())
from chai import api
family = api.load()
sample = api.Sample(aft=api.AFT(age=38.0, n_spontaneous=240, lengths=[13.5] * 60 + [11.0] * 20, dpar=2.1),
                    grains=[api.Grain(age=24.1, radius=55, eU=18), api.Grain(age=27.9, radius=78, eU=35)])
post = family[100].infer(sample, n=500, seed=3, T_present=(5.0, 15.0))
old = api.Sample(aft=api.AFT(age=180.0, n_spontaneous=900, lengths=[13.0] * 80, dpar=2.1),
                 grains=[api.Grain(age=150.0, radius=60, eU=30)])
fam = family.infer(old, n=500, seed=3, T_present=(5.0, 15.0))
print(json.dumps({"T": post.T.tolist(), "p_long": fam.choice.p, "member": fam.choice.member, "t_max": [family[100].t_max, family[250].t_max],
                  "prior_t_max": family[250].prior["t_max"],
                  "native": [m for m in sys.modules if m.startswith("chai.forward.native")]}))
"""


def test_api_runs_without_the_compiled_oracles_or_the_checkout(tmp_path):
    """From a directory that is not the repository, with the oracles unimportable,
    ``api.load()`` reaches the shipped family, its 100 Ma member gives the same draws as the run
    directory does in the checkout, where it reads the corpus's own prior."""
    # The same thread count as this process, or float32 sums reduce in a different order and the draws
    # drift by ~1e-4 (test_examples and test_fit cap this process at 4 threads).
    script = f"import torch; torch.set_num_threads({torch.get_num_threads()})\n" + _NO_ORACLES
    out = subprocess.run([sys.executable, "-c", script], cwd=tmp_path, capture_output=True,
                         text=True, timeout=300)
    assert out.returncode == 0, out.stderr
    res = json.loads(out.stdout.strip().splitlines()[-1])
    assert res["native"] == []
    # the family, with the long member's own prior read from its latent (no corpus on this path)
    assert res["t_max"] == [100.0, 250.0] and res["prior_t_max"] == 250.0
    assert res["member"] == 250 and res["p_long"] > 0.99            # a 180 Ma AFT age is the long member's
    if not SHIPPED_FILE.exists():
        return
    sample = api.Sample(aft=api.AFT(age=38.0, n_spontaneous=240, lengths=[13.5] * 60 + [11.0] * 20, dpar=2.1),
                        grains=[api.Grain(age=24.1, radius=55, eU=18), api.Grain(age=27.9, radius=78, eU=35)])
    run = api.load(SHIPPED_FILE)
    run.net.floor_sd = float(run.net.meta["config"]["dequantise"]["noise_sd"])   # what the shipped copy's flag does
    ref = run.infer(sample, n=500, seed=3, T_present=(5.0, 15.0))
    np.testing.assert_allclose(np.asarray(res["T"]), ref.T, rtol=0, atol=1e-4)


def test_the_citation_names_this_version():
    """`CITATION.cff` and the README's citing paragraph carry the package version, so a bump
    that forgets them would publish a stale reference."""
    assert f"version: {chai.__version__}\n" in (REPO / "CITATION.cff").read_text()
    assert f"(version {chai.__version__})" in (REPO / "README.md").read_text()
