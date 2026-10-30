"""The examples a user meets first run as written: the PyPI page, the README, and ``help(chai.api)``.

Each Python block runs on its own, on the CPU, with matplotlib off-screen, so a reader can paste any
one of them.  And the synthetic samples of :func:`chai.api.example_sample` are what they say they are.
"""
from __future__ import annotations

import re
from pathlib import Path

import matplotlib
import numpy as np
import pytest
import torch

from chai import api

matplotlib.use("Agg")
REPO = Path(__file__).resolve().parents[1]


def _blocks(path: Path) -> list[str]:
    return re.findall(r"```python\n(.*?)```", path.read_text(), flags=re.DOTALL)


def _docstring_block() -> str:
    """The indented code after the first ``::`` of chai.api's module docstring."""
    doc = api.__doc__
    body = doc[doc.index("::\n") + 3:]
    lines = []
    for line in body.splitlines():
        if line and not line.startswith("    "):
            break
        lines.append(line[4:])
    return "\n".join(lines)


CASES = ([(f"README-pypi.md[{k}]", b) for k, b in enumerate(_blocks(REPO / "README-pypi.md"))]
         + [(f"README.md[{k}]", b) for k, b in enumerate(_blocks(REPO / "README.md"))]
         + [("chai.api docstring", _docstring_block())])


def test_there_are_examples_to_run():
    names = [n for n, _ in CASES]
    assert sum(n.startswith("README-pypi") for n in names) >= 3
    assert "example_sample" in _docstring_block() and "T_present" in _docstring_block()


@pytest.mark.parametrize("name, code", CASES, ids=[n for n, _ in CASES])
def test_example_runs(name, code, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    exec(compile(code, name, "exec"), {"__name__": "example"})  # noqa: S102
    import matplotlib.pyplot as plt
    plt.close("all")


@pytest.fixture(scope="module")
def family():
    # inference runs on the CPU here; under xdist, every worker taking all cores thrashes
    torch.set_num_threads(min(4, torch.get_num_threads()))
    return api.load()


def test_the_named_examples_go_to_their_members(family):
    for name, member in (("pulse", 100), ("old", 250)):
        ex = api.example_sample(family, name)
        assert ex.sample.aft is not None and len(ex.sample.aft.lengths) == 100 and len(ex.sample.grains) == 4
        lo, hi = ex.T_present
        assert lo <= ex.T[0] <= hi and hi - lo >= api.MIN_PRESENT_RANGE
        assert ex.t.size == family[member].t.size
        post = family.infer(ex.sample, n=1000, T_present=ex.T_present, seed=0)
        assert post.choice.member == member
        if member == 250:
            assert post.choice.p > 0.95
        q = post.quantiles((0.025, 0.975))
        truth = np.interp(post.t, ex.t, ex.T)
        assert np.mean((truth >= q[0] - 1) & (truth <= q[1] + 1)) > 0.9     # one sample: the 95% band, loosely


def test_a_random_example_is_reproducible_and_from_the_prior(family):
    a, b = api.example_sample(family, "random", seed=4), api.example_sample(family, "random", seed=4)
    np.testing.assert_array_equal(a.T, b.T)
    assert a.sample == b.sample
    prior = family[100].prior
    assert a.T.max() <= prior.get("T_max", prior.get("T_top")) + 1e-6       # T_top: the name in the checkpoints
    with pytest.raises(ValueError, match="name must be one of"):
        api.example_sample(family, "nope")
