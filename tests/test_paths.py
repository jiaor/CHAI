"""Path representation: the fixed grid and movement between grids."""
from __future__ import annotations

import pytest
import torch

from chai.paths import fixed_grid, refine, resample, validate

T_MAX = 100.0


@pytest.mark.parametrize("spacing", ["uniform", "log"])
@pytest.mark.parametrize("n", [2, 8, 64])
def test_fixed_grid_obeys_the_canonical_convention(spacing, n):
    g = fixed_grid(n, T_MAX, spacing)
    assert g.shape == (n,)
    assert g[0] == 0.0
    assert g[-1] == pytest.approx(T_MAX)
    assert torch.all(torch.diff(g) > 0)
    validate(g[None], torch.zeros(1, n))


def test_log_grid_concentrates_on_the_recent_past():
    log_g, uni_g = fixed_grid(32, T_MAX, "log"), fixed_grid(32, T_MAX, "uniform")
    # half the log grid's nodes fall within the first tenth of the uniform grid's
    assert (log_g < uni_g[1]).sum() > (uni_g < uni_g[1]).sum()


def test_fixed_grid_rejects_bad_arguments():
    with pytest.raises(ValueError, match="at least 2 nodes"):
        fixed_grid(1, T_MAX)
    with pytest.raises(ValueError, match="t_max must be positive"):
        fixed_grid(8, 0.0)
    with pytest.raises(ValueError, match="t_first must be in"):
        fixed_grid(8, T_MAX, "log", t_first=T_MAX)
    with pytest.raises(ValueError, match="spacing must be"):
        fixed_grid(8, T_MAX, "geometric")


def test_resample_is_exact_on_a_linear_path():
    t = torch.tensor([[0.0, 10.0, 100.0]])
    T = torch.tensor([[10.0, 20.0, 110.0]])  # slope 1 K/Ma throughout
    q = torch.tensor([0.0, 5.0, 37.0, 100.0])
    assert torch.allclose(resample(t, T, q), 10.0 + q[None])


def test_resample_recovers_the_control_points():
    t = torch.tensor([[0.0, 3.0, 40.0, 100.0]])
    T = torch.tensor([[12.0, 95.0, 30.0, 140.0]])
    assert torch.allclose(resample(t, T, t[0]), T)


def test_resample_clamps_outside_the_path():
    t = torch.tensor([[0.0, 50.0]])
    T = torch.tensor([[10.0, 90.0]])
    got = resample(t, T, torch.tensor([-5.0, 25.0, 500.0]))
    assert torch.allclose(got, torch.tensor([[10.0, 50.0, 90.0]]))


def test_resample_onto_a_refinement_is_lossless():
    """A refined grid contains every control point, so nothing is lost."""
    t = torch.tensor([[0.0, 7.0, 31.0, 100.0]])
    T = torch.tensor([[15.0, 88.0, 42.0, 130.0]])
    t_fine, T_fine = refine(t, T, 8)
    assert torch.allclose(resample(t, T, t_fine[0]), T_fine, atol=1e-6)


def test_resample_is_differentiable_wrt_temperature():
    t = torch.tensor([[0.0, 20.0, 100.0]])
    T = torch.tensor([[10.0, 60.0, 120.0]], requires_grad=True)
    resample(t, T, torch.tensor([10.0])).sum().backward()
    # a query midway through the first segment splits its gradient evenly
    assert torch.allclose(T.grad, torch.tensor([[0.5, 0.5, 0.0]]))


def test_resample_is_batched_over_independent_paths():
    t = torch.tensor([[0.0, 50.0, 100.0], [0.0, 10.0, 100.0]])
    T = torch.tensor([[0.0, 50.0, 100.0], [0.0, 100.0, 100.0]])
    got = resample(t, T, torch.tensor([25.0]))
    assert torch.allclose(got, torch.tensor([[25.0], [100.0]]))


def test_resample_rejects_mismatched_shapes():
    with pytest.raises(ValueError, match="same shape"):
        resample(torch.zeros(1, 3), torch.zeros(1, 4), torch.zeros(2))
    with pytest.raises(ValueError, match=r"expected \(B, N\)"):
        resample(torch.zeros(3), torch.zeros(3), torch.zeros(2))


def test_piecewise_grid_extends_the_uniform_one():
    """A piecewise grid: 64 nodes to 100 Ma, then 32 to 250 Ma."""
    g = fixed_grid(96, 250.0, "piecewise:64@100,32@250")
    assert g.shape == (96,)
    validate(g[None], torch.zeros(1, 96))
    assert torch.equal(g[:64], fixed_grid(64, 100.0, "uniform"))
    assert torch.allclose(torch.diff(g[63:]), torch.full((32,), 150.0 / 32, dtype=g.dtype))
    assert g[-1] == 250.0


def test_piecewise_spacing_round_trips_and_rejects_bad_specs():
    from chai.paths import parse_piecewise, piecewise_spacing

    assert parse_piecewise("piecewise:64@100,32@250") == ([64, 32], [100.0, 250.0])
    assert piecewise_spacing([64, 32], [100.0, 250.0]) == "piecewise:64@100,32@250"
    # one segment is the uniform grid
    assert torch.equal(fixed_grid(8, 10.0, "piecewise:8@10"), fixed_grid(8, 10.0, "uniform"))
    with pytest.raises(ValueError, match="nodes, not"):
        fixed_grid(95, 250.0, "piecewise:64@100,32@250")
    with pytest.raises(ValueError, match="not t_max"):
        fixed_grid(96, 200.0, "piecewise:64@100,32@250")
    with pytest.raises(ValueError, match="increasing"):
        fixed_grid(96, 250.0, "piecewise:64@300,32@250")
    with pytest.raises(ValueError, match="expected"):
        fixed_grid(96, 250.0, "piecewise:64-100")
