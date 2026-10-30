"""The batched tridiagonal solver must match a dense solve exactly enough."""
import pytest
import torch

from chai.forward.torchimpl.tridiag import solve_tridiagonal


def _dense(sub, diag, sup):
    """Build the dense matrix the three bands describe."""
    n = diag.shape[-1]
    m = torch.diag_embed(diag)
    if n > 1:
        m = m + torch.diag_embed(sup[..., :-1], offset=1)
        m = m + torch.diag_embed(sub[..., 1:], offset=-1)
    return m


def _solve_unbatched(A, b):
    """One LAPACK call per matrix.  A batched ``torch.linalg.solve`` deadlocks in MKL once
    ``torch.set_num_threads`` has been called in the process (torch 2.14.0+cu130; test_fit and
    test_examples call it), so the dense reference is solved matrix by matrix."""
    if A.dim() == 2:
        return torch.linalg.solve(A, b)
    return torch.stack([_solve_unbatched(a, x) for a, x in zip(A, b)])


def _diagonally_dominant(batch, n, seed=0, dtype=torch.float64):
    g = torch.Generator().manual_seed(seed)
    sub = torch.rand(batch, n, generator=g, dtype=dtype)
    sup = torch.rand(batch, n, generator=g, dtype=dtype)
    sub[:, 0] = 0.0
    sup[:, -1] = 0.0
    # the Crank-Nicolson sign convention: diag = -(positive + sub + sup)
    diag = -(sub + sup + torch.rand(batch, n, generator=g, dtype=dtype) + 0.1)
    rhs = torch.randn(batch, n, generator=g, dtype=dtype)
    return sub, diag, sup, rhs


@pytest.mark.parametrize("n", [1, 2, 3, 4, 5, 7, 8, 16, 63, 64, 128, 512])
def test_matches_dense_solve(n):
    sub, diag, sup, rhs = _diagonally_dominant(6, n, seed=n)
    got = solve_tridiagonal(sub, diag, sup, rhs)
    want = _solve_unbatched(_dense(sub, diag, sup), rhs.unsqueeze(-1)).squeeze(-1)
    assert torch.allclose(got, want, atol=1e-10, rtol=1e-8)


def test_residual_is_small_at_realistic_size():
    sub, diag, sup, rhs = _diagonally_dominant(32, 512, seed=3)
    x = solve_tridiagonal(sub, diag, sup, rhs)
    resid = (_dense(sub, diag, sup) @ x.unsqueeze(-1)).squeeze(-1) - rhs
    assert resid.abs().max() < 1e-10 * rhs.abs().max()


def test_float32_is_usable():
    sub, diag, sup, rhs = _diagonally_dominant(8, 256, seed=5, dtype=torch.float32)
    got = solve_tridiagonal(sub, diag, sup, rhs)
    want = _solve_unbatched(_dense(sub, diag, sup).double(),
                              rhs.double().unsqueeze(-1)).squeeze(-1)
    assert (got.double() - want).abs().max() < 1e-5 * want.abs().max()


def test_batching_is_independent():
    sub, diag, sup, rhs = _diagonally_dominant(8, 64, seed=7)
    full = solve_tridiagonal(sub, diag, sup, rhs)
    one = solve_tridiagonal(*[x[3:4] for x in (sub, diag, sup, rhs)])
    assert torch.allclose(full[3:4], one, atol=1e-14)


def test_gradients_match_the_dense_solve():
    sub, diag, sup, rhs = _diagonally_dominant(3, 32, seed=11)
    grads = []
    for fn in (lambda *a: solve_tridiagonal(*a),
               lambda s, d, u, r: _solve_unbatched(_dense(s, d, u), r.unsqueeze(-1)).squeeze(-1)):
        args = [x.clone().requires_grad_(True) for x in (sub, diag, sup, rhs)]
        fn(*args).square().sum().backward()
        grads.append([a.grad for a in args])
    for g1, g2 in zip(*grads):
        assert torch.allclose(g1, g2, atol=1e-8)


def test_rejects_mismatched_shapes():
    sub, diag, sup, rhs = _diagonally_dominant(2, 8)
    with pytest.raises(ValueError, match="same shape"):
        solve_tridiagonal(sub, diag, sup, rhs[:, :-1])
