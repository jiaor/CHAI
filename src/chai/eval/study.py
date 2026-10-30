"""Shared machinery for the grid-convergence studies.

Both Stage 1 studies ask the same question of different truths --
``scripts/node_count_study.py`` of a Gaussian process with a dialled-in
correlation length, ``scripts/prior_study.py`` of the actual prior -- so the
forward evaluation, the error metrics and the grid parser live here rather than
in either script.  Keeping one definition is what makes the two sets of numbers
comparable.
"""
from __future__ import annotations

import time

import numpy as np

from chai.forward.native import calculate_aft, calculate_rdaam_debug
from chai.forward.native.ketcham import LENGTH_BINS
from chai.paths import fixed_grid

__all__ = ["build_grid", "compare", "observables", "w1_length"]


def observables(t: np.ndarray, T: np.ndarray, grain: dict) -> dict:
    """Every Goal 1 observable for one path, plus the oracles' internal cost."""
    t0 = time.perf_counter()
    aft = calculate_aft(t, T, kinetic_par=grain["dpar"])
    aft_ms = (time.perf_counter() - t0) * 1e3

    t0 = time.perf_counter()
    he = calculate_rdaam_debug(t, T, grain["radius"], grain["U"], grain["Th"])
    he_ms = (time.perf_counter() - t0) * 1e3

    pdf = aft.length_pdf
    total = pdf.sum()
    return {
        "aft_age": aft.age,
        "mean_length": aft.mean_length,
        "length_pdf": pdf / total if total > 0 else pdf,
        "he_age": he.age,
        "he_corrected": he.corrected_age,
        "he_nodes": he.n_nodes,
        "aft_ms": aft_ms,
        "he_ms": he_ms,
    }


def w1_length(p: np.ndarray, q: np.ndarray) -> float:
    """Wasserstein-1 distance between two length distributions, in microns."""
    width = LENGTH_BINS[1] - LENGTH_BINS[0]
    return float(np.abs(np.cumsum(p) - np.cumsum(q)).sum() * width)


def compare(obs: dict, ref: dict) -> dict:
    """One row of representation error: ``obs`` against the dense reference.

    Ages are relative, mean length is absolute (microns, and the measurement is
    reported that way), the length distribution is a transport distance.
    """
    return {
        "d_aft_age": abs(obs["aft_age"] - ref["aft_age"]) / ref["aft_age"],
        "d_mean_length": abs(obs["mean_length"] - ref["mean_length"]),
        "d_length_w1": w1_length(obs["length_pdf"], ref["length_pdf"]),
        "d_he_age": abs(obs["he_age"] - ref["he_age"]) / ref["he_age"],
        "d_he_corr": abs(obs["he_corrected"] - ref["he_corrected"]) / ref["he_corrected"],
        "he_nodes": obs["he_nodes"],
        "aft_ms": obs["aft_ms"],
        "he_ms": obs["he_ms"],
        "ref_aft_age": ref["aft_age"],
        "ref_he_age": ref["he_age"],
        "ref_he_nodes": ref["he_nodes"],
    }


def build_grid(spec: str, n: int, t_max: float) -> np.ndarray:
    """Parse a placement spec into grid times.

    Where the first node of a log grid sits is not a detail: it sets how much of
    the budget goes to the recent past, and with it how coarse the deep past
    becomes.  A 4-node ``log:0.1`` grid over 100 Ma spends two nodes inside the
    first 0.1 Ma and then jumps 3 to 100 Ma.  ``piecewise:64@100,32@250`` is
    passed through to :func:`chai.paths.fixed_grid`; its counts must sum to ``n``.
    """
    if spec == "uniform":
        return fixed_grid(n, t_max, "uniform").numpy()
    if spec.startswith("log:"):
        return fixed_grid(n, t_max, "log", float(spec.split(":", 1)[1])).numpy()
    if spec.startswith("piecewise:"):
        return fixed_grid(n, t_max, spec).numpy()
    raise ValueError(f"unknown spacing spec {spec!r}")
