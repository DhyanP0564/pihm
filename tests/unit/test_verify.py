"""Checks of the three verification modes."""

from __future__ import annotations

import numpy as np
import pytest

from pihm.circuits.standard import (
    derivative_ladder_error_bound,
    published_derivative_encoding,
    structured_derivative_encoding,
    zero_encoding,
)
from pihm.operators.constraints import zero_row
from pihm.operators.derivative import derivative_matrix
from pihm.verify import verify_approximation, verify_exact, verify_probe


@pytest.mark.parametrize("n", range(1, 4))
def test_verify_exact_passes_for_an_exact_encoder(n: int) -> None:
    result = verify_exact(structured_derivative_encoding(n), derivative_matrix(n))
    assert result.mode == "exact"
    assert result.passed
    assert result.max_error <= result.tolerance  # type: ignore[operator]


def test_verify_exact_fails_for_a_deliberately_wrong_target() -> None:
    n = 2
    wrong = derivative_matrix(n).toarray() + np.eye(1 << n)
    result = verify_exact(structured_derivative_encoding(n), wrong)
    assert not result.passed
    assert result.max_error > 0.5


@pytest.mark.parametrize("n", [2, 3, 5])
def test_verify_probe_passes_against_a_sparse_target(n: int, rng: np.random.Generator) -> None:
    """A sparse scipy target is never densified; only matrix-vector products are used."""
    result = verify_probe(structured_derivative_encoding(n), derivative_matrix(n), rng)
    assert result.mode == "probe"
    assert result.passed
    assert "e_first" in result.details["probe_errors"]


def test_verify_probe_on_zero_encoding(rng: np.random.Generator) -> None:
    n = 3
    result = verify_probe(zero_encoding(-0.4, n), zero_row(-0.4, n), rng, n_random=8)
    assert result.passed


def test_verify_approximation_reports_but_never_fails() -> None:
    n = 3
    G = derivative_matrix(n).toarray()
    S_norm = float(max(np.linalg.norm(G @ G.T, ord=1), np.linalg.norm(G.T @ G, ord=1)))
    alpha = 2.0 / S_norm
    E = published_derivative_encoding(n, alpha=alpha)
    bound = E.alpha * derivative_ladder_error_bound(n, alpha)
    result = verify_approximation(E, G, bound)
    assert result.passed  # by design: "report, don't gate"
    assert result.details["within_bound"]
    assert result.tolerance is None
