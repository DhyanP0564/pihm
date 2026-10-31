"""Checks of the standard-form encoders: structured exact and published."""

from __future__ import annotations

from itertools import pairwise

import numpy as np
import pytest

from pihm.circuits.primitives import top_left_block
from pihm.circuits.standard import (
    collapse_encoding,
    derivative_ladder_error_bound,
    published_derivative_encoding,
    structured_derivative_encoding,
    zero_encoding,
)
from pihm.operators.constraints import collapse_row, zero_row
from pihm.operators.derivative import derivative_matrix


@pytest.mark.parametrize("n", range(1, 5))
def test_structured_derivative_encoding_is_exact(n: int) -> None:
    """G = R0(2 U_odd) Lambda is exact to machine precision, alpha = N(N-1)."""
    E = structured_derivative_encoding(n)
    N = 1 << n
    np.testing.assert_allclose(top_left_block(E), derivative_matrix(n).toarray(), atol=1e-9)
    assert E.alpha == pytest.approx(N * (N - 1))
    assert E.n_sys_qubits == n


@pytest.mark.parametrize("n", range(2, 7))
@pytest.mark.nightly
def test_published_derivative_encoding_matches_its_own_error_bound(n: int) -> None:
    """Measured entrywise error stays within derivative_ladder_error_bound's prediction.

    Excludes n=1: the bound is a leading-order (small-angle) approximation, not valid once the
    largest per-gate argument is not well inside arcsin's domain -- the same scope limit that an
    earlier implementation already measured and documented for the identical mechanism.
    """
    G = derivative_matrix(n).toarray()
    S_norm = float(max(np.linalg.norm(G @ G.T, ord=1), np.linalg.norm(G.T @ G, ord=1)))
    alpha = 2.0 / S_norm
    E = published_derivative_encoding(n, alpha=alpha)
    measured = float(np.max(np.abs(top_left_block(E) - G)))
    bound = E.alpha * derivative_ladder_error_bound(n, alpha)
    assert measured <= bound


@pytest.mark.nightly
def test_published_derivative_encoding_error_decreases_with_n() -> None:
    """The entrywise error shrinks sharply with n (the paper's own "exponentially decreasing")."""
    errors = []
    for n in range(2, 7):
        G = derivative_matrix(n).toarray()
        E = published_derivative_encoding(n)
        errors.append(float(np.max(np.abs(top_left_block(E) - G))))
    assert all(later < earlier / 5 for earlier, later in pairwise(errors))


@pytest.mark.parametrize("n", range(1, 4))
@pytest.mark.parametrize("x", [-1.0, -0.62, 0.0, 0.37, 0.91, 1.0])
def test_zero_encoding_matches_zero_row(n: int, x: float) -> None:
    """B(x) = sqrt(N)|0><tau(x)| exactly, using the raw (not per-point-renormalised) tau."""
    E = zero_encoding(x, n)
    N = 1 << n
    assert E.alpha == pytest.approx(np.sqrt(2 * N))
    np.testing.assert_allclose(top_left_block(E), zero_row(x, n).toarray(), atol=1e-8)


@pytest.mark.parametrize("value", [1.7, -1.7])
def test_collapse_encoding_matches_collapse_row(value: float) -> None:
    """D0(x) = B(x)/value exactly, including a sign flip for a negative value."""
    n = 2
    E = collapse_encoding(0.3, value, n)
    np.testing.assert_allclose(top_left_block(E), collapse_row(0.3, value, n).toarray(), atol=1e-8)


def test_collapse_encoding_rejects_zero_and_complex_values() -> None:
    with pytest.raises(ValueError, match="non-zero"):
        collapse_encoding(0.3, 0.0, 2)
    with pytest.raises(NotImplementedError, match="real"):
        collapse_encoding(0.3, 1.0 + 2.0j, 2)
