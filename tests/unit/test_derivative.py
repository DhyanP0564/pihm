"""The derivative matrix, its factorisations, and its subnormalisations."""

from __future__ import annotations

import numpy as np
import pytest

from pihm.operators import banded_factors, derivative_factorisation, derivative_matrix

#: ||G||_2, then alpha/||G||_2 for the structured exact encoding
#: (alpha = N(N-1)) and for the paper's published prefactor (2^(n-1) + 2^n) ||G||_S.
APPENDIX_A4 = {
    2: (7.39, 1.623, 49.9),
    3: (30.1, 1.858, 468.0),
    4: (121.0, 1.986, 4.05e3),
    5: (484.0, 2.051, 3.36e4),
    6: (1935.0, 2.084, 2.73e5),
    7: (7739.0, 2.100, 2.20e6),
    8: (30957.0, 2.109, 1.77e7),
    9: (123827.0, 2.113, 1.42e8),
    10: (495308.0, 2.115, 1.14e9),
}


def test_entries_follow_the_formula() -> None:
    """G_kp = 2p / sqrt(c_k) for p > k with p - k odd, and zero elsewhere."""
    G = derivative_matrix(4).toarray()
    for k in range(16):
        for p in range(16):
            expected = 2 * p / (np.sqrt(2) if k == 0 else 1.0) if p > k and (p - k) % 2 else 0.0
            assert G[k, p] == pytest.approx(expected, abs=0)


@pytest.mark.parametrize("n", range(0, 7))
def test_derivative_is_nilpotent(n: int) -> None:
    G = derivative_matrix(n).toarray()
    assert np.all(np.linalg.matrix_power(G, 2**n) == 0)


@pytest.mark.parametrize("n", [0, 1])
def test_tiny_registers(n: int) -> None:
    B, D = banded_factors(n)
    assert B.shape == D.shape == (2**n, 2**n)
    assert np.isclose(B.toarray()[0, 0], 2.0)


def test_banded_factor_entries() -> None:
    """B~: diagonal (2, 1, 1, ...), second superdiagonal (-sqrt2, -1, ...); D~: (2 sqrt2, 4, 6)."""
    B, D = banded_factors(2)
    np.testing.assert_allclose(
        B.toarray(),
        [[2, 0, -np.sqrt(2), 0], [0, 1, 0, -1], [0, 0, 1, 0], [0, 0, 0, 1]],
    )
    np.testing.assert_allclose(
        D.toarray(),
        [[0, 2 * np.sqrt(2), 0, 0], [0, 0, 4, 0], [0, 0, 0, 6], [0, 0, 0, 0]],
    )


@pytest.mark.parametrize(("n", "expected"), sorted(APPENDIX_A4.items()))
def test_subnormalisations_reproduce_appendix_a4(
    n: int, expected: tuple[float, float, float]
) -> None:
    """alpha_G = N(N-1) is within 2.12 of ||G||_2; the published prefactor grows like N^3."""
    norm, structured, published = expected
    G = derivative_matrix(n).toarray()
    size = 2**n
    measured = np.linalg.norm(G, 2)
    G_S = max(np.abs(G @ G.T).sum(axis=0).max(), np.abs(G.T @ G).sum(axis=0).max())
    assert measured == pytest.approx(norm, rel=5e-3)
    assert size * (size - 1) / measured == pytest.approx(structured, abs=1e-3)
    assert (2 ** (n - 1) + 2**n) * G_S / measured == pytest.approx(published, rel=1e-2)


def test_factorisation_factors_have_the_documented_form() -> None:
    R0, U_odd, Lambda = derivative_factorisation(3)
    np.testing.assert_allclose(R0.diagonal(), [1 / np.sqrt(2)] + [1.0] * 7)
    assert U_odd.sum() == sum(1 for k in range(8) for p in range(8) if p > k and (p - k) % 2)
    np.testing.assert_array_equal(Lambda.diagonal(), np.arange(8.0))
