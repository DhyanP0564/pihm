"""The normalised Chebyshev basis: conventions, evaluation, fitting and the scale factor."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.polynomial import chebyshev as npcheb
from numpy.polynomial import polynomial as nppoly

from pihm import basis


def test_scale_follows_the_paper() -> None:
    """s_0 = 2^(n/2) and s_k = 2^((n-1)/2), as in the paper's rows sqrt2 [T_0/sqrt2, T_1, ...]."""
    np.testing.assert_allclose(basis.chebyshev_scale(0), [1.0])
    np.testing.assert_allclose(basis.chebyshev_scale(1), [np.sqrt(2), 1.0])
    np.testing.assert_allclose(basis.chebyshev_scale(3), [2 * np.sqrt(2), 2, 2, 2, 2, 2, 2, 2])


@pytest.mark.parametrize("n", [-1, 1.5, True])
def test_bad_qubit_counts_are_rejected(n: object) -> None:
    with pytest.raises(ValueError, match="qubit count"):
        basis.chebyshev_scale(n)  # type: ignore[arg-type]


def test_tau_shapes() -> None:
    assert basis.tau(0.3, 3).shape == (8,)
    assert basis.tau(np.array([0.1, 0.2]), 3).shape == (2, 8)
    assert basis.tau(np.zeros((2, 5)), 2).shape == (2, 5, 4)


@pytest.mark.gotcha
@pytest.mark.parametrize("x", [np.float32(0.3), 1, np.array([0.5], dtype=np.float16)])
def test_tau_is_always_double_precision(x: object) -> None:
    """Lower-precision input never produces lower-precision output."""
    assert basis.tau(x, 4).dtype == np.float64  # type: ignore[arg-type]


@pytest.mark.parametrize("x", [1.0000001, -2.0, np.nan, np.inf])
def test_tau_rejects_points_outside_the_interval(x: float) -> None:
    with pytest.raises(ValueError, match=r"\[-1, 1\]"):
        basis.tau(x, 3)


def test_tau_is_raw_not_unit_norm() -> None:
    """The feature vector is the paper's raw tau; its norm varies with x."""
    norms = np.linalg.norm(basis.tau(np.array([-1.0, 0.0, 0.5, 1.0]), 3), axis=1)
    assert np.ptp(norms) > 0.1
    np.testing.assert_allclose(norms[-1], np.sqrt((1 + 2 * 7) / 8))


def test_evaluate_matches_the_chebyshev_series(rng: np.random.Generator) -> None:
    b = rng.normal(size=16)
    x = rng.uniform(-1, 1, 20)
    np.testing.assert_allclose(
        basis.evaluate(b, x), npcheb.chebval(x, basis.to_chebyshev(b, 4)), atol=1e-14
    )
    with pytest.raises(ValueError, match="power of two"):
        basis.evaluate(np.ones(6), 0.1)


def test_coefficient_conventions_round_trip(rng: np.random.Generator) -> None:
    c = rng.normal(size=8)
    np.testing.assert_allclose(basis.to_chebyshev(basis.to_normalised(c, 3), 3), c)
    # b = 2^((n-1)/2) S c with S = diag(sqrt(c_k)), c_0 = 2
    S = np.ones(8)
    S[0] = np.sqrt(2)
    np.testing.assert_allclose(basis.to_normalised(c, 3), 2.0 ** (2 / 2) * S * c)


def test_fit_is_exact_for_polynomials(rng: np.random.Generator) -> None:
    c = rng.normal(size=8)
    b = basis.fit(lambda x: npcheb.chebval(x, c), 3)
    np.testing.assert_allclose(basis.to_chebyshev(b, 3), c, atol=1e-14)


def test_fit_truncates_the_series() -> None:
    """Fitting at a smaller register keeps the leading coefficients (no aliasing)."""
    c = np.arange(1.0, 17.0)
    b = basis.fit(lambda x: npcheb.chebval(x, c), 2)
    np.testing.assert_allclose(basis.to_chebyshev(b, 2), c[:4], atol=1e-12)


def test_fit_on_several_axes(rng: np.random.Generator) -> None:
    """Separable fields fit to Kronecker products; axis 0 is the slow index."""
    cx, cy = rng.normal(size=4), rng.normal(size=8)
    b = basis.fit(lambda x, y: npcheb.chebval(x, cx) * npcheb.chebval(y, cy), (2, 3))
    expected = np.kron(basis.to_normalised(cx, 2), basis.to_normalised(cy, 3))
    np.testing.assert_allclose(b, expected, atol=1e-13)
    constant = basis.fit(lambda x, y: np.ones_like(x) * 2.0, (1, 1))
    np.testing.assert_allclose(constant, [2.0 * 2.0, 0, 0, 0], atol=1e-14)


def test_fit_needs_enough_points() -> None:
    with pytest.raises(ValueError, match="interpolation points"):
        basis.fit(np.cos, 4, points=8)


def test_scale_factor_is_the_weighted_integral(rng: np.random.Generator) -> None:
    """||b||^2 = (N/pi) int f^2 / sqrt(1-x^2) dx exactly, for a polynomial of degree < N."""
    n = 4
    size = 2**n
    coefficients = rng.normal(size=size)

    def f(x: np.ndarray) -> np.ndarray:
        return np.asarray(nppoly.polyval(x, coefficients))

    # Gauss-Chebyshev quadrature is exact for f^2 (degree < 2m) with m nodes.
    m = 4 * size
    nodes = np.cos(np.pi * (2 * np.arange(m) + 1) / (2 * m))
    integral = np.pi / m * np.sum(f(nodes) ** 2)
    assert basis.scale_factor(f, n) == pytest.approx(size / np.pi * integral, rel=1e-12)


def test_nodes_and_transform() -> None:
    np.testing.assert_allclose(basis.nodes(1), [np.cos(np.pi / 4), -np.cos(np.pi / 4)])
    U = basis.chebyshev_gauss_transform(2)
    np.testing.assert_allclose(U[0], basis.tau(basis.nodes(2)[0], 2))
