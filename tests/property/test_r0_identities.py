"""Rung R0: the operator identities, over random resolutions and points.

Each identity is checked at the level that defines it. For example, the lift identity
``x^p <tau(x)|_n = <tau(x)|_{n+1} M_{x^p}`` is tested at random points, not merely against
another implementation of the same matrix.
"""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse.linalg as spla
from numpy.polynomial import chebyshev as npcheb

from pihm import basis
from pihm.operators import (
    banded_factors,
    derivative_factorisation,
    derivative_matrix,
    embedding,
    fold,
    fold_levels,
    multiplication,
    power_of_x,
)

NS = range(1, 9)


def _points(rng: np.random.Generator, count: int = 9) -> np.ndarray:
    return np.concatenate([[-1.0, 1.0, 0.0], rng.uniform(-1.0, 1.0, count)])


def _power(values: np.ndarray, degree: int) -> np.ndarray:
    """Row-wise Kronecker power of feature vectors (one row per point)."""
    result = values
    for _ in range(degree - 1):
        result = (result[:, :, None] * values[:, None, :]).reshape(values.shape[0], -1)
    return result


@pytest.mark.parametrize("n", NS)
def test_derivative_differentiates(n: int, rng: np.random.Generator) -> None:
    """<tau(x)| G b is the derivative of <tau(x)| b, for random b and x."""
    b = rng.normal(size=2**n)
    x = _points(rng)
    expected = npcheb.chebval(x, npcheb.chebder(basis.to_chebyshev(b, n)))
    got = basis.tau(x, n) @ (derivative_matrix(n) @ b)
    np.testing.assert_allclose(got, expected, rtol=0, atol=1e-13 * np.abs(expected).max())


@pytest.mark.parametrize("n", range(1, 11))
def test_banded_factors_reproduce_the_derivative(n: int) -> None:
    """B~ G = D~ exactly, and G = B~^-1 D~."""
    G = derivative_matrix(n)
    B, D = banded_factors(n)
    scale = 2.0 ** (n + 1)
    assert abs(B @ G - D).max() <= 1e-15 * scale * 2**n
    solved = spla.spsolve_triangular(B.tocsr(), D.toarray(), lower=False)
    np.testing.assert_allclose(solved, G.toarray(), rtol=0, atol=1e-13 * scale**2)


@pytest.mark.parametrize("n", range(1, 11))
def test_structured_factorisation_of_the_derivative(n: int) -> None:
    """G = R0 (2 U_odd) Lambda exactly, at matrix level."""
    R0, U_odd, Lambda = derivative_factorisation(n)
    G = derivative_matrix(n)
    assert abs(R0 @ (2 * U_odd) @ Lambda - G).max() <= 1e-15 * 4.0**n


@pytest.mark.parametrize("n", range(1, 9))
def test_lambda_is_a_z_string_lcu(n: int) -> None:
    """Lambda = (N-1)/2 I - sum_j 2^(j-1) Z_j, with Z_j on bit j of the index; alpha = N - 1."""
    size = 2**n
    index = np.arange(size)
    z_strings = [1.0 - 2.0 * ((index >> j) & 1) for j in range(n)]
    diagonal = (size - 1) / 2 - sum(2.0 ** (j - 1) * z for j, z in enumerate(z_strings))
    _, _, Lambda = derivative_factorisation(n)
    np.testing.assert_array_equal(diagonal, Lambda.diagonal())
    assert (size - 1) / 2 + sum(2.0 ** (j - 1) for j in range(n)) == size - 1


@pytest.mark.parametrize("n", NS)
def test_multiplication_lift_identity(n: int, rng: np.random.Generator) -> None:
    """x^p <tau(x)|_n = <tau(x)|_{n+1} M_{x^p} for every p <= N."""
    x = _points(rng)
    for p in sorted({p for p in (0, 1, 2, 3, 7, 2**n) if p <= 2**n}):
        lifted = basis.tau(x, n + 1) @ power_of_x(p, n)
        np.testing.assert_allclose(lifted, x[:, None] ** p * basis.tau(x, n), rtol=0, atol=1e-13)


@pytest.mark.parametrize("n", NS)
def test_multiplication_by_a_general_polynomial(n: int, rng: np.random.Generator) -> None:
    """The lift identity for random a(x), given as monomials or as a Chebyshev series."""
    x = _points(rng)
    degree = min(2**n, 6)
    coefficients = rng.normal(size=degree + 1)
    for basis_name, values in (
        ("monomial", np.polynomial.polynomial.polyval(x, coefficients)),
        ("chebyshev", npcheb.chebval(x, coefficients)),
    ):
        M = multiplication(coefficients, n, basis=basis_name)  # type: ignore[arg-type]
        np.testing.assert_allclose(
            basis.tau(x, n + 1) @ M, values[:, None] * basis.tau(x, n), rtol=0, atol=1e-12
        )


@pytest.mark.parametrize("n", range(1, 7))
def test_lift_is_repeated_square_multiplication_on_one_more_qubit(n: int) -> None:
    """M_{x^p}(n) = (M_x^sq(n+1))^p M_1(n) while p <= N."""
    step = power_of_x(1, n + 1, mode="square")
    running = embedding(n)
    for p in range(0, 2**n + 1):
        assert abs(running - power_of_x(p, n)).max() <= 1e-13
        running = step @ running


@pytest.mark.parametrize("n", NS)
def test_square_is_the_truncated_lift(n: int, rng: np.random.Generator) -> None:
    """The square form is the lift truncated to N modes, re-expressed on n qubits (factor sqrt2)."""
    coefficients = rng.normal(size=min(2**n, 5) + 1)
    lift = multiplication(coefficients, n, mode="lift")
    square = multiplication(coefficients, n, mode="square")
    assert abs(np.sqrt(2) * square - lift[: 2**n]).max() <= 1e-14
    x = _points(rng)
    b = rng.normal(size=2**n)
    exact = basis.tau(x, n + 1) @ (lift @ b)
    truncated = basis.tau(x, n) @ (square @ b)
    tail = basis.tau(x, n + 1)[:, 2**n :] @ (lift @ b)[2**n :]
    np.testing.assert_allclose(truncated, exact - tail, atol=1e-12)


@pytest.mark.parametrize(("degree", "n"), [(2, n) for n in range(1, 7)] + [(3, 1), (3, 2), (3, 3)])
@pytest.mark.parametrize("weight", ["none", "x"])
def test_fold_identity(degree: int, n: int, weight: str, rng: np.random.Generator) -> None:
    """<tau(x)|_n^(x)p = <tau(x)|_{n+l} N (times x for N_x), at random points."""
    x = _points(rng)
    expected = _power(basis.tau(x, n), degree)
    if weight == "x":
        expected = x[:, None] * expected
    N = fold(degree, n, weight=weight)  # type: ignore[arg-type]
    got = basis.tau(x, n + fold_levels(degree)) @ N
    np.testing.assert_allclose(got, expected, rtol=0, atol=1e-14)


@pytest.mark.parametrize(("degree", "n"), [(2, n) for n in range(1, 7)] + [(3, 1), (3, 2), (3, 3)])
@pytest.mark.parametrize("weight", ["none", "x"])
def test_nodal_fold_equals_the_coefficient_fold(degree: int, n: int, weight: str) -> None:
    """The nodal fold equals the dealiased product-to-sum fold (R0)."""
    nodal = fold(degree, n, basis="nodal", weight=weight)  # type: ignore[arg-type]
    coefficient = fold(degree, n, weight=weight)  # type: ignore[arg-type]
    assert abs(nodal - coefficient).max() <= 1e-14 * 2**n


@pytest.mark.parametrize("ns", [(1, 2), (2, 1), (2, 3), (1, 1, 2)])
@pytest.mark.parametrize("fold_basis", ["coefficient", "nodal"])
def test_multi_axis_fold_is_a_pointwise_product(
    ns: tuple[int, ...], fold_basis: str, rng: np.random.Generator
) -> None:
    """For unequal axes, f_u * f_w = <tau'| N (u (x) w) at random points."""
    size = int(np.prod([2**k for k in ns]))
    u, w = rng.normal(size=size), rng.normal(size=size)
    N = fold(2, ns, basis=fold_basis)  # type: ignore[arg-type]
    folded = N @ np.kron(u, w)
    for _ in range(5):
        point = rng.uniform(-1.0, 1.0, len(ns))
        t: np.ndarray = np.ones(1)
        t_lifted: np.ndarray = np.ones(1)
        for coordinate, k in zip(point, ns, strict=True):
            t = np.kron(t, basis.tau(coordinate, k))
            t_lifted = np.kron(t_lifted, basis.tau(coordinate, k + 1))
        assert abs((t @ u) * (t @ w) - t_lifted @ folded) <= 1e-13 * size


@pytest.mark.parametrize("n", NS)
def test_chebyshev_gauss_transform(n: int, rng: np.random.Generator) -> None:
    """U is orthogonal and maps coefficients to values at the nodes (the nodal fold's basis)."""
    U = basis.chebyshev_gauss_transform(n)
    np.testing.assert_allclose(U @ U.T, np.eye(2**n), atol=1e-14)
    np.testing.assert_allclose(U, basis.tau(basis.nodes(n), n), atol=1e-14 * 2**n)
    b = rng.normal(size=2**n)
    np.testing.assert_allclose(U @ b, basis.evaluate(b, basis.nodes(n)), atol=1e-13 * 2**n)


@pytest.mark.parametrize("n", NS)
def test_feature_vector_is_a_combination_of_two_product_states(
    n: int, rng: np.random.Generator
) -> None:
    """tau(x) = 2^{-(n-1)/2} R0 (v(z) + v(z*))/2 with v(z) = (x)_j (1, z^(2^j)).

    With z = exp(i theta) and x = cos(theta), T_k(x) = Re z^k, and v(z)_k = z^k is a product
    state over the qubits. So the raw feature vector is an LCU of two product states with the
    same R0 that appears in the derivative's factorisation.
    """
    R0, _, _ = derivative_factorisation(n)
    for theta in rng.uniform(0.0, np.pi, 5):
        z = np.exp(1j * theta)
        product = np.ones(1, dtype=complex)
        for j in reversed(range(n)):  # Kronecker order: the most significant qubit first
            product = np.kron(product, np.array([1.0, z ** (2**j)]))
        combination = 2.0 ** (-(n - 1) / 2) * (R0 @ ((product + product.conj()) / 2))
        np.testing.assert_allclose(combination, basis.tau(np.cos(theta), n), atol=1e-13)
