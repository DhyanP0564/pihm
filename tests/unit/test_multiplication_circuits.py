"""The multiplication lift as circuits, against the operators."""

from __future__ import annotations

import numpy as np
import pytest

from pihm.circuits.multiplication import (
    embed,
    multiplication_encoding,
    multiply_by_x_encoding,
    polynomial_encoding,
    shift_down_encoding,
)
from pihm.circuits.primitives import identity_encoding, top_left_block
from pihm.operators.multiplication import embedding, multiplication


@pytest.mark.parametrize("n", [1, 2, 3])
def test_shift_down_is_the_truncated_shift(n: int) -> None:
    """D maps |k> to |k-1> and annihilates |0>, at alpha = 1."""
    N = 1 << n
    D = shift_down_encoding(n)
    assert D.alpha == 1.0
    np.testing.assert_allclose(top_left_block(D), np.eye(N, k=1), atol=1e-14)


@pytest.mark.parametrize("n", [1, 2, 3])
def test_multiply_by_x_is_the_square_multiplication(n: int) -> None:
    """The square M_x is the Galerkin-truncated multiplication by x."""
    Mx = multiply_by_x_encoding(n)
    expected = multiplication([0.0, 1.0], n, mode="square").toarray()
    np.testing.assert_allclose(top_left_block(Mx), expected, atol=1e-13)
    assert Mx.alpha == pytest.approx(1.0 / np.sqrt(2.0) + 0.5)


@pytest.mark.parametrize("n", [1, 2])
def test_embed_is_m1(n: int) -> None:
    """embed(I) is the paper's M_1 = sqrt(2) [I; 0]."""
    M1 = embed(identity_encoding(n))
    assert M1.shape == (2 << n, 1 << n)
    np.testing.assert_allclose(top_left_block(M1), embedding(n).toarray(), atol=1e-14)


@pytest.mark.parametrize(
    "coefficients",
    [(0.0, -2.0), (1.0, 0.0, -1.0), (1.0, 0.0, -2.0, 0.0, 1.0), (0.5, 1.0, -3.0)],
    ids=["-2x", "1-x^2", "(1-x^2)^2", "mixed"],
)
@pytest.mark.parametrize("n", [2, 3])
def test_multiplication_encoding_is_the_exact_lift(coefficients: tuple[float, ...], n: int) -> None:
    """M_a from n to n+1 qubits matches operators.multiplication(mode='lift') exactly."""
    Ma = multiplication_encoding(coefficients, n)
    expected = multiplication(coefficients, n, mode="lift").toarray()
    assert Ma.shape == expected.shape
    np.testing.assert_allclose(top_left_block(Ma), expected, atol=1e-12)


def test_polynomial_encoding_alpha_is_the_l1_sum_of_the_powers() -> None:
    """The LCU of the powers of M_x has alpha = sum_j |a_j| alpha_x^j."""
    alpha_x = 1.0 / np.sqrt(2.0) + 0.5
    P = polynomial_encoding((1.0, 0.0, -2.0), 3)
    assert P.alpha == pytest.approx(1.0 + 2.0 * alpha_x**2)


def test_multiplication_encoding_refuses_a_degree_beyond_the_lift() -> None:
    """A multiplier of degree above N does not fit in one extra qubit."""
    with pytest.raises(ValueError, match="exceeds N"):
        multiplication_encoding((0.0, 0.0, 0.0, 1.0), 1)


def test_polynomial_encoding_refuses_the_zero_polynomial() -> None:
    """A zero multiplier has no block encoding."""
    with pytest.raises(ValueError, match="nonzero"):
        polynomial_encoding((0.0, 0.0), 2)
