"""The expression language: evaluation, the whitelist, and the special functions."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.special import eval_legendre, lpmv

from pihm.problems.expressions import compile_expression, expression_variables


def test_evaluates_elementwise_and_broadcasts_constants() -> None:
    f = compile_expression("exp(-t)*sin(2*pi*x)", ["t", "x"])
    t, x = np.meshgrid([0.0, 0.5], [0.25, -0.25], indexing="ij")
    np.testing.assert_allclose(f(t, x), np.exp(-t) * np.sin(2 * np.pi * x))
    constant = compile_expression("-1/8", ["x"])
    assert constant(np.zeros(3)).shape == (3,)
    np.testing.assert_array_equal(constant(np.zeros(3)), -0.125)
    assert float(compile_expression("1/3", [])()) == pytest.approx(1 / 3)


def test_accepts_complex_arguments() -> None:
    """Maclaurin coefficients are computed on the unit circle, so sources take complex input."""
    f = compile_expression("x*exp(-2*x)*sin(4*x)", ["x"])
    z = np.exp(1j * np.linspace(0, np.pi, 5))
    np.testing.assert_allclose(f(z), z * np.exp(-2 * z) * np.sin(4 * z))


@pytest.mark.parametrize(
    "text",
    [
        "__import__('os')",
        "x.real",
        "x[0]",
        "exp(x, out=x)",
        "(lambda y: y)(x)",
        "x if x else 1",
        "x < 1",
        "open('f')",
        "True*x",
        "'a'",
        "y + 1",
        "x @ x",
    ],
)
def test_refuses_anything_outside_the_whitelist(text: str) -> None:
    with pytest.raises(ValueError, match=r"allowed|parse"):
        compile_expression(text, ["x"])


def test_refuses_bad_variable_names() -> None:
    with pytest.raises(ValueError, match="distinct"):
        compile_expression("x", ["x", "x"])
    with pytest.raises(ValueError, match="clash"):
        compile_expression("pi", ["pi"])
    with pytest.raises(ValueError, match="parse"):
        compile_expression("x +", ["x"])


def test_wrong_argument_count_is_an_error() -> None:
    with pytest.raises(TypeError, match="expected 2"):
        compile_expression("x*t", ["t", "x"])(1.0)


def test_free_variables() -> None:
    assert expression_variables("cos(pi*x/2)*sinh(pi*(y + 1)/2)/sinh(pi)") == {"x", "y"}
    assert expression_variables("legendre(4, x)") == {"x"}


@pytest.mark.parametrize("degree", range(8))
def test_legendre_functions(degree: int) -> None:
    x = np.linspace(-1, 1, 41)
    np.testing.assert_allclose(
        compile_expression(f"legendre({degree}, x)", ["x"])(x), eval_legendre(degree, x)
    )
    np.testing.assert_allclose(
        compile_expression(f"assoc_legendre(1, {degree}, x)", ["x"])(x),
        lpmv(1, degree, x),
        atol=1e-12,
    )
    h = 1e-5
    inner = x[1:-1]
    derivative = compile_expression(f"dlegendre({degree}, x)", ["x"])(inner)
    central = (eval_legendre(degree, inner + h) - eval_legendre(degree, inner - h)) / (2 * h)
    np.testing.assert_allclose(derivative, central, atol=1e-7)


def test_legendre_identity_used_by_the_corrected_variants() -> None:
    """P_l^1 = -sqrt(1 - x^2) P_l'(x), with the Condon-Shortley phase of scipy's lpmv."""
    x = np.linspace(-1, 1, 201)
    for degree in (1, 5, 6):
        identity = compile_expression(f"-sqrt(1 - x**2)*dlegendre({degree}, x)", ["x"])(x)
        np.testing.assert_allclose(identity, lpmv(1, degree, x), atol=1e-13)


def test_legendre_orders_must_be_whole() -> None:
    with pytest.raises(ValueError, match="integer"):
        compile_expression("legendre(1.5, x)", ["x"])(0.3)
