"""Known functions: Maclaurin and Chebyshev representations, and pointwise terms."""

from __future__ import annotations

import math

import numpy as np
import pytest

from pihm import basis
from pihm.operators import collapse_row, power_of_x
from pihm.problems.functions import (
    maclaurin_coefficients,
    polynomial_coefficients,
    polynomial_values,
    source_coefficients,
    term_values,
)
from pihm.problems.spec import Factor, Polynomial, Source, Term


def test_maclaurin_coefficients_are_exact() -> None:
    """The Cauchy integral reproduces the known series to rounding."""
    np.testing.assert_allclose(
        maclaurin_coefficients(lambda z: np.exp(-2 * z), 7),
        [(-2.0) ** p / math.factorial(p) for p in range(8)],
        atol=1e-15,
    )
    np.testing.assert_allclose(
        maclaurin_coefficients(lambda z: z * np.exp(2 * z), 5),
        [0.0] + [2.0 ** (p - 1) / math.factorial(p - 1) for p in range(1, 6)],
        atol=1e-15,
    )
    with pytest.raises(ValueError, match="degree"):
        maclaurin_coefficients(np.exp, 64)


def test_polynomial_coefficients_are_exact() -> None:
    polynomial = Polynomial("x", (1.0, -2.0, 1.0))
    np.testing.assert_allclose(
        polynomial_coefficients(polynomial, 3), basis.fit(lambda x: (x - 1) ** 2, 3), atol=1e-15
    )
    np.testing.assert_allclose(polynomial_values(polynomial, [0.0, 1.0, 2.0]), [1.0, 0.0, 1.0])
    with pytest.raises(ValueError, match="does not fit"):
        polynomial_coefficients(Polynomial("x", (0.0, 0.0, 0.0, 0.0, 1.0)), 2)


@pytest.mark.parametrize("degree", [3, 5, 7])
def test_maclaurin_source_is_the_papers_sum_of_lifts(degree: int) -> None:
    """Eq. 14: sum_p c_p M_{x^p} D0(x_s) applied to the solution is the Maclaurin polynomial."""
    n = 3
    source = Source("x*exp(-2*x)*sin(4*x)", "maclaurin", degree)
    series = maclaurin_coefficients(lambda z: z * np.exp(-2 * z) * np.sin(4 * z), degree)
    b = basis.fit(np.cos, n)
    D = collapse_row(0.3, float(basis.evaluate(b, 0.3)), n)  # so that D0 b is exactly 1
    lifted = sum(c * (power_of_x(p, n) @ (D @ b)) for p, c in enumerate(series))
    np.testing.assert_allclose(source_coefficients(source, ["x"], [n + 1]), lifted, atol=1e-13)


def test_chebyshev_source_is_the_fit() -> None:
    source = Source("exp(-2*x)")
    np.testing.assert_allclose(
        source_coefficients(source, ["x"], [4]), basis.fit(lambda x: np.exp(-2 * x), 4)
    )
    with pytest.raises(ValueError, match="one axis"):
        source_coefficients(Source("exp(t + x)", "maclaurin", 2), ["t", "x"], [2, 2])


def test_term_values() -> None:
    x = np.linspace(-1, 1, 5)
    term = Term(3.0, (Factor("f", (1,)), Factor("f", (0,))), Polynomial("x", (0.0, 1.0)))
    values: dict[tuple[int, ...], np.ndarray] = {(1,): np.cos(x), (0,): np.sin(x)}
    got = term_values(term, lambda factor: values[factor.derivative], {"x": x})
    np.testing.assert_allclose(got, 3 * x * np.cos(x) * np.sin(x))
    source = Term(-1.0, (), source=Source("exp(-2*x)", "maclaurin", 1))
    exact = term_values(source, lambda factor: x, {"x": x})
    np.testing.assert_allclose(exact, -np.exp(-2 * x))  # the exact function, not its series
    constant = term_values(Term(2.5), lambda factor: x, {"x": x})
    np.testing.assert_allclose(constant, 2.5)
