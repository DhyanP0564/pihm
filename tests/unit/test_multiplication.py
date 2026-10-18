"""Multiplication operators: modes, bases, dtypes, bandwidths and accuracy."""

from __future__ import annotations

import numpy as np
import pytest

from pihm import basis
from pihm.operators import embedding, multiplication, power_of_x


def test_lift_refuses_a_degree_beyond_the_register() -> None:
    with pytest.raises(ValueError, match="larger lift"):
        power_of_x(5, 2)
    assert power_of_x(4, 2).shape == (8, 4)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"mode": "wide"}, "mode"),
        ({"basis": "legendre"}, "basis"),
    ],
)
def test_bad_options_are_rejected(kwargs: dict[str, str], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        multiplication([1.0, 2.0], 2, **kwargs)  # type: ignore[arg-type]


def test_empty_coefficients_are_rejected() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        multiplication([], 2)


def test_zero_multiplier_gives_the_zero_operator() -> None:
    assert multiplication([0.0, 0.0], 3).nnz == 0


def test_complex_coefficients_give_a_complex_operator() -> None:
    assert multiplication([1.0, 2j], 2).dtype == np.complex128
    assert multiplication([1.0, 2.0], 2).dtype == np.float64


@pytest.mark.parametrize("p", range(0, 5))
def test_square_power_is_banded(p: int) -> None:
    """The square M_{x^p} has bandwidth p (the property the descriptor LCU relies on)."""
    rows, columns = power_of_x(p, 4, mode="square").nonzero()
    assert np.max(np.abs(rows - columns), initial=0) == p


def test_embedding_is_the_papers_m1() -> None:
    np.testing.assert_allclose(embedding(2).toarray(), power_of_x(0, 2).toarray())
    np.testing.assert_allclose(embedding(1, levels=2).toarray()[:2], 2 * np.eye(2))
    with pytest.raises(ValueError, match="non-negative"):
        embedding(1, levels=-1)


def test_smooth_multiplier_through_its_chebyshev_series() -> None:
    """A smooth a(x), given by its Chebyshev series, multiplies to truncation accuracy."""
    n = 4
    a = basis.to_chebyshev(basis.fit(np.exp, n), n)  # degree 15 <= N = 16
    product = multiplication(a, n, basis="chebyshev") @ basis.fit(np.sin, n)
    expected = basis.fit(lambda x: np.exp(x) * np.sin(x), n + 1)
    np.testing.assert_allclose(product, expected, atol=1e-14)
