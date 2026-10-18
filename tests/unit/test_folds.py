"""Product folds: shapes, options, and products of fitted functions."""

from __future__ import annotations

import numpy as np
import pytest

from pihm import basis
from pihm.operators import fold, fold_levels


@pytest.mark.parametrize(("degree", "levels"), [(2, 1), (3, 2), (4, 2), (5, 3)])
def test_levels(degree: int, levels: int) -> None:
    assert fold_levels(degree) == levels


@pytest.mark.parametrize(("degree", "n"), [(2, 3), (3, 2), (4, 1)])
def test_shape(degree: int, n: int) -> None:
    size = 2**n
    assert fold(degree, n).shape == (size << fold_levels(degree), size**degree)


def test_bad_options_are_rejected() -> None:
    with pytest.raises(ValueError, match="two copies"):
        fold(1, 2)
    with pytest.raises(ValueError, match="single axis"):
        fold(2, (1, 1), weight="x")
    with pytest.raises(ValueError, match="basis"):
        fold(2, 1, basis="fourier")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="weight"):
        fold(2, 1, weight="y")  # type: ignore[arg-type]


def test_fold_multiplies_fitted_polynomials_exactly(rng: np.random.Generator) -> None:
    """N_1 (b_f (x) b_g) = b_{fg} on n + 1 qubits, and N_x gives b_{x f g}."""
    n = 3
    cf, cg = rng.normal(size=8), rng.normal(size=8)

    def f(x: np.ndarray) -> np.ndarray:
        return np.asarray(np.polynomial.chebyshev.chebval(x, cf))

    def g(x: np.ndarray) -> np.ndarray:
        return np.asarray(np.polynomial.chebyshev.chebval(x, cg))

    both = np.kron(basis.fit(f, n), basis.fit(g, n))
    np.testing.assert_allclose(
        fold(2, n) @ both, basis.fit(lambda x: f(x) * g(x), n + 1), atol=1e-13
    )
    np.testing.assert_allclose(
        fold(2, n, weight="x") @ both, basis.fit(lambda x: x * f(x) * g(x), n + 1), atol=1e-13
    )


def test_cubic_fold_multiplies_three_fields(rng: np.random.Generator) -> None:
    n = 2
    parts = [rng.normal(size=4) for _ in range(3)]
    fns = [lambda x, c=c: np.polynomial.chebyshev.chebval(x, c) for c in parts]
    stacked = np.kron(np.kron(*(basis.fit(fn, n) for fn in fns[:2])), basis.fit(fns[2], n))
    expected = basis.fit(lambda x: fns[0](x) * fns[1](x) * fns[2](x), n + 2)
    np.testing.assert_allclose(fold(3, n) @ stacked, expected, atol=1e-13)
