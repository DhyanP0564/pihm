"""Constraint rows: zero, collapse, ratio and datum, all on the raw feature vector."""

from __future__ import annotations

from typing import get_args

import numpy as np
import pytest
import scipy.sparse as sp

from pihm import basis
from pihm.operators import (
    CONSTRAINT_BUILDERS,
    ConstraintKind,
    collapse_row,
    datum_rows,
    embedding,
    fold,
    ratio_rows,
    zero_row,
)


def _solution(x: np.ndarray) -> np.ndarray:
    return np.asarray((0.5 + x) * np.exp(-2 * x))


def test_zero_row_evaluates_and_broadcasts(rng: np.random.Generator) -> None:
    """B(x) b = f(x) sqrt(N) e_0, the coefficients of the constant function f(x)."""
    n = 4
    b = rng.normal(size=16)
    for x in (-1.0, 0.3, 1.0):
        out = zero_row(x, n) @ b
        np.testing.assert_allclose(out[1:], 0.0)
        assert out[0] == pytest.approx(np.sqrt(16) * basis.evaluate(b, x))
        np.testing.assert_allclose(basis.evaluate(out, rng.uniform(-1, 1, 3)), basis.evaluate(b, x))


def test_zero_row_annihilates_a_function_vanishing_there() -> None:
    b = basis.fit(_solution, 5)  # vanishes at x = -1/2
    assert abs(zero_row(-0.5, 5) @ b).max() < 1e-14


def test_collapse_gives_the_constant_one_on_the_true_solution() -> None:
    n = 5
    b = basis.fit(_solution, n)
    x_s = 0.3
    out = collapse_row(x_s, float(_solution(np.array(x_s))), n) @ b
    np.testing.assert_allclose(out, np.sqrt(32) * np.eye(32)[0], atol=1e-13)
    with pytest.raises(ValueError, match="non-zero"):
        collapse_row(0.3, 0.0, n)


@pytest.mark.gotcha
def test_double_collapse_is_consistent_only_with_raw_tau() -> None:
    """Two collapses of the same point must agree with one collapse, lifted.

    The product of two collapsed copies (the constant 1 times itself) must equal one collapsed
    copy embedded one qubit up. With the raw feature vector this holds exactly. The earlier code
    per-point renormalisation scales one side by 1/||tau|| and the other by 1/||tau||^2, which
    biases doubled-space kernels.
    """
    n = 4
    b = basis.fit(_solution, n)
    x_s = 0.3
    v = float(_solution(np.array(x_s)))
    D = collapse_row(x_s, v, n)
    single = embedding(n) @ (D @ b)
    double = fold(2, n) @ np.kron(D @ b, D @ b)
    np.testing.assert_allclose(double, single, atol=1e-13)
    t = basis.tau(x_s, n)
    renormalised = sp.csr_array(D * np.linalg.norm(t) ** -1)
    single_r = embedding(n) @ (renormalised @ b)
    double_r = fold(2, n) @ np.kron(renormalised @ b, renormalised @ b)
    assert abs(double_r - single_r).max() > 1e-3


def test_ratio_rows_pin_a_direction(rng: np.random.Generator) -> None:
    y = rng.normal(size=5)
    F = ratio_rows(y, anchor=2).toarray()
    np.testing.assert_allclose(F @ y, 0.0, atol=1e-14)
    np.testing.assert_allclose(F @ (3.7 * y), 0.0, atol=1e-13)
    assert np.linalg.matrix_rank(F) == 4
    assert abs(F @ rng.normal(size=5)).max() > 1e-3
    with pytest.raises(ValueError, match="non-zero"):
        ratio_rows([1.0, 0.0], anchor=1)
    with pytest.raises(ValueError, match="anchor"):
        ratio_rows([1.0, 2.0], anchor=2)


def _field(t: np.ndarray | float, y: np.ndarray | float) -> np.ndarray:
    return np.asarray(np.exp(-t) * np.cos(2 * y) + 0.3 * np.multiply(t, y))


@pytest.mark.parametrize("axis", [0, 1])
def test_datum_rows_hold_on_the_true_field_only(axis: int) -> None:
    """The datum slice f|_{x_axis = -1} = g vanishes on the true field (to truncation error)."""
    ns = (4, 4)
    b = basis.fit(_field, ns)
    if axis == 0:
        g = basis.fit(lambda y: _field(-1.0, y), 4)
        anchor = (-1.0, 0.4)
    else:
        g = basis.fit(lambda t: _field(t, -1.0), 4)
        anchor = (0.4, -1.0)
    rows = datum_rows(ns, axis, -1.0, g, anchor, float(_field(*map(np.array, anchor))))
    assert abs(rows @ b).max() < 1e-10
    wrong = basis.fit(lambda t, y: _field(t, y) + 0.1 * y**2, ns)
    assert abs(rows @ wrong).max() > 1e-3


def test_datum_rows_validate_input() -> None:
    with pytest.raises(ValueError, match="one coordinate per axis"):
        datum_rows((1, 1), 0, -1.0, np.ones(2), (0.0,), 1.0)
    with pytest.raises(ValueError, match="non-zero"):
        datum_rows((1, 1), 0, -1.0, np.ones(2), (0.0, 0.0), 0.0)


def test_builders_cover_every_kind() -> None:
    assert set(CONSTRAINT_BUILDERS) == set(get_args(ConstraintKind))
