"""Checks of the minimax edge filter."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.polynomial.chebyshev import chebval

from pihm.qsp.minimax import (
    edge_degree,
    edge_suppression,
    minimax_degree,
    minimax_filter,
    minimax_suppression,
    minimax_values,
)
from pihm.qsp.phases import find_phases

X = np.linspace(-1.0, 1.0, 4001)


def _by_hand(x: np.ndarray, degree: int, x_excited: float) -> np.ndarray:
    """T_d(w(x)) / T_d(w(-1)) from numpy's own Chebyshev evaluator."""
    t_d = np.zeros(degree + 1)
    t_d[-1] = 1.0
    w = (2.0 * x - (x_excited + 1.0)) / (1.0 - x_excited)
    anchor = (-2.0 - (x_excited + 1.0)) / (1.0 - x_excited)
    return np.asarray(chebval(w, t_d) / chebval(anchor, t_d))


@pytest.mark.parametrize("degree", [1, 2, 5, 6, 31, 80])
@pytest.mark.parametrize("x_excited", [-0.8, 0.0])
def test_the_closed_form_matches_the_chebyshev_polynomial(degree: int, x_excited: float) -> None:
    # Odd degrees included: T_d is odd there, so on [x_1, 1] the ratio carries (-1)^d.
    assert (
        np.max(np.abs(minimax_values(X, degree, x_excited) - _by_hand(X, degree, x_excited)))
        < 1e-12
    )


@pytest.mark.parametrize("degree", [3, 40, 500])
@pytest.mark.parametrize("x_excited", [-0.999, -0.5, 0.5])
def test_the_coefficients_interpolate_the_closed_form(degree: int, x_excited: float) -> None:
    # To the values' own conditioning: with x_1 near -1, p is steep at the anchor, where the nodes'
    # rounding reaches about 1e-12 at degree 500.
    f = minimax_filter(degree, x_excited)
    assert f.degree == degree
    assert np.max(np.abs(chebval(X, f.coefficients) - minimax_values(X, degree, x_excited))) < 1e-11


@pytest.mark.parametrize("degree", [3, 40, 500, 5000])
@pytest.mark.parametrize("x_excited", [-0.9999, -0.9, 0.0])
def test_the_coefficients_alternate_and_their_weight_is_one(degree: int, x_excited: float) -> None:
    # sum |p_k| = p(-1) = 1 when (-1)^k p_k >= 0, so GQSP realises the filter unrescaled.
    c = minimax_filter(degree, x_excited).coefficients
    signed = c * (-1.0) ** np.arange(degree + 1)
    assert np.min(signed) > -1e-13  # the far coefficients sit at the transform's rounding
    # To the anchor's conditioning, as above (find_phases rescales to 0.99 regardless).
    assert np.sum(np.abs(c)) == pytest.approx(1.0, abs=1e-10)


@pytest.mark.parametrize(("x_ground", "x_excited"), [(-1.0, -0.99), (-0.995, -0.99), (-0.5, 0.2)])
@pytest.mark.parametrize("suppression", [1e-2, 1e-8])
def test_the_degree_is_the_least_reaching_the_suppression(
    x_ground: float, x_excited: float, suppression: float
) -> None:
    d = minimax_degree(x_ground, x_excited, suppression)
    assert minimax_suppression(d, x_ground, x_excited) <= suppression
    assert minimax_suppression(d - 1, x_ground, x_excited) > suppression
    # The suppression is measured, not only formed: the excited interval's sup over the ground.
    excited = np.linspace(x_excited, 1.0, 20001)
    measured = np.max(np.abs(minimax_values(excited, d, x_excited))) / abs(
        minimax_values(np.array([x_ground]), d, x_excited)[0]
    )
    assert measured == pytest.approx(minimax_suppression(d, x_ground, x_excited), rel=1e-9)


def test_gqsp_realises_the_filter() -> None:
    f = minimax_filter(200, -0.97)
    phases = find_phases(f.coefficients.astype(np.complex128))
    assert phases.max_error() < 1e-12


def test_bad_input_is_refused() -> None:
    with pytest.raises(ValueError, match="inside"):
        minimax_values(X, 3, 1.0)
    with pytest.raises(ValueError, match="below"):
        minimax_degree(-0.5, -0.6, 1e-3)
    with pytest.raises(ValueError, match="suppression"):
        minimax_degree(-1.0, -0.6, 1.0)
    with pytest.raises(ValueError, match="max_degree"):
        minimax_filter(10, -0.5, max_degree=5)
    with pytest.raises(ValueError, match="heights"):
        edge_degree(0.5, 0.2, 1e-6)
    with pytest.raises(ValueError, match="suppression"):
        edge_degree(0.0, 0.2, 1.0)


@pytest.mark.parametrize(
    ("ground", "excited"), [(0.0, 0.1), (1e-3, 1e-2), (1e-9, 1e-5), (0.0, 1e-13), (5e-14, 1e-13)]
)
def test_the_rule_from_heights_is_the_least_degree_reaching_the_suppression(
    ground: float, excited: float
) -> None:
    # Heights keep their precision where x = h - 1 would round to -1 (the last two cases).
    d = edge_degree(ground, excited, 1e-6)
    assert float(edge_suppression(ground, d, excited)) <= 1e-6
    assert float(edge_suppression(ground, d - 1, excited)) > 1e-6
    if excited >= 1e-5:
        assert d == minimax_degree(ground - 1.0, excited - 1.0, 1e-6)
        assert float(edge_suppression(ground, d, excited)) == pytest.approx(
            minimax_suppression(d, ground - 1.0, excited - 1.0), rel=1e-9
        )
