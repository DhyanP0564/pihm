"""Checks of the QITE and minimax comparison at like error."""

from __future__ import annotations

import math

import numpy as np
import pytest
import scipy.sparse as sp

from pihm.qsp.filter_comparison import (
    compare_state,
    compare_suppression,
    preparations,
    qite_suppression_degree,
    spectrum,
)
from pihm.qsp.filters import qite_coefficients
from pihm.qsp.minimax import minimax_filter
from pihm.qsp.prepare import apply_qite_filter


def _matrix(rng: np.random.Generator, singular: np.ndarray, rows: int) -> np.ndarray:
    """Return a real matrix with these singular values and random singular vectors."""
    columns = singular.size
    u = np.linalg.qr(rng.normal(size=(rows, rows)))[0][:, :columns]
    v = np.linalg.qr(rng.normal(size=(columns, columns)))[0]
    return np.asarray(u @ np.diag(singular) @ v.T)


@pytest.mark.parametrize("gap", [1e-2, 1e-4])
@pytest.mark.parametrize("offset", [0.0, 0.5])
@pytest.mark.parametrize("suppression", [1e-3, 1e-8])
def test_qite_never_beats_the_minimax_filter_at_equal_suppression(
    gap: float, offset: float, suppression: float
) -> None:
    # Chebyshev's theorem: no polynomial of the same degree suppresses [x_1, 1] further.
    x0 = -1.0 + offset * gap
    result = compare_suppression(x0, x0 + gap, suppression)
    assert result.qite is not None
    assert result.qite.degree >= result.minimax.degree


def test_qite_is_not_certified_where_its_tail_would_be_below_double_precision() -> None:
    # One gap above -1 the ground value is about eta, so eta = 1e-8 needs a tail near 1e-16.
    result = compare_suppression(-0.99, -0.98, 1e-8)
    assert result.qite is None
    assert result.minimax.degree > 0


def test_qite_suppression_degree_is_certified() -> None:
    # Rebuild the chosen filter and measure its suppression directly.
    x0, x1, eta = -1.0, -0.99, 1e-6
    choice = qite_suppression_degree(x0, x1, eta)
    assert choice is not None
    g = math.exp(-choice.tau * (1.0 + x0))
    e = math.exp(-choice.tau * (1.0 + x1))
    f = qite_coefficients(choice.tau, tolerance=(eta * g - e) / (1.0 + eta))
    assert f.degree == choice.degree
    excited = np.linspace(x1, 1.0, 20001)
    measured = np.max(np.abs(np.polynomial.chebyshev.chebval(excited, f.coefficients)))
    ground = abs(np.polynomial.chebyshev.chebval(x0, f.coefficients))
    assert measured / ground <= eta


def test_the_production_rule_certifies_at_the_edge_and_not_above_it() -> None:
    # tau = ln(1/eta)/gap with tail eta/10 is certified where the ground state is at -1; a ground
    # state one gap above -1 has value eta there, below the absolute tail's reach.
    at_edge = compare_suppression(-1.0, -0.99, 1e-6)
    assert at_edge.production_suppression < 1.2e-6
    above = compare_suppression(-0.99, -0.98, 1e-6)
    assert above.production_suppression > 1e-2


@pytest.mark.parametrize("target", [1, 3])
def test_the_spectral_infidelity_is_the_direct_one(target: int) -> None:
    rng = np.random.default_rng(7)
    singular = np.sort(rng.uniform(1.0, 3.0, 8))
    singular[:target] = np.linspace(0.0, 1e-9, target) if target > 1 else 0.3
    R = _matrix(rng, singular, 10)
    alpha = 4.0
    phi0 = rng.normal(size=8)
    phi0 /= np.linalg.norm(phi0)
    spec = spectrum(R, alpha, phi0, target)
    assert np.all(np.diff(spec.values) >= 0.0)
    assert np.sum(spec.weights) == pytest.approx(1.0)

    f = qite_coefficients(3.0, tolerance=1e-14)
    state = apply_qite_filter(sp.csr_array(R), alpha, f.coefficients, phi0.astype(np.complex128))
    vh = np.linalg.svd(R)[2]
    kernel = vh[np.argsort(np.linalg.svd(R)[1])[:target]]
    projection = kernel.T @ (kernel @ phi0)
    projection /= np.linalg.norm(projection)
    direct = 1.0 - abs(np.vdot(projection, state)) ** 2 / np.vdot(state, state).real
    values = np.asarray(np.polynomial.chebyshev.chebval(spec.values, f.coefficients), np.float64)
    assert spec.infidelity(values[None, :])[0] == pytest.approx(direct, rel=1e-9, abs=1e-15)
    assert spec.probability(values[None, :])[0] == pytest.approx(np.vdot(state, state).real)


def test_both_measured_degrees_reach_epsilon_through_the_t2_engine() -> None:
    rng = np.random.default_rng(3)
    singular = np.r_[0.02, np.linspace(0.3, 2.0, 7)]
    R = _matrix(rng, singular, 9)
    alpha = 2.5
    phi0 = np.full(8, 1.0 / math.sqrt(8.0))
    spec = spectrum(R, alpha, phi0, 1)
    epsilon = 1e-6
    result = compare_state(spec, epsilon)
    assert result.qite_degree is not None
    assert result.qite_tau is not None
    assert result.minimax_degree <= result.minimax_certified

    target = np.linalg.svd(R)[2][np.argmin(np.linalg.svd(R)[1])]
    csr = sp.csr_array(R)

    def infidelity(coefficients: np.ndarray) -> float:
        state = apply_qite_filter(csr, alpha, coefficients, phi0.astype(np.complex128))
        return float(1.0 - abs(np.vdot(target, state)) ** 2 / np.vdot(state, state).real)

    minimax = minimax_filter(result.minimax_degree, spec.x_excited).coefficients
    assert infidelity(minimax) == pytest.approx(result.minimax_infidelity, rel=1e-6)
    assert infidelity(minimax) <= epsilon
    s = math.sqrt(epsilon)
    lam = 2.0 * (spec.singular / alpha) ** 2
    f = np.exp(-result.qite_tau * lam)
    g = float(f[0]) * spec.gamma
    e = math.sqrt(float((f[1:] ** 2) @ spec.weights[1:]))
    full = qite_coefficients(result.qite_tau, tolerance=(s * g - e) / (1.0 + s)).coefficients
    qite = full[: result.qite_degree + 1]
    assert infidelity(qite) == pytest.approx(result.qite_infidelity, rel=1e-6)
    assert infidelity(qite) <= epsilon


def test_preparations_counts_as_t3_does() -> None:
    assert preparations(1.0) == 1
    assert preparations(0.5) == 1
    assert preparations(0.25) == 3  # arcsin(1/2) = pi/6, k = floor(1.5) = 1
    with pytest.raises(ValueError, match="positive"):
        preparations(0.0)
