r"""Checks of GQSP phase finding.

Covers the Weiss complement and the realised-polynomial check; the inverse NLFT itself is checked
in ``test_qsp_nlft.py``, and the :math:`\Pi P(W)\Pi` side is checked with tier T2.
"""

from __future__ import annotations

import numpy as np
import pytest

from pihm.qsp.filters import qite_coefficients, qite_filter
from pihm.qsp.phases import (
    Method,
    angles_from_sequence,
    find_phases,
    realised_coefficients,
    realised_polynomial,
    spectral_complement,
)


@pytest.mark.parametrize("tau", [1.0, 50.0, 1000.0])
def test_spectral_complement_satisfies_the_completeness_identity(tau: float) -> None:
    P = qite_coefficients(tau, tolerance=1e-10).coefficients.astype(complex)
    poly, Q, tail_energy, _target = spectral_complement(P)
    assert tail_energy < 1e-9
    grid = 4 * len(P)
    z = np.exp(2j * np.pi * np.arange(grid) / grid)
    p_val = np.polyval(poly[::-1], z)
    q_val = np.polyval(Q[::-1], z)
    completeness = np.abs(p_val) ** 2 + np.abs(q_val) ** 2
    assert np.max(np.abs(completeness - 1.0)) < 1e-8


@pytest.mark.parametrize("tau", [1.0, 50.0])
@pytest.mark.parametrize("method", ["fast", "layer_stripping"])
def test_find_phases_realises_the_target_within_tolerance(tau: float, method: Method) -> None:
    # The realised polynomial must match the target to <= 1e-12 on a
    # 4d-point unit-circle grid.
    P = qite_coefficients(tau, tolerance=1e-10).coefficients.astype(complex)
    phases = find_phases(P, method=method)
    assert phases.max_error() <= 1e-12


def test_find_phases_at_realistic_r1_scale() -> None:
    # R1 at n = 5: alpha_R = 994, and a flat gap 7.04. Degree in the
    # low thousands -- large enough to exercise the fast (O(d log^2 d)) path meaningfully.
    qf = qite_filter(alpha_r=994.0, gap=7.04)
    assert qf.degree > 1000
    phases = find_phases(qf.coefficients.astype(complex))
    assert phases.degree == qf.degree
    assert phases.max_error() <= 1e-12


def test_angles_from_sequence_rejects_mismatched_lengths() -> None:
    with pytest.raises(ValueError, match="same length"):
        angles_from_sequence(np.array([1.0, 0.0]), np.array([1.0, 0.0, 0.0]))


def test_angles_from_sequence_rejects_an_unknown_method() -> None:
    P = qite_coefficients(1.0, tolerance=1e-10).coefficients.astype(complex)
    _poly, Q, _tail, _target = spectral_complement(P)
    with pytest.raises(ValueError, match="unknown method"):
        angles_from_sequence(P, Q, method="bogus")  # type: ignore[arg-type]


def test_fast_and_layer_stripping_agree_on_the_realised_polynomial() -> None:
    P = qite_coefficients(20.0, tolerance=1e-10).coefficients.astype(complex)
    fast = find_phases(P, method="fast")
    reference = find_phases(P, method="layer_stripping")
    assert fast.theta == pytest.approx(reference.theta, abs=1e-9)
    grid = 4 * (fast.degree + 1)
    z = np.exp(2j * np.pi * np.arange(grid) / grid)
    realised_fast = realised_polynomial(fast.theta, fast.phi, fast.lam, z)
    realised_reference = realised_polynomial(reference.theta, reference.phi, reference.lam, z)
    assert np.max(np.abs(realised_fast - realised_reference)) < 1e-8


@pytest.mark.parametrize("degree", [0, 1, 2, 5, 64, 300])
def test_realised_coefficients_are_the_realised_polynomial(degree: int) -> None:
    """The coefficient-space product evaluates to the running recursion's values."""
    rng = np.random.default_rng(degree)
    theta = rng.uniform(0.0, np.pi, degree + 1)
    phi = rng.uniform(-np.pi, np.pi, degree + 1)
    z = np.exp(2j * np.pi * rng.uniform(size=40))
    coefficients = realised_coefficients(theta, phi, 0.7)
    assert coefficients.shape == (degree + 1,)
    np.testing.assert_allclose(
        np.polyval(coefficients[::-1], z), realised_polynomial(theta, phi, 0.7, z), atol=1e-13
    )


def test_max_error_is_the_grid_maximum_of_the_realised_polynomial() -> None:
    """max_error on the 4(d+1)-point grid equals the point-by-point evaluation's maximum."""
    rng = np.random.default_rng(3)
    P = rng.normal(size=40) + 1j * rng.normal(size=40)
    phases = find_phases(P)
    points = 4 * (phases.degree + 1)
    z = np.exp(2j * np.pi * np.arange(points) / points)
    realised = realised_polynomial(phases.theta, phases.phi, phases.lam, z)
    expected = np.max(np.abs(realised - np.polyval(phases.target[::-1], z)))
    assert phases.max_error() == pytest.approx(expected, abs=1e-13)
    assert phases.max_error(points=7) >= 0.0
