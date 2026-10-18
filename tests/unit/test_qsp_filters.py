"""Checks of the QITE filter: a priori tau, coefficients and degree."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.polynomial.chebyshev import chebval
from scipy.special import ive

from pihm.qsp.filters import (
    DEFAULT_EPSILON,
    _exp_scaled_bessel_i_fft,
    asymptotic_degree,
    imaginary_time,
    qite_coefficients,
    qite_filter,
)


def test_qite_coefficients_matches_the_bessel_formula_by_hand() -> None:
    # p_k = (2 - delta_k0) (-1)^k e^{-tau} I_k(tau), checked against scipy.special.ive
    # (an independent source: ive(k, a) = I_k(a) e^{-a}) at tau = 1.
    f = qite_coefficients(1.0, tolerance=1e-12)
    assert f.coefficients[0] == pytest.approx(ive(0, 1.0), abs=1e-14)
    assert f.coefficients[1] == pytest.approx(-2.0 * ive(1, 1.0), abs=1e-14)
    assert f.coefficients[2] == pytest.approx(2.0 * ive(2, 1.0), abs=1e-14)


@pytest.mark.parametrize("tau", [0.1, 1.0, 10.0, 1000.0])
def test_qite_coefficients_sum_of_absolute_values_is_one(tau: float) -> None:
    # Untruncated, the coefficients' absolute values sum to exactly one.
    f = qite_coefficients(tau, tolerance=1e-10)
    assert f.weight == pytest.approx(1.0, abs=1e-8)


@pytest.mark.parametrize("tau", [0.5, 5.0, 100.0, 1.0e4])
def test_qite_coefficients_truncation_error_matches_its_own_certified_bound(tau: float) -> None:
    # Independent check: evaluate the truncated polynomial on a fine grid against the actual
    # target function e^{-tau(1+x)}, via a different code path (numpy's own Chebyshev evaluator).
    # The achieved sup-norm error should sit at or below the certified tail_bound.
    tolerance = 1e-9
    f = qite_coefficients(tau, tolerance=tolerance)
    x = np.linspace(-1.0, 1.0, 4001)
    realised = chebval(x, f.coefficients)
    target = np.exp(-tau * (1.0 + x))
    achieved = float(np.max(np.abs(realised - target)))
    assert achieved <= max(f.tail_bound, tolerance) * 1.1


def test_qite_coefficients_rejects_bad_input() -> None:
    with pytest.raises(ValueError, match="positive tau"):
        qite_coefficients(0.0)
    with pytest.raises(ValueError, match="0 < tolerance < 1"):
        qite_coefficients(1.0, tolerance=1.5)


def test_qite_coefficients_raises_when_max_degree_is_too_small() -> None:
    with pytest.raises(ValueError, match="max_degree"):
        qite_coefficients(1.0e6, tolerance=1e-10, max_degree=8)


def test_qite_coefficients_does_not_fabricate_convergence_inside_the_plateau() -> None:
    # I_k(tau) for a large tau stays near-flat for k << sqrt(2*tau) before its Gaussian roll-off
    # begins (tau=1e6: the plateau extends past k=2000, the true degree is ~6467). A max_degree
    # capping the search inside that plateau must not mistake still-significant coefficients for
    # the float64 noise floor and report a fabricated near-zero tail_bound: either it raises, or
    # the tail_bound it reports is honest against the true (independently measured) error.
    tau, tolerance = 1.0e6, 1e-10
    for max_degree in (500, 1000, 2000, 4000):
        try:
            f = qite_coefficients(tau, tolerance=tolerance, max_degree=max_degree)
        except ValueError:
            continue
        x = np.linspace(-1.0, 1.0, 4001)
        realised = chebval(x, f.coefficients)
        target = np.exp(-tau * (1.0 + x))
        true_error = float(np.max(np.abs(realised - target)))
        assert true_error <= max(f.tail_bound, tolerance) * 1.1, (
            f"max_degree={max_degree}: claimed tail_bound={f.tail_bound:.3e} but the true error "
            f"is {true_error:.3e}"
        )


def test_imaginary_time_matches_the_closed_form_by_hand() -> None:
    # tau = alpha_R^2 / (2 Delta) * ln(1/epsilon_f).
    tau = imaginary_time(alpha_r=4.0, gap=2.0, epsilon_f=1.0e-6)
    assert tau == pytest.approx(16.0 / 4.0 * np.log(1.0e6), rel=1e-12)


def test_imaginary_time_rejects_bad_input() -> None:
    with pytest.raises(ValueError, match="alpha_R"):
        imaginary_time(0.0, 1.0, 1e-6)
    with pytest.raises(ValueError, match="gap"):
        imaginary_time(1.0, 0.0, 1e-6)
    with pytest.raises(ValueError, match="epsilon_f"):
        imaginary_time(1.0, 1.0, 1.5)


def test_asymptotic_degree_matches_the_closed_form_by_hand() -> None:
    d = asymptotic_degree(tau=10.0, epsilon_t=1.0e-6)
    assert d == int(np.ceil(np.sqrt(2.0 * 10.0 * np.log(2.0 / 1.0e-6))))


def test_asymptotic_degree_is_the_right_order_of_magnitude_against_the_rigorous_degree() -> None:
    # asymptotic_degree is an estimate, not a certified bound (module docstring); it should still
    # land within a small constant factor of qite_coefficients's own rigorous degree.
    tau, epsilon_t = 1.0e4, 1.0e-10
    rigorous = qite_coefficients(tau, tolerance=epsilon_t).degree
    estimate = asymptotic_degree(tau, epsilon_t)
    assert 0.3 * rigorous <= estimate <= 3.0 * rigorous


def test_qite_filter_composes_imaginary_time_and_qite_coefficients() -> None:
    alpha_r, gap = 14.0, 7.04
    result = qite_filter(alpha_r, gap)
    assert result.tau == pytest.approx(imaginary_time(alpha_r, gap, DEFAULT_EPSILON))
    assert result.weight == pytest.approx(1.0, abs=1e-6)
    # epsilon_t defaults to epsilon / 10: tighter than a bare epsilon call.
    looser = qite_coefficients(result.tau, tolerance=DEFAULT_EPSILON)
    assert result.degree >= looser.degree


def test_qite_filter_honours_an_explicit_epsilon_t() -> None:
    alpha_r, gap = 14.0, 7.04
    tau = imaginary_time(alpha_r, gap, DEFAULT_EPSILON)
    explicit = qite_filter(alpha_r, gap, epsilon_t=1.0e-8)
    direct = qite_coefficients(tau, tolerance=1.0e-8)
    assert explicit.degree == direct.degree


def test_qite_coefficients_refuses_a_tau_beyond_any_allowed_degree_at_once() -> None:
    """R5a's faithful Laplace at n = 4 has tau near 1e22: refused, not a 32 PiB Bessel grid."""
    with pytest.raises(ValueError, match="needs degree > max_degree"):
        qite_coefficients(1e22, tolerance=1e-7)


@pytest.mark.parametrize("a", [1.0e2, 1.0e6, 1.0e9])
def test_the_fft_bessel_values_match_ive_to_rounding(a: float) -> None:
    # The fallback against scipy's own values where both exist: absolute, since the far tail's
    # values are below any relative resolution.
    kmax = int(12.0 * np.sqrt(a)) + 10
    fft = _exp_scaled_bessel_i_fft(a, kmax)
    assert np.max(np.abs(fft - ive(np.arange(kmax + 1), a))) < 2e-16


@pytest.mark.parametrize("a", [2.0e9, 5.0e9])
def test_the_fft_bessel_values_keep_their_mass_where_ive_fails(a: float) -> None:
    # sum_k (2 - delta_k0) I_k(a) e^{-a} = 1. Formed as exp(a(cos t - 1)), the integrand's rounding
    # grew with a and this sum missed 1 by 5e-8 at a = 1.8e9.
    kmax = int(12.0 * np.sqrt(a))
    fft = _exp_scaled_bessel_i_fft(a, kmax)
    assert abs(fft[0] + 2.0 * np.sum(fft[1:]) - 1.0) < 1e-11


def test_the_certified_tail_holds_where_ive_fails() -> None:
    # tau = 1.8e9 takes the FFT fallback; the partial sum near x = -1, where the filter lives,
    # must sit within the certified tail of e^{-tau(1+x)}.
    tau = 1.8e9
    f = qite_coefficients(tau, tolerance=5e-9)
    # x = cos(pi - phi), so T_k(x) = (-1)^k cos(k phi) and 1 + x = 2 sin^2(phi/2), both formed
    # without cancelling at phi -> 0.
    phi = np.array([0.0, 1e-5, 3e-5, 1e-4])
    k = np.arange(f.degree + 1)
    values = np.cos(np.outer(phi, k)) @ (f.coefficients * (-1.0) ** k)
    exact = np.exp(-2.0 * tau * np.sin(phi / 2.0) ** 2)
    assert np.max(np.abs(values - exact)) <= f.tail_bound


@pytest.mark.parametrize(("tau", "tolerance"), [(7.87e10, 1e-7), (1.44e11, 1e-3)])
def test_qite_coefficients_refuses_a_degree_between_the_ceiling_and_twice_it(
    tau: float, tolerance: float
) -> None:
    # The degree needed lies above max_degree but below the early refusal's 2x margin, so the
    # capped range ends inside the coefficients' plateau, each coefficient small (~1e-7) but their
    # sum not. Read as noise, it would certify a degree near the ceiling.
    with pytest.raises(ValueError, match="max_degree"):
        qite_coefficients(tau, tolerance=tolerance)
