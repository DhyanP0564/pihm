r"""The QITE filter: a priori imaginary time, coefficients and degree.

The primary filter is imaginary time,
:math:`f_\tau(x) = e^{-\tau(1+x)}`, applied to :math:`H' = 2H/\alpha_R^2 - I` whose spectrum
:math:`x \in [-1, 1]`. Its Chebyshev coefficients have the closed form

.. math::

    f_\tau(x) = \sum_{k=0}^\infty p_k T_k(x), \qquad
    p_k = (2 - \delta_{k0})(-1)^k e^{-\tau} I_k(\tau),

with :math:`I_k` the modified Bessel function of the first kind (from
:math:`e^{cx} = I_0(c) + 2\sum_{k\ge1} I_k(c) T_k(x)` at :math:`c = -\tau`, and
:math:`I_k(-\tau) = (-1)^k I_k(\tau)`). Every :math:`p_k` share one sign pattern and
:math:`\sum_k|p_k| = e^{-\tau}\big(I_0(\tau) + 2\sum_{k\ge1}I_k(\tau)\big) = e^{-\tau}e^\tau = 1`
exactly — the filter never needs rescaling.

:func:`imaginary_time` and :func:`asymptotic_degree` are the a priori formulas of the method,
:math:`\tau=\frac{\alpha_R^2}{2\Delta}\ln\frac1{\varepsilon_f}` and
:math:`d\approx\sqrt{2\tau\ln(2/\varepsilon_t)}`. :func:`qite_coefficients` computes the
coefficients themselves and a *rigorous* degree instead of that asymptotic estimate: it sums the
tail :math:`\sum_{k>d}|p_k|`, an upper bound on the resulting sup-norm truncation error since
every :math:`|T_k(x)|\le1`, and returns the smallest ``degree`` whose own tail is below the
requested tolerance — a certified bound, not a guess (:func:`asymptotic_degree` is kept only as
a cheap cross-check; the two agree to within a small constant factor
at every :math:`\tau` this module was checked against).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Final

import numpy as np
from numpy.typing import NDArray
from scipy.special import erfcinv, ive

__all__ = [
    "DEFAULT_EPSILON",
    "MAX_DEGREE",
    "QiteFilter",
    "asymptotic_degree",
    "imaginary_time",
    "qite_coefficients",
    "qite_filter",
]

#: The headline infidelity target.
DEFAULT_EPSILON: Final[float] = 1.0e-6

#: The largest filter degree the coefficient search considers by default (2^20).
MAX_DEGREE: Final[int] = 1 << 20


def imaginary_time(alpha_r: float, gap: float, epsilon_f: float) -> float:
    r"""Return the a priori imaginary time.

    .. math:: \tau = \frac{\alpha_R^2}{2\Delta}\ln\frac1{\varepsilon_f}

    Parameters
    ----------
    alpha_r
        The block encoding's own subnormalisation, from the construction.
    gap
        The absolute spectral gap :math:`\Delta = \lambda_1 - \lambda_0`
        (:attr:`pihm.classical.ReferenceState.gap`).
    epsilon_f
        The target leakage of the excited-state weight under the ideal, untruncated filter.

    Returns
    -------
    float
        :math:`\tau`, in the units of :math:`H' = 2H/\alpha_R^2 - I`.
    """
    if alpha_r <= 0.0:
        raise ValueError(f"imaginary_time needs a positive alpha_R, got {alpha_r}")
    if gap <= 0.0:
        raise ValueError(f"imaginary_time needs a positive spectral gap, got {gap}")
    if not 0.0 < epsilon_f < 1.0:
        raise ValueError(f"imaginary_time needs 0 < epsilon_f < 1, got {epsilon_f}")
    return (alpha_r**2) / (2.0 * gap) * math.log(1.0 / epsilon_f)


def asymptotic_degree(tau: float, epsilon_t: float) -> int:
    r"""Return the a priori degree estimate :math:`d\approx\sqrt{2\tau\ln(2/\varepsilon_t)}`.

    This is the method's own closed-form estimate, kept as a cheap cross-check on
    :func:`qite_coefficients`'s rigorous, tail-sum-certified degree — not itself a certified
    bound (it can under- or over-shoot by a small constant factor).
    """
    if tau <= 0.0:
        raise ValueError(f"asymptotic_degree needs a positive tau, got {tau}")
    if not 0.0 < epsilon_t < 1.0:
        raise ValueError(f"asymptotic_degree needs 0 < epsilon_t < 1, got {epsilon_t}")
    return math.ceil(math.sqrt(2.0 * tau * math.log(2.0 / epsilon_t)))


def _exp_scaled_bessel_i_fft(
    a: float,
    kmax: int,
    grid_per_order: int = 4,
    grid_per_width: int = 64,
    min_grid: int = 1 << 12,
) -> NDArray[np.float64]:
    r""":math:`I_k(a)e^{-a}` for :math:`k=0,\dots,\text{kmax}`, from one real FFT.

    Evaluates the integral representation :math:`I_k(a)e^{-a} = \frac1\pi\int_0^\pi
    e^{a(\cos t - 1)}\cos(kt)\,dt` (Abramowitz & Stegun 9.6.19) by the trapezoid rule, which
    converges spectrally since the integrand is analytic and :math:`2\pi`-periodic. The exponent
    is formed as :math:`-2a\sin^2(t/2)`, which keeps full relative precision at the peak, so the
    values are accurate to about :math:`5\times10^{-17}` absolute at every ``a`` (checked against
    :func:`scipy.special.ive` throughout its own valid range, :math:`a\le10^9`) and
    :math:`\sum_k(2-\delta_{k0})I_k(a)e^{-a}=1` holds to :math:`10^{-12}` up to
    :math:`a=10^{11}`. Unlike ``ive`` it has no upper limit on ``a``, which is why
    :func:`_exp_scaled_bessel_i` falls back to this once ``ive`` itself starts returning ``nan``.
    """
    target = max(
        min_grid,
        grid_per_order * (kmax + 1),
        int(grid_per_width * math.sqrt(max(a, 1.0))) + 1,
    )
    grid = 1 << (int(target) - 1).bit_length()
    # a(cos t - 1) = -2a sin^2(t/2): the difference form cancels near the peak at t = 0, where a
    # large a turns cos's rounding into the integrand's leading error.
    t = np.arange(grid) * (np.pi / grid)
    np.sin(t, out=t)
    t *= t
    t *= -2.0 * a
    np.exp(t, out=t)
    return (np.fft.rfft(t).real / grid)[: kmax + 1].copy()


def _exp_scaled_bessel_i(a: float, kmax: int) -> NDArray[np.float64]:
    r""":math:`I_k(a)e^{-a}` for :math:`k=0,\dots,\text{kmax}`, :math:`a\ge0`, any ``a``.

    ``scipy.special.ive`` (accurate to about :math:`10^{-16}` relative) is used wherever it
    stays finite; :func:`_exp_scaled_bessel_i_fft` covers the rest, since ``ive`` silently
    returns ``nan`` once ``a`` passes roughly :math:`1.9\times10^9`, a limitation of the
    underlying Cephes routine rather than of the values themselves.
    """
    if a < 0.0:
        raise ValueError(f"_exp_scaled_bessel_i: a must be non-negative, got {a}")
    if np.isfinite(ive(0, a)):
        chunk = 1 << 16
        parts = []
        k = 0
        while k <= kmax:
            upto = min(k + chunk, kmax + 1)
            parts.append(ive(np.arange(k, upto), a))
            k = upto
        magnitude = np.concatenate(parts).astype(np.float64)
        if np.all(np.isfinite(magnitude)):
            return magnitude
    return _exp_scaled_bessel_i_fft(a, kmax)


@dataclass(frozen=True)
class QiteFilter:
    r"""The QITE filter's realised Chebyshev coefficients, truncated to a certified tolerance.

    Attributes
    ----------
    tau
        The imaginary time this filter was built for.
    coefficients
        :math:`p_0,\dots,p_d`, low-to-high degree, the :math:`T_k` basis on :math:`[-1, 1]`.
    tail_bound
        The certified upper bound on :math:`\sup_x|f_\tau(x) - \sum_{k\le d}p_kT_k(x)|`
        (:math:`\sum_{k>d}|p_k|`).
    """

    tau: float
    coefficients: NDArray[np.float64]
    tail_bound: float

    @property
    def degree(self) -> int:
        """The realised degree :math:`d` (``len(coefficients) - 1``)."""
        return int(self.coefficients.shape[0] - 1)

    @property
    def weight(self) -> float:
        r""":math:`\sum_k|p_k|`. Exactly 1 for the untruncated filter."""
        return float(np.sum(np.abs(self.coefficients)))


def qite_coefficients(
    tau: float,
    *,
    tolerance: float = 1.0e-10,
    max_degree: int = MAX_DEGREE,
) -> QiteFilter:
    r"""Return the QITE filter's coefficients, truncated to the smallest degree meeting a tolerance.

    Parameters
    ----------
    tau
        The imaginary time (:func:`imaginary_time`).
    tolerance
        The target sup-norm truncation-error bound (:math:`\varepsilon_t`, an absolute bound on
        :attr:`QiteFilter.tail_bound`).
    max_degree
        The largest degree the search will consider.

    Returns
    -------
    QiteFilter

    Raises
    ------
    ValueError
        If no degree up to ``max_degree`` reaches ``tolerance`` (the degree needed scales as
        :math:`O(\sqrt\tau)`; raise ``max_degree`` or loosen ``tolerance``), or if ``tolerance``
        is at or below the float64 noise floor of the coefficients at this ``tau`` (no degree can
        certify it in double precision).
    """
    if tau <= 0.0:
        raise ValueError(f"qite_coefficients needs a positive tau, got {tau}")
    if not 0.0 < tolerance < 1.0:
        raise ValueError(f"qite_coefficients needs 0 < tolerance < 1, got {tolerance}")

    # The two-sided tail past degree d is erfc(d / sqrt(2 tau)) to leading order, so no degree
    # below half of sqrt(2 tau) erfcinv(tolerance) can reach the tolerance: refuse before an
    # evaluation whose grid resolves a peak of width 1/sqrt(tau).
    least = 0.5 * math.sqrt(2.0 * tau) * float(erfcinv(tolerance))
    if least > max_degree:
        raise ValueError(
            f"qite_coefficients: tau={tau} needs degree > max_degree={max_degree} to reach "
            f"tolerance={tolerance:.3e} (at least {least:.3e} from the tail's Gaussian bound; the "
            "degree needed scales roughly as O(sqrt(tau)), so raise max_degree"
        )

    # The coefficients decay like exp(-k^2/(2*tau)) (the Gaussian tail of I_k(tau) in k), so the
    # needed degree is ~sqrt(2*tau*ln(1/tolerance)); start from twice that and double geometrically,
    # so almost every call needs only one Bessel evaluation.
    estimate = int(2.0 * math.sqrt(2.0 * tau * max(math.log(1.0 / tolerance), 1.0) + 1.0)) + 32
    kmax = min(max_degree, max(64, estimate))

    while True:
        magnitude = _exp_scaled_bessel_i(tau, kmax)

        # I_k(tau)e^{-tau} is positive and falls monotonically in k; past the requested range's
        # own tail it is float64 FFT noise, not signal (only reached via the FFT fallback, since
        # `ive` itself does not leave a noise floor). Summing that noise into the tail sum below
        # would swamp a tight tolerance and could make the cumulative sum non-monotone, so trim it
        # first, using the tail of the computed range as its own noise estimate.
        #
        # That estimate is trustworthy only once the range has actually reached the coefficients'
        # own asymptotic decay. If `kmax` is capped short of it (this loop's own doubling always
        # gets there eventually, but a caller-supplied `max_degree` can cap it first — I_k(tau) has
        # a near-flat plateau for k << sqrt(2*tau) before the Gaussian roll-off begins), the "tail"
        # sampled here is still genuine signal, not noise, and is comparable in size to `tolerance`
        # rather than to machine epsilon: measured directly, a capped kmax inside that plateau
        # reads noise/peak ~ 0.75 there, against ~1e-23 once the range truly reaches the tail.
        # Trusting the larger reading would zero out real signal and fabricate a tail_bound of
        # exactly 0 instead of the true, much larger error — silently wrong, not merely imprecise.
        # Requiring `noise` to sit far below `tolerance` itself (not a fixed constant, since
        # genuine noise is always at the float64 floor regardless of tau, tens of orders of
        # magnitude under any `tolerance` this is called with) rejects that plateau reading and
        # falls through to growing kmax again, or eventually to the max_degree error below — never
        # to a false convergence.
        noise = float(np.max(np.abs(magnitude[(3 * kmax) // 4 :]))) if kmax >= 8 else 0.0
        if noise > 1.0e-3 * tolerance:
            noise = 0.0
        np.clip(magnitude, 0.0, None, out=magnitude)
        running_max = np.maximum.accumulate(magnitude[::-1])[::-1]
        cut = np.nonzero(running_max <= 4.0 * noise)[0]
        if cut.size:
            magnitude[int(cut[0]) :] = 0.0

        magnitude[1:] *= 2.0  # (2 - delta_{k0}) for k >= 1
        # tail[d] = sum_{k >= d} magnitude[k]. The range stops at kmax, and the coefficients sum to
        # one exactly, so the mass past it (or trimmed from it) is 1 minus what was kept; every
        # tail carries that, or a capped range would certify only the part it computed.
        tail = np.cumsum(magnitude[::-1])[::-1]
        tail += max(0.0, 1.0 - float(tail[0]))

        below = np.nonzero(tail[1:] < tolerance)[0]
        if below.size:
            degree = int(below[0])
            remaining = float(tail[degree + 1])
            sign = np.where(np.arange(degree + 1) % 2 == 0, 1.0, -1.0)
            coefficients = magnitude[: degree + 1] * sign
            return QiteFilter(tau=tau, coefficients=coefficients, tail_bound=remaining)

        if tolerance <= 4.0 * noise:
            raise ValueError(
                f"qite_coefficients: tolerance {tolerance:.3e} is at or below the float64 noise "
                f"floor {4.0 * noise:.3e} of the coefficients at tau={tau} — no degree can "
                "certify it in double precision; loosen tolerance"
            )
        if kmax >= max_degree:
            break
        kmax = min(max_degree, 2 * kmax)

    raise ValueError(
        f"qite_coefficients: tau={tau} needs degree > max_degree={max_degree} to reach "
        f"tolerance={tolerance:.3e} (achieved tail bound {float(tail[-1]):.3e} at max_degree; "
        "the degree needed scales roughly as O(sqrt(tau)), so raise max_degree"
    )


def qite_filter(
    alpha_r: float,
    gap: float,
    *,
    epsilon: float = DEFAULT_EPSILON,
    epsilon_t: float | None = None,
    max_degree: int = MAX_DEGREE,
) -> QiteFilter:
    r"""Compose :func:`imaginary_time` and :func:`qite_coefficients` with its own defaults.

    Parameters
    ----------
    alpha_r
        The block encoding's own subnormalisation.
    gap
        The absolute spectral gap :math:`\Delta`.
    epsilon
        :math:`\varepsilon_f`, the filter's own leakage target. Defaults to
        :data:`DEFAULT_EPSILON`.
    epsilon_t
        :math:`\varepsilon_t`, the truncation tolerance. Defaults to ``epsilon / 10``.
    max_degree
        Forwarded to :func:`qite_coefficients`.

    Returns
    -------
    QiteFilter
    """
    if epsilon_t is None:
        epsilon_t = epsilon / 10.0
    tau = imaginary_time(alpha_r, gap, epsilon)
    return qite_coefficients(tau, tolerance=epsilon_t, max_degree=max_degree)
