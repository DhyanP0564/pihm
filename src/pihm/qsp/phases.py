r"""GQSP phases from the Weiss complement and the inverse nonlinear Fourier transform.

:func:`find_phases` takes a target polynomial :math:`P` (:math:`\sup_{|z|=1}|P|\le1`, e.g. a
:class:`~pihm.qsp.filters.QiteFilter`'s coefficients as a Laurent series) and returns the GQSP
phase angles :math:`(\theta,\phi,\lambda)` for which the circuit
:math:`R_0\prod_k\big[C_0W\,R_k\big]` realises :math:`P` in its top-left block. Two steps:

1. **The Weiss complement** (:func:`spectral_complement`): the unique minimum-phase (outer)
   :math:`Q` with :math:`|P|^2+|Q|^2=1` on the unit circle, computed by spectral
   (Kolmogorov/cepstrum) factorisation — homomorphic filtering of :math:`\log(1-|P|^2)` — rather
   than by root-finding, which is unstable well before the degrees the filters
   reach (Berntson & Sunderhauf, "Complementary polynomials in quantum signal processing",
   arXiv:2406.04246).
2. **The inverse nonlinear Fourier transform** (:mod:`pihm.qsp.nlft`): :math:`(P,Q)` is the SU(2)
   nonlinear Fourier transform of a sequence :math:`F`; inverting that map and converting :math:`F`
   to :math:`(\theta,\phi,\lambda)` is :func:`angles_from_sequence`'s own job. Minimum-phase
   :math:`Q` is exactly the condition under which the inversion is numerically stable — see
   :mod:`pihm.qsp.nlft`'s own module docstring for why the two problems coincide at all.

:func:`realised_polynomial` reconstructs :math:`P` from :math:`(\theta,\phi,\lambda)` directly
(no circuit simulation), the check the pipeline requires: "the realised
polynomial must match the target to :math:`\le10^{-12}` on a :math:`4d`-point unit-circle grid."
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Literal, TypeAlias

import numpy as np
from numpy.typing import NDArray

from pihm.qsp.nlft import _LPoly, inlft_fast, inlft_layer_stripping

__all__ = [
    "Method",
    "Phases",
    "angles_from_sequence",
    "find_phases",
    "realised_coefficients",
    "realised_polynomial",
    "spectral_complement",
]


def spectral_complement(
    P: NDArray[np.complexfloating],
    *,
    oversample: int = 64,
    margin: float = 1.0e-2,
    tail_tolerance: float = 1.0e-11,
    max_oversample: int = 1 << 20,
) -> tuple[NDArray[np.complex128], NDArray[np.complex128], float, float]:
    r"""Return the unique minimum-phase complement :math:`Q` of ``P``, by spectral factorisation.

    Given :math:`P` (coefficients low-to-high degree, length :math:`d+1`), computes the unique
    minimum-phase (outer) :math:`Q`, also length :math:`d+1`, with
    :math:`|P(e^{i\theta})|^2+|Q(e^{i\theta})|^2=1` on the unit circle — via FFTs only, with
    neither an optimisation loop nor polynomial root-finding (root-finding-based completion
    becomes unstable well before the degrees the filters reach, Berntson & Sunderhauf,
    arXiv:2406.04246).

    Method (homomorphic/cepstral minimum-phase reconstruction): sample
    :math:`M(\theta)=1-|P(e^{i\theta})|^2` (:math:`\ge0` once :math:`P` is rescaled to sup-norm
    :math:`1-\text{margin}<1`) on a dense FFT grid, take its log, inverse-FFT to the cepstral
    domain, zero the anti-causal half and double the causal half (the homomorphic minimum-phase
    window), FFT back and exponentiate to reconstruct a spectrum with the same magnitude but
    minimum phase, then inverse-FFT to read off :math:`Q`'s own coefficients.

    This module's convention wants every root of :math:`Q` *inside* the unit disc — the mirror of
    the standard DSP convention for a causal sequence (which the homomorphic algorithm above
    naturally targets, with every root *outside*) — so the raw result is reversed and conjugated
    before returning.

    :math:`P` must first be rescaled so :math:`\sup_\theta|P(e^{i\theta})|<1` strictly (``margin``
    is how far below 1), since :math:`M` touching 0 exactly would put a log singularity on the FFT
    grid that no finite grid resolves. The grid size doubles from ``oversample*(d+1)`` until
    ``tail_energy`` (the norm of the reconstructed coefficients beyond degree :math:`d`, relative
    to the degree-:math:`d` part) meets ``tail_tolerance`` or ``max_oversample`` is reached.

    Parameters
    ----------
    P
        Target polynomial coefficients, low-to-high degree, any complex values, length :math:`d+1`.
    oversample, margin, tail_tolerance, max_oversample
        As described above.

    Returns
    -------
    poly, Q, tail_energy, target
        ``poly`` is ``P`` rescaled to sup-norm ``1 - margin`` (``P / target``); ``Q`` is the
        minimum-phase complement (length :math:`d+1`, complex128); ``tail_energy`` is the
        convergence diagnostic above (smaller is better-converged); ``target`` is the positive
        real scalar ``P`` was divided by.
    """
    d = len(P) - 1
    coefficients = np.asarray(P, dtype=np.complex128)
    oversample = max(int(oversample), 1)

    while True:
        grid = 1
        while grid < oversample * (d + 1):
            grid *= 2

        Pf = np.fft.fft(coefficients, grid)
        sup = float(np.max(np.abs(Pf)))
        target = sup / (1.0 - margin) if sup > 0 else 1.0
        poly = coefficients / target

        # R(z) = log sqrt(1 - |b(z)|^2), step 1 of the Weiss algorithm (arXiv:2406.04246;
        # arXiv:2510.00443 Section 6).
        log_magnitude = np.abs(Pf)
        del Pf
        log_magnitude /= target
        log_magnitude *= log_magnitude
        np.subtract(1.0, log_magnitude, out=log_magnitude)
        np.clip(log_magnitude, 1.0e-300, None, out=log_magnitude)
        np.log(log_magnitude, out=log_magnitude)
        log_magnitude *= 0.5

        # Step 2: G = R - i*H(R) via the causal cepstral window (the discrete Hilbert transform,
        # applied by zeroing the anti-causal half of the cepstrum and doubling the causal half).
        cepstrum = np.fft.ifft(log_magnitude)
        del log_magnitude
        cepstrum[1 : grid // 2] *= 2.0
        cepstrum[grid // 2 + 1 :] = 0.0  # the Nyquist bin keeps weight 1; grid is a power of two

        # Step 3: a(z) = exp(G(z)), then back to coefficients.
        spectrum = np.fft.fft(cepstrum)
        del cepstrum
        np.exp(spectrum, out=spectrum)
        full = np.fft.ifft(spectrum)
        del spectrum

        tail_energy = float(
            np.linalg.norm(full[d + 1 :]) / (np.linalg.norm(full[: d + 1]) + 1.0e-300)
        )
        if tail_energy < tail_tolerance or oversample >= max_oversample:
            Q = np.conj(full[: d + 1][::-1])
            return poly, Q, tail_energy, target
        oversample *= 4


# ------------------------------------------------------------- NLFT sequence -> GQSP phase angles

#: Unit-circle sample points used to pin the GQSP global phase (see ``_recover_global_phase``).
#: One point would do in exact arithmetic; a small handful, combined by the |P|-weighted sum
#: there, stays well-conditioned when P is tiny at any single chosen point — as it is for this
#: module's own imaginary-time filters, whose P(1) = e^{-tau} underflows.
_LAMBDA_PROBE_POINTS: Final[int] = 8

Method: TypeAlias = Literal["fast", "layer_stripping"]


def angles_from_sequence(
    P: NDArray[np.complexfloating], Q: NDArray[np.complexfloating], *, method: Method = "fast"
) -> tuple[NDArray[np.float64], NDArray[np.float64], float]:
    r"""Return the GQSP phase angles :math:`(\theta,\phi,\lambda)` for the pair ``(P, Q)``.

    Parameters
    ----------
    P
        Target polynomial coefficients, low-to-high degree, length :math:`d+1`,
        :math:`\sup_{|z|=1}|P|\le1`.
    Q
        The minimum-phase complement, same length (:func:`spectral_complement`'s own output). The
        stability of both inverse-NLFT paths is conditional on this; a non-minimum-phase ``Q`` is
        not an error, but the result is not covered by that guarantee.
    method
        ``"fast"`` (:func:`~pihm.qsp.nlft.inlft_fast`, :math:`O(d\log^2d)`) or
        ``"layer_stripping"`` (:func:`~pihm.qsp.nlft.inlft_layer_stripping`, :math:`O(d^2)`, the
        reference).

    Returns
    -------
    theta, phi, lam
        Two float64 arrays of length :math:`d+1` and one float, in the convention of GQSP's own
        rotation gate :math:`R(\theta,\phi,\lambda)=\begin{pmatrix}e^{i(\lambda+\phi)}\cos\theta &
        e^{i\lambda}\sin\theta\\e^{i\phi}\sin\theta & -\cos\theta\end{pmatrix}` (Motlagh & Wiebe,
        arXiv:2308.01501).

    Notes
    -----
    The textbook NLFT-to-QSP dictionary reads :math:`\psi_k=\arctan|F_k|`,
    :math:`\phi_k=\arg F_k` for the NLFT literature's own SU(2) parameterisation. GQSP's rotation
    carries the phase on one row only and reserves a separate global-phase slot, so the dictionary
    picks up boundary terms. Taking :math:`(a,b)=(Q\text{ on }[-d,0],\ P\text{ on }[0,d])`, the
    exact correspondence (verified by NLFT round-trip to machine precision, pinned by this
    module's own tests) is

    .. math::

        \theta_k &= \arctan|F_k| \quad (k = 0,\dots,d-1), \qquad
        \theta_d = \tfrac\pi2 - \arctan|F_d| \\
        \phi_d &= \arg F_d, \qquad
        \phi_{d-1} = \arg F_d - \arg F_{d-1}, \qquad
        \phi_k = \arg F_{k+1} - \arg F_k - \pi \ \ (k=0,\dots,d-2)

    all mod :math:`2\pi`. :math:`\lambda` enters the GQSP product only through the first rotation's
    first column, which the whole product left-multiplies, so it is a pure global phase on the
    realised :math:`P(U)` block — recovered by ``_recover_global_phase`` below.
    """
    P = np.asarray(P, dtype=np.complex128).reshape(-1)
    Q = np.asarray(Q, dtype=np.complex128).reshape(-1)
    if P.shape != Q.shape:
        raise ValueError(f"P and Q must have the same length, got {P.shape} and {Q.shape}")

    d = P.shape[0] - 1
    a = _LPoly(Q, support_start=-d)
    b = _LPoly(P, support_start=0)

    if method == "fast":
        F = inlft_fast(a, b)
    elif method == "layer_stripping":
        F = inlft_layer_stripping(a, b)
    else:
        raise ValueError(f"unknown method {method!r}, expected 'fast' or 'layer_stripping'")

    magnitude = np.abs(F)
    angle = np.angle(F)

    theta = np.arctan(magnitude)
    theta[d] = 0.5 * np.pi - theta[d]

    phi = np.empty(d + 1, dtype=np.float64)
    phi[d] = angle[d]
    if d >= 1:
        phi[d - 1] = _wrap(angle[d] - angle[d - 1])
    if d >= 2:
        phi[: d - 1] = _wrap(angle[1:d] - angle[: d - 1] - np.pi)

    lam = _recover_global_phase(theta, phi, P)
    return theta, phi, lam


def _wrap(x: NDArray[np.float64]) -> NDArray[np.float64]:
    r"""Wrap angles into :math:`(-\pi, \pi]`, matching :func:`numpy.angle`'s own range."""
    return (np.asarray(x) + np.pi) % (2.0 * np.pi) - np.pi


def _column(
    theta: NDArray[np.float64], phi: NDArray[np.float64], z: NDArray[np.complex128]
) -> NDArray[np.complex128]:
    r"""Return the GQSP product's realised top row, :math:`\lambda=0`, at each point of ``z``.

    Shared core of ``_recover_global_phase`` and :func:`realised_polynomial`: the same
    running :math:`2\times1` first-column recursion, :math:`O(d)` per point.
    """
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    eiphi = np.exp(1j * phi)
    top = np.full(z.shape, eiphi[0] * cos_t[0], dtype=np.complex128)
    bot = np.full(z.shape, sin_t[0] + 0j, dtype=np.complex128)
    for k in range(1, theta.shape[0]):
        t0 = top * z
        top = eiphi[k] * (cos_t[k] * t0 + sin_t[k] * bot)
        bot = sin_t[k] * t0 - cos_t[k] * bot
    return top


def _recover_global_phase(
    theta: NDArray[np.float64], phi: NDArray[np.float64], P: NDArray[np.complex128]
) -> float:
    r"""Return the :math:`\lambda` that makes the angle product reproduce ``P`` exactly.

    Evaluates the :math:`\lambda=0` product at :data:`_LAMBDA_PROBE_POINTS` roots of unity
    (:func:`_column`) and reads the phase off the :math:`|P|`-weighted inner product against
    ``P``'s own values there, so a probe point where ``P`` nearly vanishes contributes nearly
    nothing instead of dominating with a meaningless angle.
    """
    z = np.exp(2j * np.pi * np.arange(_LAMBDA_PROBE_POINTS) / _LAMBDA_PROBE_POINTS)
    top = _column(theta, phi, z)
    target = np.polyval(np.asarray(P, dtype=np.complex128)[::-1], z)
    overlap = np.sum(np.conj(top) * target)
    if overlap == 0:
        return 0.0
    return float(np.angle(overlap))


def realised_polynomial(
    theta: NDArray[np.float64], phi: NDArray[np.float64], lam: float, z: NDArray[np.complexfloating]
) -> NDArray[np.complex128]:
    r"""Return the GQSP product's realised :math:`P(z)` at each point of ``z`` — no circuit.

    This is the "realised vs target polynomial" check: evaluate the same
    :math:`2\times2` rotation product :func:`angles_from_sequence` inverted, directly as complex
    arithmetic, and compare against the target the phases were found for. Acceptance is
    :math:`\le10^{-12}` on a :math:`4d`-point unit-circle grid.

    Parameters
    ----------
    theta, phi, lam
        A phase triple, e.g. from :func:`find_phases`.
    z
        Points on the unit circle to evaluate at.

    Returns
    -------
    NDArray[np.complex128]
        :math:`P(z)`, one value per point of ``z``.
    """
    points = np.asarray(z, dtype=np.complex128)
    result: NDArray[np.complex128] = np.exp(1j * lam) * _column(theta, phi, points)
    return result


def realised_coefficients(
    theta: NDArray[np.float64], phi: NDArray[np.float64], lam: float
) -> NDArray[np.complex128]:
    r"""Return the GQSP product's realised :math:`P` as coefficients, low to high degree.

    The same product :func:`realised_polynomial` evaluates point by point, taken in coefficient
    space: with :math:`v_0=(e^{i\phi_0}\cos\theta_0,\ \sin\theta_0)` and the degree-one factors

    .. math::

       M_k(z)=\begin{pmatrix} e^{i\phi_k}\cos\theta_k\,z & e^{i\phi_k}\sin\theta_k \\
       \sin\theta_k\,z & -\cos\theta_k \end{pmatrix},

    :math:`P(z)=e^{i\lambda}\,[M_d(z)\cdots M_1(z)\,v_0]_0`. The factors are multiplied in
    pairs, level by level (each level's polynomial products by FFT), so the whole product costs
    :math:`O(d\log^2 d)` rather than the :math:`O(d)` per point of the running recursion.

    Parameters
    ----------
    theta, phi, lam
        A phase triple, e.g. from :func:`find_phases`.

    Returns
    -------
    NDArray[np.complex128]
        :math:`d+1` coefficients.
    """
    d = int(theta.shape[0]) - 1
    lead = np.exp(1j * phi[0]) * np.cos(theta[0])
    tail = complex(np.sin(theta[0]))
    if d == 0:
        return np.array([np.exp(1j * lam) * lead], dtype=np.complex128)
    count = 1 << (d - 1).bit_length()
    factors = np.zeros((count, 2, 2, 2), dtype=np.complex128)
    factors[:, 0, 0, 0] = factors[:, 1, 1, 0] = 1.0  # identity padding
    rotate = np.exp(1j * phi[1:])
    cos_t, sin_t = np.cos(theta[1:]), np.sin(theta[1:])
    factors[:d] = 0.0
    factors[:d, 0, 0, 1] = rotate * cos_t
    factors[:d, 0, 1, 0] = rotate * sin_t
    factors[:d, 1, 0, 1] = sin_t
    factors[:d, 1, 1, 0] = -cos_t
    while factors.shape[0] > 1:
        width = 2 * factors.shape[-1] - 1
        size = 1 << (width - 1).bit_length()
        later = np.fft.fft(factors[1::2], n=size)
        earlier = np.fft.fft(factors[0::2], n=size)
        factors = np.fft.ifft(np.einsum("bijk,bjlk->bilk", later, earlier), axis=-1)[..., :width]
    product = factors[0]
    top = product[0, 0] * lead + product[0, 1] * tail
    result: NDArray[np.complex128] = np.exp(1j * lam) * top[: d + 1]
    return result


@dataclass(frozen=True)
class Phases:
    r"""The GQSP phase triple for a target polynomial, with its own realised-polynomial check.

    Attributes
    ----------
    theta, phi
        Length :math:`d+1` float64 arrays (:func:`angles_from_sequence`).
    lam
        The global phase.
    target
        The target ``P`` these phases were found for (rescaled by :attr:`rescale`).
    rescale
        The positive real scalar the original ``P`` was divided by
        (:func:`spectral_complement`'s own ``target``) to reach sup-norm :math:`1-\text{margin}`.
    tail_energy
        :func:`spectral_complement`'s own convergence diagnostic.
    """

    theta: NDArray[np.float64]
    phi: NDArray[np.float64]
    lam: float
    target: NDArray[np.complex128]
    rescale: float
    tail_energy: float

    @property
    def degree(self) -> int:
        """The realised degree :math:`d`."""
        return int(self.theta.shape[0] - 1)

    def max_error(self, points: int | None = None) -> float:
        r"""Return the realised-polynomial check, :math:`\max_z|P_{\rm real}(z)-P_{\rm target}(z)|`.

        Parameters
        ----------
        points
            Unit-circle grid size. Defaults to :math:`4(d+1)`.
        """
        if points is None:
            points = 4 * (self.degree + 1)
        realised = realised_coefficients(self.theta, self.phi, self.lam)
        size = max(realised.size, self.target.size)
        difference = np.zeros(size, dtype=np.complex128)
        difference[: realised.size] += realised
        difference[: self.target.size] -= self.target
        # On the grid z^points = 1, so coefficients fold modulo points before one FFT evaluates
        # the difference at every grid point, z_j = exp(2 pi i j / points).
        folded = np.zeros(points, dtype=np.complex128)
        np.add.at(folded, np.arange(size) % points, difference)
        values = points * np.fft.ifft(folded)
        return float(np.max(np.abs(values)))


def find_phases(
    P: NDArray[np.complexfloating],
    *,
    method: Method = "fast",
    margin: float = 1.0e-2,
    tail_tolerance: float = 1.0e-11,
) -> Phases:
    r"""Find the GQSP phases realising ``P``.

    Composes :func:`spectral_complement` and :func:`angles_from_sequence`.

    Parameters
    ----------
    P
        Target polynomial coefficients, low-to-high degree, :math:`\sup_{|z|=1}|P|\le1` (e.g. a
        :class:`~pihm.qsp.filters.QiteFilter`'s ``coefficients``).
    method
        Forwarded to :func:`angles_from_sequence`.
    margin, tail_tolerance
        Forwarded to :func:`spectral_complement`.

    Returns
    -------
    Phases
    """
    P = np.asarray(P, dtype=np.complex128).reshape(-1)
    poly, Q, tail_energy, target = spectral_complement(
        P, margin=margin, tail_tolerance=tail_tolerance
    )
    theta, phi, lam = angles_from_sequence(poly, Q, method=method)
    return Phases(
        theta=theta, phi=phi, lam=lam, target=poly, rescale=target, tail_energy=tail_energy
    )
