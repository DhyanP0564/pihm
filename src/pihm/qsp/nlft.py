r"""The SU(2) nonlinear Fourier transform: forward, and inverse by two equivalent methods.

The NLFT of a finitely supported sequence :math:`F=(F_k)` is the pair :math:`(a,b)` built from the
SU(2) product rule :math:`(a_1,b_1)*(a_2,b_2)=(a_1a_2-b_1b_2^*,\ a_1b_2+b_1a_2^*)`, with the
Schwarz reflection :math:`p^*(z)=\sum_k\overline{p_k}\,z^{-k}` (:func:`nlft`). :func:`inlft_fast`
and :func:`inlft_layer_stripping` invert that map — given :math:`(a,b)`, they return :math:`F`.

GQSP and the SU(2) NLFT problem are equivalent, not merely analogous (Laneve, "Generalized
quantum signal processing and non-linear Fourier transform are equivalent", arXiv:2503.03026,
Theorem 3.3): :mod:`pihm.qsp.phases` converts :math:`F` into GQSP's own phase angles. Layer
stripping (:func:`inlft_layer_stripping`) is the textbook :math:`O(d^2)` recursion; the inverse
nonlinear FAST Fourier transform (:func:`inlft_fast`) is a divide-and-conquer factorisation of the
same recursion, :math:`O(d\log^2d)` (Roos, "Inverse nonlinear fast Fourier transform on SU(2) with
applications to quantum signal processing", arXiv:2505.12615, Section 5). Both are numerically
stable only when :math:`a` is *outer* — has no zeros in the closed unit disc — which is exactly
what :func:`pihm.qsp.phases.spectral_complement`'s minimum-phase complement guarantees
(arXiv:2505.12615, Theorem 5.6).

It is an independent, dependency-free implementation of the cited literature (plain NumPy/SciPy
throughout), the numerically stable replacement for GQSP's own layer-stripping phase extraction.
"""

from __future__ import annotations

from typing import Final

import numpy as np
from numpy.typing import NDArray
from scipy.signal import fftconvolve

__all__ = [
    "inlft_fast",
    "inlft_layer_stripping",
    "nlft",
]

# Below this operand length, direct convolution beats `fftconvolve`'s own FFT setup cost (measured
# on this module's own recursion, complex128 throughout): a safe power-of-two threshold that keeps
# every small leaf of the divide-and-conquer recursion on the direct path.
_DIRECT_CONV_MAX: Final[int] = 256


class _LPoly:
    r"""A Laurent polynomial :math:`p(z)=\sum_j\text{coeffs}_j\,z^{\text{support\_start}+j}`.

    Deliberately minimal — only the operations :func:`inlft_fast` and
    :func:`inlft_layer_stripping` use. Coefficients are always complex128: the recursion mixes
    conjugates and products freely, and a real dtype would silently upcast mid-recursion anyway.
    """

    __slots__ = ("coeffs", "support_start")

    def __init__(self, coeffs: NDArray[np.complexfloating] | list[complex], support_start: int = 0):
        self.coeffs = np.asarray(coeffs, dtype=np.complex128).reshape(-1)
        self.support_start = int(support_start)

    def __len__(self) -> int:
        return int(self.coeffs.shape[0])

    def at(self, k: int) -> complex:
        """Return the coefficient of :math:`z^k` — 0 outside the stored support."""
        j = k - self.support_start
        if 0 <= j < self.coeffs.shape[0]:
            return complex(self.coeffs[j])
        return 0.0 + 0.0j

    def star(self) -> _LPoly:
        r"""Return the Schwarz reflection :math:`p^*(z)=\sum_k\overline{p_k}\,z^{-k}`."""
        return _LPoly(np.conj(self.coeffs)[::-1], -(self.support_start + self.coeffs.shape[0] - 1))

    def sharp(self) -> _LPoly:
        r"""``star()`` with the support shifted back up by the operand's own length.

        The recursion's other involution alongside :meth:`star` (same coefficients, different
        placement).
        """
        s = self.star()
        s.support_start += self.coeffs.shape[0]
        return s

    def __mul__(self, other: _LPoly) -> _LPoly:
        a, b = self.coeffs, other.coeffs
        if a.size == 0 or b.size == 0:
            return _LPoly(np.zeros(0, dtype=np.complex128), 0)
        product = (
            np.convolve(a, b)
            if min(a.size, b.size) <= _DIRECT_CONV_MAX
            else fftconvolve(a, b, mode="full")
        )
        return _LPoly(product, self.support_start + other.support_start)

    def _aligned(self, other: _LPoly) -> tuple[NDArray[np.complex128], NDArray[np.complex128], int]:
        lo = min(self.support_start, other.support_start)
        hi = max(
            self.support_start + self.coeffs.shape[0], other.support_start + other.coeffs.shape[0]
        )
        u = np.zeros(hi - lo, dtype=np.complex128)
        v = np.zeros(hi - lo, dtype=np.complex128)
        u_start = self.support_start - lo
        v_start = other.support_start - lo
        u[u_start : u_start + self.coeffs.shape[0]] = self.coeffs
        v[v_start : v_start + other.coeffs.shape[0]] = other.coeffs
        return u, v, lo

    def __add__(self, other: _LPoly) -> _LPoly:
        u, v, lo = self._aligned(other)
        return _LPoly(u + v, lo)

    def __sub__(self, other: _LPoly) -> _LPoly:
        u, v, lo = self._aligned(other)
        return _LPoly(u - v, lo)

    def truncate(self, m: int, n: int) -> _LPoly:
        """Keep only exponents in ``[m, n]``, zero-filling outside the stored support."""
        out = np.zeros(max(n - m + 1, 0), dtype=np.complex128)
        lo = max(m, self.support_start)
        hi = min(n, self.support_start + self.coeffs.shape[0] - 1)
        if hi >= lo:
            src = lo - self.support_start
            out[lo - m : hi - m + 1] = self.coeffs[src : src + (hi - lo + 1)]
        return _LPoly(out, m)

    def shift(self, k: int) -> _LPoly:
        r"""Multiply by :math:`z^k`."""
        return _LPoly(self.coeffs, self.support_start + k)


# ---------------------------------------------------------------------------------- forward NLFT


def nlft(F: NDArray[np.complexfloating], support_start: int = 0) -> tuple[_LPoly, _LPoly]:
    r"""Return the SU(2) nonlinear Fourier transform :math:`(a, b)` of the sequence ``F``.

    Built by repeatedly applying the product rule
    :math:`(a_1,b_1)*(a_2,b_2)=(a_1a_2-b_1b_2^*,\ a_1b_2+b_1a_2^*)`, :math:`O(d\log^2d)` by divide
    and conquer. Used as the round-trip oracle :func:`inlft_fast`/:func:`inlft_layer_stripping`
    are tested against.
    """
    values = np.asarray(F, dtype=np.complex128).reshape(-1)

    def rec(lo: int, hi: int) -> tuple[_LPoly, _LPoly]:
        if hi - lo <= 0:
            return _LPoly([1.0]), _LPoly([0.0])
        if hi - lo == 1:
            f = values[lo]
            den = np.sqrt(1.0 + (f * np.conj(f)).real)
            return _LPoly([1.0 / den]), _LPoly([f / den], support_start + lo)
        mid = (lo + hi) // 2
        a1, b1 = rec(lo, mid)
        a2, b2 = rec(mid, hi)
        return a1 * a2 - b1 * b2.star(), a1 * b2 + b1 * a2.star()

    return rec(0, values.shape[0])


# ---------------------------------------------------------------------------------- inverse NLFT


def _check_inlft_inputs(a: _LPoly, b: _LPoly) -> None:
    if len(a) != len(b):
        raise ValueError(
            f"(a, b) must have equally long supports to be in the image of the NLFT, got "
            f"len(a)={len(a)}, len(b)={len(b)}"
        )
    if a.support_start + len(a) != 1:
        raise ValueError(
            "`a` must be supported on [-d, 0] (its top exponent must be 0), got support "
            f"[{a.support_start}, {a.support_start + len(a) - 1}]"
        )


def inlft_layer_stripping(a: _LPoly, b: _LPoly) -> NDArray[np.complex128]:
    r"""Invert the NLFT by layer stripping — :math:`O(d^2)`, the correctness reference.

    Numerically stable when ``a`` is outer (arXiv:2505.12615, Theorem 5.6) — the condition
    :func:`pihm.qsp.phases.spectral_complement` guarantees.
    """
    _check_inlft_inputs(a, b)
    d = len(a) - 1
    sup_start = b.support_start
    a_star = a.star()
    b = b.shift(-sup_start)

    ac = a_star.truncate(0, d).coeffs.copy()
    bc = b.truncate(0, d).coeffs.copy()

    F = np.zeros(d + 1, dtype=np.complex128)
    for k in range(d + 1):
        n = d - k + 1
        f = bc[0] / ac[0]
        F[k] = f
        r = np.sqrt(1.0 + (f * np.conj(f)).real)
        new_a = (ac + np.conj(f) * bc) / r
        new_b = (bc - f * ac) / r
        if n > 1:
            ac = new_a[: n - 1]
            bc = new_b[1:n]
        else:
            ac = new_a[:1]
            bc = np.zeros(0, dtype=np.complex128)
    return F


def inlft_fast(a: _LPoly, b: _LPoly) -> NDArray[np.complex128]:
    r"""Invert the NLFT by the divide-and-conquer inverse nonlinear FAST Fourier transform.

    :math:`O(d\log^2d)` time, :math:`O(d)` space (arXiv:2505.12615, Section 5); same output and
    the same stability hypothesis as :func:`inlft_layer_stripping`, at a fraction of the cost once
    :math:`d` reaches the tens of thousands. Recursion depth is :math:`\log_2d`.
    """
    _check_inlft_inputs(a, b)
    sup_start = b.support_start
    F, _xi, _eta = _inlft_recurse(a.star(), b.shift(-sup_start))
    return F


def _inlft_recurse(a_star: _LPoly, b: _LPoly) -> tuple[NDArray[np.complex128], _LPoly, _LPoly]:
    r"""Run the divide-and-conquer core.

    Returns ``(F, xi, eta)``, where ``(xi, eta)`` is the "partial transform" pair that lets the
    two halves recombine without re-running either — the step that turns layer stripping's
    :math:`O(d^2)` into :math:`O(d\log^2d)` (arXiv:2505.12615, Section 5).
    """
    n = len(a_star)

    if n == 0:
        return np.zeros(0, dtype=np.complex128), _LPoly([0.0]), _LPoly([1.0])
    if n == 1:
        f = b.at(0) / a_star.at(0)
        r = np.sqrt(1.0 + (f * np.conj(f)).real)
        return np.array([f], dtype=np.complex128), _LPoly([f / r]), _LPoly([1.0 / r])

    m = -(-n // 2)  # ceil(n / 2)

    F_up, xi_m, eta_m = _inlft_recurse(a_star.truncate(0, m - 1), b.truncate(0, m - 1))

    a_m_star = eta_m.star() * a_star + xi_m.star() * b
    b_m = (eta_m * b - xi_m * a_star).shift(-m)

    F_down, xi_mn, eta_mn = _inlft_recurse(
        a_m_star.truncate(0, n - m - 1), b_m.truncate(0, n - m - 1)
    )

    xi_n = eta_m.sharp() * xi_mn + xi_m * eta_mn
    eta_n = eta_m * eta_mn - xi_m.sharp() * xi_mn

    return np.concatenate([F_up, F_down]), xi_n, eta_n
