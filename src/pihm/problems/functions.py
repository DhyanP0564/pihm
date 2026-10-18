r"""The known functions of a problem, as values and as Chebyshev coefficients.

A problem mentions known functions in three places: polynomial multipliers
(:class:`~pihm.problems.spec.Polynomial`), sources (:class:`~pihm.problems.spec.Source`) and
datum slices. This module turns each into what the assembler and the checks need:

* :func:`polynomial_coefficients` and :func:`source_coefficients` give normalised Chebyshev
  coefficients on a register, exactly for a polynomial and in the chosen representation for a
  source. The paper's representation is the Maclaurin polynomial of Eq. 14,
  :math:`r(x) \approx \sum_{p \le \bar p} c_p x^p` (:func:`maclaurin_coefficients`); the
  corrected one is the Chebyshev series of :math:`r` itself.
* :func:`term_values` evaluates a term pointwise, given the values of its factors, for residual
  checks and for the boundary-value reference.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np
from numpy.polynomial import chebyshev as npcheb
from numpy.polynomial import polynomial as nppoly
from numpy.typing import ArrayLike, NDArray

from pihm import basis
from pihm.problems.expressions import compile_expression
from pihm.problems.spec import Factor, Polynomial, Source, Term

__all__ = [
    "MACLAURIN_POINTS",
    "maclaurin_coefficients",
    "polynomial_coefficients",
    "polynomial_values",
    "source_coefficients",
    "source_values",
    "term_values",
]

#: Points on the unit circle used to compute Maclaurin coefficients by a discrete Cauchy
#: integral. The aliasing error is the size of the coefficient of degree ``p + MACLAURIN_POINTS``,
#: negligible for the entire functions that appear as sources.
MACLAURIN_POINTS = 128


def polynomial_values(polynomial: Polynomial, x: ArrayLike) -> NDArray[Any]:
    """Evaluate a :class:`~pihm.problems.spec.Polynomial` at the points ``x`` of its axis."""
    return np.asarray(nppoly.polyval(np.asarray(x), np.asarray(polynomial.coefficients)))


def polynomial_coefficients(polynomial: Polynomial, n: int) -> NDArray[np.float64]:
    r"""Return a polynomial's normalised Chebyshev coefficients on :math:`n` qubits, exactly.

    Raises
    ------
    ValueError
        If the degree does not fit the register (:math:`\deg \ge N`).
    """
    size = 1 << n
    if polynomial.degree >= size:
        raise ValueError(f"degree {polynomial.degree} does not fit {n} qubits")
    standard = np.zeros(size)
    series = npcheb.poly2cheb(np.asarray(polynomial.coefficients[: polynomial.degree + 1]))
    standard[: series.size] = series
    return np.asarray(basis.to_normalised(standard, n), dtype=np.float64)


def maclaurin_coefficients(f: Callable[..., ArrayLike], degree: int) -> NDArray[np.float64]:
    r"""Return the Maclaurin coefficients :math:`c_0, \dots, c_{\bar p}` of an analytic ``f``.

    They are computed by the Cauchy integral :math:`c_p = \frac{1}{2\pi i}\oint f(z) z^{-p-1} dz`
    on the unit circle, discretised at :data:`MACLAURIN_POINTS` points (a fast Fourier
    transform), which is exact to rounding for the entire functions used as sources. ``f`` must
    accept complex arguments.

    Parameters
    ----------
    f
        A vectorised function, analytic on the closed unit disc and real on the real line.
    degree
        The truncation order :math:`\bar p`.

    Returns
    -------
    numpy.ndarray
        The real coefficients, length :math:`\bar p + 1`.

    Examples
    --------
    >>> np.round(maclaurin_coefficients(np.exp, 3), 12)
    array([1.        , 1.        , 0.5       , 0.16666667])
    """
    if degree < 0 or degree >= MACLAURIN_POINTS // 2:
        raise ValueError(f"degree must be in [0, {MACLAURIN_POINTS // 2}), got {degree}")
    z = np.exp(2j * np.pi * np.arange(MACLAURIN_POINTS) / MACLAURIN_POINTS)
    coefficients = np.fft.fft(np.asarray(f(z), dtype=np.complex128)) / MACLAURIN_POINTS
    return np.asarray(coefficients[: degree + 1].real, dtype=np.float64)


def source_values(source: Source, axes: Sequence[str], *coordinates: ArrayLike) -> NDArray[Any]:
    """Evaluate a source's *exact* function at points (its representation is ignored)."""
    return compile_expression(source.expression, axes)(*coordinates)


def source_coefficients(source: Source, axes: Sequence[str], ns: Sequence[int]) -> NDArray[Any]:
    r"""Return a source's normalised coefficients on registers ``ns``, as it is represented.

    ``"chebyshev"``: the truncated Chebyshev series of the function (by :func:`pihm.basis.fit`).
    ``"maclaurin"``: the exact coefficients of its degree-:math:`\bar p` Maclaurin polynomial,
    which is the paper's :math:`\sum_{p \le \bar p} c_p \mathbb{M}_{x^p}` applied to the
    constant 1 (Eq. 14; one axis only).

    Raises
    ------
    ValueError
        For a Maclaurin source on several axes, or of a degree that does not fit the register.
    """
    compiled = compile_expression(source.expression, axes)
    if source.representation == "chebyshev":
        return basis.fit(compiled, list(ns))
    if len(axes) != 1:
        raise ValueError("Maclaurin sources are defined on one axis only")
    assert source.degree is not None  # guaranteed by Source's validation
    series = maclaurin_coefficients(compiled, source.degree)
    return polynomial_coefficients(Polynomial(axes[0], tuple(float(c) for c in series)), ns[0])


def term_values(
    term: Term,
    factor_values: Callable[[Factor], NDArray[Any]],
    coordinates: Mapping[str, NDArray[Any]],
) -> NDArray[Any]:
    r"""Evaluate one term pointwise.

    Parameters
    ----------
    term
        The term.
    factor_values
        Returns the values of a factor (a field derivative) at the points.
    coordinates
        The points, one array per axis name, all of the same shape.

    Returns
    -------
    numpy.ndarray
        :math:`c\,m\,\prod_i \partial^{\alpha_i} u` at the points; a source term uses its
        exact function.
    """
    shape = np.broadcast_shapes(*(np.shape(value) for value in coordinates.values()))
    value: NDArray[Any] = np.full(shape, term.coefficient, dtype=np.float64)
    if term.multiplier is not None:
        value = value * polynomial_values(term.multiplier, coordinates[term.multiplier.axis])
    if term.source is not None:
        names = list(coordinates)
        value = value * source_values(term.source, names, *(coordinates[a] for a in names))
    for factor in term.factors:
        value = value * factor_values(factor)
    return value
