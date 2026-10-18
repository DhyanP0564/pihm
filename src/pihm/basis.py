r"""The normalised Chebyshev basis of the latent model and its conventions.

A register of :math:`n` qubits carries :math:`N = 2^n` Chebyshev coefficients. The model is
:math:`f_q(x) = \langle\tau(x)|\psi\rangle`, with the **raw** feature vector

.. math::

   \tau(x)_k = \phi_k(x) = T_k(x) / s_k, \qquad s_0 = 2^{n/2}, \quad s_{k \ge 1} = 2^{(n-1)/2},

exactly as the paper prints it :cite:`wu2025pihm` (its supplementary §A). The feature vector is
*not* unit-norm. Renormalising it point by point would change the constraint weights, so this
library never does.

Coefficient vectors come in two conventions. Standard Chebyshev coefficients :math:`c`, with
:math:`f = \sum_k c_k T_k`, map to normalised ones :math:`b`, with :math:`f = \sum_k b_k \phi_k`,
by :math:`b_k = s_k c_k`. Fields with several axes are flattened in Kronecker order, with axis 0
the most significant (slowest-varying) index.

Examples
--------
The constant function 1 has normalised coefficients :math:`\sqrt{N}\,e_0`:

>>> import numpy as np
>>> b = np.zeros(8)
>>> b[0] = np.sqrt(8)
>>> float(evaluate(b, 0.37))
1.0
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
import scipy.fft
from numpy.polynomial import chebyshev as npcheb
from numpy.typing import ArrayLike, NDArray

__all__ = [
    "FIT_OVERSAMPLING",
    "chebyshev_gauss_transform",
    "chebyshev_scale",
    "evaluate",
    "fit",
    "nodes",
    "scale_factor",
    "tau",
    "to_chebyshev",
    "to_normalised",
]

#: The default number of interpolation nodes per retained mode in :func:`fit`.
FIT_OVERSAMPLING = 64


def _check_qubits(n: int) -> int:
    """Validate a qubit count and return :math:`N = 2^n`."""
    if isinstance(n, bool) or not isinstance(n, (int, np.integer)) or n < 0:
        raise ValueError(f"the qubit count must be a non-negative integer, got {n!r}")
    return 1 << int(n)


def chebyshev_scale(n: int) -> NDArray[np.float64]:
    r"""Return the normalisation :math:`s_k`, so that :math:`\phi_k = T_k / s_k`.

    Parameters
    ----------
    n
        Qubits in the register; the result has :math:`N = 2^n` entries.

    Returns
    -------
    numpy.ndarray
        :math:`s_0 = 2^{n/2}` and :math:`s_k = 2^{(n-1)/2}` for :math:`k \ge 1`.

    Examples
    --------
    >>> chebyshev_scale(2)
    array([2.        , 1.41421356, 1.41421356, 1.41421356])
    """
    size = _check_qubits(n)
    scale = np.full(size, 2.0 ** ((n - 1) / 2))
    scale[0] = 2.0 ** (n / 2)
    return scale


def _as_points(x: float | ArrayLike) -> NDArray[np.float64]:
    """Return ``x`` as a float64 array of points in :math:`[-1, 1]`."""
    points = np.asarray(x, dtype=np.float64)
    if not np.all(np.isfinite(points)) or np.any(np.abs(points) > 1.0):
        raise ValueError("Chebyshev points must lie in [-1, 1]")
    return points


def tau(x: float | ArrayLike, n: int) -> NDArray[np.float64]:
    r"""Return the raw feature vector :math:`\tau(x)` (not normalised).

    Parameters
    ----------
    x
        A point in :math:`[-1, 1]`, or an array of points.
    n
        Qubits in the register.

    Returns
    -------
    numpy.ndarray
        Shape ``(N,)`` for a scalar ``x``, otherwise ``x.shape + (N,)``; always float64.

    Raises
    ------
    ValueError
        If a point lies outside :math:`[-1, 1]`.

    Notes
    -----
    The polynomials are evaluated by the three-term recurrence
    (:func:`numpy.polynomial.chebyshev.chebvander`), which is stable on :math:`[-1, 1]`.
    """
    points = _as_points(x)
    size = _check_qubits(n)
    # chebvander promotes a scalar to one dimension; restore the documented shape.
    values = npcheb.chebvander(points.reshape(-1), size - 1) / chebyshev_scale(n)
    return values.reshape((*points.shape, size))


def evaluate(coefficients: ArrayLike, x: float | ArrayLike) -> NDArray[Any]:
    r"""Evaluate :math:`f(x) = \langle\tau(x)|b\rangle` from normalised coefficients ``b``.

    Parameters
    ----------
    coefficients
        Normalised coefficients, length :math:`N = 2^n`.
    x
        A point or array of points in :math:`[-1, 1]`.

    Returns
    -------
    numpy.ndarray
        The values, with the shape of ``x``.
    """
    b = np.asarray(coefficients)
    size = b.shape[-1]
    n = size.bit_length() - 1
    if size != 1 << n:
        raise ValueError(f"the coefficient count must be a power of two, got {size}")
    return np.asarray(tau(x, n) @ b)


def to_normalised(c: ArrayLike, n: int) -> NDArray[Any]:
    """Convert standard Chebyshev coefficients to normalised ones, :math:`b_k = s_k c_k`."""
    return np.asarray(c) * chebyshev_scale(n)


def to_chebyshev(b: ArrayLike, n: int) -> NDArray[Any]:
    """Convert normalised coefficients to standard Chebyshev ones, :math:`c_k = b_k / s_k`."""
    return np.asarray(b) / chebyshev_scale(n)


def nodes(n: int) -> NDArray[np.float64]:
    r"""Return the Chebyshev–Gauss nodes :math:`\bar x_i = \cos(\pi(2i+1)/2N)`, :math:`i < N`."""
    size = _check_qubits(n)
    return np.cos(np.pi * (2 * np.arange(size) + 1) / (2 * size))


def chebyshev_gauss_transform(n: int) -> NDArray[np.float64]:
    r"""Return the orthogonal Chebyshev–Gauss transform :math:`\mathbb{U}`.

    Row :math:`i` of :math:`\mathbb{U}` is the raw feature vector at node :math:`\bar x_i`, so
    :math:`\mathbb{U} b` is the vector of values of :math:`f` at the nodes. Discrete Chebyshev
    orthogonality makes :math:`\mathbb{U}` orthogonal: the feature states at the nodes are
    orthonormal. The nodal product fold is built on it.

    Parameters
    ----------
    n
        Qubits in the register.

    Returns
    -------
    numpy.ndarray
        The ``(N, N)`` orthogonal matrix :math:`\mathbb{U}_{ij} = \tau(\bar x_i)_j`.

    Examples
    --------
    >>> U = chebyshev_gauss_transform(3)
    >>> bool(np.allclose(U @ U.T, np.eye(8)))
    True

    Notes
    -----
    The entries are computed as :math:`\cos(\pi m / 2N)/s_k` with the integer
    :math:`m = k(2i+1) \bmod 4N`, so every angle lies in :math:`[0, 2\pi)` and each entry is
    accurate to rounding. The three-term recurrence used by :func:`tau` accumulates an error of
    order :math:`k\varepsilon`, which would show as a loss of orthogonality of order
    :math:`10^{-12}` by :math:`n = 8`.
    """
    size = _check_qubits(n)
    k = np.arange(size, dtype=np.int64)
    m = (k[None, :] * (2 * k[:, None] + 1)) % (4 * size)
    return np.cos(np.pi * m / (2 * size)) / chebyshev_scale(n)


def _grid(points: int) -> NDArray[np.float64]:
    """Return ``points`` Chebyshev–Gauss nodes (any count, not only powers of two)."""
    return np.cos(np.pi * (2 * np.arange(points) + 1) / (2 * points))


def fit(
    f: Callable[..., ArrayLike],
    n: int | Sequence[int],
    *,
    points: int | Sequence[int] | None = None,
) -> NDArray[Any]:
    r"""Return the normalised coefficients of ``f``, truncated to :math:`N` terms per axis.

    ``f`` is interpolated at many Chebyshev–Gauss nodes per axis (by a type-II discrete cosine
    transform), and the series is truncated to the register size. The result is therefore the
    truncation of ``f``'s Chebyshev series, accurate to the aliasing error of the tail beyond
    ``points``. For a polynomial of degree below ``points`` it is exact to rounding.

    Parameters
    ----------
    f
        A vectorised function. For one axis it takes an array of points; for :math:`d` axes it
        takes :math:`d` arrays from a ``numpy.meshgrid(..., indexing="ij")``.
    n
        Qubits per axis: an integer for one axis, or a sequence for several.
    points
        Interpolation nodes per axis; by default :data:`FIT_OVERSAMPLING` times the register
        size (at least 256).

    Returns
    -------
    numpy.ndarray
        The normalised coefficients, flattened in Kronecker order (axis 0 slowest); length
        :math:`\prod_a N_a`.

    Examples
    --------
    >>> b = fit(lambda x: 3 * x**2, 2)  # 3x^2 = 1.5 T_0 + 1.5 T_2
    >>> bool(np.allclose(to_chebyshev(b, 2), [1.5, 0.0, 1.5, 0.0]))
    True
    """
    ns = [int(n)] if isinstance(n, (int, np.integer)) else [int(k) for k in n]
    sizes = [_check_qubits(k) for k in ns]
    if points is None:
        counts = [max(FIT_OVERSAMPLING * size, 256) for size in sizes]
    elif isinstance(points, (int, np.integer)):
        counts = [int(points)] * len(ns)
    else:
        counts = [int(p) for p in points]
    if any(count < size for count, size in zip(counts, sizes, strict=True)):
        raise ValueError("need at least as many interpolation points as retained modes")
    grids = np.meshgrid(*[_grid(count) for count in counts], indexing="ij")
    values = np.asarray(f(*grids))
    if values.shape != tuple(counts):
        values = np.broadcast_to(values, tuple(counts))
    coefficients = values
    for axis, count in enumerate(counts):
        # DCT-II: y_k = 2 sum_j v_j cos(pi k (2j+1) / 2m), so c_k = y_k / m and c_0 = y_0 / 2m.
        coefficients = scipy.fft.dct(coefficients, type=2, axis=axis) / count
        index: list[slice | int] = [slice(None)] * len(counts)
        index[axis] = 0
        coefficients[tuple(index)] /= 2.0
    truncated = coefficients[tuple(slice(0, size) for size in sizes)]
    for axis, k in enumerate(ns):
        shape = [1] * len(ns)
        shape[axis] = sizes[axis]
        truncated = truncated * chebyshev_scale(k).reshape(shape)
    return np.ascontiguousarray(truncated).reshape(-1)


def scale_factor(
    f: Callable[..., ArrayLike],
    n: int | Sequence[int],
    *,
    points: int | Sequence[int] | None = None,
) -> float:
    r"""Return the paper's scale factor :math:`\eta_e = \|b\|^2` for the solution ``f``.

    With :math:`b` the normalised coefficients of ``f`` truncated at the register size, the
    model :math:`f_Q = \sqrt{\eta}\,\langle\tau|\psi\rangle` with :math:`\psi = b/\|b\|`
    reproduces the truncated series exactly. By Parseval,

    .. math::

       \|b\|^2 \to \frac{N}{\pi}\int_{-1}^{1} \frac{f(x)^2}{\sqrt{1-x^2}}\,dx

    as the tail vanishes, so :math:`\eta_e` grows like :math:`N`. The paper's printed values
    for its ODE figures equal this quantity.

    Parameters
    ----------
    f
        The solution, vectorised as for :func:`fit`.
    n
        Qubits per axis.
    points
        Interpolation nodes per axis, as for :func:`fit`.

    Returns
    -------
    float
        :math:`\|b\|^2`.
    """
    b = fit(f, n, points=points)
    return float(np.vdot(b, b).real)
