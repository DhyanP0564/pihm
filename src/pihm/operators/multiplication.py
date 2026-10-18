r"""Multiplication by a polynomial coefficient :math:`a(x)`: the exact lift and the square form.

Multiplying a degree-:math:`(N-1)` series by :math:`a(x)` raises its degree by
:math:`\deg a`, so the exact product needs a larger register. The paper lifts one qubit, which
is exact whenever :math:`\deg a \le N`:

.. math::

   a(x)\,\langle\tau(x)|_n = \langle\tau(x)|_{n+1}\,\mathbb{M}_a,
   \qquad \mathbb{M}_a \in \mathbb{R}^{2N \times N}

The *square* form is the Galerkin
truncation of the same exact product to the first :math:`N` modes, expressed in the
:math:`n`-qubit basis, which is what a same-size residual (the descriptor regime) uses. Every
:math:`s_k` grows by :math:`\sqrt2` from :math:`n` to :math:`n + 1` qubits, so the square form
equals the lift's first :math:`N` rows divided by :math:`\sqrt2`. It is banded with bandwidth
:math:`\deg a`.

Both forms follow from :math:`T_m T_k = \tfrac12 (T_{m+k} + T_{|m-k|})`, applied to the
Chebyshev series of :math:`a` and conjugated into the normalised basis:
:math:`\mathbb{M}_{jk} = s^{\rm out}_j X_{jk} / s^{\rm in}_k`.

.. note::

   The earlier square operator was a power of the truncated :math:`\mathbb{M}_x`, which
   truncates at every step. That agrees with the square form here in columns
   :math:`k \le N - \deg a` and differs only in the last :math:`\deg a - 1` columns, where
   truncating before the last factor drops terms that the exact product keeps.
"""

from __future__ import annotations

from typing import Any, Literal, TypeAlias

import numpy as np
import scipy.sparse as sp
from numpy.polynomial import chebyshev as npcheb
from numpy.typing import ArrayLike, NDArray

from pihm.basis import chebyshev_scale

__all__ = [
    "CoefficientBasis",
    "MultiplicationMode",
    "embedding",
    "multiplication",
    "power_of_x",
    "raw_product",
]

#: How a multiplication operator is shaped. ``"lift"`` is the exact map from :math:`n` to
#: :math:`n + 1` qubits (the paper's convention); ``"square"`` is its Galerkin truncation on
#: :math:`n` qubits.
MultiplicationMode: TypeAlias = Literal["lift", "square"]

#: The basis of the multiplier's expansion, in powers :math:`x^j` or polynomials :math:`T_j`.
CoefficientBasis: TypeAlias = Literal["monomial", "chebyshev"]


def raw_product(a: ArrayLike, size: int, rows: int) -> sp.csr_array:
    r"""Return the product with :math:`a(x) = \sum_m a_m T_m(x)` on *standard* coefficients.

    Column :math:`k` holds the standard Chebyshev coefficients of :math:`a(x)\,T_k(x)`, from
    :math:`T_m T_k = \tfrac12 (T_{m+k} + T_{|m-k|})`.

    Parameters
    ----------
    a
        Standard Chebyshev coefficients of the multiplier.
    size
        Number of input modes (columns).
    rows
        Number of output modes kept; terms of higher degree are dropped.

    Returns
    -------
    scipy.sparse.csr_array
        The ``(rows, size)`` product matrix.
    """
    coefficients = np.atleast_1d(np.asarray(a))
    m = np.arange(coefficients.size)[:, None]
    k = np.arange(size)[None, :]
    targets = np.concatenate([(m + k).ravel(), np.abs(m - k).ravel()])
    columns = np.concatenate([np.broadcast_to(k, (m.size, size)).ravel()] * 2)
    weights = np.concatenate(
        [np.broadcast_to(coefficients[:, None] / 2, (m.size, size)).ravel()] * 2
    )
    keep = (targets < rows) & (weights != 0)
    return sp.csr_array(
        (weights[keep], (targets[keep], columns[keep])),
        shape=(rows, size),
        dtype=np.result_type(coefficients.dtype, np.float64),
    )


def _chebyshev_series(coefficients: ArrayLike, basis: CoefficientBasis) -> NDArray[Any]:
    """Return the multiplier's standard Chebyshev coefficients, trimmed of trailing zeros."""
    values = np.atleast_1d(np.asarray(coefficients))
    if values.ndim != 1 or values.size == 0:
        raise ValueError("coefficients must be a non-empty one-dimensional sequence")
    if basis == "monomial":
        values = npcheb.poly2cheb(values)
    elif basis != "chebyshev":
        raise ValueError(f"basis must be 'monomial' or 'chebyshev', got {basis!r}")
    nonzero = np.flatnonzero(values)
    return values[: nonzero[-1] + 1] if nonzero.size else values[:1] * 0


def multiplication(
    coefficients: ArrayLike,
    n: int,
    *,
    basis: CoefficientBasis = "monomial",
    mode: MultiplicationMode = "lift",
) -> sp.csr_array:
    r"""Return the operator that multiplies a field by :math:`a(x)`.

    Parameters
    ----------
    coefficients
        The multiplier :math:`a(x) = \sum_j a_j x^j` (``basis="monomial"``) or
        :math:`\sum_j a_j T_j(x)` (``basis="chebyshev"``); real or complex.
    n
        Qubits in the input register.
    basis
        The expansion in which ``coefficients`` are given.
    mode
        ``"lift"`` for the exact ``(2N, N)`` map to :math:`n + 1` qubits, ``"square"`` for the
        ``(N, N)`` Galerkin truncation in the :math:`n`-qubit basis.

    Returns
    -------
    scipy.sparse.csr_array
        The operator in the normalised basis, float64 or complex128.

    Raises
    ------
    ValueError
        If ``mode="lift"`` and :math:`\deg a > N`, when the product does not fit in one extra
        qubit.

    Examples
    --------
    The paper's :math:`\mathbb{M}_1` embeds the :math:`n`-qubit basis into :math:`n+1` qubits:

    >>> multiplication([1.0], 1).toarray()
    array([[1.41421356, 0.        ],
           [0.        , 1.41421356],
           [0.        , 0.        ],
           [0.        , 0.        ]])
    """
    series = _chebyshev_series(coefficients, basis)
    size = int(chebyshev_scale(n).size)
    degree = series.size - 1
    if mode == "lift":
        if degree > size:
            raise ValueError(
                f"deg a = {degree} exceeds N = {size}; the product needs a larger lift"
            )
        rows, n_out = 2 * size, n + 1
    elif mode == "square":
        rows, n_out = size, n
    else:
        raise ValueError(f"mode must be 'lift' or 'square', got {mode!r}")
    X = raw_product(series, size, rows)
    return sp.csr_array(
        sp.diags_array(chebyshev_scale(n_out)[:rows]) @ X @ sp.diags_array(1.0 / chebyshev_scale(n))
    )


def power_of_x(p: int, n: int, *, mode: MultiplicationMode = "lift") -> sp.csr_array:
    r"""Return :math:`\mathbb{M}_{x^p}`, the multiplication by :math:`x^p` (a lift by default).

    Examples
    --------
    >>> bool(abs(power_of_x(1, 2)[1, 0] - 1.0) < 1e-15)  # x T_0 = T_1
    True
    """
    if p < 0:
        raise ValueError(f"p must be non-negative, got {p}")
    monomial = np.zeros(p + 1)
    monomial[p] = 1.0
    return multiplication(monomial, n, basis="monomial", mode=mode)


def embedding(n: int, levels: int = 1) -> sp.csr_array:
    r"""Return the embedding of the :math:`n`-qubit basis into :math:`n + \ell` qubits.

    The same function has coefficients :math:`2^{\ell/2} b` (zero-padded) in the larger register,
    because every :math:`s_k` scales by :math:`2^{\ell/2}`. For :math:`\ell = 1` this is the
    paper's :math:`\mathbb{M}_1 = \sqrt2\,[I;\,0]`.

    Parameters
    ----------
    n
        Qubits in the input register.
    levels
        Extra qubits :math:`\ell \ge 0`.

    Returns
    -------
    scipy.sparse.csr_array
        The ``(2^\ell N, N)`` embedding.
    """
    if levels < 0:
        raise ValueError(f"levels must be non-negative, got {levels}")
    size = int(chebyshev_scale(n).size)
    index = np.arange(size)
    return sp.csr_array(
        (np.full(size, 2.0 ** (levels / 2)), (index, index)), shape=(size << levels, size)
    )
