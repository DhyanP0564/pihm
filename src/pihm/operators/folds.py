r"""Product folds: the pointwise product of several copies of a field, as one linear map.

For :math:`p` copies of a field on :math:`n` qubits, the product has degree up to
:math:`p(N-1)`, so it lives on :math:`n + \ell` qubits with :math:`\ell = \lceil\log_2 p\rceil`:

.. math::

   \langle\tau(x)|_n^{\otimes p} = \langle\tau(x)|_{n+\ell}\,\mathbb{N}_1^{(p)}, \qquad
   x\,\langle\tau(x)|_n^{\otimes 2} = \langle\tau(x)|_{n+1}\,\mathbb{N}_x

Two constructions of the same map are provided,
because they are the two routes a circuit can take:

``"coefficient"``
    Repeated product-to-sum, :math:`T_jT_k = \tfrac12 (T_{j+k} + T_{|j-k|})`: exact and sparse.
``"nodal"``
    :math:`2^{p\ell/2}\,\mathbb{U}^T\Delta_p(\mathbb{U}E)^{\otimes p}`. Here :math:`\mathbb{U}` is
    the Chebyshev–Gauss transform on :math:`n + \ell` qubits, :math:`E` the zero-padding and
    :math:`\Delta_p = \sum_i |i\rangle\langle i|^{\otimes p}`. The product is pointwise at the
    nodes, which is what makes the fold cheap on a circuit (0 ancilla, :math:`\alpha = 2` for
    :math:`p = 2`).

On inputs of degree below :math:`N` the two agree to rounding (the dealiasing condition).
Fields with several axes fold axis by axis. The columns are ordered copy by copy, with copy 1
most significant and, within a copy, axis 0 most significant (:mod:`pihm.operators.tensor`).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Literal, TypeAlias

import numpy as np
import scipy.sparse as sp

from pihm.basis import chebyshev_gauss_transform, chebyshev_scale, nodes
from pihm.operators.multiplication import raw_product
from pihm.operators.tensor import axis_sizes, kron_all

__all__ = ["FoldBasis", "FoldWeight", "fold", "fold_levels"]

#: The construction of the fold, by repeated product-to-sum or pointwise at Chebyshev nodes.
FoldBasis: TypeAlias = Literal["coefficient", "nodal"]

#: An extra factor in the product, either none or :math:`x` (the paper's :math:`\mathbb{N}_x`).
FoldWeight: TypeAlias = Literal["none", "x"]


def fold_levels(degree: int) -> int:
    r"""Return the lift :math:`\ell = \lceil\log_2 p\rceil` that a degree-:math:`p` product needs.

    Multiplying by :math:`x` as well (the ``"x"`` weight) never needs more: the product has
    degree at most :math:`p(N-1) + 1 < 2^\ell N` whenever :math:`p \ge 2`.
    """
    if degree < 2:
        raise ValueError(f"a fold needs at least two copies, got degree {degree}")
    return math.ceil(math.log2(degree))


def _pairwise(left: int, right: int) -> sp.csr_array:
    """Raw product of a ``left``-mode and a ``right``-mode series, as a map on ``kron`` inputs."""
    i = np.repeat(np.arange(left), right)
    k = np.tile(np.arange(right), left)
    columns = np.arange(left * right)
    rows = np.concatenate([i + k, np.abs(i - k)])
    return sp.csr_array(
        (np.full(2 * columns.size, 0.5), (rows, np.concatenate([columns, columns]))),
        shape=(left + right - 1, left * right),
    )


def _coefficient_fold(degree: int, n: int, weight: FoldWeight, levels: int) -> sp.csr_array:
    """Return the single-axis fold built by repeated product-to-sum."""
    size = 1 << n
    rows_out = size << levels
    raw = sp.csr_array(sp.eye_array(size, format="csr"))
    for _ in range(degree - 1):
        raw = sp.csr_array(
            _pairwise(raw.shape[0], size) @ sp.kron(raw, sp.eye_array(size), format="csr")
        )
    if weight == "x":
        raw = sp.csr_array(raw_product([0.0, 1.0], raw.shape[0], raw.shape[0] + 1) @ raw)
    padded = sp.csr_array(sp.vstack([raw, sp.csr_array((rows_out - raw.shape[0], raw.shape[1]))]))
    column_scale = kron_all([sp.diags_array(1.0 / chebyshev_scale(n))] * degree)
    return sp.csr_array(sp.diags_array(chebyshev_scale(n + levels)) @ padded @ column_scale)


def _nodal_fold(degree: int, n: int, weight: FoldWeight, levels: int) -> sp.csr_array:
    """Return the single-axis fold built pointwise at the Chebyshev–Gauss nodes."""
    size = 1 << n
    U = chebyshev_gauss_transform(n + levels)
    values = U[:, :size]
    products = values
    for _ in range(degree - 1):
        products = (products[:, :, None] * values[:, None, :]).reshape(values.shape[0], -1)
    if weight == "x":
        products = nodes(n + levels)[:, None] * products
    return sp.csr_array(2.0 ** (degree * levels / 2) * (U.T @ products))


def _copy_major(folded: sp.csr_array, sizes: tuple[int, ...], degree: int) -> sp.csr_array:
    """Reorder columns from axis-major (as :func:`kron_all` makes them) to copy-major."""
    shape = [size for size in sizes for _ in range(degree)]
    order = [axis * degree + copy for copy in range(degree) for axis in range(len(sizes))]
    permutation = np.arange(int(np.prod(shape))).reshape(shape).transpose(order).ravel()
    return sp.csr_array(folded[:, permutation])


def fold(
    degree: int,
    ns: int | Sequence[int],
    *,
    basis: FoldBasis = "coefficient",
    weight: FoldWeight = "none",
) -> sp.csr_array:
    r"""Return the fold :math:`\mathbb{N}` that maps :math:`p` copies of a field to their product.

    Parameters
    ----------
    degree
        Number of copies :math:`p \ge 2` (2 for a quadratic term, 3 for a cubic one).
    ns
        Qubits per axis of each copy: an integer for one axis, or a sequence for several.
    basis
        The construction, ``"coefficient"`` or ``"nodal"``. Both give the same map, to rounding.
    weight
        ``"x"`` multiplies the product by :math:`x` (the paper's :math:`\mathbb{N}_x`; one axis
        only).

    Returns
    -------
    scipy.sparse.csr_array
        The map from the ``kron`` of :math:`p` copies (each of :math:`\prod_a N_a` coefficients)
        to the product's coefficients on :math:`n_a + \ell` qubits per axis. Its shape is
        :math:`(\prod_a 2^\ell N_a,\ (\prod_a N_a)^p)`.

    Raises
    ------
    ValueError
        For fewer than two copies, an unknown mode, or the ``"x"`` weight on several axes.

    Examples
    --------
    For :math:`n = 1`, with primes marking the two-qubit basis: :math:`\tau_0^2 = \tau_0'`,
    :math:`\tau_0\tau_1 = \tau_1'` and :math:`\tau_1^2 = x^2 = \tau_0' + \tau_2'/\sqrt2`.

    >>> fold(2, 1).toarray().round(4)
    array([[1.    , 0.    , 0.    , 1.    ],
           [0.    , 1.    , 1.    , 0.    ],
           [0.    , 0.    , 0.    , 0.7071],
           [0.    , 0.    , 0.    , 0.    ]])
    """
    levels = fold_levels(degree)
    sizes = axis_sizes(ns)
    counts = [size.bit_length() - 1 for size in sizes]
    if weight not in ("none", "x"):
        raise ValueError(f"weight must be 'none' or 'x', got {weight!r}")
    if weight == "x" and len(counts) != 1:
        raise ValueError("the 'x' weight is defined for a single axis only")
    builders = {"coefficient": _coefficient_fold, "nodal": _nodal_fold}
    if basis not in builders:
        raise ValueError(f"basis must be one of {sorted(builders)}, got {basis!r}")
    per_axis = [builders[basis](degree, n, weight, levels) for n in counts]
    if len(per_axis) == 1:
        return per_axis[0]
    return _copy_major(kron_all(per_axis), sizes, degree)
