r"""The Carleman lift of a polynomial vector field to a linear system.

A polynomial system :math:`du/dt = \sum_{p \ge 1} F_p\,u^{\otimes p}` on :math:`M` components
becomes, on the monomials :math:`z_\alpha = \prod_i u_i^{\alpha_i}` of degree
:math:`1 \le |\alpha| \le K`, the linear system :math:`dz/dt = \mathbb{A}_K z`. Differentiating a
monomial,

.. math::

   \frac{dz_\alpha}{dt} = \sum_i \alpha_i\, u^{\alpha - e_i} \frac{du_i}{dt}
   = \sum_i \alpha_i \sum_p \sum_{j_1 \dots j_p} (F_p)_{i, j_1 \dots j_p}\,
     z_{\alpha - e_i + e_{j_1} + \dots + e_{j_p}},

and every monomial of degree above :math:`K` is dropped: that truncation is the Carleman
approximation. Its error is :math:`O(R^K)` with :math:`R = \|u_0\|\,\|F_2\|/|\mathrm{Re}\,
\lambda_1(F_1)|` (:func:`nonlinearity_ratio`). The truncated linear initial-value problem has a
unique solution, so a residual built on it has a one-dimensional near-kernel, in either form.

**The symmetric layout.** :math:`z` is carried once per monomial, not once per ordered tensor
index: level :math:`k` has :math:`\binom{M+k-1}{k}` components rather than :math:`M^k`. The
monomials run level by level, and within a level in the order of
:func:`itertools.combinations_with_replacement`, so the first :math:`M` are the fields themselves
in their own order (:func:`monomials`).

A linear system is the lift of order 1: its components are its fields, and
:math:`\mathbb{A}_1 = F_1`.

**The tensor layout** carries level :math:`k` as the whole tensor
:math:`u^{\otimes k}`, :math:`M^k` components in :math:`k` slots (:func:`tensor_matrix`), the
layout a register of slots has. Its matrix is a sum of structured terms that does not grow with
:math:`M` (:func:`tensor_terms`): :math:`F_p` on the slots :math:`j, \dots, j + p - 1` of every
level that occupies them, contracting them into slot :math:`j`,

.. math::

   (\mathbb{A}_K)_{k,\,k + p - 1} = \sum_{j=0}^{k-1}
   I^{\otimes j} \otimes F_p \otimes I^{\otimes (k - 1 - j)},

one term per degree and first slot, :math:`2K - 1` for a quadratic field: each slot's
:math:`F_1` serves every level that occupies it, and each pair's :math:`F_2` every level it
lowers. The symmetric layout embeds in it (:func:`symmetric_embedding`): a monomial's value sits
at every ordered index with its exponents, and the tensor layout's matrix maps that image as the
symmetric layout's does, :math:`\mathbb{A}^{\otimes}V = V\mathbb{A}_K`, because each term
reads its slots' indices as a monomial reads its exponents. The sum over :math:`j` is needed:
its terms are not equal on symmetric tensors, only their symmetrisations are.

Examples
--------
The logistic-type :math:`y' = -y + y^2` of rung R8, lifted to order 3:

>>> import numpy as np, scipy.sparse as sp
>>> field = [sp.csr_array([[-1.0]]), sp.csr_array([[1.0]])]
>>> carleman_matrix(field, 3).toarray()
array([[-1.,  1.,  0.],
       [ 0., -2.,  2.],
       [ 0.,  0., -3.]])
>>> moments([0.5], monomials(1, 3))
array([0.5  , 0.25 , 0.125])
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from itertools import combinations_with_replacement, product
from math import comb
from typing import Any

import numpy as np
import scipy.sparse as sp
from numpy.typing import ArrayLike, NDArray

from pihm.operators import SparseArray

__all__ = [
    "TensorTerm",
    "carleman_dimension",
    "carleman_matrix",
    "moments",
    "monomials",
    "nonlinearity_ratio",
    "symmetric_embedding",
    "tensor_matrix",
    "tensor_moments",
    "tensor_terms",
]


def monomials(fields: int, order: int) -> tuple[tuple[int, ...], ...]:
    """Return the exponents of every monomial of degree 1 to ``order`` in ``fields`` variables.

    Parameters
    ----------
    fields
        The number of components :math:`M` of the base state.
    order
        The lift's order :math:`K`.

    Returns
    -------
    tuple of tuple of int
        Level by level, and within a level in the order of
        :func:`itertools.combinations_with_replacement`; the first ``fields`` are the unit
        vectors, the fields themselves.

    Raises
    ------
    ValueError
        Unless both counts are at least 1.

    Examples
    --------
    >>> monomials(2, 2)
    ((1, 0), (0, 1), (2, 0), (1, 1), (0, 2))
    """
    if fields < 1 or order < 1:
        raise ValueError(f"need at least one field and order 1, got {fields} and {order}")
    found: list[tuple[int, ...]] = []
    for level in range(1, order + 1):
        for combination in combinations_with_replacement(range(fields), level):
            exponents = [0] * fields
            for index in combination:
                exponents[index] += 1
            found.append(tuple(exponents))
    return tuple(found)


def carleman_dimension(fields: int, order: int) -> int:
    r"""Return :math:`\sum_{k=1}^K \binom{M+k-1}{k}`, the count of :func:`monomials`."""
    return sum(comb(fields + level - 1, level) for level in range(1, order + 1))


def _tensor_indices(column: int, fields: int, degree: int) -> tuple[int, ...]:
    r"""Return :math:`(j_1, \dots, j_p)` of the column :math:`j_1 M^{p-1} + \dots + j_p`."""
    indices = []
    for _ in range(degree):
        column, remainder = divmod(column, fields)
        indices.append(remainder)
    return tuple(reversed(indices))


def carleman_matrix(vector_field: Sequence[SparseArray], order: int) -> sp.csr_array:
    r"""Return the truncated Carleman matrix :math:`\mathbb{A}_K` of a polynomial vector field.

    Parameters
    ----------
    vector_field
        :math:`[F_1, F_2, \dots, F_P]`, with :math:`F_p` of shape :math:`(M, M^p)` acting on
        :math:`u^{\otimes p}` (column :math:`j_1 M^{p-1} + \dots + j_p`). Any of them may be zero;
        a constant term is not a Carleman system (the vector field must vanish at
        :math:`u = 0`).
    order
        The lift's order :math:`K`.

    Returns
    -------
    scipy.sparse.csr_array
        :math:`\mathbb{A}_K` on the monomials of :func:`monomials`, square of side
        :func:`carleman_dimension`.

    Raises
    ------
    ValueError
        For an empty vector field or a term of the wrong shape.
    """
    if not vector_field:
        raise ValueError("the vector field needs at least its linear part F_1")
    fields = vector_field[0].shape[0]
    for degree, term in enumerate(vector_field, start=1):
        if term.shape != (fields, fields**degree):
            raise ValueError(
                f"F_{degree} must have shape ({fields}, {fields**degree}), got {term.shape}"
            )
    keys = monomials(fields, order)
    position = {key: index for index, key in enumerate(keys)}
    # The non-zeros of every F_p, grouped by the component i they drive.
    driving: dict[int, list[tuple[tuple[int, ...], complex]]] = {}
    for degree, term in enumerate(vector_field, start=1):
        coo = sp.coo_array(term)
        for driven, column, value in zip(coo.row, coo.col, coo.data, strict=True):
            if value != 0:
                indices = _tensor_indices(int(column), fields, degree)
                driving.setdefault(int(driven), []).append((indices, value))
    rows: list[int] = []
    columns: list[int] = []
    values: list[complex] = []
    for row, key in enumerate(keys):
        for i, exponent in enumerate(key):
            if not exponent:
                continue
            for indices, value in driving.get(i, ()):
                target = list(key)
                target[i] -= 1
                for j in indices:
                    target[j] += 1
                if sum(target) <= order:
                    rows.append(row)
                    columns.append(position[tuple(target)])
                    values.append(exponent * value)
    dtype = np.result_type(*(term.dtype for term in vector_field), np.float64)
    size = len(keys)
    matrix = sp.coo_array(
        (np.asarray(values, dtype=dtype), (np.asarray(rows), np.asarray(columns))),
        shape=(size, size),
    )
    return sp.csr_array(matrix)


def moments(data: ArrayLike, keys: Sequence[tuple[int, ...]]) -> NDArray[np.float64]:
    r"""Return :math:`z_\alpha = \prod_i y_i^{\alpha_i}` for each monomial, the lifted data."""
    y = np.atleast_1d(np.asarray(data, dtype=np.float64))
    return np.array([float(np.prod(y ** np.asarray(key))) for key in keys])


def nonlinearity_ratio(vector_field: Sequence[SparseArray], data: ArrayLike) -> float:
    r"""Return :math:`R = \|u_0\|\,\|F_2\|_2/|\mathrm{Re}\,\lambda_1(F_1)|`, the lift's error ratio.

    :math:`\lambda_1(F_1)` is the eigenvalue of the linear part with the largest real part, the
    least damped mode. :math:`R < 1` is the regime in which the truncation error falls
    geometrically with the order; a linear field has :math:`R = 0`.
    :math:`F_2` acts only on :math:`u\otimes u`, so its norm is taken on the symmetric tensors,
    the norm of its symmetrised form, whichever order the equations write each product in.

    Raises
    ------
    ValueError
        If the least damped mode is not damped, where :math:`R` is undefined.
    """
    y = np.atleast_1d(np.asarray(data, dtype=np.float64))
    if len(vector_field) < 2 or vector_field[1].nnz == 0:
        return 0.0
    linear = np.asarray(sp.csr_array(vector_field[0]).toarray())
    damping = float(np.max(np.linalg.eigvals(linear).real))
    if damping >= 0.0:
        raise ValueError("the ratio R needs a damped linear part (every Re lambda < 0)")
    quadratic = np.asarray(sp.csr_array(vector_field[1]).toarray())
    fields = quadratic.shape[0]
    swapped = quadratic.reshape(fields, fields, fields).transpose(0, 2, 1).reshape(fields, -1)
    symmetric = (quadratic + swapped) / 2.0
    return float(np.linalg.norm(y) * np.linalg.norm(symmetric, 2) / abs(damping))


# ---------------------------------------------------------------------------- the tensor layout


@dataclass(frozen=True)
class TensorTerm:
    r"""One structured term of the tensor layout: :math:`F_p` on the slots :math:`j, \dots, j+p-1`.

    It acts on every level :math:`k \ge j + p`, which occupies those slots, contracting them into
    slot :math:`j` and moving the slots above down by :math:`p - 1`, so level :math:`k` goes to
    :math:`k - p + 1`.

    Attributes
    ----------
    degree
        :math:`p`, the degree of the field's term.
    slot
        :math:`j`, the first slot it reads and the one it writes.
    """

    degree: int
    slot: int

    def levels(self, order: int) -> range:
        """Return the input levels it acts on, those that occupy its slots."""
        return range(self.slot + self.degree, order + 1)


def tensor_terms(order: int, degrees: Sequence[int] = (1, 2)) -> tuple[TensorTerm, ...]:
    r"""Return the tensor layout's terms: one per degree :math:`p` and first slot :math:`j`.

    :math:`j + p \le K`, so there are :math:`\sum_p (K - p + 1)`, :math:`2K - 1` for a quadratic
    field, whatever the number of fields.
    """
    if order < 1:
        raise ValueError(f"the lift's order must be at least 1, got {order}")
    return tuple(
        TensorTerm(degree, slot) for degree in degrees for slot in range(order - degree + 1)
    )


def _offsets(fields: int, order: int) -> list[int]:
    """Return where each level of the tensor layout starts, level 1 first, then its end."""
    starts = [0]
    for level in range(1, order + 1):
        starts.append(starts[-1] + fields**level)
    return starts


def tensor_matrix(vector_field: Sequence[SparseArray], order: int) -> sp.csr_array:
    r"""Return :math:`\mathbb{A}_K` in the tensor layout, summed from :func:`tensor_terms`.

    Level :math:`k` holds :math:`u^{\otimes k}` in :math:`M^k` components, levels in order, slot
    0 the most significant. The side is :math:`\sum_k M^k`.
    """
    fields = vector_field[0].shape[0]
    starts = _offsets(fields, order)
    rows: list[NDArray[np.integer[Any]]] = []
    columns: list[NDArray[np.integer[Any]]] = []
    values: list[NDArray[np.generic]] = []
    for term in tensor_terms(order, range(1, len(vector_field) + 1)):
        F = sp.csr_array(vector_field[term.degree - 1])
        for level in term.levels(order):
            below = sp.eye_array(fields**term.slot, format="csr")
            above = sp.eye_array(fields ** (level - term.slot - term.degree), format="csr")
            block = sp.coo_array(sp.kron(sp.kron(below, F), above))
            rows.append(block.row + starts[level - term.degree])
            columns.append(block.col + starts[level - 1])
            values.append(block.data)
    size = starts[-1]
    matrix = sp.coo_array(
        (np.concatenate(values), (np.concatenate(rows), np.concatenate(columns))),
        shape=(size, size),
    )
    return sp.csr_array(matrix)


def symmetric_embedding(fields: int, order: int) -> sp.csr_array:
    r"""Return :math:`V`, the symmetric layout's components placed in the tensor layout.

    Column :math:`\alpha` (of :func:`monomials`) has a one at every ordered index of level
    :math:`|\alpha|` whose slots hold the exponents :math:`\alpha`, so
    :math:`V\,z(u) = (u, u^{\otimes 2}, \dots)` for the moments :math:`z(u)`.
    """
    keys = monomials(fields, order)
    position = {key: index for index, key in enumerate(keys)}
    starts = _offsets(fields, order)
    rows: list[int] = []
    columns: list[int] = []
    for level in range(1, order + 1):
        for flat, indices in enumerate(product(range(fields), repeat=level)):
            exponents = [0] * fields
            for index in indices:
                exponents[index] += 1
            rows.append(starts[level - 1] + flat)
            columns.append(position[tuple(exponents)])
    return sp.csr_array((np.ones(len(rows)), (rows, columns)), shape=(starts[-1], len(keys)))


def tensor_moments(data: ArrayLike, order: int) -> NDArray[np.float64]:
    r"""Return :math:`(u, u^{\otimes 2}, \dots, u^{\otimes K})`, the tensor layout's lifted data."""
    u = np.atleast_1d(np.asarray(data, dtype=np.float64))
    levels: list[NDArray[np.floating[Any]]] = [u]
    for _ in range(order - 1):
        levels.append(np.kron(levels[-1], u))
    return np.concatenate(levels).astype(np.float64)
