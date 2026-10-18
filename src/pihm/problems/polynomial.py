r"""A first-order system's equations as a polynomial vector field.

A system here is :math:`u_i' = \sum_p (F_p u^{\otimes p})_i` on one axis, one equation per field,
each holding the first derivative of its own field with a constant coefficient and otherwise only
constant multiples of products of the fields (:func:`vector_field`). The Carleman lift builds on
it (:mod:`pihm.assemble.systems`), and so does an initial-value reference, whose right-hand side
and Jacobian it gives in sparse form (:func:`slope`, :func:`jacobian`;
:mod:`pihm.problems.reference`).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import scipy.sparse as sp
from numpy.typing import ArrayLike, NDArray

from pihm.problems.spec import Problem, Term

__all__ = ["jacobian", "slope", "vector_field"]


def _constant(term: Term) -> float:
    """Return a term's constant factor, refusing a non-constant multiplier or a source."""
    if term.source is not None:
        raise NotImplementedError("a system's equations take no sources")
    if term.multiplier is None:
        return term.coefficient
    if not term.multiplier.is_constant:
        raise NotImplementedError("a system's coefficients must be constant")
    return term.coefficient * term.multiplier.coefficients[0]


def vector_field(problem: Problem) -> list[sp.csr_array]:
    r"""Return :math:`[F_1, \dots, F_P]`, the problem as :math:`u' = \sum_p F_p u^{\otimes p}`.

    Parameters
    ----------
    problem
        A problem on one axis whose equation :math:`i` holds :math:`c\,u_i'` once, with a constant
        :math:`c`, and otherwise constant multiples of products of the fields.

    Returns
    -------
    list of scipy.sparse.csr_array
        :math:`F_p` of shape :math:`(M, M^p)`, acting on :math:`u^{\otimes p}` (column
        :math:`j_1 M^{p-1} + \dots + j_p`), for :math:`p` up to the problem's degree.

    Raises
    ------
    NotImplementedError
        For a problem that is not such a system: several axes, a derivative of order other than
        one, a field whose derivative appears in no equation or in two, a source, or a
        non-constant coefficient.
    """
    if len(problem.axes) != 1:
        raise NotImplementedError("systems are implemented on one axis")
    names = [field.name for field in problem.fields]
    count = len(names)
    rows: list[dict[tuple[int, int], float]] = [{} for _ in range(problem.degree)]
    driven: set[int] = set()
    for equation in problem.equations:
        tops = [t for t in equation.terms if any(any(f.derivative) for f in t.factors)]
        if len(tops) != 1 or tops[0].degree != 1 or tops[0].factors[0].derivative != (1,):
            raise NotImplementedError(
                "each equation of a system holds its field's first derivative once, alone"
            )
        (top,) = tops
        i = names.index(top.factors[0].field)
        if i in driven:
            raise NotImplementedError(f"the derivative of {names[i]!r} appears in two equations")
        driven.add(i)
        leading = _constant(top)
        for term in equation.terms:
            if term is top:
                continue
            if not term.factors:
                raise NotImplementedError("a system's equations are homogeneous: no sources")
            column = 0
            for factor in term.factors:
                column = column * count + names.index(factor.field)
            key = (i, column)
            entries = rows[term.degree - 1]
            entries[key] = entries.get(key, 0.0) - _constant(term) / leading
    result = []
    for degree, entries in enumerate(rows, start=1):
        coordinates = np.array(list(entries), dtype=np.int64).reshape(-1, 2)
        result.append(
            sp.csr_array(
                (list(entries.values()), (coordinates[:, 0], coordinates[:, 1])),
                shape=(count, count**degree),
            )
        )
    return result


#: A term's non-zeros: their rows, the tensor indices of their columns, and their values.
_Entries = tuple[NDArray[Any], NDArray[Any], NDArray[Any]]


def _indices(field: Sequence[sp.csr_array]) -> list[_Entries]:
    """Return each term's non-zeros: rows, the tensor indices of their columns, and values."""
    count = field[0].shape[0]
    found = []
    for degree, term in enumerate(field, start=1):
        coo = sp.coo_array(term)
        columns = np.asarray(coo.col, dtype=np.int64)
        indices = np.empty((columns.size, degree), dtype=np.int64)
        for position in reversed(range(degree)):
            columns, indices[:, position] = np.divmod(columns, count)
        found.append((np.asarray(coo.row, dtype=np.int64), indices, np.asarray(coo.data)))
    return found


def slope(field: Sequence[sp.csr_array], u: ArrayLike) -> NDArray[np.float64]:
    r"""Return :math:`\sum_p F_p u^{\otimes p}`, each term from its non-zeros alone."""
    y = np.asarray(u, dtype=np.float64)
    result = np.zeros(field[0].shape[0])
    for rows, indices, values in _indices(field):
        np.add.at(result, rows, values * np.prod(y[indices], axis=1))
    return result


def jacobian(field: Sequence[sp.csr_array], u: ArrayLike) -> sp.csc_array:
    r"""Return :math:`\partial(\sum_p F_p u^{\otimes p})/\partial u`, sparse.

    Each non-zero :math:`v\,u_{j_1}\cdots u_{j_p}` of row :math:`i` adds, for each position
    :math:`s`, :math:`v\prod_{r \ne s} u_{j_r}` at :math:`(i, j_s)`.
    """
    y = np.asarray(u, dtype=np.float64)
    size = field[0].shape[0]
    rows_out: list[NDArray[Any]] = []
    columns_out: list[NDArray[Any]] = []
    values_out: list[NDArray[Any]] = []
    for rows, indices, values in _indices(field):
        factors = y[indices]
        for position in range(indices.shape[1]):
            others = np.prod(np.delete(factors, position, axis=1), axis=1)
            rows_out.append(rows)
            columns_out.append(indices[:, position])
            values_out.append(values * others)
    matrix = sp.coo_array(
        (np.concatenate(values_out), (np.concatenate(rows_out), np.concatenate(columns_out))),
        shape=(size, size),
    )
    return sp.csc_array(matrix)
