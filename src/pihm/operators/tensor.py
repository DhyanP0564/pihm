r"""Multi-axis conventions: Kronecker order with axis 0 most significant.

A field on :math:`d` Chebyshev axes with :math:`n_a` qubits on axis :math:`a` has normalised
coefficients :math:`b_{k_0 k_1 \dots k_{d-1}}`, flattened with :func:`numpy.kron`'s ordering:
axis 0 is the slowest-varying (most significant) index. So a separable field
:math:`f_0(x_0) f_1(x_1)` has coefficients ``kron(b0, b1)``.

.. warning::

   This is the *opposite* of Qiskit's little-endian qubit order, in which qubit 0 is the least
   significant. The circuit layer maps between the two explicitly and tests the map with an
   asymmetric two-axis operator.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, TypeAlias

import numpy as np
import scipy.sparse as sp

__all__ = ["SparseArray", "axis_sizes", "embed_axis", "kron_all"]

#: The SciPy sparse array formats that operators take and return. Builders return CSR; diagonal
#: and COO arrays appear as intermediate factors.
SparseArray: TypeAlias = (
    sp.csr_array[Any, Any] | sp.csc_array[Any] | sp.coo_array[Any, Any] | sp.dia_array[Any]
)


def axis_sizes(ns: int | Sequence[int]) -> tuple[int, ...]:
    """Return the register sizes :math:`N_a = 2^{n_a}` for per-axis qubit counts ``ns``."""
    counts = (ns,) if isinstance(ns, (int, np.integer)) else tuple(ns)
    if not counts or any(isinstance(k, bool) or int(k) < 0 for k in counts):
        raise ValueError(f"qubit counts must be non-negative integers, got {ns!r}")
    return tuple(1 << int(k) for k in counts)


def kron_all(operators: Sequence[SparseArray]) -> sp.csr_array:
    """Return the Kronecker product of ``operators``, the first being most significant."""
    if not operators:
        raise ValueError("need at least one operator")
    result = operators[0].tocsr()
    for operator in operators[1:]:
        result = sp.kron(result, operator, format="csr").tocsr()
    return result


def embed_axis(operator: SparseArray, ns: Sequence[int], axis: int) -> sp.csr_array:
    r"""Embed a single-axis operator on one axis of a multi-axis field.

    Parameters
    ----------
    operator
        Acts on axis ``axis`` alone; shape ``(M, N_axis)``, where ``M`` may differ from
        ``N_axis`` (a lift, for example).
    ns
        Qubits per axis.
    axis
        The axis the operator acts on.

    Returns
    -------
    scipy.sparse.csr_array
        :math:`I_{<a} \otimes X \otimes I_{>a}`, in the Kronecker order of this module.

    Raises
    ------
    ValueError
        If ``axis`` is out of range or ``operator`` has the wrong number of columns.
    """
    sizes = axis_sizes(ns)
    if not 0 <= axis < len(sizes):
        raise ValueError(f"axis must be in [0, {len(sizes)}), got {axis}")
    matrix = operator.tocsr()
    if matrix.shape[1] != sizes[axis]:
        raise ValueError(f"operator has {matrix.shape[1]} columns; axis {axis} has {sizes[axis]}")
    left = int(np.prod(sizes[:axis], dtype=np.int64))
    right = int(np.prod(sizes[axis + 1 :], dtype=np.int64))
    parts: list[SparseArray] = []
    if left > 1:
        parts.append(sp.eye_array(left, format="csr"))
    parts.append(matrix)
    if right > 1:
        parts.append(sp.eye_array(right, format="csr"))
    return kron_all(parts)
