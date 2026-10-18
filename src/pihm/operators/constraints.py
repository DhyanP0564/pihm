r"""Constraint rows, shared by both regimes.

Every row here is built from the **raw** feature vector :math:`\tau`:

``"zero"``
    :math:`\mathbb{B}(x) = \sqrt N\,|0\rangle\langle\tau(x)|`, the invariant constraint
    :math:`f(x) = 0` (the paper's :math:`\mathbb{B}_n(x)`).
``"collapse"``
    :math:`\mathbb{D}^{(0)}(x_s) = \mathbb{B}(x_s)/v` with :math:`v = f(x_s)` known. It maps the
    state to the constant function :math:`f_q(x_s)/v`, which is 1 on the true solution. It
    injects sources and lifts linear terms into a doubled space.
``"ratio"``
    :math:`\mathbb{F} = y_a(I - |a\rangle\langle a|) - w\langle a|` with
    :math:`w = \sum_{c \ne a} y_c|c\rangle`. It annihilates exactly the vectors parallel to the
    data :math:`y`, which is how vector initial data enter (Carleman, coupled fields).
``"datum"``
    :math:`\bigl(\mathbb{S}_a(x^*) - \hat g\,\tau(x_s)^{\top}/v\bigr)`, an inhomogeneous boundary or
    initial *function* :math:`g` on the slice :math:`x_a = x^*`, imposed without doubling.

Placing a row on a field or a derivative block (the ``on``) is the
assembler's job. The builders here act on one field's coefficients.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any, Final, Literal, TypeAlias

import numpy as np
import scipy.sparse as sp
from numpy.typing import ArrayLike, NDArray

from pihm.basis import tau
from pihm.operators.tensor import axis_sizes, embed_axis

__all__ = [
    "CONSTRAINT_BUILDERS",
    "ConstraintKind",
    "collapse_row",
    "datum_rows",
    "ratio_rows",
    "zero_row",
]

#: The four kinds of constraint row.
ConstraintKind: TypeAlias = Literal["zero", "collapse", "ratio", "datum"]


def zero_row(x: float, n: int) -> sp.csr_array:
    r"""Return the zero row :math:`\mathbb{B}(x) = \sqrt N\,|0\rangle\langle\tau(x)|`.

    It imposes the invariant constraint :math:`f(x) = 0`.

    Parameters
    ----------
    x
        The constraint point in :math:`[-1, 1]`.
    n
        Qubits in the register.

    Returns
    -------
    scipy.sparse.csr_array
        The rank-one ``(N, N)`` matrix whose only non-zero row is :math:`\sqrt N\,\tau(x)`. It
        maps coefficients :math:`b` to :math:`f(x)\,\sqrt N e_0`, which represents the constant
        function :math:`f(x)`.

    Examples
    --------
    The paper's :math:`\mathbb{B}_2(x) = \sqrt2\,[T_0/\sqrt2, T_1, T_2, T_3]` in row 0, at
    :math:`x = 1`:

    >>> zero_row(1.0, 2).toarray()[0]
    array([1.        , 1.41421356, 1.41421356, 1.41421356])
    """
    row = np.sqrt(2.0**n) * tau(x, n)
    size = row.size
    return sp.csr_array(
        (row, (np.zeros(size, dtype=np.int64), np.arange(size))), shape=(size, size)
    )


def collapse_row(x: float, value: complex, n: int) -> sp.csr_array:
    r"""Return :math:`\mathbb{D}^{(0)}(x_s) = \mathbb{B}(x_s)/v`, the collapse onto a known datum.

    Parameters
    ----------
    x
        The regular point :math:`x_s` at which the solution value is known.
    value
        The known value :math:`v = f(x_s)`, non-zero, possibly complex.
    n
        Qubits in the register.

    Returns
    -------
    scipy.sparse.csr_array
        :math:`\mathbb{B}(x_s)/v`. Applied to the true solution's coefficients it returns
        :math:`\sqrt N e_0`, the constant function 1, whatever the solution's scale. Because
        :math:`\tau` is raw, any number of collapses stay mutually consistent.

    Raises
    ------
    ValueError
        If ``value`` is zero.
    """
    if value == 0:
        raise ValueError("a collapse needs a non-zero datum value")
    return sp.csr_array(zero_row(x, n) / value)


def ratio_rows(data: ArrayLike, anchor: int = 0) -> sp.csr_array:
    r"""Return :math:`\mathbb{F} = y_a(I - |a\rangle\langle a|) - w\langle a|`, pinning a direction.

    Parameters
    ----------
    data
        The data vector :math:`y` (length :math:`M`), for example multi-component initial data.
    anchor
        The component :math:`a` used as the reference; :math:`y_a` must be non-zero.

    Returns
    -------
    scipy.sparse.csr_array
        The ``(M, M)`` operator with :math:`(\mathbb{F}v)_c = y_a v_c - y_c v_a` for
        :math:`c \ne a` and :math:`(\mathbb{F}v)_a = 0`. Its kernel is exactly the span of
        :math:`y`, so the rows fix the ratios of the components and leave the scale free.

    Raises
    ------
    ValueError
        If the anchor is out of range or :math:`y_a = 0`.
    """
    y = np.atleast_1d(np.asarray(data))
    if y.ndim != 1 or not 0 <= anchor < y.size:
        raise ValueError("data must be a vector and the anchor one of its indices")
    if y[anchor] == 0:
        raise ValueError("the anchor component of the data must be non-zero")
    others = np.flatnonzero(np.arange(y.size) != anchor)
    rows = np.concatenate([others, others])
    columns = np.concatenate([others, np.full(others.size, anchor)])
    values = np.concatenate([np.full(others.size, y[anchor]), -y[others]])
    return sp.csr_array((values, (rows, columns)), shape=(y.size, y.size))


def datum_rows(
    ns: Sequence[int],
    axis: int,
    at: float,
    datum: ArrayLike,
    anchor: Sequence[float],
    anchor_value: complex,
) -> sp.csr_array:
    r"""Return the datum-slice rows :math:`\mathbb{S}_a(x^*) - \hat g\,\tau(x_s)^{\top}/v`.

    The rows impose :math:`f|_{x_a = x^*} = g` for a known function :math:`g` of the other axes,
    homogeneously in :math:`f`. The scale is fixed through a regular point :math:`x_s` where
    :math:`f(x_s) = v` is known.

    * :math:`\mathbb{S}_a(x^*) = I \otimes \mathbb{B}(x^*) \otimes I` (on axis :math:`a`)
      evaluates :math:`f` on the slice and broadcasts it as a constant along axis :math:`a`.
    * :math:`\hat g` is :math:`g` broadcast the same way, so on the true solution both terms
      represent the same field and the rows vanish.

    Parameters
    ----------
    ns
        Qubits per axis of the field.
    axis
        The axis :math:`a` that is fixed on the slice.
    at
        The slice position :math:`x^*` on that axis.
    datum
        Normalised coefficients of :math:`g` on the *other* axes, in Kronecker order.
    anchor
        The regular point :math:`x_s`, one coordinate per axis.
    anchor_value
        The known value :math:`v = f(x_s)`, non-zero.

    Returns
    -------
    scipy.sparse.csr_array
        A square matrix on the field's :math:`\prod_a N_a` coefficients.

    Raises
    ------
    ValueError
        If the shapes are inconsistent or ``anchor_value`` is zero.
    """
    sizes = axis_sizes(ns)
    if len(anchor) != len(sizes):
        raise ValueError("the anchor needs one coordinate per axis")
    if anchor_value == 0:
        raise ValueError("the anchor value must be non-zero")
    other = [size for index, size in enumerate(sizes) if index != axis]
    g = np.asarray(datum).reshape(other)
    constant_along_axis = np.zeros(sizes[axis])
    constant_along_axis[0] = np.sqrt(sizes[axis])
    g_hat = np.expand_dims(g, axis) * constant_along_axis.reshape(
        [sizes[axis] if index == axis else 1 for index in range(len(sizes))]
    )
    tau_anchor: NDArray[Any] = np.ones(1)
    for coordinate, count in zip(anchor, ns, strict=True):
        tau_anchor = np.kron(tau_anchor, tau(coordinate, count))
    slice_rows = embed_axis(zero_row(at, ns[axis]), ns, axis)
    # The injected term is rank one, and g_hat is non-zero only on the constant mode of the
    # sliced axis, so it is built as a sparse outer product: prod_{b != a} N_b dense rows.
    g_flat = g_hat.reshape(-1)
    rows = np.flatnonzero(g_flat)
    injection = sp.csr_array(
        (
            np.outer(g_flat[rows], tau_anchor / anchor_value).ravel(),
            (np.repeat(rows, tau_anchor.size), np.tile(np.arange(tau_anchor.size), rows.size)),
        ),
        shape=(g_flat.size, tau_anchor.size),
    )
    return sp.csr_array(slice_rows - injection)


#: Builders by kind, for dispatch from a problem specification.
CONSTRAINT_BUILDERS: Final[Mapping[ConstraintKind, Callable[..., sp.csr_array]]] = {
    "zero": zero_row,
    "collapse": collapse_row,
    "ratio": ratio_rows,
    "datum": datum_rows,
}
