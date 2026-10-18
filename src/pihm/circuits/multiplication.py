r"""Multiplication by a polynomial coefficient as circuits: the lift :math:`\mathbb{M}_a`.

The circuit counterpart of :func:`pihm.operators.multiplication`. On :math:`m` qubits, multiplying
the normalised Chebyshev basis by :math:`x` is, from :math:`xT_0=T_1` and
:math:`xT_k=\tfrac12(T_{k+1}+T_{k-1})`,

.. math::

   \mathbb{M}_x = \tfrac12(U + D) + \Bigl(\tfrac1{\sqrt2}-\tfrac12\Bigr)
   \bigl(|1\rangle\langle0| + |0\rangle\langle1|\bigr),

with :math:`D=\sum_{k\ge1}|k-1\rangle\langle k|` and :math:`U=D^\dagger` the *truncated* shifts
(:func:`shift_down_encoding`, an in-place subtraction of 1 whose borrow flag is post-selected, so
:math:`|0\rangle` is annihilated rather than wrapped) and the correction a Pauli :math:`X` on the
lowest qubit where every other qubit is :math:`|0\rangle`: a three-term LCU
(:func:`multiply_by_x_encoding`), :math:`\alpha=1/\sqrt2+1/2`.

The paper's lift :math:`\mathbb{M}_a` maps :math:`n` qubits into :math:`n+1` exactly. With
:math:`\mathbb{M}_1=\sqrt2\,[I;\,0]` the embedding, and :math:`a(x)=
\sum_j a_jx^j`, :math:`\mathbb{M}_a = \sum_j a_j(\mathbb{M}_x)^j\,\mathbb{M}_1` with
:math:`\mathbb{M}_x` the square form on :math:`n+1` qubits: an embedded series has degree
below :math:`N`, so :math:`\deg a\le N` multiplications never reach the truncated mode :math:`2N`
and each is exact (:func:`multiplication_encoding`).
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import XGate

from pihm.circuits.arithmetic import register_subtractor
from pihm.circuits.compose import adjoint, lcu_sum, pad_rows, product
from pihm.circuits.primitives import BlockEncoding, identity_encoding

__all__ = [
    "embed",
    "multiplication_encoding",
    "multiply_by_x_encoding",
    "polynomial_encoding",
    "shift_down_encoding",
]


def shift_down_encoding(n_qubits: int) -> BlockEncoding:
    r"""Return the truncated down shift :math:`D=\sum_{k\ge1}|k-1\rangle\langle k|`, exactly.

    An in-place subtraction of 1 (:func:`~pihm.circuits.arithmetic.register_subtractor` with its
    subtrahend register set to 1 and restored): :math:`|k\rangle\mapsto|k-1\rangle` for
    :math:`k\ge1`, while :math:`|0\rangle` sets the borrow flag and so vanishes on
    post-selection. :math:`\alpha=1`; ancilla: the subtrahend register, the carry and the borrow.
    """
    m = n_qubits
    system = list(range(m))
    subtrahend = list(range(m, 2 * m))
    carry, borrow = 2 * m, 2 * m + 1
    qc = QuantumCircuit(2 * m + 2, name="D")
    qc.x(subtrahend[0])
    qc.compose(register_subtractor(m), qubits=[carry, *subtrahend, *system, borrow], inplace=True)
    qc.x(subtrahend[0])
    M = 1 << m
    return BlockEncoding(qc, 1.0, m, m + 2, (M, M), label="D")


def _boundary_encoding(n_qubits: int) -> BlockEncoding:
    r"""Return :math:`|1\rangle\langle0|+|0\rangle\langle1|`: :math:`X_0` where the rest is zero.

    One ancilla flips, and flips back only when every qubit but the lowest is :math:`|0\rangle`
    (:func:`~pihm.circuits.primitives.zero_projector`'s construction on those qubits); then
    :math:`X` on the lowest qubit. For one qubit it is :math:`X` itself, with no ancilla.
    """
    m = n_qubits
    M = 1 << m
    if m == 1:
        qc = QuantumCircuit(1, name="E")
        qc.x(0)
        return BlockEncoding(qc, 1.0, 1, 0, (M, M), label="E")
    anc = m
    qc = QuantumCircuit(m + 1, name="E")
    qc.x(anc)
    qc.append(XGate().control(m - 1, ctrl_state=0), [*range(1, m), anc])
    qc.x(0)
    return BlockEncoding(qc, 1.0, m, 1, (M, M), label="E")


def multiply_by_x_encoding(n_qubits: int) -> BlockEncoding:
    r"""Return the square :math:`\mathbb{M}_x` on ``n_qubits`` qubits, exactly (module docstring).

    Parameters
    ----------
    n_qubits
        Qubits in the Chebyshev register.

    Returns
    -------
    BlockEncoding
        Shape ``(M, M)``, :math:`\alpha=1/\sqrt2+1/2`.
    """
    D = shift_down_encoding(n_qubits)
    correction = 1.0 / np.sqrt(2.0) - 0.5
    return lcu_sum(
        [adjoint(D, label="U"), D, _boundary_encoding(n_qubits)],
        [0.5, 0.5, correction],
        label="Mx",
    )


def polynomial_encoding(coefficients: Sequence[float], n_qubits: int) -> BlockEncoding:
    r"""Return the square :math:`\sum_j a_j(\mathbb{M}_x)^j` on ``n_qubits`` qubits.

    The powers are :func:`~pihm.circuits.compose.product`\ s of :func:`multiply_by_x_encoding`,
    combined by :func:`~pihm.circuits.compose.lcu_sum`. On a lifted register it is exact on every
    embedded series (module docstring).

    Parameters
    ----------
    coefficients
        :math:`a_0, a_1, \dots` in powers of :math:`x`; real, not all zero.
    n_qubits
        Qubits in the register it acts on.

    Returns
    -------
    BlockEncoding
        Shape ``(M, M)``.
    """
    values = [float(c) for c in coefficients]
    while values and values[-1] == 0.0:
        values.pop()
    if not values:
        raise ValueError("polynomial_encoding needs a nonzero polynomial")
    Mx = multiply_by_x_encoding(n_qubits)
    powers = [identity_encoding(n_qubits), Mx]
    while len(powers) < len(values):
        powers.append(product(Mx, powers[-1]))
    terms = [powers[j] for j, a in enumerate(values) if a != 0.0]
    weights = [a for a in values if a != 0.0]
    return lcu_sum(terms, weights, label="Ma")


def embed(encoding: BlockEncoding) -> BlockEncoding:
    r"""Return :math:`\mathbb{M}_1X=\sqrt2\,[X;\,0]`: ``encoding`` on a lifted register.

    The paper's embedding :math:`\mathbb{M}_1` of the :math:`n`-qubit basis into :math:`n+1`
    qubits: :func:`~pihm.circuits.compose.pad_rows`, with the :math:`\sqrt2`
    carried by :math:`\alpha`.
    """
    padded = pad_rows(encoding)
    return BlockEncoding(
        padded.circuit,
        np.sqrt(2.0) * padded.alpha,
        padded.n_sys_qubits,
        padded.n_anc_qubits,
        padded.shape,
        f"M1·{encoding.label}",
    )


def multiplication_encoding(
    coefficients: Sequence[float], n_qubits: int, label: str = ""
) -> BlockEncoding:
    r"""Return the lift :math:`\mathbb{M}_a` for :math:`a(x)=\sum_j a_jx^j`, from :math:`n` qubits.

    :func:`polynomial_encoding` on :math:`n+1` qubits after :func:`embed`: exact for
    :math:`\deg a\le N`, as :func:`pihm.operators.multiplication`'s ``mode="lift"`` is.

    Parameters
    ----------
    coefficients
        :math:`a_0, a_1, \dots` in powers of :math:`x`; real.
    n_qubits
        Qubits in the input register.
    label
        Circuit label.

    Returns
    -------
    BlockEncoding
        Shape ``(2N, N)`` on :math:`n+1` system qubits.

    Raises
    ------
    ValueError
        If :math:`\deg a > N` (the product would not fit in one extra qubit), or every
        coefficient is zero.
    """
    values = [float(c) for c in coefficients]
    while values and values[-1] == 0.0:
        values.pop()
    N = 1 << n_qubits
    if len(values) - 1 > N:
        raise ValueError(
            f"deg a = {len(values) - 1} exceeds N = {N}; the product needs a larger lift"
        )
    square = polynomial_encoding(values, n_qubits + 1)
    return product(square, embed(identity_encoding(n_qubits)), label=label or "M_a")
