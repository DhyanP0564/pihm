r"""The product fold as a circuit: :math:`\mathbb{N}_1` and :math:`\mathbb{N}_x`.

The circuit counterpart of :func:`pihm.operators.folds.fold`'s nodal construction for two copies
(:math:`p=2`, one lift level): at the Chebyshev–Gauss nodes a product is pointwise, so

.. math::

   \mathbb{N}_1 = 2\,\mathbb{U}^\dagger\Delta\,(\mathbb{U}\otimes\mathbb{U}),

with :math:`\mathbb{U}` the orthogonal Chebyshev–Gauss transform on :math:`n+1` qubits
(:func:`pihm.basis.chebyshev_gauss_transform`), each copy padded by one qubit, and :math:`\Delta`
keeping the diagonal :math:`m=m'`. The circuit applies :math:`\mathbb{U}` to both padded copies,
copies the first into the second with :math:`n+1` CNOTs (the second is then zero exactly on the
diagonal, and is post-selected), and applies :math:`\mathbb{U}^\dagger` to the first:
:math:`\alpha=2`, no other ancilla. :math:`\mathbb{N}_x` multiplies by the nodes before
:math:`\mathbb{U}^\dagger`: :math:`\bar x_m=\cos\theta_m` with :math:`\theta_m=\pi(2m+1)/2M`
linear in :math:`m`, so :math:`\mathrm{diag}(\bar x)=(D+D^\dagger)/2` for
:math:`D=\mathrm{diag}(e^{i\theta_m})`, one phase gate per qubit, combined on one ancilla
(:math:`\alpha` stays 2).

:math:`\mathbb{U}` is placed as a dense gate, standing for the discrete cosine transform that
is budgeted as a cited primitive (Klappenecker–Rötteler); the fold's :math:`\alpha` and
ancilla are exact, and its gate counts include the dense gate's own synthesis.

The result is square on the two copies' :math:`2n` qubits: the product occupies the lowest
:math:`n+1`, and the rest of the register is returned to zero, so the block is
:math:`[\mathbb{N};\,0]` and a doubled-space row block stacks with the others.
"""

from __future__ import annotations

from typing import Literal, TypeAlias

import numpy as np
from qiskit import QuantumCircuit

from pihm.basis import chebyshev_gauss_transform
from pihm.circuits.primitives import BlockEncoding, unitary_gate

__all__ = ["FoldWeight", "fold_encoding"]

#: The fold's weight, the plain product (:math:`\mathbb{N}_1`) or the product times :math:`x`.
FoldWeight: TypeAlias = Literal["none", "x"]


def fold_encoding(n: int, weight: FoldWeight = "none", label: str = "") -> BlockEncoding:
    r"""Return the fold :math:`[\mathbb{N};\,0]` of two :math:`n`-qubit copies (module docstring).

    Parameters
    ----------
    n
        Qubits per copy; the first copy (the Kronecker product's first factor) holds the higher
        :math:`n` qubits.
    weight
        ``"none"`` for :math:`\mathbb{N}_1`, ``"x"`` for :math:`\mathbb{N}_x`.
    label
        Circuit label.

    Returns
    -------
    BlockEncoding
        Shape ``(N**2, N**2)`` on :math:`2n` system qubits, :math:`\alpha=2`; rows at and above
        :math:`2N` are zero.
    """
    if n < 1:
        raise ValueError(f"fold_encoding needs n >= 1, got {n}")
    if weight not in ("none", "x"):
        raise ValueError(f"weight must be 'none' or 'x', got {weight!r}")
    size = 1 << (n + 1)
    U = chebyshev_gauss_transform(n + 1)
    second = list(range(n))  # the second copy's system qubits, the output's lower n
    first = list(range(n, 2 * n))
    e_first, e_second = 2 * n, 2 * n + 1  # the padding qubit of each copy
    spare = list(range(2 * n + 2, 3 * n + 1))  # zeros that take the diagonal's residue
    control = 3 * n + 1 if weight == "x" else None
    total = 3 * n + 1 + int(weight == "x")
    out = [*second, e_second]
    check = [*first, e_first]

    qc = QuantumCircuit(total, name=label or ("N_x" if weight == "x" else "N_1"))
    qc.append(unitary_gate(U, "U"), out)
    qc.append(unitary_gate(U, "U"), check)
    for source, target in zip(out, check, strict=True):
        qc.cx(source, target)
    if control is not None:
        qc.h(control)
        for bit, qubit in enumerate(out):
            angle = np.pi * (1 << bit) / size
            qc.p(angle, qubit)
            qc.cp(-2.0 * angle, control, qubit)
        qc.global_phase += np.pi / (2 * size)
        qc.p(-np.pi / size, control)
        qc.h(control)
    qc.append(unitary_gate(U.T, "U†"), out)
    qc.swap(e_second, n)
    for qubit, zero in zip(range(n + 1, 2 * n), spare, strict=True):
        qc.swap(qubit, zero)
    dim = 1 << (2 * n)
    name = qc.name
    return BlockEncoding(qc, 2.0, 2 * n, total - 2 * n, (dim, dim), name)
