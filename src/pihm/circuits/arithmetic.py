r"""Reversible arithmetic: the Cuccaro ripple-carry adder and the register subtractor.

:func:`ripple_carry_adder` is the standard MAJ/UMA construction (Cuccaro, Draper, Kutin and
Moulton, "A new quantum ripple-carry addition circuit," quant-ph/0410184): :math:`O(n)`
Toffolis, one carry-in ancilla and one overflow qubit, both restored (carry-in) or freshly
written (overflow) rather than consumed. :func:`register_subtractor` inverts it to give the
in-place subtraction with a borrow flag that serves as the arithmetic core of the
structured :math:`\mathbb{G}` encoding's :math:`U_{\rm odd}` SELECT step.
"""

from __future__ import annotations

from qiskit import QuantumCircuit

__all__ = ["register_subtractor", "ripple_carry_adder"]


def _maj(qc: QuantumCircuit, c: int, b: int, a: int) -> None:
    r"""Apply Cuccaro's MAJ gate: (c, b, a) -> (c, a XOR b, MAJ(a, b, c)).

    :math:`\mathrm{MAJ}(a,b,c) = ab \oplus ac \oplus bc`, the carry out of :math:`a+b+c`; the
    incoming carry ``c`` is left as :math:`a \oplus c` (verified by direct symbolic expansion:
    CNOT(a,b) then CNOT(a,c) then a Toffoli controlled on the new (c, b)).
    """
    qc.cx(a, b)
    qc.cx(a, c)
    qc.ccx(c, b, a)


def _uma(qc: QuantumCircuit, c: int, b: int, a: int) -> None:
    r"""Apply Cuccaro's UMA gate, undoing :func:`_maj` while writing the sum bit into ``b``.

    Given the state :func:`_maj` leaves behind, this restores ``a`` and ``c`` to their values
    before :func:`_maj` and leaves ``b`` holding :math:`a \oplus b \oplus c` (the sum bit with
    the true incoming carry) -- verified by direct symbolic expansion, the mirror image of
    :func:`_maj`'s own.
    """
    qc.ccx(c, b, a)
    qc.cx(a, c)
    qc.cx(c, b)


def ripple_carry_adder(n_qubits: int, label: str = "") -> QuantumCircuit:
    r"""Return the Cuccaro adder on two ``n_qubits``-wide registers.

    Qubit layout, ``2 * n_qubits + 2`` qubits total:

    * ``carry_in`` (index 0): must start :math:`|0\rangle`; restored to :math:`|0\rangle`.
    * ``a`` (indices ``1 .. n_qubits``): the addend; restored unchanged.
    * ``b`` (indices ``n_qubits + 1 .. 2 * n_qubits``): becomes :math:`a + b \bmod 2^{n}`.
    * ``overflow`` (index ``2 * n_qubits + 1``): must start :math:`|0\rangle`; becomes the true
      carry out of :math:`a + b` (1 iff :math:`a + b \ge 2^n`).

    ``n_qubits`` MAJ gates (ripple the carry forward, in place, through ``a``), one CNOT to copy
    the final carry into ``overflow``, then ``n_qubits`` UMA gates in reverse (ripple back,
    restoring ``a`` and ``carry_in`` while writing the sum into ``b``).

    Parameters
    ----------
    n_qubits
        Width of each register (at least 1).
    label
        Circuit label.

    Returns
    -------
    qiskit.circuit.QuantumCircuit
        A ``2 * n_qubits + 2``-qubit circuit, :math:`O(n_{\rm qubits})` Toffolis.

    Raises
    ------
    ValueError
        If ``n_qubits < 1``.
    """
    if n_qubits < 1:
        raise ValueError("ripple_carry_adder needs at least one qubit per register")
    n = n_qubits
    carry_in = 0
    a = list(range(1, n + 1))
    b = list(range(n + 1, 2 * n + 1))
    overflow = 2 * n + 1
    qc = QuantumCircuit(2 * n + 2, name=label or "ADD")

    carry = carry_in
    for i in range(n):
        _maj(qc, carry, b[i], a[i])
        carry = a[i]
    qc.cx(a[-1], overflow)
    for i in reversed(range(n)):
        carry = carry_in if i == 0 else a[i - 1]
        _uma(qc, carry, b[i], a[i])
    return qc


def register_subtractor(n_qubits: int, label: str = "") -> QuantumCircuit:
    r"""Return the in-place subtractor :math:`b \leftarrow b - a`, with a borrow flag.

    The exact inverse of :func:`ripple_carry_adder` (:meth:`~qiskit.circuit.QuantumCircuit.inverse`
    of a MAJ/UMA circuit reverses instruction order and inverts each Toffoli/CNOT, which are
    self-inverse, so it is itself MAJ/UMA-shaped). On the same qubit layout, with ``carry_in`` and
    ``borrow`` both starting :math:`|0\rangle`:

    * ``a`` is left unchanged.
    * ``b`` becomes :math:`(b - a) \bmod 2^n`.
    * ``borrow`` (the ``overflow`` qubit of the adder) becomes 1 exactly when :math:`b < a`
      (a genuine underflow), 0 otherwise.

    Derivation: writing the adder's action as :math:`(c, a, b, z) \mapsto (c, a, a + b + c \bmod
    2^n, z \oplus [a+b+c \ge 2^n])` for *arbitrary* inputs (not only :math:`c=z=0`; the three-gate
    MAJ/UMA identities do not use :math:`c=0` anywhere), inverting the (bijective, for fixed
    :math:`a, c`) map :math:`b \mapsto a+b+c \bmod 2^n` gives :math:`b = (\text{input} - a - c)
    \bmod 2^n`, and the carry condition on that reconstructed :math:`b` is exactly
    :math:`\text{input} < a + c`. At :math:`c = 0` this is the wanted borrow flag
    :math:`\text{input} < a`.

    Parameters
    ----------
    n_qubits
        Width of each register.
    label
        Circuit label.

    Returns
    -------
    qiskit.circuit.QuantumCircuit
        A ``2 * n_qubits + 2``-qubit circuit: ``carry_in`` (0), ``a`` (1..n), ``b`` (n+1..2n),
        ``borrow`` (2n+1), the same layout as :func:`ripple_carry_adder`.
    """
    return ripple_carry_adder(n_qubits, label=label or "SUB").inverse()
