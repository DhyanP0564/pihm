r"""Compile the term IR of the descriptor's system rows into a circuit.

:func:`compile_system` compiles a :class:`~pihm.ir.SystemIR`. Its band terms,

.. math::

   A = \sum_t c_t\,X^{x_t}Z^{z_t}\otimes\bigotimes_a S^{(s_{t,a})}\,{\rm diag}_{t,a},

become PREPARE, SELECT and PREPARE† with **factored selectors**. One register chooses the field
register's operation; each axis has one register for its diagonal (the identity, a :math:`Z_j` or
a reflection :math:`R_m`) and one for its shift :math:`S^{(s)}`. Each register is sized to the
distinct values the terms use, so SELECT applies each distinct operation once, controlled on its
own register, rather than once per term. PREPARE loads :math:`{\rm sign}(c_t)\sqrt{|c_t|/\alpha}`
at each term's joint index of the registers and the unpreparation :math:`\sqrt{|c_t|/\alpha}`
(:func:`~pihm.circuits.compose.real_state_preparation`), so :math:`\alpha = \sum_t|c_t|`, the IR's
own.

**The field register's operation.** The terms whose axis factors and :math:`X^x` agree sum to
:math:`X^xD`, :math:`D` diagonal. Where :math:`D` has one non-zero entry, :math:`D_{cc}`, as a
matrix unit placed on one block does, its :math:`2^{n_f}` Pauli words are one term
:math:`D_{cc}X^x\Pi_c` at the same :math:`\alpha`: a flag qubit, post-selected, marks every
field index but :math:`c`. Otherwise, as for :math:`\tilde{\mathbb{D}}` on the recurrences'
diagonal, the Pauli words stay, since their :math:`\alpha` is the tighter.

**Matchings** (:class:`~pihm.ir.Matching`). A term whose field factor is a matching
permutes the field register's basis states (:func:`~pihm.circuits.primitives.basis_permutation`),
and its entries :math:`v_r` on the rows are then one rotation of the flag qubit,
:math:`R_Y(2\arccos v_r)`, which leaves :math:`v_r` on the flag's zero (and nothing on a row the
matching does not reach). Every matching's rotations are one uniformly controlled :math:`R_Y`,
controlled on the field register and the field operation's selector together, so their entries
cost :math:`2^{n_f + w}` CNOT for all of them, not one multiplexor each.

**Rank-one terms**, a source's :math:`|\hat r\rangle\langle\tau(x_s)|` placed on the field
register, are compiled apart and summed with the bands by
:func:`~pihm.circuits.compose.lcu_sum`. The field register's pattern of each is a few matrix
units (:func:`field_placement`); on its axis the feature map reads :math:`\tau(x_s)`
(:func:`~pihm.circuits.standard.zero_encoding`) and a rotation tree loads :math:`\hat r`
(:func:`rank_one_encoding`), so a source costs two ancilla and no arithmetic.

**Registers** (:class:`Layout`). The state's register is the field register above the axes', in
Kronecker order (the first axis most significant), as :func:`pihm.assemble.assemble` lays the
state out. A lifted axis's extra output qubit sits above the field register, so the state fills
the leading amplitudes and a lifted row keeps its axis's low qubits (:func:`circuit_rows`); a
reflection of the stacked residual counts those qubits as ancilla
(:func:`~pihm.circuits.compose.stacked_reflection`). The ancilla are then one overflow qubit per
axis that shifts, and the selectors.

**Truncated shifts.** A band's shift takes column :math:`c` to row :math:`c - s`, and vanishes
where that row does not exist. On the axis's output register widened by its overflow qubit, the
cyclic addition of :math:`-s` does exactly that: a row below zero or past the register lands in
the overflow half, which the ancilla-zero projection discards. Adding a constant is an increment
per set bit, each a ladder of multi-controlled :math:`X` gates, so the overflow qubit is the
shift's only ancilla and the descriptor's ancilla stay :math:`O(\log n)`.
"""

from __future__ import annotations

from collections.abc import Hashable, Sequence
from dataclasses import dataclass, replace
from typing import Any, Generic, TypeVar

import numpy as np
from numpy.typing import NDArray
from qiskit import QuantumCircuit
from qiskit.circuit.library import MCPhaseGate, UCRYGate, XGate

from pihm.circuits.compose import _select, lcu_sum, pad_rows, product, real_state_preparation
from pihm.circuits.primitives import (
    BlockEncoding,
    basis_permutation,
    inverse_circuit,
    zero_projector,
)
from pihm.circuits.standard import zero_encoding
from pihm.ir import Band, Diagonal, Matching, Outer, SystemIR, Term, pauli_word

__all__ = [
    "Layout",
    "circuit_rows",
    "compile_system",
    "field_placement",
    "place",
    "rank_one_encoding",
    "shift_circuit",
]

#: Relative size below which an entry of a rank-one term's field pattern is taken as zero.
_TOLERANCE = 1.0e-12

_T = TypeVar("_T", bound=Hashable)


@dataclass(frozen=True)
class Layout:
    """Where the descriptor's registers sit in a circuit's system register.

    Attributes
    ----------
    field_qubits
        The field register's qubits :math:`n_f`.
    qubits_in, qubits_out
        Each axis's register on the input and the output side; the output has one qubit more on
        a lifted axis.
    """

    field_qubits: int
    qubits_in: tuple[int, ...]
    qubits_out: tuple[int, ...]

    @classmethod
    def of(cls, ir: SystemIR) -> Layout:
        """Return the layout of ``ir``'s registers."""
        return cls(ir.field_qubits, ir.qubits_in, ir.qubits_out)

    @property
    def state_qubits(self) -> int:
        """The state's qubits, the field register's and every axis's input register's."""
        return self.field_qubits + sum(self.qubits_in)

    @property
    def n_sys(self) -> int:
        """The system register's qubits, the state's and each lifted axis's extra output qubit."""
        return self.field_qubits + sum(self.qubits_out)

    @property
    def shape(self) -> tuple[int, int]:
        """The shape of an operator from the state to the system register."""
        return 1 << self.n_sys, 1 << self.state_qubits

    def data(self, axis: int) -> list[int]:
        """Return axis ``axis``'s input qubits, least significant first."""
        start = sum(self.qubits_in[axis + 1 :])
        return list(range(start, start + self.qubits_in[axis]))

    def field(self) -> list[int]:
        """Return the field register's qubits, least significant first."""
        start = sum(self.qubits_in)
        return list(range(start, start + self.field_qubits))

    def output(self, axis: int) -> list[int]:
        """Return axis ``axis``'s output qubits: its input qubits, then its lift qubit if any."""
        qubits = self.data(axis)
        lifted = [out > n for out, n in zip(self.qubits_out, self.qubits_in, strict=True)]
        if lifted[axis]:
            qubits.append(self.state_qubits + sum(lifted[:axis]))
        return qubits


def circuit_rows(layout: Layout) -> NDArray[np.intp]:
    """Return the circuit's basis index of each row of the IR's operator.

    The IR's rows are the field index, then each axis's output index in Kronecker order
    (:meth:`pihm.ir.SystemIR.to_sparse`); the circuit keeps a lifted axis's top output qubit above
    the field register (:class:`Layout`).
    """
    rows = np.arange(1 << layout.n_sys)
    remaining = rows.copy()
    index = np.zeros_like(rows)
    for axis in reversed(range(len(layout.qubits_in))):
        size = 1 << layout.qubits_out[axis]
        value = remaining % size
        remaining //= size
        for bit, qubit in enumerate(layout.output(axis)):
            index |= ((value >> bit) & 1) << qubit
    for bit, qubit in enumerate(layout.field()):
        index |= ((remaining >> bit) & 1) << qubit
    return index


def _increment(width: int) -> QuantumCircuit:
    r"""Return :math:`|k\rangle \mapsto |k + 1 \bmod 2^{\rm width}\rangle`, with no ancilla."""
    qc = QuantumCircuit(width, name="+1")
    for target in reversed(range(1, width)):
        qc.mcx(list(range(target)), target)
    qc.x(0)
    return qc


def shift_circuit(shift: int, width: int) -> QuantumCircuit:
    r"""Return the cyclic addition of :math:`-s` on ``width`` qubits, :math:`s` = ``shift``.

    An increment of the qubits from :math:`j` up for each set bit :math:`j` of :math:`|s|`, or the
    inverse for :math:`s > 0`. With the register's top qubit an overflow qubit that starts in
    :math:`|0\rangle`, it is a band's truncated shift (module documentation).
    """
    qc = QuantumCircuit(width, name=f"S({shift})")
    for bit in range(width):
        if (abs(shift) >> bit) & 1:
            qc.compose(_increment(width - bit), qubits=range(bit, width), inplace=True)
    return inverse_circuit(qc) if shift > 0 else qc


def _diagonal_circuit(diagonal: Diagonal, width: int) -> QuantumCircuit:
    r"""Return :math:`Z_j` or :math:`R_m = I - 2|m\rangle\langle m|` on ``width`` qubits."""
    qc = QuantumCircuit(width, name=f"{diagonal.kind}{diagonal.index}")
    if diagonal.kind == "z":
        qc.z(diagonal.index)
    elif diagonal.kind == "reflection":
        flips = [q for q in range(width) if not (diagonal.index >> q) & 1]
        for q in flips:
            qc.x(q)
        if width == 1:
            qc.z(0)
        else:
            qc.append(MCPhaseGate(np.pi, width - 1), range(width))
        for q in flips:
            qc.x(q)
    return qc


@dataclass(frozen=True)
class _FieldOperation:
    r"""A term's field operation: :math:`X^x` after :math:`Z^z`, or after :math:`\Pi_c`."""

    x_mask: int
    z_mask: int = 0
    column: int | None = None

    @property
    def needs_flag(self) -> bool:
        """Whether the operation post-selects the flag qubit."""
        return self.column is not None

    def circuit(self, width: int, flag: bool) -> QuantumCircuit:
        """Return the operation on ``width`` qubits, then the flag qubit if ``flag``."""
        qc = QuantumCircuit(width + int(flag), name=repr(self))
        if self.column is None:
            for q in range(width):
                if (self.z_mask >> q) & 1:
                    qc.z(q)
        else:
            qc.x(width)
            qc.append(XGate().control(width, ctrl_state=self.column), [*range(width), width])
        for q in range(width):
            if (self.x_mask >> q) & 1:
                qc.x(q)
        return qc


@dataclass(frozen=True)
class _MatchingOperation:
    """A term's field operation that is a matching: its permutation, its entries on the flag.

    A matching that is a permutation (``unitary``) has no entries to put on the flag.
    """

    matching: Matching
    unitary: bool

    @property
    def needs_flag(self) -> bool:
        """Whether the operation post-selects the flag qubit, for its entries."""
        return not self.unitary

    def circuit(self, width: int, flag: bool) -> QuantumCircuit:
        """Return the matching's permutation on ``width`` qubits, then the idle flag qubit."""
        qc = QuantumCircuit(width + int(flag), name="P")
        permutation = basis_permutation(self.matching.permutation(width))
        qc.compose(permutation, qubits=range(width), inplace=True)
        return qc

    def angles(self, width: int) -> NDArray[np.float64]:
        r"""Return the flag's angle on each row, :math:`2\arccos v_r`, and :math:`\pi` elsewhere."""
        values = np.zeros(1 << width)
        values[list(self.matching.rows)] = self.matching.values
        return np.asarray(2.0 * np.arccos(np.clip(values, -1.0, 1.0)))


_FieldOp = _FieldOperation | _MatchingOperation


def _field_terms(
    terms: Sequence[Term], field_qubits: int
) -> list[tuple[float, _FieldOp, list[Band]]]:
    """Return the band terms, with each single-entry field diagonal as one projected term.

    Terms that come to the same field operation and bands, such as a matching that is the
    identity beside a Pauli identity, are one term with their coefficients summed.
    """
    b = np.arange(1 << field_qubits)
    groups: dict[tuple[tuple[Band | Outer, ...], int], list[Term]] = {}
    found: list[tuple[float, _FieldOp, list[Band]]] = []
    for term in terms:
        if term.matching is not None:
            bands = [f for f in term.factors if isinstance(f, Band)]
            unitary = term.matching.is_permutation(field_qubits)
            images = term.matching.permutation(field_qubits)
            operation: _FieldOp = (
                _FieldOperation(0)
                if unitary and images == tuple(range(1 << field_qubits))
                else _MatchingOperation(term.matching, unitary)
            )
            found.append((term.coefficient, operation, bands))
        else:
            groups.setdefault((term.factors, term.x_mask), []).append(term)
    for (factors, x_mask), members in groups.items():
        bands = [f for f in factors if isinstance(f, Band)]
        diagonal = np.zeros(b.size)
        for t in members:
            diagonal += t.coefficient * (1.0 - 2.0 * (np.bitwise_count(b & t.z_mask) & 1))
        entries = np.flatnonzero(np.abs(diagonal) > _TOLERANCE * np.max(np.abs(diagonal)))
        if entries.size == 1 and field_qubits > 0:
            c = int(entries[0])
            found.append((float(diagonal[c]), _FieldOperation(x_mask, column=c), bands))
        else:
            found.extend((t.coefficient, _FieldOperation(x_mask, t.z_mask), bands) for t in members)
    merged: dict[tuple[_FieldOp, tuple[Band, ...]], float] = {}
    for coefficient, operation, bands in found:
        key = (operation, tuple(bands))
        merged[key] = merged.get(key, 0.0) + coefficient
    return [(c, operation, list(bands)) for (operation, bands), c in merged.items() if c != 0.0]


@dataclass(frozen=True)
class _Slot(Generic[_T]):
    """One factored selector: the distinct values its terms use, and the one that needs no gates."""

    values: tuple[_T, ...]
    neutral: _T

    @property
    def width(self) -> int:
        """The selector's qubits."""
        return (len(self.values) - 1).bit_length()

    def active(self) -> list[tuple[int, _T]]:
        """Return each value that needs gates, with its index."""
        return [(i, value) for i, value in enumerate(self.values) if value != self.neutral]


def _slot(values: Sequence[_T], neutral: _T) -> _Slot[_T]:
    """Return the selector of ``values``, the neutral value first when it is used."""
    used = set(values)
    rest = sorted(used - {neutral}, key=repr)
    return _Slot((neutral, *rest) if neutral in used else tuple(rest), neutral)


def _is_band(term: Term) -> bool:
    return all(isinstance(f, Band) for f in term.factors)


def compile_system(ir: SystemIR, label: str = "") -> BlockEncoding:
    r"""Return the block encoding of ``ir``'s operator (the module documentation).

    Parameters
    ----------
    ir
        The system rows' terms.
    label
        Circuit label.

    Returns
    -------
    BlockEncoding
        On :attr:`Layout.n_sys` system qubits, shape :attr:`Layout.shape`, with rows in
        :func:`circuit_rows`' order. :math:`\alpha` is the bands' :math:`\sum_t|c_t|` plus, for
        each rank-one entry, its weight times :func:`rank_one_encoding`'s :math:`\alpha`.

    Raises
    ------
    ValueError
        If ``ir`` has no terms.
    NotImplementedError
        For a rank-one term with a band on another axis.
    """
    if not ir.terms:
        raise ValueError("compile_system needs at least one term")
    bands = tuple(t for t in ir.terms if _is_band(t))
    encodings: list[BlockEncoding] = []
    weights: list[complex] = []
    if bands:
        encodings.append(_band_lcu(replace(ir, terms=bands)))
        weights.append(1.0)
    for encoding, weight in _rank_one_blocks(ir, [t for t in ir.terms if not _is_band(t)]):
        encodings.append(encoding)
        weights.append(weight)
    if len(encodings) == 1 and weights[0] == 1.0:
        return replace(encodings[0], label=label or "A_sys")
    return lcu_sum(encodings, weights, label=label or "A_sys")


def _rank_one_blocks(ir: SystemIR, terms: Sequence[Term]) -> list[tuple[BlockEncoding, float]]:
    """Return each rank-one term group's matrix units as block encodings, with their weights."""
    layout = Layout.of(ir)
    patterns: dict[tuple[Band | Outer, ...], NDArray[np.float64]] = {}
    size = 1 << layout.field_qubits
    for term in terms:
        pattern = patterns.setdefault(term.factors, np.zeros((size, size)))
        word = pauli_word(term.x_mask, term.z_mask, layout.field_qubits).toarray()
        pattern += term.coefficient * word
    identity = Band(0, Diagonal("identity"))
    blocks = []
    for factors, pattern in patterns.items():
        axes: list[tuple[BlockEncoding, Sequence[int]]] = []
        for a, factor in enumerate(factors):
            if isinstance(factor, Outer):
                encoding = rank_one_encoding(factor, layout.qubits_in[a], layout.qubits_out[a])
                axes.append((encoding, layout.output(a)))
            elif factor != identity:
                raise NotImplementedError("a rank-one term takes the identity on its other axes")
        scale = float(np.max(np.abs(pattern)))
        for row, column in zip(*np.nonzero(np.abs(pattern) > _TOLERANCE * scale), strict=True):
            placement = field_placement(int(row), int(column), layout.field_qubits)
            block = place([(placement, layout.field()), *axes], layout, label="rank one")
            blocks.append((block, float(pattern[row, column])))
    return blocks


def rank_one_encoding(outer: Outer, n_in: int, n_out: int) -> BlockEncoding:
    r"""Return :math:`|u\rangle\langle\tau(x)|` from ``n_in`` qubits to ``n_out``.

    :func:`~pihm.circuits.standard.zero_encoding` gives :math:`\sqrt{N}|0\rangle\langle\tau(x)|`
    (:math:`N = 2^{n_{\rm in}}`), below an idle qubit if the axis is lifted, and a rotation tree
    (:func:`~pihm.circuits.compose.real_state_preparation`) takes :math:`|0\rangle` to
    :math:`u/\|u\|`; :math:`\alpha = \sqrt2\,\|u\|`, two ancilla.
    """
    u = np.asarray(outer.left, dtype=np.float64)
    size = 1 << n_out
    row = zero_encoding(outer.at, n_in)
    if n_out > n_in:
        row = pad_rows(row)
    loader = BlockEncoding(real_state_preparation(u, "S_u"), 1.0, n_out, 0, (size, size), "S_u")
    block = product(loader, row, label="|u><tau|")
    scale = float(np.linalg.norm(u)) / np.sqrt(float(1 << n_in))
    return replace(block, alpha=block.alpha * scale)


def _band_lcu(ir: SystemIR) -> BlockEncoding:
    """Return the factored LCU of ``ir``'s band terms (the module documentation)."""
    layout = Layout.of(ir)
    entries = _field_terms(ir.terms, layout.field_qubits)
    axes = range(len(layout.qubits_in))
    field: _Slot[_FieldOp] = _slot([op for _, op, _ in entries], _FieldOperation(0))
    diagonals = [_slot([b[a].diagonal for _, _, b in entries], Diagonal("identity")) for a in axes]
    shifts = [_slot([b[a].shift for _, _, b in entries], 0) for a in axes]
    slots: list[_Slot[Any]] = [field, *(s for a in axes for s in (diagonals[a], shifts[a]))]

    shifting = [a for a in axes if shifts[a].active()]
    projecting = any(op.needs_flag for op in field.values)
    n_sys = layout.n_sys
    overflow = {a: n_sys + i for i, a in enumerate(shifting)}
    flag = [n_sys + len(shifting)] if projecting else []
    start = n_sys + len(shifting) + len(flag)
    registers = []
    for slot in slots:
        registers.append(list(range(start, start + slot.width)))
        start += slot.width
    total = start
    selector = [q for register in registers for q in register]

    alpha = float(sum(abs(coefficient) for coefficient, _, _ in entries))
    size = 1 << len(selector)
    right, left = np.zeros(size), np.zeros(size)
    for coefficient, operation, band in entries:
        keys: list[Any] = [operation]
        for a in axes:
            keys.extend((band[a].diagonal, band[a].shift))
        joint, offset = 0, 0
        for slot, key in zip(slots, keys, strict=True):
            joint |= slot.values.index(key) << offset
            offset += slot.width
        weight = np.sqrt(abs(coefficient) / alpha)
        right[joint] = np.sign(coefficient) * weight
        left[joint] = weight

    qc = QuantumCircuit(total, name="A_bands")
    if selector:
        qc.compose(real_state_preparation(right, "PREPARE"), qubits=selector, inplace=True)
    elif right[0] < 0:
        qc.global_phase += np.pi
    for index, operation in field.active():
        branch = operation.circuit(layout.field_qubits, projecting)
        _select(qc, branch, registers[0], index, [*layout.field(), *flag])
    if any(isinstance(op, _MatchingOperation) and op.needs_flag for op in field.values):
        angles = np.zeros((1 << len(registers[0]), 1 << layout.field_qubits))
        for index, operation in enumerate(field.values):
            if isinstance(operation, _MatchingOperation) and operation.needs_flag:
                angles[index] = operation.angles(layout.field_qubits)
        entries_gate = UCRYGate(list(angles.ravel()))
        qc.append(entries_gate, [*flag, *layout.field(), *registers[0]])
    for a in axes:
        diagonal_register, shift_register = registers[1 + 2 * a], registers[2 + 2 * a]
        for index, diagonal in diagonals[a].active():
            branch = _diagonal_circuit(diagonal, layout.qubits_in[a])
            _select(qc, branch, diagonal_register, index, layout.data(a))
        for index, shift in shifts[a].active():
            wide = [*layout.output(a), overflow[a]]
            _select(qc, shift_circuit(shift, len(wide)), shift_register, index, wide)
    if selector:
        unprepare = inverse_circuit(real_state_preparation(left, "PREPARE"))
        qc.compose(unprepare, qubits=selector, inplace=True)
    return BlockEncoding(qc, alpha, n_sys, total - n_sys, layout.shape, "A_bands")


def field_placement(row: int, column: int, field_qubits: int) -> BlockEncoding:
    r"""Return :math:`|{\rm row}\rangle\langle{\rm column}|` on the field register.

    :math:`X` gates take ``column`` to zero, the zero projector
    (:func:`~pihm.circuits.primitives.zero_projector`) keeps it alone, and :math:`X` gates take
    zero to ``row``: one ancilla, :math:`\alpha = 1`.
    """
    size = 1 << field_qubits
    name = f"|{row}><{column}|"
    qc = QuantumCircuit(field_qubits + 1, name=name)
    for q in range(field_qubits):
        if (column >> q) & 1:
            qc.x(q)
    qc.compose(zero_projector(field_qubits).circuit, inplace=True)
    for q in range(field_qubits):
        if (row >> q) & 1:
            qc.x(q)
    return BlockEncoding(qc, 1.0, field_qubits, 1, (size, size), name)


def place(
    parts: Sequence[tuple[BlockEncoding, Sequence[int]]], layout: Layout, label: str = ""
) -> BlockEncoding:
    r"""Return the Kronecker product of ``parts`` on ``layout``'s system register.

    Each part acts on the system qubits beside it (its system qubit :math:`k` on the
    :math:`k`-th), with ancilla of its own above the register, and the identity acts on the
    qubits no part names; :math:`\alpha` is the parts' product. A lifted part keeps its extra
    output qubit last, where :meth:`Layout.output` puts the lift.

    Raises
    ------
    ValueError
        If two parts share a qubit, or a part is not given one qubit per system qubit.
    """
    named = [q for _, qubits in parts for q in qubits]
    if len(named) != len(set(named)):
        raise ValueError("place needs parts on disjoint qubits")
    for part, qubits in parts:
        if len(qubits) != part.n_sys_qubits:
            raise ValueError(
                f"{part.label} has {part.n_sys_qubits} system qubits, not {len(qubits)}"
            )
    n_sys = layout.n_sys
    total = n_sys + sum(part.n_anc_qubits for part, _ in parts)
    qc = QuantumCircuit(total, name=label or "placed")
    start, alpha = n_sys, 1.0
    for part, qubits in parts:
        ancilla = range(start, start + part.n_anc_qubits)
        qc.compose(part.circuit, qubits=[*qubits, *ancilla], inplace=True)
        start += part.n_anc_qubits
        alpha *= part.alpha
    return BlockEncoding(qc, alpha, n_sys, total - n_sys, layout.shape, label or "placed")
