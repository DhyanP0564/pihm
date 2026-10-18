r"""The Carleman lift's tensor layout as a circuit.

The tensor layout (:mod:`pihm.carleman`) carries level :math:`k` of the lift as the whole tensor
:math:`v^{\otimes k}` in :math:`k` slots, and its matrix is :math:`2K - 1` structured terms
whatever the number of modes. Here those terms are circuits, for a Fourier–Galerkin equation in
its structured form (:class:`~pihm.problems.fluids.Galerkin`), in the variables
:math:`v = \sqrt2\,\hat u` of :func:`~pihm.problems.fluids.fourier_field`.

**The register** (:class:`TensorLayout`): :math:`K` slots of :math:`d\,m` qubits, then a level
register of :math:`\lceil\log_2(K + 1)\rceil` qubits holding :math:`k \in 1..K`. Slot 0 is the
most significant, as in :func:`pihm.carleman.tensor_matrix`, and within a slot axis 0 is. A
wavevector's component sits in two's complement, :math:`k_d \bmod 2^m`, with
:math:`2^m \ge 3\max|k_d| + 1`, so that the product of two kept modes never aliases onto a kept
mode. Level :math:`k` occupies slots :math:`0..k-1`, each holding a kept wavevector, and leaves
the rest at :math:`|0\rangle`, the zero wavevector, which no truncation keeps: these are the
**physical** states (:meth:`TensorLayout.physical`).

**The fold** (:func:`fourier_fold`). The circular convolution
:math:`(c \ast d)_k = \sum_{p + q \equiv k} c_pd_q` is :math:`N^{d/2}\,\mathrm{QFT}^\dagger\Delta
(\mathrm{QFT} \otimes \mathrm{QFT})`: the QFT takes both copies' coefficients to values on the
grid, a CNOT from each qubit of the first copy to the second leaves the second at zero exactly
where the two grid points agree (:math:`\Delta`), and the inverse QFT returns the first to
coefficients. The second copy is swapped into ancilla, so the block is
:math:`[\text{fold};\,0]` at :math:`\alpha = N^{d/2} = 2^{dm/2}`, the convolution's own norm on
the register (the Chebyshev fold's counterpart, :mod:`pihm.circuits.folds`, with the QFT native
in place of the cited transform).

**The multipliers** (:func:`multiplier_encoding`): a symbol's real magnitude on each kept
wavevector of a slot, zero on every other register value, as one uniformly controlled
:math:`R_Y` on a flag qubit, :math:`\alpha` its largest magnitude; its phase
:math:`i^{\sum a_d}` joins the term's weight. A constant input multiplier is a weight alone.

**The terms** (:func:`pihm.carleman.tensor_terms`), with :math:`\Pi_S` the level register's
projector onto the levels :math:`S` (:func:`level_projector`):

* :math:`F_1` on slot :math:`j`: its multiplier times :math:`\Pi_{k > j}`;
* :math:`F_2` on slots :math:`j, j + 1`: :math:`\Pi_{k \ge j + 2}`, then each product's
  output multiplier (the Galerkin truncation) after the fold of its input multipliers, summed,
  then the slots above :math:`j` rotated down one, the emptied slot to the top, and the level
  lowered by one (:func:`shift_level`).

Their linear combination (:func:`tensor_encoding`), weighted by :math:`T/2` and, for
:math:`F_2`, the lift's scale :math:`s/\sqrt2`, block-encodes
:math:`\mathbb{A}_K` on the physical states, the tensor layout of
:func:`~pihm.problems.fluids.fourier_field` (:func:`tensor_target`), and maps them to physical
states only: :math:`F_1`'s multiplier and each product's output are zero off the truncation, so
the leak is zero. :func:`tensor_ir` applies the same terms to each physical basis state
classically, register values and all, which checks the construction at any size.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import product as ordered

import numpy as np
import scipy.sparse as sp
from numpy.typing import NDArray
from qiskit import QuantumCircuit
from qiskit.circuit.library import QFTGate, UCRYGate, XGate

from pihm.carleman import TensorTerm, tensor_matrix, tensor_terms
from pihm.circuits.compose import embed_system, lcu_sum, product
from pihm.circuits.lcu import shift_circuit
from pihm.circuits.primitives import BlockEncoding, compile_circuit, evolve_basis
from pihm.problems.fluids import Galerkin, Product, Symbol, fourier_field

__all__ = [
    "TensorLayout",
    "fourier_fold",
    "level_projector",
    "multiplier_encoding",
    "physical_block",
    "quadratic_encoding",
    "shift_level",
    "tensor_encoding",
    "tensor_ir",
    "tensor_target",
]


@dataclass(frozen=True)
class TensorLayout:
    """The fixed-width register of a Galerkin lift's tensor layout (module documentation).

    Attributes
    ----------
    wavevectors
        The truncation's wavevectors, in :func:`~pihm.problems.fluids.fourier_field`'s order.
    width
        :math:`m`, qubits per axis in a slot.
    order
        :math:`K`, the lift's order and the number of slots.
    """

    wavevectors: tuple[tuple[int, ...], ...]
    width: int
    order: int

    @classmethod
    def of(cls, galerkin: Galerkin, order: int) -> TensorLayout:
        """Return the smallest layout whose products of kept modes never alias onto one."""
        width = max(1, math.ceil(math.log2(3 * galerkin.largest + 1)))
        return cls(galerkin.wavevectors, width, order)

    @property
    def axes(self) -> int:
        """The number of periodic axes, :math:`d`."""
        return len(self.wavevectors[0])

    @property
    def slot_qubits(self) -> int:
        """Qubits per slot, :math:`dm`."""
        return self.axes * self.width

    @property
    def level_qubits(self) -> int:
        """Qubits of the level register, enough for :math:`K`."""
        return self.order.bit_length()

    @property
    def n_sys(self) -> int:
        """The system qubits, the slots and then the level."""
        return self.order * self.slot_qubits + self.level_qubits

    def slot(self, index: int) -> list[int]:
        """Return slot ``index``'s qubits, least significant first."""
        base = (self.order - 1 - index) * self.slot_qubits
        return list(range(base, base + self.slot_qubits))

    def level(self) -> list[int]:
        """Return the level register's qubits, least significant first."""
        base = self.order * self.slot_qubits
        return list(range(base, base + self.level_qubits))

    def register(self, k: Sequence[int]) -> int:
        """Return a wavevector's value in a slot: each component mod :math:`2^m`, axis 0 first."""
        value = 0
        for component in k:
            value = (value << self.width) | (component % (1 << self.width))
        return value

    def wavevector(self, value: int) -> tuple[int, ...]:
        """Return the wavevector a slot's value holds, each component in two's complement."""
        size = 1 << self.width
        components = []
        for _ in range(self.axes):
            value, component = divmod(value, size)
            components.append(component - size if component >= size // 2 else component)
        return tuple(reversed(components))

    def index(self, level: int, slots: Sequence[int]) -> int:
        """Return the system basis index of a level and its slots' values."""
        value = level
        for position in range(self.order):
            value = (value << self.slot_qubits) | (slots[position] if position < len(slots) else 0)
        return value

    def physical(self) -> NDArray[np.int64]:
        """Return the physical states' indices, in :func:`~pihm.carleman.tensor_matrix`'s order."""
        values = [self.register(k) for k in self.wavevectors]
        found = [
            self.index(level, chosen)
            for level in range(1, self.order + 1)
            for chosen in ordered(values, repeat=level)
        ]
        return np.asarray(found, dtype=np.int64)


def multiplier_encoding(symbol: Symbol, layout: TensorLayout, label: str = "") -> BlockEncoding:
    r"""Return a symbol's magnitude on a slot's kept wavevectors, zero elsewhere, on one flag.

    One uniformly controlled :math:`R_Y`, :math:`\cos(\theta_v/2)` the magnitude over
    :math:`\alpha` at each slot value :math:`v`; :math:`\alpha` the largest magnitude. The
    symbol's phase is the caller's.
    """
    width = layout.slot_qubits
    values = np.zeros(1 << width)
    for k in layout.wavevectors:
        values[layout.register(k)] = symbol.magnitude(k)
    alpha = float(np.max(np.abs(values)))
    if alpha == 0.0:
        raise ValueError(f"the symbol {symbol} vanishes on the truncation")
    qc = QuantumCircuit(width + 1, name=label or "L")
    angles = 2.0 * np.arccos(np.clip(values / alpha, -1.0, 1.0))
    qc.append(UCRYGate([float(a) for a in angles]), [width, *range(width)])
    dim = 1 << width
    return BlockEncoding(qc, alpha, width, 1, (dim, dim), qc.name)


def fourier_fold(layout: TensorLayout) -> BlockEncoding:
    r"""Return :math:`[\text{fold};\,0]` on two slots, the first (the higher) receiving the product.

    Module documentation; :math:`\alpha = 2^{dm/2}`, and one slot's worth of ancilla takes the
    second copy's residue.
    """
    width, m = layout.slot_qubits, layout.width
    first = list(range(width, 2 * width))
    second = list(range(width))
    qc = QuantumCircuit(3 * width, name="fold")

    def axis(copy: list[int], a: int) -> list[int]:
        start = (layout.axes - 1 - a) * m
        return copy[start : start + m]

    for copy in (first, second):
        for a in range(layout.axes):
            qc.append(QFTGate(m), axis(copy, a))
    for source, target in zip(first, second, strict=True):
        qc.cx(source, target)
    for a in range(layout.axes):
        qc.append(QFTGate(m).inverse(), axis(first, a))
    for qubit in second:
        qc.swap(qubit, 2 * width + qubit)
    dim = 1 << (2 * width)
    return BlockEncoding(qc, 2.0 ** (width / 2), 2 * width, width, (dim, dim), "fold")


def _on_slots(
    encoding: BlockEncoding, layout: TensorLayout, slots: Sequence[int], label: str
) -> BlockEncoding:
    """Return an encoding on ``slots`` (the first the most significant) placed in the register."""
    positions = [q for slot in reversed(slots) for q in layout.slot(slot)]
    return embed_system(encoding, positions, len(positions), label)


def _product_encoding(term: Product, layout: TensorLayout) -> tuple[BlockEncoding, complex]:
    """Return one product on two slots, output ∘ fold ∘ (left ⊗ right), and its weight."""
    pair = 2 * layout.slot_qubits
    weight = term.left.phase * term.right.phase * term.output.phase
    encoding = fourier_fold(layout)
    inputs: list[BlockEncoding] = []
    for symbol, slot, name in ((term.left, 0, "left"), (term.right, 1, "right")):
        if not any(symbol.derivatives) and symbol.laplacian == 0:
            weight *= symbol.coefficient
            continue
        single = multiplier_encoding(symbol, layout, name)
        positions = [
            q + (layout.slot_qubits if slot == 0 else 0) for q in range(layout.slot_qubits)
        ]
        inputs.append(embed_system(single, positions, pair, name))
    for multiplier in inputs:
        encoding = product(encoding, multiplier)
    output = multiplier_encoding(term.output, layout, "output")
    top = list(range(layout.slot_qubits, pair))
    encoding = product(embed_system(output, top, pair, "output"), encoding)
    return encoding, weight


def quadratic_encoding(galerkin: Galerkin, layout: TensorLayout) -> tuple[BlockEncoding, complex]:
    r"""Return :math:`F_2`'s structure on two slots, :math:`\sum` output ∘ fold ∘ (left ⊗ right).

    One product is returned with its weight; several are one linear combination, weight 1.
    Neither carries :math:`T/2` or :math:`1/\sqrt2`.
    """
    parts = [_product_encoding(term, layout) for term in galerkin.products]
    if len(parts) == 1:
        return parts[0]
    encodings, weights = zip(*parts, strict=True)
    return lcu_sum(list(encodings), list(weights), label="F2"), 1.0


def level_projector(layout: TensorLayout, levels: Sequence[int]) -> BlockEncoding:
    r"""Return :math:`\Pi_S` on the level register: a flag set unless the level is in ``levels``."""
    width = layout.level_qubits
    qc = QuantumCircuit(width + 1, name=f"P{sorted(levels)}")
    qc.x(width)
    for level in levels:
        qc.append(XGate().control(width, ctrl_state=level), [*range(width), width])
    dim = 1 << width
    return BlockEncoding(qc, 1.0, width, 1, (dim, dim), qc.name)


def shift_level(layout: TensorLayout, slot: int) -> QuantumCircuit:
    r"""Return the unitary after :math:`F_2` on slots ``slot``, ``slot + 1``.

    The slots from ``slot + 1`` up rotate down one, the emptied one to the top, and the level
    falls by one.
    """
    qc = QuantumCircuit(layout.n_sys, name="shift")
    for lower in range(slot + 1, layout.order - 1):
        for a, b in zip(layout.slot(lower), layout.slot(lower + 1), strict=True):
            qc.swap(a, b)
    qc.compose(shift_circuit(1, layout.level_qubits), qubits=layout.level(), inplace=True)
    return qc


def _term(
    term: TensorTerm,
    layout: TensorLayout,
    linear: BlockEncoding,
    quadratic: BlockEncoding,
) -> BlockEncoding:
    """Return one tensor term on the whole register (module documentation)."""
    n = layout.n_sys
    j = term.slot
    levels = list(term.levels(layout.order))
    projector = embed_system(level_projector(layout, levels), layout.level(), n, "P")
    if term.degree == 1:
        placed = embed_system(linear, layout.slot(j), n, f"F1[{j}]")
        return product(placed, projector, f"F1[{j}]")
    pair = [*layout.slot(j + 1), *layout.slot(j)]
    placed = embed_system(quadratic, pair, n, f"F2[{j}]")
    shift = shift_level(layout, j)
    dim = 1 << n
    moved = BlockEncoding(shift, 1.0, n, 0, (dim, dim), "shift")
    return product(moved, product(placed, projector), f"F2[{j}]")


def tensor_encoding(
    galerkin: Galerkin, order: int, scale: float = 1.0, label: str = ""
) -> BlockEncoding:
    r"""Return a block encoding of :math:`\mathbb{A}_K` in the tensor layout (module documentation).

    Parameters
    ----------
    galerkin
        The equation's structured form.
    order
        :math:`K`.
    scale
        The lift's level scale :math:`s` (:func:`pihm.assemble.systems.lift_scale`), which
        multiplies :math:`F_2`.
    label
        Circuit label.

    Returns
    -------
    BlockEncoding
        On :attr:`TensorLayout.n_sys` system qubits, :math:`2K - 1` terms in one linear
        combination; on the physical states it is :func:`tensor_target`.
    """
    layout = TensorLayout.of(galerkin, order)
    half = galerkin.time / 2.0
    linear = multiplier_encoding(galerkin.linear, layout, "F1")
    quadratic, weight = quadratic_encoding(galerkin, layout)
    terms, weights = [], []
    for term in tensor_terms(order, (1, 2) if galerkin.products else (1,)):
        terms.append(_term(term, layout, linear, quadratic))
        if term.degree == 1:
            weights.append(half * galerkin.linear.phase)
        else:
            weights.append(half * scale * weight / math.sqrt(2.0))
    return lcu_sum(terms, weights, label=label or f"A[K={order}]")


def physical_block(
    encoding: BlockEncoding, layout: TensorLayout
) -> tuple[NDArray[np.complex128], float]:
    r"""Return the encoding's block on the physical states, and its largest leak out of them.

    One statevector evolution per physical state, the circuit compiled once: the block's rows
    and columns are :meth:`TensorLayout.physical`'s, and the leak is the largest magnitude the
    block puts on any other system state,
    :math:`\alpha\,\|(I - \Pi)\langle 0|U|0\rangle\Pi\|_{\max}`.
    """
    physical = layout.physical()
    compiled = compile_circuit(encoding.circuit)
    outside = np.ones(1 << layout.n_sys, dtype=bool)
    outside[physical] = False
    block = np.empty((physical.size, physical.size), dtype=np.complex128)
    leak = 0.0
    for column, index in enumerate(physical):
        image = encoding.alpha * evolve_basis(compiled, int(index))[: 1 << layout.n_sys]
        block[:, column] = image[physical]
        leak = max(leak, float(np.max(np.abs(image[outside]), initial=0.0)))
    return block, leak


def tensor_target(galerkin: Galerkin, order: int, scale: float = 1.0) -> sp.csr_array:
    r"""Return :math:`\mathbb{A}_K` in the tensor layout, the lift of the Fourier variables.

    Of :func:`~pihm.problems.fluids.fourier_field`'s vector field.

    The lift's scale multiplies :math:`F_2`. Rows and columns are :meth:`TensorLayout.physical`'s.
    """
    linear, quadratic = fourier_field(galerkin)
    return tensor_matrix([linear, quadratic * scale], order)


def tensor_ir(galerkin: Galerkin, order: int, scale: float = 1.0) -> sp.csr_array:
    r"""Return the circuit's terms applied to each physical basis state, classically.

    Each term is applied as its circuit acts on register values: the level's projector, the
    multipliers on each slot's two's-complement wavevector, the fold's sum taken modulo
    :math:`2^m` per axis, the output's truncation, the slots' rotation and the level's decrement.
    The result is indexed by :meth:`TensorLayout.physical`, with any image outside the physical
    states an error, so it checks the construction's algebra, and the leak, at any size.
    """
    layout = TensorLayout.of(galerkin, order)
    physical = layout.physical()
    position = {int(index): i for i, index in enumerate(physical)}
    kept = set(galerkin.wavevectors)
    half = galerkin.time / 2.0
    rows: list[int] = []
    columns: list[int] = []
    values: list[complex] = []

    def emit(target: int, column: int, value: complex) -> None:
        if value == 0:
            return
        if target not in position:
            raise ValueError(f"a term leaks from physical state {column} to {target}")
        rows.append(position[target])
        columns.append(column)
        values.append(value)

    slot_mask = (1 << layout.slot_qubits) - 1
    for column, index in enumerate(physical):
        level = int(index) >> (order * layout.slot_qubits)
        slots = [
            (int(index) >> ((order - 1 - s) * layout.slot_qubits)) & slot_mask for s in range(order)
        ]
        for term in tensor_terms(order, (1, 2) if galerkin.products else (1,)):
            j = term.slot
            if level not in term.levels(order):
                continue
            if term.degree == 1:
                k = layout.wavevector(slots[j])
                entry = galerkin.linear.value(k) if k in kept else 0.0
                emit(int(index), column, half * entry)
                continue
            p, q = layout.wavevector(slots[j]), layout.wavevector(slots[j + 1])
            # The fold adds each axis's components modulo 2^m.
            out = layout.wavevector(layout.register([a + b for a, b in zip(p, q, strict=True)]))
            entry = sum(
                t.output.value(out) * t.left.value(p) * t.right.value(q)
                for t in galerkin.products
                if out in kept
            )
            moved = [*slots[:j], layout.register(out), *slots[j + 2 :], 0]
            emit(layout.index(level - 1, moved), column, half * scale * entry / math.sqrt(2.0))
    count = len(physical)
    return sp.csr_array((values, (rows, columns)), shape=(count, count), dtype=complex)
