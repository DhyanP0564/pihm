r"""Shared circuit primitives used by every encoder.

Every encoder in this package is tracked as a :class:`BlockEncoding`: a genuine
:class:`~qiskit.circuit.QuantumCircuit` plus its subnormalisation :math:`\alpha`, never a stored
dense matrix. The circuit's qubits are always **system qubits low, ancilla qubits high**
(indices ``0 .. n_sys_qubits - 1`` then ``n_sys_qubits .. n_sys_qubits + n_anc_qubits - 1``), so
that the *ancilla-all-zero* block is the leading corner of the state vector: for a state of the
system register padded with an ancilla in :math:`|0\rangle`, evolving through the circuit and
reading off the first ``shape[0]`` amplitudes gives exactly :math:`\alpha\,\Pi_{\rm out} U
\Pi_{\rm in}` applied to that system state (:func:`apply`). This is what "exact" mode
extracts, one system basis state (one statevector evolution) at a time, rather than ever forming
the full :math:`2^{n}\times2^{n}` unitary.

Every simulation in this package is a statevector evolution on the Aer simulator
(:func:`evolve`, after :func:`compile_circuit`): memory grows with the statevector,
:math:`2^q` amplitudes for :math:`q` qubits, never with a dense operator of a whole circuit, and Aer
applies multi-controlled gates natively.

This module holds the atoms that do not depend on the term intermediate representation
(:mod:`pihm.circuits.arithmetic` for the register subtractor; :mod:`pihm.circuits.compose` for
the algebra that combines these atoms):

* :func:`zero_projector`: the rank-one projector :math:`|0\rangle\langle0|`, block-encoded from
  one ancilla.
* :func:`reflection_about_zero`: the Hermitian *unitary* :math:`I - 2|0\rangle\langle0|`, needed
  wherever a reflection (not a projector) is composed directly, with no ancilla; optionally
  controlled on further qubits, as the qubitised walk's controlled reflections
  (:func:`pihm.circuits.compose.controlled_walk`).
* :func:`chebyshev_feature_map`: the paper's Fig. S1 preparation of (a state proportional to)
  the raw feature vector :math:`\tau(x)`, independently derived (the figure's
  exact multi-controlled wiring is not reconstructable at the resolution available to this
  library) and verified against :func:`pihm.basis.tau` to machine precision.
* :func:`pauli_z_string`: a single Pauli :math:`Z` on one qubit, embedded in the system register,
  as one term of a Pauli-string LCU (used for :math:`\Lambda` in
  :mod:`pihm.circuits.standard`).
* :func:`block_encode_atom`: a one-ancilla exact dilation of an arbitrary matrix, for atoms not
  (yet) built gate by gate and for exercising :mod:`pihm.circuits.compose` on generic operators.
* :class:`ConjugatedGate`: :math:`O\,C\,O^\dagger`, whose controlled form controls :math:`C`
  alone, since :math:`OO^\dagger = I` where the controls do not hold; and
  :func:`basis_permutation`, a permutation of basis states as transpositions of that form
  (:func:`transposition`).

**Controls.** :func:`controlled_operation` controls an instruction as it stands. A conjugated
gate is controlled on its core alone, and a uniformly controlled :math:`R_Y`
(:class:`~qiskit.circuit.library.UCRYGate`, :math:`2^k` CNOT for :math:`k` controls) takes a few
further controls as more of its own, with zero angles where they do not hold, rather than
controlling each of its gates.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Final

import numpy as np
import scipy.sparse as sp
from numpy.typing import NDArray
from qiskit import QuantumCircuit, transpile
from qiskit.circuit import ControlledGate, Gate, Instruction, Qubit
from qiskit.circuit.library import (
    MCPhaseGate,
    MCXGate,
    RYGate,
    UCRYGate,
    UnitaryGate,
    XGate,
    ZGate,
)
from qiskit_aer import AerSimulator
from qiskit_aer.library import SaveStatevector, SetStatevector

from pihm.operators.tensor import SparseArray

__all__ = [
    "BlockEncoding",
    "ConjugatedGate",
    "apply",
    "basis_permutation",
    "block_encode_atom",
    "chebyshev_feature_map",
    "compile_circuit",
    "controlled_matrix",
    "controlled_operation",
    "evolve",
    "evolve_basis",
    "identity_encoding",
    "inverse_circuit",
    "native_circuit",
    "pauli_z_string",
    "reflection_about_zero",
    "top_left_block",
    "transposition",
    "unitary_gate",
    "zero_projector",
]

#: The most further controls a uniformly controlled rotation takes as its own: each doubles its
#: :math:`2^k` CNOT, where controlling each of its gates costs about :math:`6s + 10` times.
_MULTIPLEXED_CONTROLS: Final[int] = 5

#: Gate fusion is off: these circuits are mostly multi-controlled :math:`X`, which fusion cannot
#: merge, and the dense blocks it builds from the rest cost more to apply than their gates (a
#: controlled walk step 1.2 to 2.1 times faster without it from 17 to 20 qubits, the state the
#: same to :math:`10^{-16}`).
_SIMULATOR = AerSimulator(method="statevector", precision="double", fusion_enable=False)


def _append_closed(
    qc: QuantumCircuit, base: Gate, controls: list[Qubit], targets: list[Qubit]
) -> None:
    r"""Append ``base`` controlled on ``controls`` all in :math:`|1\rangle`, in Aer's native gates.

    :math:`X` is ``mcx``; :math:`Z`, a phase and a global phase are ``mcphase``;
    :math:`R_Y(\theta)` is :math:`R_Y(\theta/2)\,\mathrm{MCX}\,R_Y(-\theta/2)\,\mathrm{MCX}`
    and :math:`H` is :math:`R_Y(\pi/4)\,\mathrm{MCZ}\,R_Y(-\pi/4)`, both exact. Any other gate
    is left for :func:`~qiskit.compiler.transpile` to decompose.
    """
    k = len(controls)
    name = base.name
    if name == "x":
        qc.append(MCXGate(k), [*controls, *targets])
    elif name in ("z", "p"):
        angle = np.pi if name == "z" else float(base.params[0])
        qc.append(MCPhaseGate(angle, k), [*controls, *targets])
    elif name == "global_phase":
        angle = float(base.params[0])
        if k == 1:
            qc.p(angle, controls[0])
        else:
            qc.append(MCPhaseGate(angle, k - 1), controls)
    elif name == "ry":
        (target,) = targets
        theta = float(base.params[0])
        qc.ry(theta / 2, target)
        qc.append(MCXGate(k), [*controls, target])
        qc.ry(-theta / 2, target)
        qc.append(MCXGate(k), [*controls, target])
    elif name == "h":
        (target,) = targets
        qc.ry(-np.pi / 4, target)
        qc.append(MCPhaseGate(np.pi, k), [*controls, target])
        qc.ry(np.pi / 4, target)
    else:
        qc.append(base.control(k), [*controls, *targets])


def native_circuit(circuit: QuantumCircuit) -> QuantumCircuit:
    """Return ``circuit`` with every controlled gate on native multi-controlled gates.

    An open control is a closed one between two :math:`X` gates, so a gate carrying many
    controls stays one ``mcx`` or ``mcphase`` instruction (``_append_closed``; both Aer and
    :func:`~qiskit.compiler.transpile` know them). The :math:`X` gates are carried forward, not
    emitted in pairs: a qubit left flipped by one gate is flipped back only when a later gate
    needs it otherwise, or at the end. :math:`X` is a permutation, so this removes only
    :math:`XX = I` pairs, and every other instruction keeps its place and order.
    """
    native = QuantumCircuit(*circuit.qregs, name=circuit.name, global_phase=circuit.global_phase)
    flipped: dict[Qubit, None] = {}  # qubits carrying an X not yet undone, in flip order

    def settle(qubits: list[Qubit], wanted: set[Qubit]) -> None:
        for q in qubits:
            if (q in flipped) != (q in wanted):
                native.x(q)
                if q in flipped:
                    del flipped[q]
                else:
                    flipped[q] = None

    for inst in circuit.data:
        operation = inst.operation
        if not isinstance(operation, ControlledGate):
            settle(list(inst.qubits), set())
            native.append(operation, inst.qubits)
            continue
        k = operation.num_ctrl_qubits
        controls, targets = list(inst.qubits[:k]), list(inst.qubits[k:])
        open_controls = {q for j, q in enumerate(controls) if not (operation.ctrl_state >> j) & 1}
        settle([*controls, *targets], open_controls)
        _append_closed(native, operation.base_gate, controls, targets)
    settle(list(flipped), set())
    return native


def compile_circuit(circuit: QuantumCircuit) -> QuantumCircuit:
    """Return ``circuit`` in the Aer statevector simulator's own instruction set.

    :func:`native_circuit`, then :func:`~qiskit.compiler.transpile` at ``optimization_level=0``
    for anything else the simulator does not support; nothing is optimised away. A circuit
    simulated many times is compiled once and passed to :func:`evolve` each time.
    """
    compiled: QuantumCircuit = transpile(
        native_circuit(circuit), backend=_SIMULATOR, optimization_level=0
    )
    return compiled


def evolve(circuit: QuantumCircuit, state: NDArray[np.complexfloating]) -> NDArray[np.complex128]:
    r"""Return ``circuit`` applied to ``state``: one statevector evolution on the Aer simulator.

    A linear map: ``state`` need not be normalised (it is scaled to unit norm for the simulator
    and back afterwards). The circuit's own global phase is applied here, because a simulation
    that starts from a given statevector does not carry it.

    Parameters
    ----------
    circuit
        In the simulator's own instruction set: :func:`compile_circuit`'s output, or any circuit of
        standard gates and :class:`~qiskit.circuit.library.UnitaryGate` instructions.
    state
        Length :math:`2^q` for the circuit's :math:`q` qubits, qubit 0 least significant.

    Returns
    -------
    numpy.ndarray
        The evolved statevector, the same length.
    """
    dim = 1 << circuit.num_qubits
    if state.shape != (dim,):
        raise ValueError(f"expected a length-{dim} state, got shape {state.shape}")
    norm = float(np.linalg.norm(state))
    if norm == 0.0:
        return np.zeros(dim, dtype=np.complex128)
    qc = QuantumCircuit(circuit.num_qubits)
    qc.append(SetStatevector(np.asarray(state, dtype=np.complex128) / norm), qc.qubits)
    qc.compose(circuit, inplace=True)
    qc.global_phase = 0.0
    qc.append(SaveStatevector(circuit.num_qubits), qc.qubits)
    evolved = np.asarray(_SIMULATOR.run(qc).result().get_statevector(), dtype=np.complex128)
    scale = norm * np.exp(1j * float(circuit.global_phase))
    result: NDArray[np.complex128] = evolved * scale
    return result


def evolve_basis(circuit: QuantumCircuit, index: int) -> NDArray[np.complex128]:
    r"""Return ``circuit`` applied to the basis state :math:`|{\rm index}\rangle`.

    As :func:`evolve`, but the state is prepared by :math:`X` gates rather than loaded, so no
    statevector passes to the simulator: the faster route when the input is a basis state.
    """
    qc = QuantumCircuit(circuit.num_qubits)
    for qubit in range(circuit.num_qubits):
        if (index >> qubit) & 1:
            qc.x(qubit)
    qc.compose(circuit, inplace=True)
    qc.global_phase = 0.0
    qc.append(SaveStatevector(circuit.num_qubits), qc.qubits)
    evolved = np.asarray(_SIMULATOR.run(qc).result().get_statevector(), dtype=np.complex128)
    result: NDArray[np.complex128] = evolved * np.exp(1j * float(circuit.global_phase))
    return result


@dataclass(frozen=True)
class BlockEncoding:
    r"""A block encoding of an operator: a circuit plus its subnormalisation.

    :math:`\Pi_{\rm out} U \Pi_{\rm in} = X/\alpha`, with :math:`\Pi_{\rm in}, \Pi_{\rm out}` the
    projectors onto ancilla :math:`|0\rangle`. System qubits occupy indices
    ``0 .. n_sys_qubits - 1`` of :attr:`circuit`; ancilla qubits occupy the rest.

    Attributes
    ----------
    circuit
        The unitary :math:`U`.
    alpha
        The subnormalisation :math:`\alpha \ge \|X\|_2`.
    n_sys_qubits
        System qubits; :math:`2^{\text{n\_sys\_qubits}} \ge \max(\text{shape})`.
    n_anc_qubits
        Ancilla qubits.
    shape
        The encoded operator's true (possibly rectangular) shape.
    label
        A short, human-readable name, for diagrams and error messages.
    """

    circuit: QuantumCircuit
    alpha: float
    n_sys_qubits: int
    n_anc_qubits: int
    shape: tuple[int, int]
    label: str = ""

    def __post_init__(self) -> None:
        """Check the circuit width against the declared register split."""
        total = self.n_sys_qubits + self.n_anc_qubits
        if self.circuit.num_qubits != total:
            raise ValueError(
                f"{self.label or 'encoding'}: circuit has {self.circuit.num_qubits} qubits, "
                f"expected n_sys_qubits + n_anc_qubits = {total}"
            )

    @property
    def n_qubits(self) -> int:
        """Total qubits, system plus ancilla."""
        return self.n_sys_qubits + self.n_anc_qubits


def apply(
    encoding: BlockEncoding,
    vector: NDArray[np.complexfloating],
    compiled: QuantumCircuit | None = None,
) -> NDArray[np.complexfloating]:
    r"""Return :math:`\alpha\,\Pi_{\rm out} U \Pi_{\rm in} |{\rm vector}\rangle`.

    Evolves the ancilla (:math:`|0\rangle`) and system (``vector``, padded to
    :math:`2^{\text{n\_sys\_qubits}}`) register through :attr:`~BlockEncoding.circuit` and reads
    off the leading ``shape[0]`` amplitudes (the ancilla-:math:`|0\rangle` block), scaled by
    :math:`\alpha`. One statevector evolution (:func:`evolve`) over
    :attr:`~BlockEncoding.n_qubits` qubits, independent of how large the system register is
    relative to ``vector``'s own support.

    Parameters
    ----------
    encoding
        The block encoding.
    vector
        Length ``shape[1]``.
    compiled
        The encoding's circuit already compiled (:func:`compile_circuit`), for a caller that
        applies it many times; compiled here otherwise.

    Returns
    -------
    numpy.ndarray
        Length ``shape[0]``: the target operator applied to ``vector``, if ``encoding`` is exact.
    """
    cols = encoding.shape[1]
    if vector.shape != (cols,):
        raise ValueError(f"expected a length-{cols} vector, got shape {vector.shape}")
    sys_dim = 1 << encoding.n_sys_qubits
    if cols > sys_dim:
        raise ValueError(f"shape[1]={cols} exceeds the system register's dimension {sys_dim}")
    full = np.zeros(1 << encoding.n_qubits, dtype=complex)
    full[:cols] = vector
    circuit = compiled if compiled is not None else compile_circuit(encoding.circuit)
    evolved = evolve(circuit, full)
    return (encoding.alpha * evolved[: encoding.shape[0]]).astype(np.complex128)


def top_left_block(encoding: BlockEncoding) -> NDArray[np.complexfloating]:
    r"""Return the dense matrix :math:`\alpha\,\Pi_{\rm out} U \Pi_{\rm in}`, column by column.

    One call to :func:`apply` per system basis vector (``shape[1]`` statevector evolutions), the
    circuit compiled once.
    Only feasible for a small system register; :mod:`pihm.verify`'s ``"exact"`` mode is the
    intended caller, at small resolutions (total qubits :math:`\lesssim 20`).

    Parameters
    ----------
    encoding
        The block encoding.

    Returns
    -------
    numpy.ndarray
        Dense ``shape`` array.
    """
    rows, cols = encoding.shape
    compiled = compile_circuit(encoding.circuit)
    block = np.empty((rows, cols), dtype=complex)
    basis = np.zeros(cols, dtype=complex)
    for k in range(cols):
        basis[k] = 1.0
        block[:, k] = apply(encoding, basis, compiled)
        basis[k] = 0.0
    return block


def zero_projector(n_qubits: int, label: str = "") -> BlockEncoding:
    r"""Return a block encoding of :math:`|0\rangle\langle0|` on ``n_qubits`` system qubits.

    One ancilla. :math:`X` on the ancilla, then an ``n_qubits``-controlled-on-:math:`|0\dots
    0\rangle` :math:`X` on the ancilla: for a system input of all zeros the ancilla is flipped
    and flipped back (net: unchanged, contributing 1 to the ancilla-:math:`|0\rangle` block's
    only non-zero entry); for any other input the ancilla ends at :math:`|1\rangle`, contributing
    0. An exact permutation, :math:`\alpha = 1`.

    Parameters
    ----------
    n_qubits
        System qubits.
    label
        Circuit label.

    Returns
    -------
    BlockEncoding
        Shape ``(N, N)``, :math:`N = 2^{\text{n\_qubits}}`.
    """
    N = 1 << n_qubits
    anc = n_qubits
    qc = QuantumCircuit(n_qubits + 1, name=label or "Pi0")
    qc.x(anc)
    qc.append(XGate().control(n_qubits, ctrl_state=0), [*range(n_qubits), anc])
    return BlockEncoding(qc, 1.0, n_qubits, 1, (N, N), label or "Pi0")


def reflection_about_zero(
    n_qubits: int, label: str = "", *, n_controls: int = 0, ctrl_state: int = 0
) -> QuantumCircuit:
    r"""Return the ancilla-free reflection :math:`I - 2|0\rangle\langle0|`, optionally controlled.

    :math:`X` on every target qubit (so :math:`|0\dots0\rangle \mapsto |1\dots1\rangle`), a
    :math:`Z` on the last target controlled on every other target (a phase flip exactly on
    :math:`|1\dots1\rangle`), then :math:`X` on every target again. Only the all-zero
    computational basis state ever picks up the :math:`-1`, so the net unitary is exactly
    :math:`I - 2|0\rangle\langle0|`: Hermitian, unitary, and needing no ancilla.

    With ``n_controls`` extra qubits (placed above the targets), the same :math:`Z` is also
    controlled on them equalling ``ctrl_state``; where they do not, the two :math:`X` layers cancel
    and the circuit acts as the identity. These are the controlled reflections a controlled
    qubitised walk needs.

    Parameters
    ----------
    n_qubits
        Qubits the reflection acts on.
    label
        Circuit label.
    n_controls
        Extra control qubits, indices ``n_qubits .. n_qubits + n_controls - 1``.
    ctrl_state
        The value the control qubits must hold, little-endian over them (qiskit's convention).

    Returns
    -------
    qiskit.circuit.QuantumCircuit
        ``(n_qubits + n_controls)``-qubit circuit.
    """
    if n_qubits < 1:
        raise ValueError("reflection_about_zero needs at least one qubit")
    if n_controls < 0 or not 0 <= ctrl_state < (1 << n_controls):
        raise ValueError(f"ctrl_state={ctrl_state} does not fit {n_controls} control qubits")
    targets = list(range(n_qubits))
    gate_controls = [*targets[:-1], *range(n_qubits, n_qubits + n_controls)]
    qc = QuantumCircuit(n_qubits + n_controls, name=label or "R0")
    qc.x(targets)
    if gate_controls:
        state = ((1 << (n_qubits - 1)) - 1) | (ctrl_state << (n_qubits - 1))
        z = ZGate().control(len(gate_controls), ctrl_state=state)
        qc.append(z, [*gate_controls, targets[-1]])
    else:
        qc.z(targets[-1])
    qc.x(targets)
    return qc


def chebyshev_feature_map(x: float, n_qubits: int, label: str = "") -> BlockEncoding:
    r"""Return a block encoding of (a scalar multiple of) :math:`|0\rangle\langle\tau(x)|`.

    One ancilla. Post-selecting it to :math:`|0\rangle` leaves the system register in the raw
    feature vector's *direction*: :math:`\tau(x)/\|\tau(x)\|` (the state a
    unitary circuit produces is always unit norm, so the true, x-dependent norm
    :math:`\|\tau(x)\|` is recovered separately, in :attr:`~BlockEncoding.alpha`).

    THE MECHANISM (Hadamard-test phase kick-back, independently derived; the paper's Fig. S1
    multi-controlled wiring is not confidently reconstructable at the resolution available here;
    it builds :math:`\tau(x)` as a combination of two product states). With
    :math:`\theta = \arccos x` and :math:`k` the integer indexed by the system qubits
    (little-endian):

    1. Hadamard every qubit (ancilla and system): a uniform superposition over the ancilla and
       :math:`k \in [0, N)`.
    2. An unconditional phase :math:`\mathrm{diag}(1, e^{-i 2^j\theta})` on system qubit
       :math:`j`: the system now carries an overall phase :math:`e^{-ik\theta}`.
    3. An ancilla-controlled (control 1) phase :math:`\mathrm{diag}(1, e^{i\,2\cdot2^j\theta})`
       on system qubit :math:`j`: combined with step 2, the ancilla-1 branch nets
       :math:`e^{+ik\theta}`, the ancilla-0 branch stays at :math:`e^{-ik\theta}`.
    4. A single system-all-zero–controlled :math:`R_Y(\pi/2)` on the ancilla, fixing the relative
       weight of the constant mode (:math:`\phi_0 = T_0/2^{n/2}` against :math:`\phi_{k\ge1} =
       \sqrt2\,T_k/2^{n/2}`): only the :math:`k = 0` term, where both ancilla
       branches share the same phase, is touched.
    5. Hadamard the ancilla.

    Post-selecting the ancilla to :math:`|0\rangle` leaves the system register proportional to
    :math:`(T_0(x)/\sqrt2, T_1(x), \dots, T_{N-1}(x)) = 2^{(n-1)/2}\tau(x)`.

    Parameters
    ----------
    x
        A point in :math:`[-1, 1]`.
    n_qubits
        System qubits.
    label
        Circuit label.

    Returns
    -------
    BlockEncoding
        Shape ``(N, N)``. Only column 0 of the top-left block has the :math:`\tau(x)` structure;
        :attr:`~BlockEncoding.alpha` is set so that reading out that column
        alone (via :func:`apply` on :math:`e_0`) reproduces
        :func:`pihm.operators.constraints.zero_row`'s row 0, transposed.
    """
    theta = float(np.arccos(np.clip(x, -1.0, 1.0)))
    anc = n_qubits
    sys_qubits = list(range(n_qubits))
    qc = QuantumCircuit(n_qubits + 1, name=label or f"Utau({x:.4g})")
    qc.h(anc)
    qc.h(sys_qubits)
    for idx, q in enumerate(sys_qubits):
        qc.p(-(2**idx) * theta, q)
    for idx, q in enumerate(sys_qubits):
        qc.cp(2 * (2**idx) * theta, anc, q)
    mcry = RYGate(np.pi / 2).control(n_qubits, ctrl_state=0)
    qc.append(mcry, [*sys_qubits, anc])
    qc.h(anc)
    N = 1 << n_qubits
    return BlockEncoding(qc, 1.0, n_qubits, 1, (N, N), label or f"Utau({x:.4g})")


def pauli_z_string(n_qubits: int, qubit: int | None, label: str = "") -> BlockEncoding:
    r"""Return a block encoding (no ancilla, :math:`\alpha = 1`) of a Pauli-:math:`Z` string.

    ``qubit=None`` gives the identity; otherwise :math:`Z` on that one system qubit, identity
    elsewhere (:math:`\Lambda = \frac{N-1}2 I - \sum_j 2^{j-1} Z_j`: each term
    is already exactly unitary, so no rotation ladder is needed to realise it).

    Parameters
    ----------
    n_qubits
        System qubits.
    qubit
        Which system qubit carries the :math:`Z`, or ``None`` for the identity term.
    label
        Circuit label.

    Returns
    -------
    BlockEncoding
        Shape ``(N, N)``, :math:`\alpha = 1`, no ancilla.
    """
    if qubit is not None and not 0 <= qubit < n_qubits:
        raise ValueError(f"qubit must be in [0, {n_qubits}) or None, got {qubit}")
    qc = QuantumCircuit(n_qubits, name=label or (f"Z{qubit}" if qubit is not None else "I"))
    if qubit is not None:
        qc.z(qubit)
    N = 1 << n_qubits
    return BlockEncoding(qc, 1.0, n_qubits, 0, (N, N), label or qc.name)


def identity_encoding(n_qubits: int, label: str = "") -> BlockEncoding:
    """Return a block encoding of the identity on ``n_qubits`` system qubits (no ancilla)."""
    return pauli_z_string(n_qubits, None, label=label or "I")


def unitary_gate(U: NDArray[Any], label: str | None = None) -> UnitaryGate:
    r"""Return a :class:`~qiskit.circuit.library.UnitaryGate` for a matrix unitary by construction.

    Every dense instruction this package builds (a controlled copy of another instruction, a
    PREPARE, a rotation, a dilation) is unitary by construction, so qiskit's own unitarity check
    (a dense :math:`d\times d` product per gate, the dominant cost of building a nested LCU) is
    skipped.
    """
    return UnitaryGate(U, label=label, check_input=False)


def inverse_circuit(circuit: QuantumCircuit) -> QuantumCircuit:
    """Return ``circuit``'s inverse, instruction by instruction in reverse.

    A dense instruction's adjoint goes through :func:`unitary_gate`, and a controlled gate's is
    its base gate's inverse under the same controls (:func:`controlled_operation`), so neither
    is re-checked nor re-synthesised.
    """
    inverse = QuantumCircuit(
        *circuit.qregs, name=f"{circuit.name}†", global_phase=-circuit.global_phase
    )
    for inst in reversed(circuit.data):
        operation = inst.operation
        if isinstance(operation, UnitaryGate):
            operation = unitary_gate(operation.to_matrix().conj().T)
        elif isinstance(operation, ControlledGate) and not isinstance(operation, MCXGate):
            operation = controlled_operation(
                operation.base_gate.inverse(), operation.num_ctrl_qubits, operation.ctrl_state
            )
        else:
            operation = operation.inverse()
        inverse.append(operation, inst.qubits)
    return inverse


def controlled_operation(operation: Instruction, n_controls: int, ctrl_state: int) -> Instruction:
    """Return ``operation`` controlled on ``n_controls`` qubits equalling ``ctrl_state``.

    The control qubits come first, as in :func:`controlled_matrix`. A gate keeps its structure:
    a controlled gate gains controls on the same base gate (an :math:`X` one is an
    :class:`~qiskit.circuit.library.MCXGate`), with no decomposition synthesised here, which is
    :func:`native_circuit`'s job; a dense :class:`~qiskit.circuit.library.UnitaryGate` becomes
    :func:`controlled_matrix`'s dense controlled copy. A :class:`ConjugatedGate` is controlled on
    its core, and a uniformly controlled :math:`R_Y` takes up to a few controls as its own (module
    documentation).
    """
    if isinstance(operation, ConjugatedGate):
        return operation.controlled(n_controls, ctrl_state)
    if isinstance(operation, UCRYGate) and n_controls <= _MULTIPLEXED_CONTROLS:
        return _multiplexed(operation, n_controls, ctrl_state)
    if isinstance(operation, UnitaryGate):
        return unitary_gate(controlled_matrix(operation.to_matrix(), n_controls, ctrl_state))
    base, inner, inner_state = operation, 0, 0
    if isinstance(operation, ControlledGate):
        base = operation.base_gate
        inner, inner_state = operation.num_ctrl_qubits, operation.ctrl_state
    total = n_controls + inner
    state = ctrl_state | (inner_state << n_controls)
    if base.name == "x":
        return MCXGate(total, ctrl_state=state)
    return ControlledGate(
        f"c{total}{base.name}",
        total + base.num_qubits,
        list(base.params),
        num_ctrl_qubits=total,
        ctrl_state=state,
        base_gate=base,
    )


def _multiplexed(operation: UCRYGate, n_controls: int, ctrl_state: int) -> Instruction:
    """Return a uniformly controlled :math:`R_Y` with the further controls as its own."""
    angles = [float(a) for a in operation.params]
    k = operation.num_qubits - 1
    widened = [0.0] * (len(angles) << n_controls)
    widened[ctrl_state << k : (ctrl_state + 1) << k] = angles
    qc = QuantumCircuit(n_controls + 1 + k, name="ucry")
    # The target, the gate's own controls, then the further ones, which read as the high bits.
    own = range(n_controls + 1, n_controls + 1 + k)
    qc.append(UCRYGate(widened), [n_controls, *own, *range(n_controls)])
    return qc.to_gate(label="ucry")


class ConjugatedGate(Gate):  # type: ignore[misc]  # qiskit ships no type information
    r"""The gate :math:`O\,C\,O^\dagger`, whose controlled form controls :math:`C` alone.

    Where the controls do not hold, :math:`OO^\dagger = I`, so SELECT and the row stack pay their
    controls on the core only (:func:`controlled_operation`).

    Parameters
    ----------
    outer
        :math:`O`, on every qubit of the gate.
    core
        :math:`C`.
    core_qubits
        The gate's qubits that :math:`C` acts on, in its own order.
    """

    def __init__(self, outer: QuantumCircuit, core: Instruction, core_qubits: Sequence[int]):
        """Record the parts; the definition is built when a synthesis asks for it."""
        super().__init__("conj", outer.num_qubits, [])
        self.outer = outer
        self.core = core
        self.core_qubits = tuple(core_qubits)

    def _define(self) -> None:
        qc = QuantumCircuit(self.num_qubits, name=self.name)
        qc.compose(self.outer, inplace=True)
        qc.append(self.core, list(self.core_qubits))
        qc.compose(inverse_circuit(self.outer), inplace=True)
        self.definition = qc

    def controlled(self, n_controls: int, ctrl_state: int) -> ConjugatedGate:
        """Return the gate controlled on ``n_controls`` qubits placed first, core alone."""
        outer = QuantumCircuit(n_controls + self.num_qubits)
        outer.compose(self.outer, qubits=range(n_controls, outer.num_qubits), inplace=True)
        core = controlled_operation(self.core, n_controls, ctrl_state)
        qubits = [*range(n_controls), *(n_controls + q for q in self.core_qubits)]
        return ConjugatedGate(outer, core, qubits)

    def inverse(self, annotated: bool = False) -> ConjugatedGate:
        r"""Return :math:`O\,C^\dagger O^\dagger`."""
        return ConjugatedGate(self.outer, self.core.inverse(), self.core_qubits)


def transposition(a: int, b: int, width: int) -> ConjugatedGate:
    r"""Return the swap of basis states :math:`|a\rangle` and :math:`|b\rangle`, every other fixed.

    A CNOT cascade from one qubit where they differ leaves them differing there alone; a
    multi-controlled :math:`X` on it, controlled on their shared value of every other qubit, swaps
    them, and the cascade is undone: :math:`2(d - 1)` CNOT and one :math:`(w - 1)`-controlled
    :math:`X` for Hamming distance :math:`d` on :math:`w` qubits.
    """
    a, b = int(a), int(b)
    difference = a ^ b
    if not difference:
        raise ValueError("a transposition needs two different states")
    target = (difference & -difference).bit_length() - 1
    outer = QuantumCircuit(width, name="cascade")
    for q in range(width):
        if q != target and (difference >> q) & 1:
            outer.cx(target, q)
    unchanged = a if not (a >> target) & 1 else b
    controls = [q for q in range(width) if q != target]
    state = sum(((unchanged >> q) & 1) << i for i, q in enumerate(controls))
    core: Instruction = MCXGate(len(controls), ctrl_state=state) if controls else XGate()
    return ConjugatedGate(outer, core, [*controls, target])


def basis_permutation(images: Sequence[int], label: str = "") -> QuantumCircuit:
    r"""Return the circuit :math:`|j\rangle \mapsto |{\rm images}[j]\rangle` on its basis states.

    Each cycle :math:`(c_0\,c_1 \dots c_{L-1})` is :math:`L - 1` transpositions
    (:func:`transposition`), :math:`(c_{L-2}\,c_{L-1})` first.
    """
    images = [int(image) for image in images]
    size = len(images)
    if size < 1 or size & (size - 1) or sorted(images) != list(range(size)):
        raise ValueError("a basis permutation needs a permutation of a power-of-two range")
    width = size.bit_length() - 1
    qc = QuantumCircuit(width, name=label or "P")
    seen: set[int] = set()
    for start in range(size):
        if start in seen or images[start] == start:
            continue
        cycle = [start]
        while images[cycle[-1]] != start:
            cycle.append(images[cycle[-1]])
        seen.update(cycle)
        for i in reversed(range(len(cycle) - 1)):
            qc.append(transposition(cycle[i], cycle[i + 1], width), range(width))
    return qc


def controlled_matrix(
    U: NDArray[Any], s_qubits: int, ctrl_state: int
) -> NDArray[np.complexfloating]:
    """Return ``U`` controlled on ``s_qubits`` selector qubits equalling ``ctrl_state``.

    Built directly (a Kronecker product with a control-value projector): the controlled form of
    a dense instruction, which has no gate structure for :meth:`~qiskit.circuit.Gate.control` to
    keep (:func:`controlled_operation`; :func:`pihm.circuits.gqsp.evolve_gqsp`'s rotations).

    The qubit convention matches this module's own: append the returned matrix with the
    selector qubits listed *before* ``U``'s own target qubits
    (``qc.append(unitary_gate(controlled_matrix(U, s, i)), selector_qubits + target_qubits)``).
    """
    dim = U.shape[0]
    S = 1 << s_qubits
    P = np.zeros((S, S), dtype=complex)
    P[ctrl_state, ctrl_state] = 1.0
    return np.kron(U, P) + np.kron(np.eye(dim, dtype=complex), np.eye(S, dtype=complex) - P)


def block_encode_atom(
    A: SparseArray | NDArray[Any], alpha: float | None = None, label: str = ""
) -> BlockEncoding:
    r"""Return a one-ancilla exact block encoding of an arbitrary matrix (a classical dilation).

    Not a gate-level construction: :attr:`~BlockEncoding.circuit` holds one dense
    :class:`~qiskit.circuit.library.UnitaryGate`, built from ``A``'s SVD by the standard
    block-encoding dilation

    .. math::

       U = \begin{pmatrix} A/\alpha & \sqrt{I - AA^\dagger/\alpha^2} \\
                            \sqrt{I - A^\dagger A/\alpha^2} & -A^\dagger/\alpha \end{pmatrix},

    which is unitary exactly when :math:`\alpha \ge \|A\|_2`. It is a black-box fallback for
    an atom without a gate-level circuit, and it exercises :mod:`pihm.circuits.compose` against
    generic operators.

    Parameters
    ----------
    A
        A square or rectangular matrix, side lengths powers of two.
    alpha
        Subnormalisation; defaults to :math:`\|A\|_2` (dense SVD).
    label
        Circuit label.

    Returns
    -------
    BlockEncoding
        Shape ``A.shape``, one ancilla, exact to machine precision.

    Raises
    ------
    ValueError
        If ``max(A.shape)`` is not a power of two, or ``alpha < \|A\|_2``.
    """
    dense = np.asarray(A.todense() if sp.issparse(A) else A, dtype=complex)
    max_dim = max(dense.shape)
    n_sys = max_dim.bit_length() - 1
    if (1 << n_sys) != max_dim:
        raise ValueError(f"block_encode_atom needs a power-of-two max(shape), got {dense.shape}")
    U, s, Vh = np.linalg.svd(dense)
    if alpha is None:
        alpha = float(s[0]) if s.size else 1.0
    if s.size and s[0] > alpha * (1 + 1e-9):
        raise ValueError(f"alpha={alpha} is smaller than ||A||_2={float(s[0])}")
    dim = 1 << n_sys
    padded = np.zeros((dim, dim), dtype=complex)
    padded[: dense.shape[0], : dense.shape[1]] = dense / alpha
    Up = np.eye(dim, dtype=complex)
    Up[: U.shape[0], : U.shape[0]] = U
    Vp = np.eye(dim, dtype=complex)
    Vp[: Vh.shape[0], : Vh.shape[0]] = Vh
    sp_ = np.zeros(dim)
    sp_[: s.size] = s / alpha
    cos = np.clip(sp_, -1.0, 1.0)
    sin = np.sqrt(np.clip(1.0 - cos**2, 0.0, None))
    block = np.block(
        [
            [padded, Up @ np.diag(sin) @ Up.conj().T],
            [Vp.conj().T @ np.diag(sin) @ Vp, -padded.conj().T],
        ]
    )
    qc = QuantumCircuit(n_sys + 1, name=label or "atom")
    qc.append(unitary_gate(block, label=label or None), range(n_sys + 1))
    return BlockEncoding(qc, float(alpha), n_sys, 1, dense.shape, label or "atom")
