"""Checks of the shared circuit primitives."""

from __future__ import annotations

import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.circuit import Gate
from qiskit.circuit.library import (
    GlobalPhaseGate,
    HGate,
    PhaseGate,
    RYGate,
    UCRYGate,
    XGate,
    ZGate,
)
from qiskit.circuit.random import random_circuit
from qiskit.quantum_info import Operator, Statevector

from pihm.basis import tau
from pihm.circuits.primitives import (
    BlockEncoding,
    apply,
    basis_permutation,
    block_encode_atom,
    chebyshev_feature_map,
    compile_circuit,
    controlled_matrix,
    controlled_operation,
    evolve,
    identity_encoding,
    inverse_circuit,
    native_circuit,
    pauli_z_string,
    reflection_about_zero,
    top_left_block,
    transposition,
    unitary_gate,
    zero_projector,
)


def test_evolve_matches_an_independent_statevector_simulation(rng: np.random.Generator) -> None:
    """The Aer evolution agrees with qiskit's own Statevector, global phase and norm included."""
    circuit = random_circuit(5, 6, seed=11)
    circuit.append(ZGate().control(3, ctrl_state=0b101), [0, 1, 2, 4])
    circuit.global_phase = 0.4
    state = rng.normal(size=32) + 1j * rng.normal(size=32)
    norm = np.linalg.norm(state)
    expected = Statevector(state / norm).evolve(circuit).data * norm
    np.testing.assert_allclose(evolve(compile_circuit(circuit), state), expected, atol=1e-12)


def test_evolve_rejects_a_state_of_the_wrong_length() -> None:
    """The state must match the circuit's width."""
    with pytest.raises(ValueError, match="length-4"):
        evolve(QuantumCircuit(2), np.zeros(8, dtype=complex))


def test_evolve_maps_zero_to_zero() -> None:
    """The evolution is linear: the zero vector, which no simulator can load, maps to zero."""
    np.testing.assert_array_equal(evolve(QuantumCircuit(2), np.zeros(4, dtype=complex)), 0.0)


@pytest.mark.parametrize("n", range(1, 4))
def test_zero_projector_is_the_rank_one_projector(n: int) -> None:
    """zero_projector(n) block-encodes |0><0| exactly, alpha = 1."""
    E = zero_projector(n)
    N = 1 << n
    target = np.zeros((N, N))
    target[0, 0] = 1.0
    np.testing.assert_allclose(top_left_block(E), target, atol=1e-10)
    assert E.alpha == 1.0


@pytest.mark.parametrize("n", range(1, 4))
def test_reflection_about_zero(n: int) -> None:
    """reflection_about_zero(n) is I - 2|0><0|, unitary and self-inverse, no ancilla."""
    circuit = reflection_about_zero(n)
    assert circuit.num_qubits == n
    matrix = Operator(circuit).data
    target = np.eye(1 << n) - 2.0 * np.eye(1, 1 << n)[0][:, None] @ np.eye(1, 1 << n)
    np.testing.assert_allclose(matrix, target, atol=1e-10)
    np.testing.assert_allclose(matrix @ matrix, np.eye(1 << n), atol=1e-10)


@pytest.mark.parametrize(
    ("n", "n_controls", "ctrl_state"), [(1, 1, 0), (1, 1, 1), (3, 1, 0), (2, 2, 2), (3, 2, 3)]
)
def test_reflection_about_zero_acts_only_where_the_controls_match(
    n: int, n_controls: int, ctrl_state: int
) -> None:
    """Controlled, it is I - 2|0><0| where the controls equal ctrl_state and I elsewhere."""
    circuit = reflection_about_zero(n, n_controls=n_controls, ctrl_state=ctrl_state)
    assert circuit.num_qubits == n + n_controls
    N, C = 1 << n, 1 << n_controls
    reflection = np.eye(N)
    reflection[0, 0] = -1.0
    target = np.zeros((N * C, N * C))
    for c in range(C):
        projector = np.zeros((C, C))
        projector[c, c] = 1.0
        target += np.kron(projector, reflection if c == ctrl_state else np.eye(N))
    np.testing.assert_allclose(Operator(circuit).data, target, atol=1e-10)


def test_reflection_about_zero_rejects_a_control_value_out_of_range() -> None:
    """ctrl_state must fit the control register."""
    with pytest.raises(ValueError, match="ctrl_state"):
        reflection_about_zero(2, n_controls=1, ctrl_state=2)


@pytest.mark.parametrize("n", range(1, 4))
def test_pauli_z_string_and_identity(n: int) -> None:
    """pauli_z_string places Z on exactly one system qubit; qubit=None gives the identity."""
    N = 1 << n
    identity = top_left_block(identity_encoding(n))
    np.testing.assert_allclose(identity, np.eye(N), atol=1e-10)
    for q in range(n):
        block = top_left_block(pauli_z_string(n, q))
        diag = np.array([1.0 - 2.0 * ((k >> q) & 1) for k in range(N)])
        np.testing.assert_allclose(np.diag(block), diag, atol=1e-10)
        np.testing.assert_allclose(block, np.diag(diag), atol=1e-10)


@pytest.mark.parametrize("n", range(1, 5))
@pytest.mark.parametrize("x", [-1.0, -0.37, 0.0, 0.62, 1.0])
def test_feature_map_column_zero_is_tau_over_sqrt2(n: int, x: float) -> None:
    """Column 0 of the feature map's top-left block equals tau(x)/sqrt(2) exactly.

    This constant (not the naive but wrong alpha = sqrt(N/p(x)), where p(x) is the measured
    success probability) is what pihm.circuits.standard.zero_encoding relies on to reach the
    fixed prefactor alpha_B = sqrt(2N) for every x.
    """
    E = chebyshev_feature_map(x, n)
    N = 1 << n
    e0 = np.zeros(N, dtype=complex)
    e0[0] = 1.0
    col0 = apply(E, e0)
    np.testing.assert_allclose(col0, tau(x, n) / np.sqrt(2.0), atol=1e-9)


@pytest.mark.parametrize("shape", [(2, 2), (4, 4), (2, 4), (4, 2)])
def test_block_encode_atom(shape: tuple[int, int], rng: np.random.Generator) -> None:
    """block_encode_atom reproduces an arbitrary (possibly rectangular) matrix exactly."""
    A = rng.normal(size=shape) + 1j * rng.normal(size=shape)
    alpha = float(np.linalg.norm(A, 2)) * 1.2
    E = block_encode_atom(A, alpha=alpha)
    assert E.shape == shape
    assert E.alpha == alpha
    np.testing.assert_allclose(top_left_block(E), A, atol=1e-9)


def test_block_encode_atom_rejects_too_small_alpha(rng: np.random.Generator) -> None:
    """Alpha below ||A||_2 cannot give a valid (unitary) dilation."""
    A = rng.normal(size=(2, 2))
    with pytest.raises(ValueError, match="smaller than"):
        block_encode_atom(A, alpha=1e-6)


def test_block_encoding_rejects_a_circuit_of_the_wrong_width() -> None:
    """The dataclass checks n_sys_qubits + n_anc_qubits against the circuit's own width."""
    with pytest.raises(ValueError, match="circuit has"):
        BlockEncoding(QuantumCircuit(2), 1.0, 1, 2, (2, 2))


@pytest.mark.parametrize(
    "gate",
    [
        XGate(),
        XGate().control(2, ctrl_state=0b01),
        RYGate(0.7),
        RYGate(0.7).control(1, ctrl_state=0),
        HGate(),
        ZGate().control(2, ctrl_state=0b10),
        PhaseGate(1.1).control(1),
        GlobalPhaseGate(0.9),
    ],
    ids=["x", "open mcx", "ry", "open cry", "h", "open ccz", "cp", "global phase"],
)
@pytest.mark.parametrize(("n_controls", "ctrl_state"), [(1, 0), (2, 0b10), (3, 0b101)])
def test_controlled_operation_matches_the_dense_controlled_matrix(
    gate: Gate, n_controls: int, ctrl_state: int
) -> None:
    """Symbolic control, compiled to native gates, is controlled_matrix's dense matrix exactly."""
    width = n_controls + gate.num_qubits
    qc = QuantumCircuit(width)
    qc.append(controlled_operation(gate, n_controls, ctrl_state), range(width))
    base = Operator(gate).data if gate.num_qubits else np.exp(1j * gate.params[0]) * np.eye(1)
    expected = controlled_matrix(base, n_controls, ctrl_state)
    np.testing.assert_allclose(Operator(native_circuit(qc)).data, expected, atol=1e-12)


def test_controlled_operation_keeps_a_dense_gate_dense(rng: np.random.Generator) -> None:
    """A UnitaryGate has no structure to keep: its controlled form is the dense matrix."""
    U, _ = np.linalg.qr(rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4)))
    controlled = controlled_operation(unitary_gate(U), 2, 0b01)
    np.testing.assert_allclose(controlled.to_matrix(), controlled_matrix(U, 2, 0b01), atol=1e-14)


def test_inverse_circuit_is_the_inverse(rng: np.random.Generator) -> None:
    """inverse_circuit undoes a circuit of standard, controlled and dense gates, phase included."""
    U, _ = np.linalg.qr(rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4)))
    qc = random_circuit(4, 4, seed=3)
    qc.append(unitary_gate(U), [0, 2])
    qc.append(controlled_operation(RYGate(0.4), 2, 0b10), [1, 3, 0])
    qc.global_phase = 0.3
    native = native_circuit(qc)
    product = Operator(native_circuit(inverse_circuit(qc))).data @ Operator(native).data
    np.testing.assert_allclose(product, np.eye(16), atol=1e-12)


def test_native_circuit_carries_open_control_flips_forward() -> None:
    """Consecutive open controls on a qubit share one X pair; the operator is unchanged."""
    qc = QuantumCircuit(4)
    qc.append(controlled_operation(XGate(), 2, 0b00), [0, 1, 3])
    qc.append(controlled_operation(RYGate(0.3), 2, 0b10), [0, 1, 2])
    qc.append(controlled_operation(ZGate(), 1, 0), [0, 3])
    qc.h(1)
    native = native_circuit(qc)
    x_count = sum(1 for inst in native.data if inst.operation.name == "x")
    assert x_count == 4  # qubit 0 flipped once and restored once; qubit 1 the same
    expected = np.eye(16, dtype=complex)
    for inst in qc.data:  # each gate on its own, its open controls flipped and restored
        single = QuantumCircuit(4)
        single.append(inst.operation, inst.qubits)
        expected = Operator(native_circuit(single)).data @ expected
    np.testing.assert_allclose(Operator(native).data, expected, atol=1e-12)


@pytest.mark.parametrize("width", [1, 2, 3, 4])
def test_a_basis_permutation_sends_each_state_to_its_image(
    width: int, rng: np.random.Generator
) -> None:
    images = [int(i) for i in rng.permutation(1 << width)]
    expected = np.zeros((1 << width, 1 << width))
    expected[images, range(1 << width)] = 1.0
    circuit = basis_permutation(images)
    np.testing.assert_allclose(Operator(circuit).data, expected, atol=1e-12)
    np.testing.assert_allclose(Operator(inverse_circuit(circuit)).data, expected.T, atol=1e-12)


def test_a_controlled_transposition_controls_its_core_alone() -> None:
    """The cascade stays uncontrolled; the swap happens only where the controls hold."""
    gate = transposition(0b0110, 0b1011, 4)
    controlled = controlled_operation(gate, 2, 0b01)
    circuit = QuantumCircuit(6)
    circuit.append(controlled, range(6))
    unitary = Operator(circuit).data
    for state in range(1 << 4):
        for control in range(4):
            swapped = {0b0110: 0b1011, 0b1011: 0b0110}.get(state, state)
            image = swapped if control == 0b01 else state
            assert abs(unitary[control + 4 * image, control + 4 * state] - 1.0) < 1e-12
    assert sum(inst.operation.name == "cx" for inst in controlled.outer.data) == 2
    with pytest.raises(ValueError, match="two different states"):
        transposition(3, 3, 2)


@pytest.mark.parametrize(("own", "further"), [(1, 1), (2, 3), (3, 2)])
def test_a_uniformly_controlled_rotation_takes_further_controls_as_its_own(
    own: int, further: int, rng: np.random.Generator
) -> None:
    gate = UCRYGate(list(rng.uniform(0.0, 2.0 * np.pi, 1 << own)))
    state = int(rng.integers(0, 1 << further))
    ours = QuantumCircuit(further + 1 + own)
    ours.append(controlled_operation(gate, further, state), range(further + 1 + own))
    reference = QuantumCircuit(further + 1 + own)
    reference.append(gate.control(further, ctrl_state=state), range(further + 1 + own))
    np.testing.assert_allclose(Operator(ours).data, Operator(reference).data, atol=1e-12)
