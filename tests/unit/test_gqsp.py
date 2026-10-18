"""Checks of GQSP on the qubitised walk."""

from __future__ import annotations

import numpy as np
import pytest
from qiskit import QuantumCircuit

from pihm.circuits.assemble import encode_standard_linear
from pihm.circuits.compose import walk
from pihm.circuits.gqsp import apply_gqsp, evolve_gqsp, gqsp_circuit, rotation_gate
from pihm.circuits.primitives import BlockEncoding, apply, top_left_block
from pihm.problems.ladder import problem
from pihm.qsp.filters import qite_coefficients
from pihm.qsp.phases import find_phases


def _dense_chebyshev_sum(
    evals: np.ndarray, evecs: np.ndarray, coefficients: np.ndarray
) -> np.ndarray:
    """Return sum_k coefficients[k] * T_k(diag(evals)) reconstructed in the original basis.

    Scales evecs' own columns before the right multiplication (evecs * weights) @ evecs.conj().T,
    not (weights * evecs.conj().T): the latter broadcasts weights against the wrong axis.
    """
    weights = np.polynomial.chebyshev.chebval(evals, coefficients)
    result: np.ndarray = (evecs * weights) @ evecs.conj().T
    return result


def _r1(n: int = 2) -> tuple[BlockEncoding, BlockEncoding]:
    """R1's U_R and walk at resolution n."""
    stacked, reflection = encode_standard_linear(problem("R1"), n)
    return stacked, walk(reflection)


@pytest.mark.parametrize(("theta", "phi"), [(0.0, 0.0), (0.3, 0.7), (np.pi / 2, -1.2), (1.5, 3.0)])
def test_rotation_gate_is_unitary(theta: float, phi: float) -> None:
    R = rotation_gate(theta, phi)
    assert np.allclose(R @ R.conj().T, np.eye(2), atol=1e-12)


def test_gqsp_circuit_at_degree_zero_is_a_scalar_rotation() -> None:
    # No controlled walk at all: the leading block is e^{i lam} R(theta_0, phi_0)[0, 0] times I.
    stacked, _W = _r1()
    theta, phi, lam = np.array([0.4]), np.array([0.9]), 0.3
    block = top_left_block(gqsp_circuit(theta, phi, lam, stacked))
    expected = np.exp(1j * lam) * rotation_gate(0.4, 0.9)[0, 0]
    np.testing.assert_allclose(block, expected * np.eye(block.shape[0]), atol=1e-12)


def test_gqsp_circuit_realises_h_prime_itself_for_p_z_equals_z() -> None:
    stacked, W = _r1()
    H_prime = top_left_block(W)
    phases = find_phases(np.array([0.0, 1.0], dtype=complex))
    gqsp = gqsp_circuit(phases.theta, phases.phi, phases.lam, stacked, rescale=phases.rescale)
    assert np.max(np.abs(top_left_block(gqsp) - H_prime)) < 1e-12


def test_pi_in_w_k_pi_in_equals_chebyshev_of_h_prime() -> None:
    # The identity, re-checked directly (one matrix power via circuit composition, then
    # one restriction -- never iterated apply-then-project), since T1's correctness rests on it.
    _stacked, W = _r1()
    H_prime = top_left_block(W)
    evals, evecs = np.linalg.eigh(H_prime)
    for k in (1, 2, 3):
        qc = QuantumCircuit(W.n_qubits)
        for _ in range(k):
            qc.compose(W.circuit, inplace=True)
        Wk = BlockEncoding(qc, 1.0, W.n_sys_qubits, W.n_anc_qubits, W.shape, f"W^{k}")
        coeffs = np.zeros(k + 1)
        coeffs[k] = 1.0
        expected = _dense_chebyshev_sum(evals, evecs, coeffs)
        assert np.max(np.abs(top_left_block(Wk) - expected)) < 1e-10


@pytest.mark.parametrize("tau", [1.0, 3.0, 5.0])
def test_gqsp_circuit_matches_the_qite_filters_dense_chebyshev_sum(tau: float) -> None:
    # Does the assembled circuit realise the same polynomial qsp.prepare.apply_qite_filter applies
    # classically? Cross-checked against an independent dense eigenbasis reconstruction.
    stacked, W = _r1()
    H_prime = top_left_block(W)
    evals, evecs = np.linalg.eigh(H_prime)
    qf = qite_coefficients(tau, tolerance=1e-9)
    phases = find_phases(qf.coefficients.astype(complex))
    gqsp = gqsp_circuit(phases.theta, phases.phi, phases.lam, stacked, rescale=phases.rescale)
    expected = _dense_chebyshev_sum(evals, evecs, qf.coefficients)
    assert np.max(np.abs(top_left_block(gqsp) - expected)) < 1e-10


def test_stepwise_evolution_matches_the_whole_circuit(rng: np.random.Generator) -> None:
    # Evolving step by step (one controlled walk per simulation) is the same map as the
    # assembled d-fold circuit simulated in one go.
    stacked, _W = _r1()
    qf = qite_coefficients(3.0, tolerance=1e-9)
    phases = find_phases(qf.coefficients.astype(complex))
    gqsp = gqsp_circuit(phases.theta, phases.phi, phases.lam, stacked, rescale=phases.rescale)
    vector = rng.normal(size=4) + 1j * rng.normal(size=4)
    stepwise = apply_gqsp(
        phases.theta, phases.phi, phases.lam, stacked, vector, rescale=phases.rescale
    )
    np.testing.assert_allclose(stepwise, apply(gqsp, vector), atol=1e-12)


def test_controlled_gqsp_keeps_the_global_phase(rng: np.random.Generator) -> None:
    # Under a selector, lam is a relative phase. The selector-one branch must equal
    # the uncontrolled output, phase included, and the selector-zero branch must be untouched.
    phases = find_phases(np.array([0.2 + 0.3j, 0.5, 0.1j], dtype=complex))
    assert abs(np.sin(phases.lam)) > 0.1  # a lam that dropping it would visibly change
    stacked, _W = _r1()
    width = stacked.n_qubits + 1
    psi = rng.normal(size=1 << width) + 1j * rng.normal(size=1 << width)
    psi /= np.linalg.norm(psi)
    uncontrolled = evolve_gqsp(phases.theta, phases.phi, phases.lam, stacked, psi)
    both = np.concatenate([psi, psi]) / np.sqrt(2.0)  # selector in |+>: the high half is |1>
    controlled = evolve_gqsp(phases.theta, phases.phi, phases.lam, stacked, both, selector=True)
    np.testing.assert_allclose(controlled[: 1 << width], psi / np.sqrt(2.0), atol=1e-12)
    np.testing.assert_allclose(controlled[1 << width :], uncontrolled / np.sqrt(2.0), atol=1e-12)


def test_gqsp_circuit_rejects_mismatched_theta_and_phi() -> None:
    stacked, _W = _r1()
    with pytest.raises(ValueError, match="same shape"):
        gqsp_circuit(np.array([0.1, 0.2]), np.array([0.1]), 0.0, stacked)


def test_evolve_gqsp_rejects_a_state_of_the_wrong_length() -> None:
    stacked, _W = _r1()
    with pytest.raises(ValueError, match="state"):
        evolve_gqsp(np.array([0.1]), np.array([0.2]), 0.0, stacked, np.zeros(4, dtype=complex))


def test_gqsp_circuit_register_layout() -> None:
    stacked, W = _r1()
    phases = find_phases(np.array([1.0], dtype=complex))
    gqsp = gqsp_circuit(phases.theta, phases.phi, phases.lam, stacked, rescale=phases.rescale)
    assert gqsp.n_sys_qubits == W.n_sys_qubits
    assert gqsp.n_anc_qubits == W.n_anc_qubits + 1
    assert gqsp.n_qubits == W.n_qubits + 1
    assert gqsp.shape == W.shape


def test_a_long_evolution_logs_its_progress(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    stacked, _W = _r1()
    phases = find_phases(qite_coefficients(3.0, tolerance=1e-9).coefficients.astype(complex))
    state = np.zeros(1 << (stacked.n_qubits + 1), dtype=complex)
    state[0] = 1.0
    degree = len(phases.theta) - 1
    with caplog.at_level("INFO", logger="pihm.circuits.gqsp"):
        evolve_gqsp(phases.theta, phases.phi, phases.lam, stacked, state)
        assert not caplog.messages
        monkeypatch.setattr("pihm.circuits.gqsp.PROGRESS_SECONDS", 0.0)
        evolve_gqsp(phases.theta, phases.phi, phases.lam, stacked, state)
    assert len(caplog.messages) == degree
    assert caplog.messages[-1].startswith(f"GQSP step {degree} of {degree}: ")
