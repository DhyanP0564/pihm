"""Checks of the composition algebra: product, adjoint, lcu_sum, and the stacked reflection.

The stacked-residual identities mirror the documentation's numerical
check, now run against this package's own circuits rather
than a black-box dense isometry.
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.polynomial import chebyshev as npcheb
from qiskit.quantum_info import Operator

from pihm.circuits.assemble import encode_standard_linear
from pihm.circuits.compose import (
    adjoint,
    controlled_walk,
    embed_system,
    lcu_sum,
    pad_rows,
    product,
    stack_rows,
    stacked_reflection,
    walk,
)
from pihm.circuits.primitives import (
    BlockEncoding,
    block_encode_atom,
    identity_encoding,
    top_left_block,
    zero_projector,
)
from pihm.circuits.standard import structured_derivative_encoding
from pihm.problems.ladder import problem


def test_product_composes_matrix_multiplication(rng: np.random.Generator) -> None:
    """product(A, B) block-encodes A @ B (B applied first, per the docstring's own convention)."""
    A = rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4))
    B = rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4))
    EA = block_encode_atom(A, alpha=float(np.linalg.norm(A, 2)) * 1.1)
    EB = block_encode_atom(B, alpha=float(np.linalg.norm(B, 2)) * 1.1)
    np.testing.assert_allclose(top_left_block(product(EA, EB)), A @ B, atol=1e-9)


def test_adjoint_is_the_conjugate_transpose(rng: np.random.Generator) -> None:
    """adjoint(A) block-encodes A^dagger at the same alpha."""
    A = rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4))
    EA = block_encode_atom(A, alpha=float(np.linalg.norm(A, 2)) * 1.1)
    Ea = adjoint(EA)
    assert Ea.alpha == EA.alpha
    np.testing.assert_allclose(top_left_block(Ea), A.conj().T, atol=1e-9)


def test_lcu_sum_is_a_signed_linear_combination(rng: np.random.Generator) -> None:
    """lcu_sum([C1, C2], [w1, w2]) block-encodes w1*C1 + w2*C2, signed weights included."""
    C1 = rng.normal(size=(4, 4))
    C2 = rng.normal(size=(4, 4))
    E1 = block_encode_atom(C1, alpha=float(np.linalg.norm(C1, 2)) * 1.1)
    E2 = block_encode_atom(C2, alpha=float(np.linalg.norm(C2, 2)) * 1.1)
    combo = lcu_sum([E1, E2], [1.5, -0.7])
    np.testing.assert_allclose(top_left_block(combo), 1.5 * C1 - 0.7 * C2, atol=1e-9)
    assert combo.alpha == pytest.approx(1.5 * E1.alpha + 0.7 * E2.alpha)


def test_lcu_sum_applies_a_negative_weight_to_a_gateless_term(rng: np.random.Generator) -> None:
    """A term whose circuit has zero instructions (the identity) still gets its sign.

    Regression test: the phase used to ride on the term's first instruction, so a term with no
    instructions at all silently dropped it -- invisible whenever that term's own weight was
    positive (every prior caller: R0 and Lambda's own identity terms), and first caught when
    circuits.assemble's R2b (order 2, coefficients 1, -2, -3) put a *negative* weight on the
    identity term alongside G and G^2.
    """
    C = rng.normal(size=(4, 4))
    E = block_encode_atom(C, alpha=float(np.linalg.norm(C, 2)) * 1.1)
    combo = lcu_sum([identity_encoding(2), E], [-3.0, 1.0])
    target = -3.0 * np.eye(4) + C
    np.testing.assert_allclose(top_left_block(combo), target, atol=1e-9)


def test_product_shares_one_ancilla_register() -> None:
    """product(G, G) uses the wider factor's ancilla plus one flag, and is exactly G @ G."""
    G = structured_derivative_encoding(2)
    G2 = product(G, G)
    assert G2.n_anc_qubits == G.n_anc_qubits + 1
    block = top_left_block(G)
    np.testing.assert_allclose(top_left_block(G2), block @ block, atol=1e-12)


def test_product_with_an_ancilla_free_right_factor_needs_no_flag(
    rng: np.random.Generator,
) -> None:
    """A right factor with no ancilla cannot leave them dirty, so no flag qubit is added."""
    C = rng.normal(size=(4, 4))
    E = block_encode_atom(C, alpha=float(np.linalg.norm(C, 2)) * 1.1)
    EI = product(E, identity_encoding(2))
    assert EI.n_anc_qubits == E.n_anc_qubits
    np.testing.assert_allclose(top_left_block(EI), C, atol=1e-9)


def test_product_flags_a_right_factor_that_leaves_its_ancilla_dirty() -> None:
    """A projector's failure branch must not feed the left factor: the flag keeps it out."""
    P = zero_projector(2)
    X = np.zeros((4, 4))
    X[0, 0] = 1.0
    np.testing.assert_allclose(top_left_block(product(P, P)), X, atol=1e-12)


def test_lcu_sum_pads_prepare_and_keeps_complex_weights(rng: np.random.Generator) -> None:
    """Three terms (PREPARE padded to four) with complex weights, nested in a second LCU."""
    Cs = [rng.normal(size=(4, 4)) for _ in range(3)]
    Es = [block_encode_atom(C, alpha=float(np.linalg.norm(C, 2)) * 1.1) for C in Cs]
    weights = [0.5, -1.2j, 2.0 * np.exp(0.3j)]
    inner = lcu_sum(Es, weights)
    target = sum(w * C for w, C in zip(weights, Cs, strict=True))
    np.testing.assert_allclose(top_left_block(inner), target, atol=1e-9)
    outer = lcu_sum([inner, identity_encoding(2)], [-1.0, 0.5])
    np.testing.assert_allclose(top_left_block(outer), -target + 0.5 * np.eye(4), atol=1e-9)


def test_pad_rows_puts_the_block_above_zeros(rng: np.random.Generator) -> None:
    """pad_rows(X) is [X; 0] on one more system qubit, at the same alpha and ancilla."""
    C = rng.normal(size=(4, 4))
    E = block_encode_atom(C, alpha=float(np.linalg.norm(C, 2)) * 1.1)
    padded = pad_rows(E)
    assert (padded.shape, padded.n_sys_qubits, padded.alpha) == ((8, 4), 3, E.alpha)
    np.testing.assert_allclose(top_left_block(padded), np.vstack([C, np.zeros((4, 4))]), atol=1e-9)


def test_pad_rows_needs_rows_that_fill_the_register(rng: np.random.Generator) -> None:
    """A block with fewer rows than its register holds has no [X; 0] one qubit up."""
    C = rng.normal(size=(2, 4))
    E = block_encode_atom(C, alpha=float(np.linalg.norm(C, 2)) * 1.1)
    with pytest.raises(ValueError, match="fill"):
        pad_rows(E)


def test_stack_rows_stacks_rectangular_blocks(rng: np.random.Generator) -> None:
    """Lifted (2N x N) blocks stack as tall blocks: the reflection still gives 2H/alpha^2 - I."""
    Cs = [rng.normal(size=(8, 4)) for _ in range(2)]
    Es = [block_encode_atom(C, alpha=float(np.linalg.norm(C, 2)) * 1.1) for C in Cs]
    stacked = stack_rows(Es)
    assert stacked.shape == (16, 4)
    np.testing.assert_allclose(top_left_block(stacked), np.vstack(Cs), atol=1e-9)
    X = np.vstack(Cs)
    Hp = 2.0 * X.T @ X / stacked.alpha**2 - np.eye(4)
    np.testing.assert_allclose(top_left_block(stacked_reflection(stacked)), Hp, atol=1e-9)


def test_embed_system_is_a_kronecker_factor(rng: np.random.Generator) -> None:
    """A on qubits (1, 2) of three, B on qubit 0: their product is A ⊗ B, ancilla shared."""
    A = rng.normal(size=(4, 4))
    B = rng.normal(size=(2, 2))
    EA = block_encode_atom(A, alpha=float(np.linalg.norm(A, 2)) * 1.1)
    EB = block_encode_atom(B, alpha=float(np.linalg.norm(B, 2)) * 1.1)
    both = product(embed_system(EA, [1, 2], 3), embed_system(EB, [0], 3))
    np.testing.assert_allclose(top_left_block(both), np.kron(A, B), atol=1e-9)
    assert both.n_anc_qubits == max(EA.n_anc_qubits, EB.n_anc_qubits) + 1


def test_embed_system_refuses_positions_that_do_not_fit() -> None:
    """One distinct position per system qubit, inside the larger register."""
    with pytest.raises(ValueError, match="distinct"):
        embed_system(identity_encoding(2), [0, 0], 3)
    with pytest.raises(ValueError, match="larger register"):
        embed_system(identity_encoding(2), [2, 3], 3)


def _random_stack(rng: np.random.Generator, P: int) -> tuple[list[np.ndarray], list[BlockEncoding]]:
    """Two random square blocks (a dense one and a rank-one one) and their block encodings."""
    R1 = rng.normal(size=(P, P))
    R2 = np.outer(rng.normal(size=P), rng.normal(size=P))
    blocks = [R1, R2]
    encodings = [
        block_encode_atom(R, alpha=float(np.linalg.norm(R, 2)) * 1.1 or 1.0) for R in blocks
    ]
    return blocks, encodings


def test_stack_rows_reproduces_the_vertical_stack(rng: np.random.Generator) -> None:
    """stack_rows([R1, R2]) reads out as vstack([R1, R2]) exactly."""
    P = 4
    blocks, encodings = _random_stack(rng, P)
    stacked = stack_rows(encodings)
    block = top_left_block(stacked)
    X = np.vstack(blocks)
    np.testing.assert_allclose(block, X, atol=1e-9)
    assert stacked.alpha == pytest.approx(np.sqrt(sum(e.alpha**2 for e in encodings)))


def test_stacked_reflection_gives_two_h_over_alpha_squared_minus_i(
    rng: np.random.Generator,
) -> None:
    """Pi_in V Pi_in = 2H/alpha_R^2 - I, H = R^dagger R."""
    P = 4
    blocks, encodings = _random_stack(rng, P)
    stacked = stack_rows(encodings)
    V = stacked_reflection(stacked)
    X = np.vstack(blocks)
    H = X.T @ X
    Hp = 2.0 * H / stacked.alpha**2 - np.eye(P)
    np.testing.assert_allclose(top_left_block(V), Hp, atol=1e-9)

    matrix = Operator(V.circuit).data
    np.testing.assert_allclose(matrix, matrix.conj().T, atol=1e-9)
    np.testing.assert_allclose(matrix @ matrix, np.eye(matrix.shape[0]), atol=1e-9)


def test_walk_reproduces_chebyshev_polynomials_of_h_prime(rng: np.random.Generator) -> None:
    """Pi_in W^k Pi_in = T_k(H'), checked from one matrix power plus one restriction.

    Not an iterated project-then-apply, which does not equal the same thing for a qubitised walk.
    """
    P = 4
    blocks, encodings = _random_stack(rng, P)
    stacked = stack_rows(encodings)
    V = stacked_reflection(stacked)
    W = walk(V)
    X = np.vstack(blocks)
    H = X.T @ X
    Hp = 2.0 * H / stacked.alpha**2 - np.eye(P)
    w, S = np.linalg.eigh(Hp)

    Wmat = Operator(W.circuit).data
    power = np.eye(Wmat.shape[0], dtype=complex)
    for k in range(8):
        Tk = S @ np.diag(npcheb.chebval(w, [0] * k + [1])) @ S.T
        np.testing.assert_allclose(power[:P, :P], Tk, atol=1e-9)
        power = Wmat @ power


@pytest.mark.parametrize(("n_controls", "ctrl_state"), [(1, 0), (1, 1), (2, 2)])
def test_controlled_walk_is_the_walk_where_the_controls_match(
    rng: np.random.Generator, n_controls: int, ctrl_state: int
) -> None:
    """controlled_walk is W where the controls equal ctrl_state and the identity elsewhere."""
    _blocks, encodings = _random_stack(rng, 4)
    stacked = stack_rows(encodings)
    W = Operator(walk(stacked_reflection(stacked)).circuit).data
    C = 1 << n_controls
    target = np.zeros((C * W.shape[0], C * W.shape[0]), dtype=complex)
    for c in range(C):
        projector = np.zeros((C, C))
        projector[c, c] = 1.0
        target += np.kron(projector, W if c == ctrl_state else np.eye(W.shape[0]))
    circuit = controlled_walk(stacked, n_controls=n_controls, ctrl_state=ctrl_state)
    np.testing.assert_allclose(Operator(circuit).data, target, atol=1e-10)


def test_controlled_walk_controls_only_the_reflections() -> None:
    """Only the two multi-controlled Z gates touch the control; U_R's gates stay uncontrolled.

    Simulating the controlled walk must cost a statevector, never a dense operator
    of the whole walk. Checked on R2a's own circuit (19 walk qubits), construction only.
    """
    stacked, _reflection = encode_standard_linear(problem("R2a", "corrected"), 2)
    circuit = controlled_walk(stacked)
    control = stacked.n_qubits
    widest = max(inst.operation.num_qubits for inst in stacked.circuit.data)
    touching = [
        inst
        for inst in circuit.data
        if any(circuit.find_bit(q).index == control for q in inst.qubits)
    ]
    assert len(touching) == 2
    assert all(inst.operation.base_gate.name == "z" for inst in touching)
    others = [inst for inst in circuit.data if inst not in touching]
    assert max(inst.operation.num_qubits for inst in others) <= widest


def test_controlled_walk_needs_a_control() -> None:
    """A controlled walk without a control qubit would be a second implementation of walk()."""
    stacked, _reflection = encode_standard_linear(problem("R1"), 2)
    with pytest.raises(ValueError, match="at least one control"):
        controlled_walk(stacked, n_controls=0)
