"""The Carleman lift's tensor layout as a circuit."""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp

from pihm.assemble.systems import coupling, lift_scale
from pihm.carleman import symmetric_embedding
from pihm.circuits.primitives import BlockEncoding, compile_circuit, evolve_basis
from pihm.circuits.tensor import (
    TensorLayout,
    fourier_fold,
    multiplier_encoding,
    physical_block,
    tensor_encoding,
    tensor_ir,
    tensor_target,
)
from pihm.problems.fluids import (
    Galerkin,
    Symbol,
    burgers,
    burgers_galerkin,
    real_embedding,
    vortex_pair,
    vorticity_galerkin,
)
from pihm.problems.spec import Problem


def _block(encoding: BlockEncoding, columns: int) -> np.ndarray:
    """Return the encoding's top-left block, column by column, the circuit compiled once."""
    compiled = compile_circuit(encoding.circuit)
    size = 1 << encoding.n_sys_qubits
    return np.stack(
        [encoding.alpha * evolve_basis(compiled, c)[:size] for c in range(columns)], axis=1
    )


@pytest.mark.parametrize(
    ("galerkin", "order"),
    [(burgers_galerkin(2, 1.0, 1.0), 2), (vorticity_galerkin(1, 1.0, 1.0), 2)],
    ids=["1-D", "2-D"],
)
def test_the_fold_is_the_circular_convolution(galerkin: Galerkin, order: int) -> None:
    """[fold; 0] on two slots at alpha = 2^(dm/2): the product in the first, zero in the second."""
    layout = TensorLayout.of(galerkin, order)
    fold = fourier_fold(layout)
    size = 1 << layout.slot_qubits
    block = _block(fold, size * size)
    assert fold.alpha == pytest.approx(np.sqrt(size))
    expected = np.zeros((size * size, size * size))
    for p in range(size):
        for q in range(size):
            k = layout.register(
                [a + b for a, b in zip(layout.wavevector(p), layout.wavevector(q), strict=True)]
            )
            expected[k * size, p * size + q] = 1.0
    np.testing.assert_allclose(block, expected, atol=1e-13)


def test_a_multiplier_is_its_magnitude_on_the_truncation_and_zero_elsewhere() -> None:
    galerkin = vorticity_galerkin(2, 1.0, 1.0)
    layout = TensorLayout.of(galerkin, 1)
    symbol = Symbol(1.0, (1, 0), -1)  # psi_x's multiplier over its phase i: k_x / |k|^2
    encoding = multiplier_encoding(symbol, layout)
    block = _block(encoding, 1 << layout.slot_qubits)
    diagonal = np.zeros(1 << layout.slot_qubits)
    for k in galerkin.wavevectors:
        diagonal[layout.register(k)] = k[0] / (k[0] ** 2 + k[1] ** 2)
    np.testing.assert_allclose(block, np.diag(diagonal), atol=1e-14)
    assert encoding.alpha == pytest.approx(1.0)


@pytest.mark.parametrize("order", [1, 2])
def test_the_encoding_is_the_lift_on_the_physical_states_with_no_leak(order: int) -> None:
    """Exact on R9's smallest truncation, with the lift's scale on F_2."""
    galerkin = burgers_galerkin(2, 1.0, 1.0)
    layout = TensorLayout.of(galerkin, order)
    encoding = tensor_encoding(galerkin, order, scale=1.3)
    block, leak = physical_block(encoding, layout)
    np.testing.assert_allclose(block, tensor_target(galerkin, order, 1.3).toarray(), atol=1e-13)
    assert leak < 1e-13


@pytest.mark.slow
@pytest.mark.parametrize(
    ("galerkin", "order"),
    [(burgers_galerkin(2, 1.0, 1.0), 3), (vorticity_galerkin(1, 1.0, 1.0), 2)],
    ids=["1-D, K = 3", "2-D, K = 2"],
)
def test_the_encoding_is_exact_where_the_slots_rotate_and_in_two_dimensions(
    galerkin: Galerkin, order: int
) -> None:
    """K = 3, where F_2 moves the slots above it; and Navier-Stokes' two products."""
    layout = TensorLayout.of(galerkin, order)
    block, leak = physical_block(tensor_encoding(galerkin, order, scale=1.3), layout)
    np.testing.assert_allclose(block, tensor_target(galerkin, order, 1.3).toarray(), atol=1e-13)
    assert leak < 1e-13


@pytest.mark.parametrize(
    ("galerkin", "order"),
    [
        (burgers_galerkin(4, 1.0, 1.0), 3),
        (burgers_galerkin(8, 1.0, 1.0), 2),
        (vorticity_galerkin(4, 1.0, 1.0), 2),
        (vorticity_galerkin(5, 1.0, 0.125), 2),
        (vorticity_galerkin(2, 1.0, 1.0), 3),
    ],
    ids=["R9-M4-K3", "R9-M8-K2", "R10-M12-K2", "R10-M20-W8-K2", "R10-K_c2-K3"],
)
def test_the_ir_of_the_terms_is_the_lift(galerkin: Galerkin, order: int) -> None:
    """The circuit's terms, applied classically on register values, give A_K and never leak."""
    ir = tensor_ir(galerkin, order, scale=2.0)
    np.testing.assert_allclose(
        ir.toarray(), tensor_target(galerkin, order, 2.0).toarray(), atol=1e-13
    )


@pytest.mark.parametrize(
    ("galerkin", "problem", "order"),
    [
        (burgers_galerkin(4, 1.0, 1.0), burgers(4, 3.0, 1.0, 1.0), 3),
        (vorticity_galerkin(4, 1.0, 1.0), vortex_pair(4, 10.0, 1.0, 1.0), 2),
    ],
    ids=["R9d-M4", "R10c-M12"],
)
def test_the_target_is_the_problems_own_lift(
    galerkin: Galerkin, problem: Problem, order: int
) -> None:
    """A_tensor (S V) = (S V) A_K: the circuit's target, through the real embedding, is the lift."""
    E = real_embedding(galerkin)
    powers = [E]
    for _ in range(order - 1):
        powers.append(sp.csr_array(sp.kron(powers[-1], E)))
    S = sp.block_diag(powers)
    embed = S @ symmetric_embedding(E.shape[1], order)
    scale = lift_scale(problem, order)
    left = tensor_target(galerkin, order, scale) @ embed
    right = embed @ coupling(problem, order)
    np.testing.assert_allclose(left.toarray(), right.toarray(), atol=1e-12)
