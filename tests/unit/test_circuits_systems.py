"""First-order systems' circuits in both forms, exact against the assembly."""

from __future__ import annotations

import numpy as np
import pytest

from pihm.assemble import assemble
from pihm.assemble.systems import coupling
from pihm.circuits.assemble import encode_linear
from pihm.circuits.primitives import top_left_block
from pihm.circuits.systems import field_encoding, ratio_encoding, system_blocks
from pihm.ir import system_ir, system_matrix
from pihm.problems import ladder
from pihm.verify import verify_exact

#: The systems verified exactly at n = 2: R7, R8's lifts with and without padding, R1's mass form,
#: and R9's smallest lift in the standard form (its descriptor needs 18 qubits more than exact
#: verification takes at this size; its Gram matrix is checked below).
CASES = [
    ("R9c-M4", 2, "standard"),
    ("R7", None, "standard"),
    ("R7", None, "descriptor"),
    ("R8", 2, "standard"),
    ("R8", 2, "descriptor"),
    ("R8", 3, "standard"),
    ("R8", 3, "descriptor"),
    ("R1", None, "descriptor"),
]


@pytest.mark.parametrize(("name", "order", "regime"), CASES)
@pytest.mark.nightly
def test_the_reflection_encodes_the_assembled_hamiltonian(
    name: str, order: int | None, regime: str
) -> None:
    problem = ladder.problem(name)
    residual = assemble(problem, 2, regime=regime, carleman=order)  # type: ignore[arg-type]
    stacked, reflection = encode_linear(problem, 2, regime=regime, carleman=order)  # type: ignore[arg-type]
    H = residual.hamiltonian().toarray()
    target = 2.0 * H / stacked.alpha**2 - np.eye(H.shape[0])
    result = verify_exact(reflection, target)
    assert result.passed, result.max_error


@pytest.mark.parametrize(("order", "n"), [(None, 2), (None, 6), (2, 4), (3, 5), (5, 3)])
def test_the_mass_form_ir_sums_to_the_assembled_rows(order: int | None, n: int) -> None:
    """IR mode for the descriptor's systems, beyond any circuit's reach."""
    problem = ladder.problem("R7" if order is None else "R8")
    residual = assemble(problem, n, regime="descriptor", carleman=order)
    assembled = system_matrix(residual).toarray()
    difference = system_ir(residual).to_sparse().toarray() - assembled
    assert np.max(np.abs(difference)) < 1e-12 * max(1.0, float(np.max(np.abs(assembled))))


@pytest.mark.parametrize("regime", ["standard", "descriptor"])
def test_each_row_block_is_its_assembled_block(regime: str) -> None:
    """R8 at K = 3 (padded): the system rows and the ratio rows, each exact on its own."""
    problem = ladder.problem("R8")
    residual = assemble(problem, 2, regime=regime, carleman=3)  # type: ignore[arg-type]
    rows, ratio = system_blocks(problem, 2, regime=regime, carleman=3)  # type: ignore[arg-type]
    assembled = {block.name: block.matrix.toarray() for block in residual.blocks}
    padding = [m for name, m in assembled.items() if name.startswith("padding")]
    expected = np.vstack([assembled["equation"], *padding])
    np.testing.assert_allclose(top_left_block(rows), expected, atol=1e-12)
    np.testing.assert_allclose(top_left_block(ratio), assembled["ratio on t = -1"], atol=1e-12)


def test_a_ratio_constraint_with_one_nonzero_datum_needs_no_rank_one_term() -> None:
    """R7's data (0, pi): the identity and a projector, alpha = 2 alpha_B = 2 sqrt(2N)."""
    problem = ladder.problem("R7")
    (constraint,) = problem.constraints
    rows = ratio_encoding(problem, constraint, 3, 1)
    lifted = ratio_encoding(ladder.problem("R8"), ladder.problem("R8").constraints[0], 3, 3)
    assert rows.alpha == pytest.approx(2.0 * np.sqrt(16.0))
    assert lifted.alpha > rows.alpha


def test_a_field_matrix_is_encoded_exactly() -> None:
    rng = np.random.default_rng(3)
    F = rng.normal(size=(4, 4))
    encoding = field_encoding(F, 2)
    np.testing.assert_allclose(top_left_block(encoding), F, atol=1e-12)


def test_a_coupling_matrix_is_encoded_exactly_as_matchings() -> None:
    """The matchings' alpha is the sum of their largest entries."""
    A = coupling(ladder.problem("R9c-M4"), 2).toarray()
    F = np.zeros((16, 16))
    F[:14, :14] = A
    encoding = field_encoding(F, 4, matchings=True)
    np.testing.assert_allclose(top_left_block(encoding), F, atol=1e-12)
    assert encoding.alpha < field_encoding(F, 4).alpha


@pytest.mark.nightly
def test_r9s_descriptor_encodes_its_residual() -> None:
    """R9c-M4 at K = 2, n = 2: U_R's Gram matrix is the assembled R^T R."""
    problem = ladder.problem("R9c-M4")
    stacked, _ = encode_linear(problem, 2, regime="descriptor", carleman=2)
    R = assemble(problem, 2, regime="descriptor", carleman=2).matrix().toarray()
    S = top_left_block(stacked)
    scale = float(np.linalg.norm(R, 2)) ** 2
    np.testing.assert_allclose(S.conj().T @ S, R.T @ R, atol=1e-13 * scale)
