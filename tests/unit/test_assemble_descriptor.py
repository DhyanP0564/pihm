"""The descriptor form, checked against its definition and the standard form."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest
import scipy.sparse as sp

from pihm.assemble import PADDING_WEIGHT, Rescale, Residual, assemble, represent
from pihm.classical import reference_state, solve
from pihm.operators import banded_factors, zero_row
from pihm.problems import ladder
from pihm.problems.spec import Equation, Factor, Term

#: Every linear panel of the ladder carried with derivative blocks, each variant, at a small
#: resolution (R1, a first-order equation, is carried in mass form: ``test_assemble_systems``).
LINEAR = [
    (name, variant, (2, 3) if name.startswith("R5") else 3)
    for name in (
        *("R2a", "R2b", "R2c", "R3a", "R3b", "R3c", "R3d"),
        *("R4a", "R4b", "R4c", "R4d", "R5a", "R5b", "R5c"),
    )
    for variant in ("faithful", "corrected")
]


def _dense(matrix: sp.csr_array) -> np.ndarray:
    return np.asarray(matrix.toarray())


def _blocks(residual: Residual) -> dict[str, np.ndarray]:
    return {block.name: _dense(block.matrix) for block in residual.blocks}


@pytest.mark.parametrize(
    ("name", "variant", "layout"),
    [
        ("R2a", "faithful", ((0,), (1,), (2,), None)),
        ("R4d", "faithful", ((0,), (1,), (2,), None)),
        ("R5a", "faithful", ((0, 0), (1, 0), (2, 0), (0, 1), (0, 2), None, None, None)),
        ("R5b", "faithful", ((0, 0), (1, 0), (0, 1), (0, 2))),
        ("R5c", "corrected", ((0, 0), (1, 0), (2, 0), (0, 1), (0, 2), None, None, None)),
    ],
)
def test_the_state_carries_each_derivative_padded_to_a_power_of_two(
    name: str, variant: str, layout: tuple[tuple[int, ...] | None, ...]
) -> None:
    residual = assemble(ladder.problem(name, variant), 2, regime="descriptor")
    assert residual.derivatives == layout
    assert residual.field_qubits == (len(layout) - 1).bit_length()
    assert residual.dimension == len(layout) * 4 ** len(layout[0] or ())


def test_fig3_rows_are_the_recurrences_the_equation_and_the_padding() -> None:
    """R2a: [-D, B, 0, 0], [0, -D, B, 0], [4, 4, 1, 0], [0, 0, 0, c_pad] and B_n(-1) on f."""
    n = 3
    B, D = (_dense(factor) for factor in banded_factors(n))
    one, nil = np.eye(8), np.zeros((8, 8))
    blocks = _blocks(assemble(ladder.problem("R2a"), n, regime="descriptor", rescale="none"))
    assert list(blocks) == [
        "equation",
        "recurrence f^(0) -> f^(1) along x",
        "recurrence f^(1) -> f^(2) along x",
        "padding block 3",
        "zero f^(0) on x = -1",
    ]
    np.testing.assert_array_equal(blocks["equation"], np.block([[4 * one, 4 * one, one, nil]]))
    np.testing.assert_array_equal(
        blocks["recurrence f^(0) -> f^(1) along x"], np.block([[-D, B, nil, nil]])
    )
    np.testing.assert_array_equal(
        blocks["recurrence f^(1) -> f^(2) along x"], np.block([[nil, -D, B, nil]])
    )
    padding = np.block([[nil, nil, nil, PADDING_WEIGHT * one]])
    np.testing.assert_array_equal(blocks["padding block 3"], padding)
    zero = _dense(zero_row(-1.0, n))
    np.testing.assert_array_equal(blocks["zero f^(0) on x = -1"], np.block([[zero, nil, nil, nil]]))


def test_a_slope_constraint_sits_on_the_derivative_block_without_g() -> None:
    """R2c's f'(0) = 0 is the Dirichlet row on the f'-block."""
    n = 3
    blocks = _blocks(assemble(ladder.problem("R2c"), n, regime="descriptor", rescale="none"))
    nil = np.zeros((8, 8))
    expected = np.block([[nil, _dense(zero_row(0.0, n)), nil, nil]])
    np.testing.assert_array_equal(blocks["zero f^(1) on x = 0"], expected)


@pytest.mark.parametrize("rescale", ["block", "none"])
@pytest.mark.parametrize(("name", "variant", "n"), LINEAR)
def test_the_rows_act_on_a_represented_state_as_the_standard_form_on_its_field(
    name: str, variant: str, n: int | tuple[int, ...], rescale: Rescale
) -> None:
    """With p_j = G^j p, each block is the standard form's, the rest vanish."""
    problem = ladder.problem(name, variant)
    standard = assemble(problem, n)
    descriptor = assemble(problem, n, regime="descriptor", rescale=rescale)
    p = np.random.default_rng(7).normal(size=standard.dimension)
    w = represent(descriptor, p)
    same = {block.name: block.matrix for block in standard.blocks}
    kinds = {block.kind for block in descriptor.blocks}
    assert {"equation", "recurrence"} <= kinds
    for block in descriptor.blocks:
        expected: np.ndarray
        if block.kind in ("recurrence", "padding"):
            expected = np.zeros(block.matrix.shape[0])
            scale = abs(block.matrix) @ np.abs(w)
        else:
            expected = same[block.name] @ p
            scale = abs(same[block.name]) @ np.abs(p)
        np.testing.assert_allclose(block.matrix @ w, expected, rtol=0, atol=1e-13 * scale.max())
    assert {b.name for b in standard.blocks} <= {b.name for b in descriptor.blocks}


@pytest.mark.parametrize("name", ["R3a", "R3b"])
@pytest.mark.parametrize("n", [3, 4, 5])
def test_the_exact_kernels_coincide(name: str, n: int) -> None:
    """Legendre m = 0: ker R_desc = represent(ker R_std), exactly."""
    problem = ladder.problem(name)
    standard = reference_state(assemble(problem, n).matrix())
    residual = assemble(problem, n, regime="descriptor")
    descriptor = reference_state(residual.matrix())
    assert descriptor.trusted
    assert descriptor.singular_values[0] <= 1e-15 * descriptor.singular_values[-1]
    lifted = represent(residual, standard.vector)
    overlap = abs(np.vdot(lifted, descriptor.vector)) / np.linalg.norm(lifted)
    assert 1.0 - overlap**2 <= 1e-14


def test_a_padding_block_decouples_with_its_weight() -> None:
    residual = assemble(ladder.problem("R2a"), 3, regime="descriptor")
    R = _dense(residual.matrix())
    padding = R[:, 24:]
    rows = np.flatnonzero(np.any(padding != 0, axis=1))
    np.testing.assert_array_equal(padding[rows], PADDING_WEIGHT * np.eye(8))
    assert not np.any(R[rows, :24])
    singular = reference_state(R).singular_values
    assert np.count_nonzero(np.abs(singular - PADDING_WEIGHT) < 1e-12) >= 8


@pytest.mark.parametrize(
    ("name", "variant", "n"),
    [
        ("R1", "corrected", 6),
        ("R2a", "faithful", 6),
        ("R2b", "faithful", 6),
        ("R4a", "corrected", 6),
    ],
)
def test_the_field_reads_out_as_in_the_standard_form(name: str, variant: str, n: int) -> None:
    """The decoded field converges to the reference in both regimes, with the same scale."""
    problem = ladder.problem(name, variant)
    standard = solve(assemble(problem, n))
    descriptor = solve(assemble(problem, n, regime="descriptor"))
    assert descriptor.field_error_linf <= max(standard.field_error_linf, 1e-13)
    assert descriptor.eta == pytest.approx(descriptor.eta_ideal, rel=1e-10)
    size = descriptor.residual.dimension // len(descriptor.residual.derivatives)
    field_weight = np.vdot(descriptor.vector[:size], descriptor.vector[:size]).real
    assert descriptor.eta * field_weight == pytest.approx(standard.eta, rel=1e-10)
    assert descriptor.infidelity <= 1e-14


def test_represent_is_the_identity_in_the_standard_form() -> None:
    b = np.random.default_rng(3).normal(size=8)
    np.testing.assert_array_equal(represent(assemble(ladder.problem("R2a"), 3), b), b)
    with pytest.raises(ValueError, match="not linear"):
        represent(assemble(ladder.problem("R6a"), 2), np.ones(4))


def test_the_rescaling_scales_each_block_s_columns_by_its_weight() -> None:
    """R S, with S = diag(lambda^order) block by block and the padding unscaled."""
    problem = ladder.problem("R5c", "corrected")
    scaled = assemble(problem, (3, 3), regime="descriptor")
    plain = assemble(problem, (3, 3), regime="descriptor", rescale="none")
    assert scaled.scales == (16.0, 8.0)
    assert plain.scales == (1.0, 1.0)
    weights = [1.0, 16.0, 256.0, 8.0, 64.0, 1.0, 1.0, 1.0]
    assert scaled.derivatives[:5] == ((0, 0), (1, 0), (2, 0), (0, 1), (0, 2))
    assert scaled.block_weights == tuple(weights)
    S = np.diag(np.repeat(weights, 64))
    for a, b in zip(scaled.blocks, plain.blocks, strict=True):
        np.testing.assert_array_equal(_dense(a.matrix), _dense(b.matrix) @ S)
    np.testing.assert_array_equal(_dense(scaled.readout.decoder), _dense(plain.readout.decoder))


def test_the_reference_takes_the_physical_blocks_only() -> None:
    """The padding decouples, so its singular value (1) never stands in for the physical gap."""
    residual = assemble(ladder.problem("R2a"), 5, regime="descriptor")
    solution = solve(residual)
    physical = reference_state(sp.csr_array(residual.matrix()[:, residual.physical_columns]))
    assert solution.state.gap == physical.gap
    assert solution.state.singular_values[1] > PADDING_WEIGHT
    full = reference_state(residual.matrix()).singular_values
    assert np.any(np.abs(full[1:4] - PADDING_WEIGHT) < 1e-12)
    padding = np.setdiff1d(np.arange(residual.dimension), residual.physical_columns)
    assert not np.any(solution.vector[padding])


def test_descriptor_options_are_checked() -> None:
    with pytest.raises(ValueError, match="ablation of the standard form"):
        assemble(ladder.problem("R2a"), 3, regime="descriptor", truncation="tau")
    with pytest.raises(ValueError, match="no block rescaling"):
        assemble(ladder.problem("R2a"), 3, rescale="block")
    with pytest.raises(NotImplementedError, match="doubled space"):
        assemble(ladder.problem("R6a"), 2, regime="descriptor")
    laplace = ladder.problem("R5a")
    mixed = dataclasses.replace(
        laplace, equations=(Equation((Term(1.0, (Factor("f", (1, 1)),)),)),)
    )
    with pytest.raises(NotImplementedError, match="mixed derivative"):
        assemble(mixed, 2, regime="descriptor")
