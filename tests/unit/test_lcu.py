"""The descriptor's term-IR compiler and whole-residual encoder."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from pihm.assemble import Residual, assemble
from pihm.circuits.assemble import descriptor_blocks, encode_descriptor_linear
from pihm.circuits.lcu import (
    Layout,
    circuit_rows,
    compile_system,
    field_placement,
    place,
    rank_one_encoding,
    shift_circuit,
)
from pihm.circuits.primitives import BlockEncoding, identity_encoding, top_left_block
from pihm.ir import Band, Diagonal, Outer, SystemIR, Term, system_ir, system_matrix
from pihm.problems import ladder


def _descriptor(name: str, ns: tuple[int, ...], rescale: str = "block") -> Residual:
    return assemble(
        ladder.problem(name, "corrected"),
        ns,
        regime="descriptor",
        rescale=rescale,  # type: ignore[arg-type]
    )


def _bands(ir: SystemIR) -> SystemIR:
    terms = tuple(t for t in ir.terms if all(isinstance(f, Band) for f in t.factors))
    return dataclasses.replace(ir, terms=terms)


def _in_circuit_rows(matrix: np.ndarray, layout: Layout) -> np.ndarray:
    placed = np.zeros(layout.shape, dtype=matrix.dtype)
    placed[circuit_rows(layout), :] = matrix
    return placed


@pytest.mark.parametrize("shift", [-3, -2, -1, 1, 2, 3])
@pytest.mark.parametrize(("n_in", "n_out"), [(2, 2), (2, 3), (3, 3)])
def test_a_shift_with_an_overflow_qubit_is_the_truncated_shift(
    shift: int, n_in: int, n_out: int
) -> None:
    """Column c goes to row c - s where that row exists, and vanishes elsewhere."""
    width = n_out + 1
    encoding = BlockEncoding(shift_circuit(shift, width), 1.0, n_out, 1, (1 << n_out, 1 << n_in))
    expected = Band(shift, Diagonal("identity")).matrix(n_in, n_out).toarray()
    np.testing.assert_allclose(top_left_block(encoding), expected, atol=1e-12)


@pytest.mark.parametrize("rescale", ["block", "none"])
@pytest.mark.parametrize(
    ("name", "ns"),
    [
        ("R1", (3,)),
        ("R2a", (2,)),
        ("R3a", (2,)),
        ("R4a", (2,)),
        ("R5a", (1, 1)),
        ("R5b", (1, 1)),
        ("R5c", (1, 1)),
    ],
)
def test_the_compiled_bands_are_the_ir_s_bands(
    name: str, ns: tuple[int, ...], rescale: str
) -> None:
    """Every band of the system rows, lifted or not, rescaled or not, on one axis or two."""
    ir = _bands(system_ir(_descriptor(name, ns, rescale)))
    encoding = compile_system(ir)
    layout = Layout.of(ir)
    assert encoding.alpha == pytest.approx(ir.alpha)
    expected = _in_circuit_rows(ir.to_sparse().toarray(), layout)
    np.testing.assert_allclose(top_left_block(encoding), expected, atol=1e-12 * encoding.alpha)


def test_two_axes_compile_with_the_first_lifted() -> None:
    """Kronecker positions, a lift on an axis that is not the last, and each axis's own slots."""
    identity = Diagonal("identity")
    terms = (
        Term(0.7, 0, 0, (Band(0, identity), Band(1, Diagonal("z", 1)))),
        Term(-0.4, 1, 0, (Band(-1, Diagonal("reflection", 1)), Band(0, identity))),
        Term(0.25, 1, 1, (Band(2, Diagonal("z", 0)), Band(-1, identity))),
        Term(1.1, 0, 1, (Band(1, identity), Band(2, Diagonal("reflection", 3)))),
    )
    ir = SystemIR(terms, field_qubits=1, qubits_in=(1, 2), qubits_out=(2, 2))
    encoding = compile_system(ir)
    expected = _in_circuit_rows(ir.to_sparse().toarray(), Layout.of(ir))
    np.testing.assert_allclose(top_left_block(encoding), expected, atol=1e-12)


def test_circuit_rows_is_the_identity_unless_an_axis_is_lifted() -> None:
    flat = Layout.of(system_ir(_descriptor("R2a", (3,))))
    assert flat.qubits_in == flat.qubits_out
    np.testing.assert_array_equal(circuit_rows(flat), np.arange(1 << flat.n_sys))
    lifted = Layout.of(system_ir(_descriptor("R4a", (3,))))
    rows = circuit_rows(lifted)
    assert sorted(rows.tolist()) == list(range(1 << lifted.n_sys))
    # The state's rows come first: every row with the lift qubit clear lands below the state's size.
    size = 1 << lifted.state_qubits
    low = [r for r in range(1 << lifted.n_sys) if not (r >> lifted.qubits_in[0]) & 1]
    assert all(rows[r] < size for r in low)


def test_placement_puts_a_matrix_unit_on_the_field_register() -> None:
    placement = field_placement(2, 1, 2)
    expected = np.zeros((4, 4))
    expected[2, 1] = 1.0
    np.testing.assert_allclose(top_left_block(placement), expected, atol=1e-12)
    layout = Layout(2, (1,), (1,))
    with pytest.raises(ValueError, match="disjoint"):
        place([(placement, [1, 2]), (identity_encoding(1), [1])], layout)


@pytest.mark.parametrize("name", ["R1", "R4a", "R4b"])
def test_the_system_block_carries_the_sources(name: str) -> None:
    """The bands and each source's rank-one term (polynomial, Chebyshev) sum to the system rows."""
    residual = _descriptor(name, (2,))
    system = descriptor_blocks(residual)[0]
    layout = Layout.of(system_ir(residual))
    expected = _in_circuit_rows(system_matrix(residual).toarray(), layout)
    np.testing.assert_allclose(top_left_block(system), expected, atol=1e-11 * system.alpha)


@pytest.mark.parametrize(("name", "n"), [("R5a", 2), ("R5b", 2), ("R5c", 2), ("R4a", 3)])
def test_each_constraint_block_is_its_assembled_rows(name: str, n: int) -> None:
    """Invariant rows (one on f_t) and datum slices on two axes, and a lifted row, in place."""
    residual = _descriptor(name, (n,) * len(ladder.problem(name, "corrected").fields[0].axes))
    layout = Layout.of(system_ir(residual))
    assembled = [block for block in residual.blocks if block.kind == "constraint"]
    encodings = descriptor_blocks(residual)[1:]
    assert len(encodings) == len(assembled)
    for encoding, block in zip(encodings, assembled, strict=True):
        rows = np.zeros(layout.shape)
        rows[: block.matrix.shape[0]] = block.matrix.toarray()
        expected = _in_circuit_rows(rows, layout)
        np.testing.assert_allclose(top_left_block(encoding), expected, atol=1e-11 * encoding.alpha)


@pytest.mark.parametrize("name", ["R2a", "R3a", "R4a"])
def test_the_reflection_encodes_the_rescaled_hamiltonian(name: str) -> None:
    """V's block is 2H/alpha_R^2 - I for H = (RS)^T RS, on the state's register."""
    p = ladder.problem(name, "corrected")
    stacked, V = encode_descriptor_linear(p, 2)
    H = assemble(p, 2, regime="descriptor").hamiltonian().toarray()
    expected = 2.0 * H / stacked.alpha**2 - np.eye(H.shape[0])
    assert V.shape == H.shape
    np.testing.assert_allclose(top_left_block(V), expected, atol=1e-10)


def test_the_unscaled_ablation_encodes_its_own_hamiltonian() -> None:
    p = ladder.problem("R2c", "corrected")
    stacked, V = encode_descriptor_linear(p, 2, rescale="none")
    H = assemble(p, 2, regime="descriptor", rescale="none").hamiltonian().toarray()
    expected = 2.0 * H / stacked.alpha**2 - np.eye(H.shape[0])
    np.testing.assert_allclose(top_left_block(V), expected, atol=1e-10)


@pytest.mark.parametrize("name", ["R2a", "R4a"])
def test_the_ancilla_grow_as_log_n(name: str) -> None:
    """The system rows' ancilla are O(log n), a source's (R4a) included."""
    counts = [compile_system(system_ir(_descriptor(name, (n,)))).n_anc_qubits for n in range(2, 9)]
    assert max(counts) - min(counts) <= 2
    assert max(counts) <= 14


@pytest.mark.parametrize(("n_in", "n_out"), [(2, 2), (2, 3)])
def test_a_rank_one_factor_reads_the_feature_vector_and_loads_its_coefficients(
    n_in: int, n_out: int
) -> None:
    outer = Outer(tuple(np.linspace(-1.0, 2.0, 1 << n_out)), 0.3)
    encoding = rank_one_encoding(outer, n_in, n_out)
    assert encoding.alpha == pytest.approx(np.sqrt(2.0) * np.linalg.norm(outer.left))
    expected = outer.matrix(n_in, n_out).toarray()
    np.testing.assert_allclose(top_left_block(encoding), expected, atol=1e-12)
