"""The generic standard-form linear circuit compiler."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from pihm.assemble import assemble
from pihm.circuits.assemble import _one_axis_blocks, encode_linear, encode_standard_linear
from pihm.circuits.primitives import top_left_block
from pihm.circuits.standard import DERIVATIVE_ENCODERS, StandardEncoder
from pihm.problems.ladder import problem
from pihm.problems.spec import Constraint


def _check(name: str, n: int | tuple[int, ...], variant: str | None = None) -> None:
    p = problem(name) if variant is None else problem(name, variant)
    stacked, V = encode_standard_linear(p, n)
    Hp_circuit = top_left_block(V)
    H_classical = assemble(p, n).hamiltonian().toarray()
    Hp_classical = 2.0 * H_classical / stacked.alpha**2 - np.eye(H_classical.shape[0])
    np.testing.assert_allclose(Hp_circuit.real, Hp_classical, atol=1e-8)
    np.testing.assert_allclose(Hp_circuit.imag, 0.0, atol=1e-8)


@pytest.mark.parametrize("n", [3, 4])
def test_r1_matches_the_classical_hamiltonian(n: int) -> None:
    """R1 (order 1, no invariant constraint) needs no product(), so this stays fast."""
    _check("R1", n)


@pytest.mark.parametrize("name", ["R2a", "R2b", "R2c"])
def test_fig3_panels_match_the_classical_hamiltonian(name: str) -> None:
    """Every Fig. 3 panel (order 2, needs G^2) matches."""
    _check(name, 2)


@pytest.mark.parametrize("name", ["R3a", "R3b"])
@pytest.mark.nightly
def test_fig4_legendre_panels_match_the_classical_hamiltonian(name: str) -> None:
    """The Legendre panels (lifted, multipliers of degree 2) match, lifted rows included."""
    _check(name, 2)


@pytest.mark.slow
@pytest.mark.parametrize("name", ["R3c", "R3d"])
def test_fig4_associated_legendre_panels_match_the_classical_hamiltonian(name: str) -> None:
    """The associated Legendre panels (the paper's degree-4 multipliers) match: 19 qubits."""
    _check(name, 2)


@pytest.mark.parametrize(
    ("name", "variant"),
    [
        ("R4a", "faithful"),
        ("R4b", "faithful"),
        ("R4a", "corrected"),
        ("R4b", "corrected"),
        ("R4c", "corrected"),
        ("R4d", "corrected"),
    ],
)
@pytest.mark.nightly
def test_fig5_panels_match_the_classical_hamiltonian(name: str, variant: str) -> None:
    """The sources: the paper's M_{x^p} D0 (faithful) and the Chebyshev loader (corrected)."""
    _check(name, 2, variant)


@pytest.mark.slow
@pytest.mark.parametrize("name", ["R4c", "R4d"])
def test_fig5_maclaurin_sources_of_degree_5_and_7_match(name: str) -> None:
    """The paper's degree-5 and degree-7 Maclaurin sources: 21 and 23 qubits at n = 2."""
    _check(name, 2, "faithful")


@pytest.mark.parametrize("variant", ["faithful", "corrected"])
@pytest.mark.parametrize("name", ["R5a", "R5b", "R5c"])
@pytest.mark.nightly
def test_fig6_panels_match_the_classical_hamiltonian(name: str, variant: str) -> None:
    """Laplace, heat and wave on two axes: Kronecker products, and the corrected datum slices."""
    _check(name, (2, 2), variant)


@pytest.mark.nightly
def test_axes_may_take_different_resolutions() -> None:
    """One n per axis: the heat equation at n = (2, 3)."""
    _check("R5b", (2, 3), "corrected")


def test_the_resolution_must_name_every_axis() -> None:
    """A two-axis problem needs one n, or two."""
    with pytest.raises(ValueError, match="2 axes"):
        encode_standard_linear(problem("R5a"), (2, 2, 2))


@pytest.mark.parametrize("variant", ["faithful", "corrected"])
@pytest.mark.parametrize("name", ["R6a", "R6b"])
@pytest.mark.nightly
def test_fig7_panels_match_the_classical_hamiltonian(name: str, variant: str) -> None:
    """The doubled space: every term folded by N_1 or N_x, the copies' blocks side by side."""
    _check(name, 2, variant)


def test_a_lifted_problem_widens_the_register_by_one_qubit() -> None:
    """R3's residual is (2N x N) per block: the lifted rows sit on n + 1 qubits."""
    stacked, V = encode_standard_linear(problem("R3a"), 2)
    assert V.shape == (4, 4)
    assert stacked.shape == (2 * 8, 4)


def test_stack_rows_alpha_matches_the_stacked_norm() -> None:
    """alpha_R is a quadrature (l2), not l1, combination of the block alphas."""
    stacked, _ = encode_standard_linear(problem("R2a"), 2)
    assert stacked.alpha > 0


def test_out_of_scope_problems_are_refused_clearly() -> None:
    """A datum slice on one axis is outside the compiler's coverage."""
    datum = Constraint("datum", "f", "x", 0.5, datum="1")
    with pytest.raises(NotImplementedError, match="datum slices"):
        encode_standard_linear(dataclasses.replace(problem("R1"), constraints=(datum,)), 3)


@pytest.mark.parametrize("encoder", ["structured", "published"])
def test_the_encoder_changes_only_the_derivative(encoder: StandardEncoder) -> None:
    # R2a is G² + 4G + 4I with one invariant row: alpha_A = (alpha_G + 2)² in either encoder, and
    # the row, which reads no derivative, is the same atom.
    blocks = _one_axis_blocks(problem("R2a", "corrected"), 3, encoder)
    alpha_g = DERIVATIVE_ENCODERS[encoder](3).alpha
    assert blocks[0].alpha == pytest.approx((alpha_g + 2.0) ** 2)
    assert (
        blocks[1].alpha == _one_axis_blocks(problem("R2a", "corrected"), 3, "structured")[1].alpha
    )


def test_the_descriptor_has_no_derivative_encoder_to_choose() -> None:
    with pytest.raises(ValueError, match="no G to encode"):
        encode_linear(problem("R1"), 3, regime="descriptor", encoder="published")
