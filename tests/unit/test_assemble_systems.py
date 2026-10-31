"""First-order systems in both forms, checked against their definition."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest
import scipy.sparse as sp

from pihm.assemble import PADDING_WEIGHT, Residual, assemble, system_state
from pihm.assemble.systems import chain_scale, coupling, lift_scale, vector_field
from pihm.basis import tau
from pihm.carleman import carleman_matrix, moments, monomials
from pihm.classical import solve
from pihm.operators import banded_factors, derivative_matrix
from pihm.problems import ladder
from pihm.problems.spec import Constraint, Equation, Factor, Polynomial, Term

#: Each system of the ladder, with a Carleman order where it is lifted.
SYSTEMS = [("R7", None), ("R8", 2), ("R8", 3), ("R8", 5)]


def _block(residual: Residual, name: str) -> np.ndarray:
    (block,) = [b for b in residual.blocks if b.name == name]
    return np.asarray(block.matrix.toarray())


def test_the_problems_become_their_vector_fields() -> None:
    (linear,) = vector_field(ladder.problem("R7"))
    np.testing.assert_allclose(linear.toarray(), [[0.0, 1.0], [-(np.pi**2), 0.0]])
    linear, quadratic = vector_field(ladder.problem("R8"))
    np.testing.assert_allclose(linear.toarray(), [[-1.5]])
    np.testing.assert_allclose(quadratic.toarray(), [[1.5]])


@pytest.mark.parametrize(("name", "order"), SYSTEMS)
def test_the_state_is_one_block_per_component_padded(name: str, order: int | None) -> None:
    residual = assemble(ladder.problem(name), 3, carleman=order)
    count = 2 if order is None else order
    blocks = 1 << (count - 1).bit_length()
    assert residual.is_system
    assert len(residual.monomials) == len(residual.derivatives) == blocks
    assert residual.monomials.count(None) == residual.derivatives.count(None) == blocks - count
    assert residual.carleman_order == (order or 1)
    assert residual.dimension == blocks * 8
    assert residual.field_blocks == ((0, 1) if name == "R7" else (0,))


@pytest.mark.parametrize(("name", "order"), SYSTEMS)
def test_the_descriptor_equations_are_the_standard_ones_times_b(
    name: str, order: int | None
) -> None:
    """Mass form: (I (x) B~)(I (x) G - A (x) I) = I (x) D~ - A (x) B~, exactly."""
    n = 4
    problem = ladder.problem(name)
    standard = _block(assemble(problem, n, carleman=order), "equation")
    descriptor = _block(assemble(problem, n, regime="descriptor", carleman=order), "equation")
    B, _ = banded_factors(n)
    components = standard.shape[0] // (1 << n)
    mass = np.kron(np.eye(components), B.toarray())
    np.testing.assert_allclose(mass @ standard, descriptor, atol=1e-12 * np.abs(standard).max())


@pytest.mark.parametrize(("name", "order"), SYSTEMS)
def test_the_ratio_rows_vanish_on_the_moments_of_the_data_alone(
    name: str, order: int | None
) -> None:
    n = 3
    residual = assemble(ladder.problem(name), n, carleman=order)
    (constraint,) = residual.problem.constraints
    rows = _block(residual, f"ratio on {constraint.axis} = -1")
    keys = [key for key in residual.monomials if key is not None]
    data = [0.0, np.pi] if name == "R7" else [0.5]
    tau_0 = tau(-1.0, n)
    size = 1 << n
    pinned = np.zeros(residual.dimension)
    for i, value in enumerate(moments(data, keys)):
        # coefficients whose value at the point is the moment: tau(-1) scaled
        pinned[i * size : (i + 1) * size] = value * tau_0 / (tau_0 @ tau_0)
    assert np.max(np.abs(rows @ pinned)) < 1e-14
    assert np.linalg.matrix_rank(rows) == len(residual.monomials) - 1


def test_a_zero_datum_is_held_as_a_zero_constraint() -> None:
    """R7's ratio row, divided by y_v, is exactly B(-1) on u."""
    residual = assemble(ladder.problem("R7"), 3)
    rows = _block(residual, "ratio on x = -1")
    expected = np.zeros_like(rows)
    expected[0, :8] = np.sqrt(8.0) * tau(-1.0, 3)
    np.testing.assert_allclose(rows, expected, atol=1e-15)


def test_padding_blocks_decouple_with_their_weight() -> None:
    residual = assemble(ladder.problem("R8"), 3, carleman=3)
    padding = _block(residual, "padding block 3")
    np.testing.assert_array_equal(padding[:, 24:], PADDING_WEIGHT * np.eye(8))
    R = residual.matrix().toarray()
    np.testing.assert_array_equal(R[:, 24:][np.any(R[:, :24] != 0, axis=1)], 0.0)


@pytest.mark.parametrize("regime", ["standard", "descriptor"])
def test_the_reference_state_is_the_kernel_of_the_linear_system(regime: str) -> None:
    """R7 is linear: at a resolved n its exact solution satisfies every row to truncation."""
    residual = assemble(ladder.problem("R7"), 6, regime=regime)  # type: ignore[arg-type]
    z = system_state(residual)
    R = residual.matrix()
    sigma_max = np.linalg.norm(R.toarray(), 2)
    assert np.linalg.norm(R @ z) / (sigma_max * np.linalg.norm(z)) < 1e-12


def test_both_forms_share_their_kernel() -> None:
    problem = ladder.problem("R8")
    states = [
        solve(assemble(problem, 6, regime=r, carleman=4)).vector for r in ("standard", "descriptor")
    ]
    assert 1.0 - abs(np.vdot(*states)) ** 2 < 1e-12


def test_the_carleman_error_falls_with_the_order() -> None:
    """The error falls as O(R^K) with R = 1/2 once the time axis is resolved (n = 5)."""
    errors = [
        solve(assemble(ladder.problem("R8"), 5, regime="descriptor", carleman=k)).field_error_linf
        for k in (2, 3, 4, 5, 6)
    ]
    ratios = np.array(errors[1:]) / np.array(errors[:-1])
    assert np.all(ratios < 0.6)
    assert errors[-1] < 1e-3


def test_the_fields_are_decoded_with_one_scale() -> None:
    solution = solve(assemble(ladder.problem("R7"), 5, regime="descriptor"))
    assert solution.field.shape == solution.reference.shape == (2, *solution.grid[0].shape)
    assert solution.field_error_linf < 1e-12
    assert solution.eta == pytest.approx(solution.eta_ideal, rel=1e-12)


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"carleman": 2, "problem": "R7"}, "is linear"),
        ({"problem": "R8"}, "ratio rows pin"),
        ({"carleman": 0, "problem": "R8"}, "at least 1"),
        ({"carleman": 2, "problem": "R8", "truncation": "tau"}, "no truncation"),
        ({"carleman": 2, "problem": "R8", "nonlinearity": "doubled"}, "only a Carleman lift"),
    ],
)
def test_inconsistent_options_are_refused(change: dict[str, object], message: str) -> None:
    options = dict(change)
    problem = ladder.problem(str(options.pop("problem")))
    with pytest.raises(ValueError, match=message):
        assemble(problem, 3, **options)  # type: ignore[arg-type]


def _r7_with(equations: tuple[Equation, ...]) -> object:
    return dataclasses.replace(ladder.problem("R7"), equations=equations)


@pytest.mark.parametrize(
    ("equations", "message"),
    [
        (
            (
                Equation((Term(1.0, (Factor("u", (2,)),)), Term(-1.0, (Factor("v", (0,)),)))),
                Equation((Term(1.0, (Factor("v", (1,)),)), Term(1.0, (Factor("u", (0,)),)))),
            ),
            "first derivative once",
        ),
        (
            (
                Equation((Term(1.0, (Factor("u", (1,)),)), Term(-1.0, (Factor("v", (0,)),)))),
                Equation((Term(1.0, (Factor("u", (1,)),)), Term(1.0, (Factor("u", (0,)),)))),
            ),
            "two equations",
        ),
        (
            (
                Equation((Term(1.0, (Factor("u", (1,)),)), Term(-1.0))),
                Equation((Term(1.0, (Factor("v", (1,)),)), Term(1.0, (Factor("u", (0,)),)))),
            ),
            "homogeneous",
        ),
        (
            (
                Equation(
                    (
                        Term(1.0, (Factor("u", (1,)),)),
                        Term(1.0, (Factor("v", (0,)),), Polynomial("x", (0.0, 1.0))),
                    )
                ),
                Equation((Term(1.0, (Factor("v", (1,)),)), Term(1.0, (Factor("u", (0,)),)))),
            ),
            "constant",
        ),
    ],
)
def test_problems_outside_the_class_are_refused(
    equations: tuple[Equation, ...], message: str
) -> None:
    with pytest.raises(NotImplementedError, match=message):
        vector_field(_r7_with(equations))  # type: ignore[arg-type]


def test_a_system_takes_ratio_constraints_only() -> None:
    problem = dataclasses.replace(
        ladder.problem("R7"), constraints=(Constraint("zero", "u", "x", -1.0),)
    )
    with pytest.raises(NotImplementedError, match="ratio constraints only"):
        assemble(problem, 3)


def test_the_standard_equations_are_g_minus_a() -> None:
    n = 3
    residual = assemble(ladder.problem("R7"), n)
    equation = _block(residual, "equation")
    G = derivative_matrix(n).toarray()
    A = np.array([[0.0, 1.0], [-(np.pi**2), 0.0]])
    np.testing.assert_allclose(equation, np.kron(np.eye(2), G) - np.kron(A, np.eye(8)))
    assert sp.issparse(residual.blocks[0].matrix)


def test_a_first_order_equation_is_carried_in_mass_form_in_the_descriptor() -> None:
    """R1 in the descriptor is one block, D~ + 2B~, with the standard form's state."""
    n = 5
    descriptor = assemble(ladder.problem("R1"), n, regime="descriptor")
    standard = assemble(ladder.problem("R1"), n)
    assert descriptor.is_system
    assert not standard.is_system
    assert descriptor.derivatives == ((0,),)
    assert descriptor.field_qubits == 0
    B, D = (m.toarray() for m in banded_factors(n))
    np.testing.assert_allclose(_block(descriptor, "equation"), D + 2.0 * B, atol=1e-14)
    ground = [solve(r).vector for r in (descriptor, standard)]
    assert 1.0 - abs(np.vdot(*ground)) ** 2 < 1e-14


def test_r1s_chain_ablation_is_the_descriptors_chain() -> None:
    """R1's chain ablation carries its derivative as a block again, with lambda = 4."""
    residual = assemble(ladder.problem("R1"), 4, regime="descriptor", layout="chain")
    assert not residual.is_system
    assert residual.derivatives == ((0,), (1,))
    assert residual.scales == (4.0,)


@pytest.mark.parametrize(("name", "order"), [("R7", None), ("R8", 3)])
def test_a_systems_chain_reads_out_the_mass_forms_field(name: str, order: int | None) -> None:
    problem = ladder.problem(name)
    mass = solve(assemble(problem, 5, regime="descriptor", carleman=order))
    chain = solve(assemble(problem, 5, regime="descriptor", carleman=order, layout="chain"))
    count = len(chain.residual.monomials) - chain.residual.monomials.count(None)
    assert chain.residual.derivatives[count // 2 : count] == ((1,),) * (count // 2)
    assert chain.residual.scales[0] == pytest.approx(chain_scale(problem, order or 1))
    np.testing.assert_allclose(chain.field, mass.field, atol=1e-10 * np.abs(mass.field).max())
    if order is None:  # a linear system's exact solution is the chain's kernel, blocks and all
        z = system_state(chain.residual)
        R = chain.residual.matrix().toarray()
        assert np.linalg.norm(R @ z) < 1e-9 * np.linalg.norm(z) * np.abs(R).max()


def test_a_chains_ratio_rows_leave_the_derivatives_free() -> None:
    residual = assemble(ladder.problem("R7"), 3, regime="descriptor", layout="chain")
    rows = _block(residual, "ratio on x = -1")
    assert not np.any(rows[:, 16:32])


def test_the_chain_scale_is_decision_33s_rule() -> None:
    """The scale is 2 rho: R7's roots are +-i pi, R1's -2 (the single-field rule's 4)."""
    assert chain_scale(ladder.problem("R7"), 1) == pytest.approx(2 * np.pi)


@pytest.mark.parametrize(("name", "regime"), [("R7", "standard"), ("R2a", "descriptor")])
def test_the_chain_layout_is_the_descriptors_first_order_ablation(name: str, regime: str) -> None:
    variant = "corrected"
    with pytest.raises(ValueError, match="chain layout"):
        assemble(ladder.problem(name, variant), 3, regime=regime, layout="chain")  # type: ignore[arg-type]


def test_the_lift_scale_is_the_data_norm_past_one() -> None:
    """The lift scale is s = max(|u_0|, 1), and 1 without a lift."""
    assert lift_scale(ladder.problem("R7"), 1) == 1.0
    assert lift_scale(ladder.problem("R8"), 3) == 1.0
    assert lift_scale(ladder.problem("R9c-M4"), 3) == 1.0
    assert lift_scale(ladder.problem("R9e-M4"), 3) == pytest.approx(10.0)


def test_the_scaled_lift_is_a_diagonal_similarity_of_the_unscaled_one() -> None:
    """D^-1 A_K D with D = diag(s^k): the same lifted system in w = u / s."""
    problem = ladder.problem("R9e-M4")
    order, scale = 3, 10.0
    unscaled = carleman_matrix(vector_field(problem), order).toarray()
    levels = np.array([sum(key) for key in monomials(4, order)], dtype=float)
    D = np.diag(scale**levels)
    np.testing.assert_allclose(
        coupling(problem, order).toarray(), np.linalg.inv(D) @ unscaled @ D, atol=1e-12
    )


@pytest.mark.parametrize("regime", ["standard", "descriptor"])
@pytest.mark.nightly
def test_a_scaled_lift_reads_out_the_unscaled_lifts_field(regime: str) -> None:
    """The decoded field is the unscaled truncated lift's, exponentiated apart (R9e-M4, K = 3)."""
    from scipy.sparse.linalg import expm_multiply

    problem = ladder.problem("R9e-M4")
    solution = solve(assemble(problem, 4, regime=regime, carleman=3))  # type: ignore[arg-type]
    assert solution.residual.lift_scale == pytest.approx(10.0)
    (times,) = solution.grid
    unscaled = carleman_matrix(vector_field(problem), 3).tocsc()
    start = moments([10.0, 0.0, 0.0, 0.0], monomials(4, 3))
    lifted = expm_multiply(unscaled, start, start=0.0, stop=2.0, num=times.size, endpoint=True)
    np.testing.assert_allclose(solution.field, lifted[:, :4].T, atol=1e-6 * 10.0)
