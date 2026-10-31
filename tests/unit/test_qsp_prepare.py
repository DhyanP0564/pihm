"""Checks of tier T2 (exact polynomial emulation)."""

from __future__ import annotations

import math

import numpy as np
import pytest
from numpy.polynomial.chebyshev import chebval

from pihm.assemble import assemble
from pihm.circuits.assemble import encode_descriptor_linear, encode_standard_linear
from pihm.classical import physical_reference
from pihm.problems.ladder import problem
from pihm.qsp.filters import DEFAULT_EPSILON, QiteFilter, qite_coefficients
from pihm.qsp.initial import geometric_state
from pihm.qsp.minimax import MinimaxFilter, edge_suppression
from pihm.qsp.prepare import apply_qite_filter, infidelity_bound, paper_imaginary_time, t2


def test_apply_qite_filter_matches_an_independent_dense_eigenbasis_computation() -> None:
    # Cross-check the sparse matvec recurrence against a completely different code path: build H'
    # densely and apply the same polynomial via its eigendecomposition instead.
    p = problem("R1")
    n = 3
    stacked, _reflection = encode_standard_linear(p, n)
    residual = assemble(p, n)
    R = residual.matrix()
    alpha_r = stacked.alpha
    coefficients = np.array([0.5, -0.3, 0.1, -0.02])
    phi0 = geometric_state(residual.dimension, -0.5)

    sparse_result = apply_qite_filter(R, alpha_r, coefficients, phi0)

    H = (R.conj().T @ R).toarray()
    H_prime = 2.0 * H / alpha_r**2 - np.eye(H.shape[0])
    evals, evecs = np.linalg.eigh(H_prime)
    dense_result = evecs @ (chebval(evals, coefficients) * (evecs.conj().T @ phi0))

    assert np.max(np.abs(sparse_result - dense_result)) < 1e-10


def test_t2_prepares_r1_close_to_the_classical_reference() -> None:
    result = t2(problem("R1"), 4)
    assert result.infidelity < 1e-6
    assert 0.0 < result.success_probability <= 1.0
    assert result.phases.max_error() <= 1e-12


@pytest.mark.parametrize("name", ["R2a", "R2b"])
def test_t2_prepares_the_well_conditioned_r2_panels_close_to_reference(name: str) -> None:
    result = t2(problem(name), 2)
    assert result.infidelity < 1e-8


def test_t2_infidelity_shrinks_as_epsilon_tightens() -> None:
    p = problem("R1")
    infidelities = [t2(p, 3, epsilon=eps).infidelity for eps in (1e-2, 1e-4, 1e-6, 1e-8)]
    assert infidelities == sorted(infidelities, reverse=True)
    assert infidelities[0] > infidelities[-1]


def test_t2_geometric_state_keeps_the_larger_measured_probability() -> None:
    # R1's own solution decays from x = -1, so its overlap with r = -0.5's alternating-sign
    # geometric state should exceed r = +0.5's -- confirms both are genuinely tried, not just the
    # first one taken.
    p = problem("R1")
    n = 4
    result = t2(p, n, geometric_r=(0.5, -0.5))
    stacked, _reflection = encode_standard_linear(p, n)
    residual = assemble(p, n)
    R = residual.matrix()
    probabilities = {}
    for r in (0.5, -0.5):
        phi0 = geometric_state(residual.dimension, r)
        raw = apply_qite_filter(R, stacked.alpha, result.filter.coefficients, phi0)
        probabilities[r] = float(np.vdot(raw, raw).real)
    assert result.success_probability == pytest.approx(max(probabilities.values()))
    assert probabilities[0.5] != pytest.approx(probabilities[-0.5])


def test_t2_uniform_and_random_kinds_run() -> None:
    p = problem("R1")
    uniform_result = t2(p, 3, kind="uniform")
    random_result = t2(p, 3, kind="random", rng=np.random.default_rng(0))
    assert 0.0 <= uniform_result.success_probability <= 1.0
    assert 0.0 <= random_result.success_probability <= 1.0


def test_t2_rejects_an_unknown_kind() -> None:
    with pytest.raises(ValueError, match="unknown kind"):
        t2(problem("R1"), 3, kind="bogus")  # type: ignore[arg-type]


def test_t2_rejects_an_empty_geometric_r() -> None:
    with pytest.raises(ValueError, match="geometric_r"):
        t2(problem("R1"), 3, geometric_r=())


def test_t2_r2c_is_poorly_conditioned_by_default_but_tau_truncation_fixes_it() -> None:
    # R2's stiffest panel ({1, 5, 400}) has a genuinely degenerate near-null residual under the
    # paper's own untruncated construction at small n, unlike R2a/R2b at the same n (both < 1e-8
    # above) -- T2 correctly reports this rather than a fabricated small infidelity. The
    # tau-truncation ablation (assemble(..., truncation="tau")) resolves it (checked directly:
    # kernel_dimension drops from 3 to 1 and gap_ratio grows by over 300 orders of magnitude).
    untruncated = t2(problem("R2c"), 2)
    assert untruncated.infidelity > 1e-2
    assert untruncated.reference.kernel_dimension > 1

    truncated = t2(problem("R2c"), 2, truncation="tau")
    assert truncated.reference.kernel_dimension == 1
    assert truncated.reference.trusted
    assert truncated.infidelity < 1e-8


@pytest.mark.nightly
def test_t2_projects_onto_the_doubled_kernel() -> None:
    """R6a: the filter's gap is above the kernel; T2 prepares phi_0's projection onto it."""
    result = t2(problem("R6a", "corrected"), 2)
    assert result.reference.kernel_count() == 10
    assert result.gap > 1.0
    assert result.infidelity <= 1e-10


def test_t2_takes_an_explicit_imaginary_time() -> None:
    """The gap's own tau passed back in gives the same preparation."""
    baseline = t2(problem("R1"), 3)
    assert isinstance(baseline.filter, QiteFilter)
    again = t2(problem("R1"), 3, tau=baseline.filter.tau)
    assert again.filter.degree == baseline.filter.degree
    np.testing.assert_allclose(again.state, baseline.state, atol=1e-14)


def test_paper_imaginary_time_is_the_paper_rule_on_this_filter() -> None:
    """Tau = t alpha_R^2 / (2 ||H||_F), t = lambda_max / (lambda_2 (n - 1)) on H / ||H||_F."""
    from pihm.assemble import assemble
    from pihm.circuits.assemble import encode_standard_linear
    from pihm.classical import reference_state

    residual = assemble(problem("R2a", "corrected"), 3)
    values = reference_state(residual.matrix()).singular_values ** 2
    alpha = encode_standard_linear(problem("R2a", "corrected"), 3)[0].alpha
    t = values[-1] / (values[1] * 2)
    expected = t * alpha**2 / (2.0 * np.sqrt(np.sum(values**2)))
    assert paper_imaginary_time(problem("R2a", "corrected"), 3) == pytest.approx(expected)


@pytest.mark.parametrize(("name", "n"), [("R1", 3), ("R2a", 3), ("R3a", 2), ("R6a", 2), ("R4d", 3)])
@pytest.mark.nightly
def test_t2_infidelity_is_within_its_a_priori_bound(name: str, n: int) -> None:
    # The predicted bound, from the spectrum's edge, gamma, tau and the tail only;
    # R3a at n = 2 has its ground state 0.56 gaps above -1, R6a a ten-dimensional kernel.
    result = t2(problem(name, "corrected"), n)
    assert 0.0 < result.gamma <= 1.0
    assert result.infidelity <= result.infidelity_bound


def test_the_bound_is_not_certified_where_the_tail_reaches_the_target() -> None:
    # R4d at n = 2: the ground state sits 3.9 gaps above -1, so its filtered value is below the
    # tail, and the rule's filter misses epsilon.
    result = t2(problem("R4d", "corrected"), 2)
    assert math.isinf(result.infidelity_bound)
    assert result.infidelity > 1e-6


def test_infidelity_bound_matches_its_formula_by_hand() -> None:
    qf = qite_coefficients(50.0, tolerance=1e-8)
    singular = np.array([0.1, 0.5, 1.0])
    f = np.exp(-2.0 * 50.0 * (singular / 2.0) ** 2)
    gamma = 0.8
    expected = ((0.6 * f[1] + qf.tail_bound) / (gamma * f[0] - qf.tail_bound)) ** 2
    assert infidelity_bound(singular, 2.0, qf, gamma, 1) == pytest.approx(expected, rel=1e-12)
    spread = gamma * (f[0] - f[1])
    expected_two = ((0.6 * f[2] + spread + qf.tail_bound) / (gamma * f[1] - qf.tail_bound)) ** 2
    assert infidelity_bound(singular, 2.0, qf, gamma, 2) == pytest.approx(expected_two, rel=1e-12)


@pytest.mark.parametrize(("name", "n"), [("R1", 3), ("R2a", 3), ("R3a", 3), ("R4d", 3)])
def test_t2_prepares_the_descriptor_form_on_its_physical_subspace(name: str, n: int) -> None:
    """The initial state lies on the field block; the gap and the target are the physical ones."""
    p = problem(name, "corrected")
    result = t2(p, n, regime="descriptor")
    residual = assemble(p, n, regime="descriptor")
    size = residual.dimension // len(residual.derivatives)
    padding = [i for i, d in enumerate(residual.derivatives) if d is None]
    assert np.linalg.norm(result.initial[size:]) == 0.0
    for i in padding:
        assert np.linalg.norm(result.state[i * size : (i + 1) * size]) <= 1e-14
    assert result.gap == pytest.approx(physical_reference(residual).gap)
    assert result.alpha_r == pytest.approx(encode_descriptor_linear(p, n)[0].alpha)
    assert result.infidelity <= result.infidelity_bound


@pytest.mark.parametrize(("name", "n", "regime"), [("R1", 4, "standard"), ("R2a", 4, "descriptor")])
def test_the_minimax_filter_holds_qite_s_guarantee_at_fewer_degrees(
    name: str, n: int, regime: str
) -> None:
    """The same suppression and success probability, about 1.9 times fewer degrees."""
    spec = problem(name, "corrected")
    qite = t2(spec, n, regime=regime)  # type: ignore[arg-type]
    minimax = t2(spec, n, regime=regime, filter="minimax")  # type: ignore[arg-type]
    assert isinstance(minimax.filter, MinimaxFilter)
    assert 1.85 < qite.filter.degree / minimax.filter.degree < 1.97
    assert minimax.success_probability == pytest.approx(qite.success_probability, rel=1e-3)
    assert minimax.infidelity <= minimax.infidelity_bound <= DEFAULT_EPSILON


def test_the_minimax_degree_is_the_least_at_suppression_epsilon() -> None:
    result = t2(problem("R1"), 4, filter="minimax")
    ground, excited = 2.0 * (result.reference.singular_values[:2] / result.alpha_r) ** 2
    degree = result.filter.degree
    assert float(edge_suppression(ground, degree, excited)) <= DEFAULT_EPSILON
    assert float(edge_suppression(ground, degree - 1, excited)) > DEFAULT_EPSILON


def test_an_imaginary_time_or_an_unknown_filter_is_refused() -> None:
    with pytest.raises(ValueError, match="QITE filter only"):
        t2(problem("R1"), 3, filter="minimax", tau=1.0)
    with pytest.raises(ValueError, match="unknown filter"):
        t2(problem("R1"), 3, filter="chebyshev")  # type: ignore[arg-type]


@pytest.mark.parametrize("regime", ["standard", "descriptor"])
@pytest.mark.parametrize(("name", "order"), [("R7", None), ("R8", 3)])
def test_t2_prepares_a_system_within_its_bound(name: str, order: int | None, regime: str) -> None:
    result = t2(problem(name), 3, regime=regime, carleman=order, filter="minimax")  # type: ignore[arg-type]
    assert result.reference.trusted
    assert result.infidelity <= DEFAULT_EPSILON
    assert result.infidelity <= result.infidelity_bound


@pytest.mark.nightly
def test_t2_prepares_a_strong_lift_within_epsilon_once_its_levels_are_scaled() -> None:
    """At a/nu = 10 the level-one initial state reaches the scaled lift's ground."""
    result = t2(problem("R9e-M4"), 3, regime="descriptor", carleman=4, filter="minimax")
    assert result.gamma**2 > 1e-3
    assert result.infidelity <= DEFAULT_EPSILON
