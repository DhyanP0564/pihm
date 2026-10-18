"""Checks of tier T1 (full circuit simulation).

R1 across n in the default suite. R2's circuits have 20 qubits at n = 2 and take about 4 s per
GQSP step on a laptop, so under ``slow`` one panel (R2b, at a looser filter accuracy and so a
lower degree) stands for them; the p4_t1 campaign records T1 for every panel at the production
parameters.
"""

from __future__ import annotations

import numpy as np
import pytest

from pihm.problems.ladder import problem
from pihm.qsp.feasibility import Budget, InfeasibleError
from pihm.qsp.prepare import t1, t2


@pytest.mark.parametrize("n", [2, 3])
def test_t1_agrees_with_t2_for_r1(n: int) -> None:
    # The acceptance bar: T1 vs T2 state difference <= 1e-10.
    result = t1(problem("R1"), n, geometric_r=(-0.5,))
    assert result.t2_agreement <= 1e-10
    assert result.filtered_agreement <= 1e-12


def test_t1_reports_the_circuit_width() -> None:
    # R1 at n = 2: 2 system qubits, 7 walk ancilla, and the signal qubit.
    assert t1(problem("R1"), 2, geometric_r=(-0.5,)).n_qubits == 10


def test_t1_state_probability_and_infidelity_match_t2() -> None:
    p = problem("R1")
    r1 = t1(p, 2)
    r2 = t2(p, 2)
    assert np.max(np.abs(r1.state - r2.state)) <= 1e-10
    assert abs(r1.success_probability - r2.success_probability) < 1e-10
    assert abs(r1.infidelity - r2.infidelity) < 1e-8
    assert r1.initial_kind == r2.initial_kind
    assert np.array_equal(r1.initial, r2.initial)


@pytest.mark.slow
def test_t1_agrees_with_t2_for_r1_at_n4() -> None:
    # 14 qubits, degree 1,522: the densest R1 case T1 is run at by default.
    result = t1(problem("R1"), 4, geometric_r=(-0.5,))
    assert result.t2_agreement <= 1e-10


@pytest.mark.slow
def test_t1_agrees_with_t2_for_r2() -> None:
    # R2b at n = 2: 13 qubits, degree 199 at epsilon = 1e-2, p about 0.79.
    result = t1(problem("R2b", "corrected"), 2, geometric_r=(-0.5,), epsilon=1e-2)
    assert result.n_qubits == 13
    assert result.t2_agreement <= 1e-10
    assert result.filtered_agreement <= 1e-12


def test_t1_refuses_a_run_over_its_budget() -> None:
    # Refused before anything is simulated: the estimate covers both initial-state candidates.
    with pytest.raises(InfeasibleError, match="memory") as refused:
        t1(problem("R1"), 2, budget=Budget(1e6, 3600.0, "override"))
    assert refused.value.estimate.simulations == 2


def test_t1_keeps_the_estimate_it_ran_under() -> None:
    result = t1(problem("R1"), 2, geometric_r=(-0.5,), budget=Budget(1e12, 3600.0, "override"))
    assert result.cost is not None
    assert result.cost.n_qubits == result.n_qubits
    assert result.cost.degree == result.filter.degree


def test_t1_rejects_an_empty_geometric_r() -> None:
    with pytest.raises(ValueError, match="geometric_r"):
        t1(problem("R1"), 2, geometric_r=())


def test_t1_rejects_an_unknown_kind() -> None:
    with pytest.raises(ValueError, match="unknown kind"):
        t1(problem("R1"), 2, kind="bogus")  # type: ignore[arg-type]


def test_t1_agrees_with_t2_in_the_descriptor_form() -> None:
    """The descriptor's GQSP circuit, 9 qubits for R1 in mass form at n = 2, against T2."""
    result = t1(problem("R1"), 2, geometric_r=(-0.5,), regime="descriptor")
    assert result.t2_agreement <= 1e-10
    assert result.filtered_agreement <= 1e-12
    assert result.n_qubits == 9


def test_t1_agrees_with_t2_under_the_minimax_filter() -> None:
    result = t1(problem("R1"), 2, geometric_r=(-0.5,), filter="minimax")
    assert result.t2_agreement <= 1e-10
    assert result.filtered_agreement <= 1e-10
