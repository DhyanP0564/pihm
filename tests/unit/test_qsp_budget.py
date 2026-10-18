"""Checks of the state-preparation error budget."""

from __future__ import annotations

import math

import numpy as np
import pytest

from pihm import classical
from pihm.problems.ladder import problem
from pihm.qsp.budget import state_budget, state_distance
from pihm.qsp.prepare import t2


def test_state_distance_ignores_the_global_phase() -> None:
    a = np.array([1.0, 0.0], dtype=np.complex128)
    b = np.array([np.cos(0.1), np.sin(0.1)], dtype=np.complex128)
    assert state_distance(a, 1j * a) == pytest.approx(0.0, abs=1e-15)
    assert state_distance(a, np.exp(0.7j) * b) == pytest.approx(2.0 * np.sin(0.05), rel=1e-12)
    assert math.isnan(state_distance(a, np.zeros(2, dtype=np.complex128)))


def test_state_distance_resolves_below_the_overlap_form() -> None:
    # sqrt(2 - 2|<a|b>|) cancels to about 1e-8; the difference's norm does not.
    a = np.array([1.0, 0.0], dtype=np.complex128)
    b = np.array([np.cos(1e-10), np.sin(1e-10)], dtype=np.complex128)
    assert state_distance(a, b) == pytest.approx(1e-10, rel=1e-6)


@pytest.mark.parametrize(("name", "n"), [("R1", 3), ("R2a", 3), ("R3a", 3)])
def test_the_budget_closes_on_well_resolved_problems(name: str, n: int) -> None:
    result = t2(problem(name), n)
    budget = state_budget(result)
    assert budget.measured <= budget.t2 * (1.0 + 1e-9)
    assert budget.filter > budget.truncation
    assert budget.phases < 1e-12
    assert budget.measured == pytest.approx(math.sqrt(result.infidelity), rel=1e-2)


def test_truncation_dominates_where_the_ground_state_sits_above_the_edge() -> None:
    # R4d at n = 2: lambda_0 / Delta = 3.9, the tail reaches the ground state's filtered value
    # ; the budget shows it is the truncation, not the imaginary time.
    budget = state_budget(t2(problem("R4d"), 2))
    assert budget.truncation > 1e4 * budget.filter
    assert budget.measured <= budget.t2 * (1.0 + 1e-9)


def test_the_minimax_filter_leaves_only_the_recurrences_rounding() -> None:
    result = t2(problem("R2a"), 3, filter="minimax")
    budget = state_budget(result)
    # Nothing is truncated: the term is T2's recurrence against the exact spectral evaluation.
    assert budget.truncation < 1e-4 * budget.filter
    assert budget.measured <= budget.t2 * (1.0 + 1e-9)


def test_past_the_dense_ceiling_the_budget_decomposes_the_physical_columns(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The descriptor's padding decouples: its own SVD of the physical columns, as the dense one."""
    expected = state_budget(t2(problem("R2a"), 3, regime="descriptor", filter="minimax"))
    monkeypatch.setattr(classical, "DENSE_COLUMNS", 16)
    result = t2(problem("R2a"), 3, regime="descriptor", filter="minimax")
    assert result.reference.right_vectors is None
    budget = state_budget(result)
    assert budget.filter == pytest.approx(expected.filter, rel=1e-6)
    assert budget.measured == pytest.approx(expected.measured, rel=1e-6)
    assert budget.phases == pytest.approx(expected.phases, rel=1e-3, abs=1e-15)
