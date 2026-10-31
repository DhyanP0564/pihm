"""The pipeline's ``"stateprep"`` and ``"readout"`` stages."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from pihm.pipeline.spec import RunSpec
from pihm.pipeline.stages import run
from pihm.pipeline.summary import keep_best_candidate
from pihm.problems.ladder import problem
from pihm.problems.spec import Constraint, Problem


def test_the_spec_hash_changes_with_kind_and_epsilon() -> None:
    base = RunSpec(problem=problem("R1"), n=(3,), stage="stateprep")
    assert base.to_dict()["kind"] == "geometric"
    assert base.to_dict()["epsilon"] == 1e-6
    uniform = RunSpec(problem=problem("R1"), n=(3,), stage="stateprep", kind="uniform")
    looser = RunSpec(problem=problem("R1"), n=(3,), stage="stateprep", epsilon=1e-4)
    hashes = {base.config_hash, uniform.config_hash, looser.config_hash}
    assert len(hashes) == 3


def test_stateprep_run_prepares_r1_close_to_the_classical_reference() -> None:
    spec = RunSpec(problem=problem("R1"), n=(3,), stage="stateprep")
    record = run(spec)
    assert record.status == "ok"
    assert record.metrics["infidelity"] < 1e-6
    assert 0.0 < record.metrics["success_probability"] <= 1.0
    assert record.metrics["phases_max_error"] <= 1e-12
    assert record.metrics["alpha_R"] > 0


def test_stateprep_run_honours_truncation_for_r2c() -> None:
    spec = RunSpec(problem=problem("R2c"), n=(2,), stage="stateprep", truncation="tau")
    record = run(spec)
    assert record.status == "ok"
    assert record.metrics["trusted"] is True
    assert record.metrics["infidelity"] < 1e-8


def test_stateprep_run_records_an_out_of_scope_problem_as_failed() -> None:
    spec = RunSpec(problem=_one_axis_datum(), n=(3,), stage="stateprep")
    record = run(spec)
    assert record.status == "failed"
    assert record.error is not None
    assert "datum slices" in record.error


def test_a_stateprep_run_keeps_its_field_beside_the_ground_states() -> None:
    """The prepared field is the one its metrics judge, on the build's grid."""
    record = run(RunSpec(problem=problem("R7"), n=(3,), stage="stateprep", filter="minimax"))
    arrays = record.arrays
    assert set(arrays) == {"field", "ground", "reference", "grid_0"}
    assert arrays["field"].shape == arrays["ground"].shape == (2, arrays["grid_0"].size)
    error = float(np.max(np.abs(arrays["field"] - arrays["reference"])))
    assert error == record.metrics["field_err_linf"]
    assert (
        float(np.max(np.abs(arrays["ground"] - arrays["reference"])))
        == (record.metrics["eps_disc"])
    )


def test_the_spec_hash_changes_with_tier_and_shots() -> None:
    base = RunSpec(problem=problem("R1"), n=(2,), stage="stateprep")
    assert base.to_dict()["tier"] == "T2"
    circuit = RunSpec(problem=problem("R1"), n=(2,), stage="stateprep", tier="T1")
    sampled = RunSpec(problem=problem("R1"), n=(2,), stage="readout", tier="T1", shots=10)
    assert len({base.config_hash, circuit.config_hash, sampled.config_hash}) == 3


def test_a_t1_run_records_its_agreement_with_t2_and_its_estimate() -> None:
    record = run(RunSpec(problem=problem("R1"), n=(2,), stage="stateprep", tier="T1"))
    assert record.status == "ok"
    assert record.metrics["tier"] == "T1"
    assert record.metrics["n_qubits"] == 10
    assert record.metrics["t2_agreement"] <= 1e-10
    assert record.metrics["filtered_agreement"] <= 1e-12
    assert record.metrics["predicted_seconds"] > 0


def test_a_t1_run_over_its_budget_is_recorded_as_infeasible(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PIHM_MEMORY_BUDGET_GB", "0.001")
    monkeypatch.setenv("PIHM_TIME_BUDGET_HOURS", "1")
    record = run(RunSpec(problem=problem("R1"), n=(2,), stage="stateprep", tier="T1"))
    assert record.status == "infeasible"
    assert record.error is None
    assert record.metrics["predicted_peak_gib"] > record.metrics["budget_gib"]
    assert "memory" in record.metrics["infeasible_reasons"]
    assert record.metrics["budget_source"] == "override"


def test_a_readout_run_records_the_field_and_its_agreement_with_t2() -> None:
    spec = RunSpec(problem=problem("R1"), n=(2,), stage="readout", tier="T1", shots=1000)
    record = run(spec)
    assert record.status == "ok"
    assert record.metrics["t2_agreement"] < 1e-10
    assert record.metrics["sampled_departure"] > 0
    assert set(record.arrays) == {"points", "field", "exact_field", "reference", "sampled_field"}
    assert run(spec).metrics["sampled_departure"] == record.metrics["sampled_departure"]


def test_a_readout_run_rejects_a_non_geometric_initial_state() -> None:
    record = run(RunSpec(problem=problem("R1"), n=(2,), stage="readout", kind="uniform"))
    assert record.status == "failed"
    assert record.error is not None
    assert "geometric" in record.error


def test_one_candidate_per_run_keeps_what_one_run_over_both_keeps() -> None:
    """Split by initial state, the kept record is the run over both candidates, to the bit."""
    both = run(RunSpec(problem=problem("R1"), n=(2,), stage="stateprep", tier="T1"))
    alone = [
        run(RunSpec(problem=problem("R1"), n=(2,), stage="stateprep", tier="T1", geometric_r=(r,)))
        for r in (0.5, -0.5)
    ]
    rows = [{**record.metrics, "status": record.status} for record in alone]
    (kept,) = keep_best_candidate(rows)
    for key in ("success_probability", "infidelity", "t2_agreement", "filtered_agreement"):
        assert kept[key] == both.metrics[key]
    assert both.config_hash not in {record.config_hash for record in alone}


def test_a_doubled_space_run_records_its_kernel_and_the_product_state() -> None:
    """The doubled space's record: its kernel, the exact target's residual, the product fidelity."""
    record = run(RunSpec(problem=problem("R6a", "corrected"), n=(2,), stage="stateprep"))
    assert record.status == "ok"
    assert record.metrics["kernel_count"] == 10
    assert record.metrics["target_residual"] <= 1e-14  # Eq. 29's solution is a polynomial
    assert 0.0 <= record.metrics["product_fidelity"] <= 1.0
    assert record.arrays == {}, "a doubled space's prepared state decodes to no field"


def _one_axis_datum() -> Problem:
    """R1 with a datum slice, which the compiler takes only on several axes."""
    datum = Constraint("datum", "f", "x", 0.5, datum="1")
    return dataclasses.replace(problem("R1"), constraints=(datum,))


def test_a_t3_run_records_the_estimate_at_its_resolution() -> None:
    """Tier T3 of the stateprep stage: the direct estimate, and whether it passes the ceiling."""
    record = run(RunSpec(problem=problem("R2b", "corrected"), n=(3,), stage="stateprep", tier="T3"))
    assert record.status == "ok"
    assert record.metrics["method"] == "direct"
    assert record.metrics["beyond_ceiling"] is False
    assert record.metrics["gate_count"] > 0


def test_a_linear_t2_run_checks_its_field_against_the_bound() -> None:
    """The field check: the decoded field within eps_disc plus the readout bound."""
    record = run(RunSpec(problem=problem("R3a", "corrected"), n=(3,), stage="stateprep"))
    metrics = record.metrics
    assert metrics["field_within_bound"] is True
    assert metrics["field_err_linf"] <= metrics["field_bound"]
    assert metrics["state_deviation"] > 0.0


def test_a_protocol_run_holds_every_time_rule() -> None:
    """One record: the gap's tau, the gap underestimated x2 and x10, and the paper's rule."""
    record = run(RunSpec(problem=problem("R1"), n=(3,), stage="protocol"))
    metrics = record.metrics
    assert record.status == "ok"
    assert metrics["gap_half_tau"] == pytest.approx(2.0 * metrics["gap_tau"])
    assert metrics["gap_tenth_tau"] == pytest.approx(10.0 * metrics["gap_tau"])
    assert metrics["gap_reaches_epsilon"] is True
    assert metrics["paper_degree"] < metrics["gap_degree"]


def test_a_random_initial_state_reproduces_from_its_configuration() -> None:
    """The random control: the state is seeded by the configuration hash."""
    spec = RunSpec(problem=problem("R1"), n=(2,), stage="stateprep", kind="random")
    first, second = run(spec).metrics, run(spec).metrics
    assert first["gamma2"] == second["gamma2"]
    assert first["infidelity"] == second["infidelity"]
    other = run(RunSpec(problem=problem("R1"), n=(2,), stage="stateprep", kind="random", label="x"))
    assert other.metrics["gamma2"] != first["gamma2"]


def test_a_t2_run_records_its_infidelity_bound() -> None:
    """The state check: the infidelity beside its a priori bound."""
    metrics = run(RunSpec(problem=problem("R1"), n=(3,), stage="stateprep")).metrics
    assert metrics["infidelity"] <= metrics["infidelity_bound"] <= 10.0 * metrics["infidelity"]
    assert 0.0 < metrics["gamma2"] <= 1.0


def test_a_linear_t2_run_records_its_error_budget() -> None:
    """The field within eps_disc plus the budget's filter and truncation terms.

    The budget bounds the state's distance by its chain's links, never less than the distance
    itself, so it is never tighter than the bound at the measured distance.
    """
    metrics = run(RunSpec(problem=problem("R1"), n=(3,), stage="stateprep")).metrics
    assert metrics["field_within_budget"] is True
    assert metrics["field_bound"] <= metrics["field_budget"]
    assert metrics["budget_phases"] < metrics["budget_truncation"] < metrics["budget_filter"]


def test_the_field_bound_holds_where_the_infidelity_rounds_to_zero() -> None:
    # R5a in the descriptor at n = 3 per axis prepares its state so well that 1 - |<a|b>|² rounds
    # to zero; the state is still 2.4e-8 away, which the bound must count.
    spec = RunSpec(
        problem=problem("R5a", "corrected"), n=(3, 3), stage="stateprep", regime="descriptor"
    )
    metrics = run(spec).metrics
    # Rounding level only: 0 on macOS, 4e-16 to 1.3e-15 across Linux runs. The distance's own
    # square is about 6e-16, so 1 - |<a|b>|^2 cannot resolve it.
    assert metrics["infidelity"] < 1e-14
    assert metrics["state_deviation"] > 1e-9
    assert metrics["field_within_bound"] is True
