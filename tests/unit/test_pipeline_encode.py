"""The pipeline's ``"encode"`` stage."""

from __future__ import annotations

import dataclasses

from pihm.pipeline.spec import RunSpec
from pihm.pipeline.stages import run
from pihm.problems.ladder import problem
from pihm.problems.spec import Constraint, Problem


def test_the_spec_hash_changes_with_verify_and_resources() -> None:
    base = RunSpec(problem=problem("R1"), n=(3,), stage="encode")
    assert base.to_dict()["verify"] == "exact"
    assert base.to_dict()["resources"] is True
    probe = RunSpec(problem=problem("R1"), n=(3,), stage="encode", verify="probe")
    none_check = RunSpec(problem=problem("R1"), n=(3,), stage="encode", verify="none")
    no_resources = RunSpec(problem=problem("R1"), n=(3,), stage="encode", resources=False)
    hashes = {base.config_hash, probe.config_hash, none_check.config_hash, no_resources.config_hash}
    assert len(hashes) == 4


def test_encode_run_matches_the_classical_hamiltonian_exactly() -> None:
    spec = RunSpec(problem=problem("R1"), n=(3,), stage="encode", verify="exact", resources=True)
    record = run(spec)
    assert record.status == "ok"
    assert record.metrics["verify_passed"] is True
    assert record.metrics["verify_max_error"] < 1e-10
    assert record.metrics["alpha_R"] > 0
    assert record.metrics["cx_count"] > 0
    assert (
        record.metrics["n_qubits_total"]
        == record.metrics["n_sys_qubits"] + record.metrics["n_anc_qubits"]
    )


def test_a_descriptor_encode_run_verifies_its_rescaled_hamiltonian() -> None:
    spec = RunSpec(problem=problem("R2a", "corrected"), n=(2,), stage="encode", regime="descriptor")
    record = run(spec)
    assert record.status == "ok"
    assert record.metrics["verify_passed"] is True
    assert record.metrics["verify_max_error"] < 1e-10
    assert (record.metrics["rescale"], record.metrics["scales"]) == ("block", [4.0])
    assert record.metrics["n_sys_qubits"] == 4  # the state: two field qubits and the axis's two
    assert "rescale" not in run(dataclasses.replace(spec, regime="standard")).metrics


def test_encode_run_skips_expensive_work_when_asked() -> None:
    spec = RunSpec(problem=problem("R1"), n=(3,), stage="encode", verify="none", resources=False)
    record = run(spec)
    assert record.status == "ok"
    assert record.metrics["verify_passed"] is None
    assert record.metrics["verify_max_error"] is None
    assert record.metrics["resources_measured"] is False
    assert record.metrics["cx_count"] is None
    assert record.timings["verify"] < 0.01
    assert record.timings["resources"] < 0.01


def test_encode_run_records_an_out_of_scope_problem_as_failed() -> None:
    spec = RunSpec(
        problem=_one_axis_datum(), n=(3,), stage="encode", verify="none", resources=False
    )
    record = run(spec)
    assert record.status == "failed"
    assert record.error is not None
    assert "datum slices" in record.error


def test_encode_run_writes_no_arrays() -> None:
    spec = RunSpec(problem=problem("R1"), n=(3,), stage="encode", verify="none", resources=False)
    record = run(spec)
    assert record.arrays == {}


def _one_axis_datum() -> Problem:
    """R1 with a datum slice, which the compiler takes only on several axes."""
    datum = Constraint("datum", "f", "x", 0.5, datum="1")
    return dataclasses.replace(problem("R1"), constraints=(datum,))
