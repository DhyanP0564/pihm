"""Checks of the audit of a campaign's refusals, interruptions and sources."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from pihm.pipeline.audit import (
    audit,
    foreign_sources,
    interruptions,
    slow_refusals,
    step_seconds,
    unrecorded,
)
from pihm.pipeline.records import JSONValue, Provenance, Record, Status, write_record

_REFUSED_ON_TIME = "time: 237 h estimated, 12 h budget (slurm)"


def _record(
    r: float,
    status: Status,
    step: float,
    *,
    problem: str = "R5a",
    host: str = "nid1",
    reasons: str = _REFUSED_ON_TIME,
) -> Record:
    """Return a T1 record of ``problem`` at 19 qubits, candidate ``r``, whose step read ``step``."""
    metrics: dict[str, JSONValue] = {
        "problem": problem,
        "variant": "corrected",
        "regime": "standard",
        "n": 3,
        "qubits": [3, 3],
        "n_qubits": 19,
        "degree": 49058,
        "geometric_r": [r],
        "predicted_seconds_per_step": step,
        "predicted_seconds": step * 49059,
    }
    if status == "infeasible":
        metrics["infeasible_reasons"] = reasons
    provenance = Provenance(None, None, "0", "3.11", "linux", host, {}, {}, "2026-09-28")
    spec: dict[str, JSONValue] = {"problem": problem, "geometric_r": [r]}
    return Record(spec=spec, status=status, metrics=metrics, provenance=provenance)


def _pair(first: Record, second: Record) -> list[tuple[Path, Record]]:
    return [(Path("a"), first), (Path("b"), second)]


def test_the_step_is_read_from_the_record_or_from_its_predicted_time() -> None:
    assert step_seconds({"predicted_seconds_per_step": 0.25}) == 0.25
    assert step_seconds({"predicted_seconds_per_step": float("nan")}) is None
    two_candidates = {"predicted_seconds": 22.0, "degree": 10, "geometric_r": [0.5, -0.5]}
    assert step_seconds(two_candidates) == pytest.approx(1.0)
    assert step_seconds({"predicted_seconds": 11.0, "degree": 10}) == pytest.approx(1.0)
    assert step_seconds({"degree": 10}) is None


def test_a_refusal_many_times_its_twins_step_is_suspect() -> None:
    findings = slow_refusals(
        _pair(_record(0.5, "infeasible", 17.4, host="nid002537"), _record(-0.5, "ok", 0.325))
    )
    (finding,) = findings
    assert "R5a n=3x3" in finding
    assert "nid002537" in finding
    assert "54 times" in finding


def test_a_refusal_finds_its_twin_by_the_runs_specification() -> None:
    """A refusal records fewer metrics than a finished run: the rescaling is read from the spec."""
    refused, twin = _record(0.5, "infeasible", 27.0), _record(-0.5, "ok", 0.784)
    refused = replace(refused, spec={**refused.spec, "rescale": "block"})
    twin = replace(
        twin,
        spec={**twin.spec, "rescale": "block"},
        metrics={**twin.metrics, "rescale": "block"},
    )
    (finding,) = slow_refusals(_pair(refused, twin))
    assert "34 times" in finding


def test_a_refusal_near_its_twins_step_is_believed() -> None:
    assert not slow_refusals(_pair(_record(0.5, "infeasible", 0.99), _record(-0.5, "ok", 0.33)))


def test_only_a_refusal_on_time_and_with_a_twin_is_judged() -> None:
    memory = _record(0.5, "infeasible", 17.4, reasons="memory: 40 GiB estimated, 8 GiB budget")
    assert not slow_refusals(_pair(memory, _record(-0.5, "ok", 0.325)))
    # Another circuit of the same width is no twin: its gate count, and so its step, differ.
    other = _record(-0.5, "ok", 0.325, problem="R3a")
    assert not slow_refusals(_pair(_record(0.5, "infeasible", 17.4), other))


def _logs(
    tmp_path: Path,
    job: int,
    task: int,
    config: str,
    *,
    cancelled: bool,
    memory: bool = False,
    err: str | None = None,
) -> None:
    """Write one Slurm array task's ``.out`` and ``.err`` (``err``, if given) into ``tmp_path``."""
    stem = tmp_path / f"pihm-p4_t1-{job}_{task}"
    stem.with_suffix(".out").write_text(
        f"p4_t1: run {task} of 96\n"
        f"2026-09-27 21:22:33,756 run R4a/faithful stage=stateprep n=[3] ({config})\n"
    )
    limit = f"*** JOB {job} ON nid002810 CANCELLED AT 2026-09-28T21:22:27 DUE TO TIME LIMIT ***\n"
    oom = f"slurmstepd: error: Detected 1 oom_kill event in StepId={job}.0.\n"
    ending = limit if cancelled else oom if memory else ""
    stem.with_suffix(".err").write_text(ending if err is None else err)


def test_a_task_cancelled_at_its_time_limit_without_an_ok_record_is_a_finding(
    tmp_path: Path,
) -> None:
    refused = _record(-0.5, "infeasible", 1.2)
    _logs(tmp_path, 50032427, 51, refused.config_hash, cancelled=True)
    (finding,) = interruptions(tmp_path, [(Path("a"), refused)])
    assert refused.config_hash in finding
    assert "its record is infeasible" in finding
    _logs(tmp_path, 50032427, 52, "0123456789abcdef", cancelled=True)
    assert "its record is missing" in interruptions(tmp_path, [])[1]


def test_a_cancellation_that_a_later_attempt_or_an_ok_record_answers_is_not(
    tmp_path: Path,
) -> None:
    done = _record(0.5, "ok", 0.7)
    _logs(tmp_path, 50011393, 53, "0123456789abcdef", cancelled=True)
    _logs(tmp_path, 50032427, 53, done.config_hash, cancelled=False)
    _logs(tmp_path, 50032427, 54, done.config_hash, cancelled=True)
    assert not interruptions(tmp_path, [(Path("a"), done)])


def test_a_task_killed_out_of_memory_is_a_finding(tmp_path: Path) -> None:
    _logs(tmp_path, 50032427, 60, "0123456789abcdef", cancelled=False, memory=True)
    (finding,) = interruptions(tmp_path, [])
    assert finding.startswith("killed out of memory: R4a/faithful")


def test_a_task_that_crashed_is_a_finding(tmp_path: Path) -> None:
    crash = (
        "srun: error: nid002553: task 0: Bus error (core dumped)\nsrun: Terminating StepId=1.0\n"
    )
    _logs(tmp_path, 50244871, 49, "0123456789abcdef", cancelled=False, err=crash)
    (finding,) = interruptions(tmp_path, [])
    assert finding.startswith("crashed (bus error): R4a/faithful n=[3] (0123456789abcdef)")


def test_a_run_with_no_record_is_named_once(tmp_path: Path) -> None:
    """A task with no log never started; one Slurm ended is :func:`interruptions`' finding."""
    done = _record(0.5, "ok", 0.7)
    expected = [
        (done.config_hash, "done"),
        ("1" * 16, "ended"),
        ("2" * 16, "lost"),
        ("3" * 16, "odd"),
    ]
    _logs(tmp_path, 50232456, 0, done.config_hash, cancelled=False)
    _logs(tmp_path, 50232456, 1, "1" * 16, cancelled=True)
    _logs(tmp_path, 50232456, 3, "3" * 16, cancelled=False, err="Traceback: lost\n")
    findings = unrecorded(expected, [(Path("a"), done)], tmp_path)
    assert findings == [
        f"never started: no record and no log: task 2, lost ({'2' * 16})",
        f"no record: task 3, odd ({'3' * 16}); pihm-p4_t1-50232456_3.err ends Traceback: lost",
    ]
    assert unrecorded(expected[:3], [], None)[2] == f"no record: task 2, lost ({'2' * 16})"


def test_records_of_other_source_are_named_once_per_source() -> None:
    mine, other = _record(0.5, "ok", 0.3), _record(-0.5, "ok", 0.3)
    stamped = [(Path("a"), mine), (Path("b"), other)]
    (finding,) = foreign_sources(stamped, {"0123456789abcdef"})
    assert finding.startswith("2 records made by unrecorded source")
    assert not foreign_sources([], {"0123456789abcdef"})
    accepted = replace(other, provenance=replace(other.provenance, source_digest="b" * 16))
    assert len(foreign_sources([(Path("b"), accepted)], {"a" * 16, "b" * 16})) == 0


def test_the_audit_reads_a_results_directory(tmp_path: Path) -> None:
    slow, twin = _record(0.5, "infeasible", 17.4), _record(-0.5, "ok", 0.325)
    write_record(slow, tmp_path / "stateprep" / "slow")
    write_record(twin, tmp_path / "stateprep" / "twin")
    (tmp_path / "logs").mkdir()
    _logs(tmp_path / "logs", 50012458, 76, slow.config_hash, cancelled=True)
    findings = audit(tmp_path)
    assert [finding.split(":")[0] for finding in findings[:2]] == [
        "suspect refusal",
        "cancelled at the time limit",
    ]
    assert findings[2].startswith("2 records made by unrecorded source")
