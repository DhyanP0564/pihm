r"""Checks on a campaign's records before its refusals and gaps are believed.

A T1 or readout run times one step of its circuit on the node it lands on and refuses itself when
the degree times that step exceeds its budget (:mod:`pihm.qsp.feasibility`). A node shared with a
busy neighbour can read a step many times its steady cost, even after a warm-up and a median, and
a run can slow down after it has passed its estimate; neither shows in the record. Three checks
catch them, and the runs that left no record:

* :func:`slow_refusals`: a run refused on time whose step read more than :data:`SLOW_STEP_FACTOR`
  times the median step of every other record of the same circuit (the other initial-state
  candidate, and any earlier attempt). Width alone is no guide, since circuits of the same width
  differ in gate count. Across campaign ``p4_t1``'s cluster runs, the same circuit read within a
  factor of 3.5 of itself on a quiet node, and 6 to 54 times its steady step on a contended one.
* :func:`interruptions`: a Slurm task whose latest attempt was cancelled at its time limit,
  killed out of memory or crashed (a bus error or segmentation fault, as srun reports it; its
  ``.err`` in the campaign's ``logs/``) and whose run has no ``ok`` record, so its record, if any,
  is older than the attempt. An earlier attempt that a later one replaced is history, not a gap.
* :func:`unrecorded`: a run of the campaign with no record that no such ending explains: a Slurm
  task with no log never started (a launch failure).

A fourth check guards the records' provenance. :func:`foreign_sources` names the records made by
source other than the tree's (:func:`pihm.pipeline.records.source_digest`) or an earlier one the
campaign list accepts (:meth:`pihm.pipeline.results.Results.accepted_sources`): a copy on the
cluster not
synced, or a record from before a change.

:func:`audit` runs all four over a results directory; ``examples/run_campaign.py --collect`` prints
what it finds.
"""

from __future__ import annotations

import re
import statistics
from collections import defaultdict
from collections.abc import Collection, Iterable, Mapping, Sequence
from pathlib import Path
from typing import Final

from pihm.pipeline.records import JSONValue, Record, read_record, source_digest

__all__ = [
    "SLOW_STEP_FACTOR",
    "audit",
    "foreign_sources",
    "interruptions",
    "slow_refusals",
    "step_seconds",
    "unrecorded",
]

#: How many times the median step at its width a refusal's step may read before it is suspect.
SLOW_STEP_FACTOR: Final[float] = 5.0

_RUN_LINE: Final = re.compile(r"run (\S+) stage=\S+ n=(\[[^\]]*\]) \(([0-9a-f]+)\)")
_ARRAY_LOG: Final = re.compile(r"-(\d+)_(\d+)\.err$")
#: What Slurm writes to a task's ``.err`` when it ends one, and what a finding calls it.
_ENDINGS: Final[tuple[tuple[str, str], ...]] = (
    ("DUE TO TIME LIMIT", "cancelled at the time limit"),
    ("oom_kill", "killed out of memory"),
    ("Out Of Memory", "killed out of memory"),
)
#: What srun writes to a task's ``.err`` when the run's process dies of a signal.
_CRASH: Final = re.compile(r"srun: error: \S+: task \d+: ([A-Z][A-Za-z ]*[a-z])")


def step_seconds(metrics: Mapping[str, JSONValue]) -> float | None:
    """Return the step a record's estimate measured, in seconds, or ``None`` if it has none.

    A record that does not keep the step itself gives it as its predicted time over its steps: the
    degree plus the opening rotation, once per initial-state candidate (one for a readout).
    """
    step = metrics.get("predicted_seconds_per_step")
    if isinstance(step, (int, float)):
        return float(step) if step == step else None
    seconds, degree = metrics.get("predicted_seconds"), metrics.get("degree")
    if not isinstance(seconds, (int, float)) or not isinstance(degree, int):
        return None
    candidates = metrics.get("geometric_r")
    simulations = metrics.get("simulations")
    if not isinstance(simulations, int):
        simulations = len(candidates) if isinstance(candidates, list) else 1
    return float(seconds) / (simulations * (degree + 1))


def _circuit(record: Record) -> tuple[JSONValue, ...] | None:
    """Return what fixes a record's simulated step, or ``None`` if the record has no width.

    The step is the controlled walk of the regime's encoding at the problem's resolution, its
    Carleman order and its rescaling, and a readout's carries one more qubit, the selector; the
    filter and the initial state do not enter it. The order and the rescaling are read from the
    run's specification, since a refusal records fewer metrics than a finished run.
    """
    metrics = record.metrics
    qubits = metrics.get("n_qubits")
    if not isinstance(qubits, int):
        return None
    shape = metrics.get("qubits")
    return (
        metrics.get("regime"),
        metrics.get("problem"),
        metrics.get("variant"),
        tuple(shape) if isinstance(shape, list) else metrics.get("n"),
        record.spec.get("carleman"),
        record.spec.get("rescale"),
        qubits,
    )


def _name(metrics: Mapping[str, JSONValue]) -> str:
    """Return a record's problem, as its key names it, and resolution, for a finding."""
    qubits = metrics.get("qubits")
    n = "x".join(str(q) for q in qubits) if isinstance(qubits, list) else metrics.get("n")
    printed = "/faithful" if metrics.get("variant") == "faithful" else ""
    return f"{metrics.get('problem')}{printed} n={n}"


def slow_refusals(
    records: Sequence[tuple[Path, Record]], factor: float = SLOW_STEP_FACTOR
) -> list[str]:
    """Return a finding for each refusal on time whose step is ``factor`` times its circuit's.

    The median is over every other record of the same circuit with a measured step, ``ok`` or
    refused. A circuit with no other record gives no finding, and neither do two candidates
    refused together on the same slow node.
    """
    steps: dict[tuple[JSONValue, ...], list[tuple[Path, float]]] = defaultdict(list)
    for directory, record in records:
        circuit, step = _circuit(record), step_seconds(record.metrics)
        if circuit is not None and step is not None:
            steps[circuit].append((directory, step))
    findings = []
    for directory, record in records:
        reasons = record.metrics.get("infeasible_reasons")
        circuit, step = _circuit(record), step_seconds(record.metrics)
        if record.status != "infeasible" or not isinstance(reasons, str) or "time:" not in reasons:
            continue
        if circuit is None or step is None:
            continue
        others = [s for d, s in steps[circuit] if d != directory]
        if not others:
            continue
        typical = statistics.median(others)
        if step > factor * typical:
            findings.append(
                f"suspect refusal: {_name(record.metrics)} ({record.config_hash}) on "
                f"{record.provenance.hostname} read {step:.3g} s a step, {step / typical:.0f} "
                f"times the median {typical:.3g} s of its circuit's other "
                f"{'record' if len(others) == 1 else f'{len(others)} records'}; rerun it on "
                f"another node"
            )
    return findings


def _ending(text: str) -> str | None:
    """Return how Slurm ended an attempt, from its ``.err``, or ``None`` if nothing ended it."""
    named = next((name for mark, name in _ENDINGS if mark in text), None)
    if named is not None:
        return named
    crash = _CRASH.search(text)
    return f"crashed ({crash.group(1).lower()})" if crash else None


def _latest_logs(logs: Path) -> dict[str, Path]:
    """Return each task's latest ``.err`` (its highest job ID), by task number or file name."""
    latest: dict[str, tuple[int, Path]] = {}
    for err in logs.glob("*.err"):
        match = _ARRAY_LOG.search(err.name)
        task, job = (match.group(2), int(match.group(1))) if match else (err.name, 0)
        if task not in latest or job > latest[task][0]:
            latest[task] = (job, err)
    return {task: err for task, (_, err) in latest.items()}


def interruptions(logs: Path, records: Iterable[tuple[Path, Record]]) -> list[str]:
    """Return a finding for each task whose latest attempt Slurm ended: time, memory or a crash.

    ``logs`` holds the Slurm arrays' ``<job>_<task>.out`` and ``.err`` pairs; the ``.out`` names
    the run (:func:`pihm.pipeline.stages.run`'s first line). A task's latest attempt is its
    highest job ID; an ended run that has an ``ok`` record gives no finding.
    """
    status = {record.config_hash: record.status for _, record in records}
    findings = []
    for err in sorted(_latest_logs(logs).values()):
        ending = _ending(err.read_text(errors="replace"))
        if ending is None:
            continue
        out = err.with_suffix(".out")
        match = _RUN_LINE.search(out.read_text(errors="replace")) if out.exists() else None
        if match is None:
            findings.append(f"{ending}: {err.name}, whose run is not named")
            continue
        key, n, config = match.groups()
        held = status.get(config)
        if held == "ok":
            continue
        findings.append(
            f"{ending}: {key} n={n} ({config}), {err.name}; its record is {held or 'missing'}"
        )
    return findings


def unrecorded(
    expected: Sequence[tuple[str, str]],
    records: Iterable[tuple[Path, Record]],
    logs: Path | None,
) -> list[str]:
    """Return a finding for each run with no record that no ending of Slurm's explains.

    ``expected`` holds the campaign's runs in task order, each its configuration hash and name.
    In an array, a task with no log never started; one whose log Slurm did not end is named with
    its log's last line. A task that :func:`interruptions` names is left to it.
    """
    held = {record.config_hash for _, record in records}
    latest = _latest_logs(logs) if logs is not None else {}
    array = any(task.isdigit() for task in latest)
    findings = []
    for task, (config, name) in enumerate(expected):
        if config in held:
            continue
        err = latest.get(str(task))
        if err is None:
            reason = "never started: no record and no log" if array else "no record"
            findings.append(f"{reason}: task {task}, {name} ({config})")
            continue
        text = err.read_text(errors="replace")
        if _ending(text) is None:
            last = next((line for line in reversed(text.splitlines()) if line.strip()), "")
            findings.append(
                f"no record: task {task}, {name} ({config}); {err.name} ends "
                f"{last.strip()[:100] or 'empty'}"
            )
    return findings


def foreign_sources(records: Iterable[tuple[Path, Record]], accepted: Collection[str]) -> list[str]:
    """Return a finding for each source outside ``accepted`` that made some of the records."""
    made: dict[str | None, list[Path]] = defaultdict(list)
    for directory, record in records:
        if record.provenance.source_digest not in accepted:
            made[record.provenance.source_digest].append(directory)
    return [
        f"{len(directories)} records made by "
        f"{'unrecorded source' if source is None else f'source {source}'}, not this tree's or "
        f"an accepted one ({', '.join(sorted(accepted))}), the first {directories[0]}"
        for source, directories in sorted(made.items(), key=lambda item: str(item[0]))
    ]


def audit(
    root: Path,
    expected: Sequence[tuple[str, str]] = (),
    accepted: Collection[str] | None = None,
) -> list[str]:
    """Return every finding of the module's checks under ``root``.

    ``expected`` is the campaign's runs for :func:`unrecorded` (none skips it), and ``accepted``
    the source digests whose records stand, by default the tree's alone.
    """
    records = [(meta.parent, read_record(meta.parent)) for meta in sorted(root.rglob("meta.json"))]
    logs = root / "logs"
    ended = interruptions(logs, records) if logs.is_dir() else []
    missing = unrecorded(expected, records, logs if logs.is_dir() else None)
    sources = accepted if accepted is not None else {source_digest()}
    return slow_refusals(records) + ended + missing + foreign_sources(records, sources)
