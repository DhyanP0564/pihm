"""Checks of the resource estimates made before a circuit simulation."""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys
import textwrap
import time
from types import SimpleNamespace

import pytest
from qiskit import QuantumCircuit

from pihm.circuits.assemble import encode_standard_linear
from pihm.circuits.primitives import BlockEncoding
from pihm.problems.ladder import problem
from pihm.qsp import feasibility
from pihm.qsp.feasibility import (
    BASELINE_BYTES,
    DEFAULT_HOURS,
    MACHINE_FRACTION,
    STATEVECTOR_COPIES,
    Budget,
    InfeasibleError,
    estimate_gqsp,
    require,
)

_GIB = 2.0**30
_ROOMY = Budget(1e15, 1e9, "override")


def _r1(n: int = 2) -> BlockEncoding:
    stacked, _reflection = encode_standard_linear(problem("R1"), n)
    return stacked


def test_the_memory_estimate_follows_the_statevector() -> None:
    stacked = _r1()
    estimate = estimate_gqsp(stacked, 10, simulations=2)
    assert estimate.n_qubits == stacked.n_qubits + 1
    assert estimate.statevector_bytes == 16 * 2**estimate.n_qubits
    assert estimate.peak_bytes >= BASELINE_BYTES + STATEVECTOR_COPIES * estimate.statevector_bytes
    assert estimate.seconds == pytest.approx(2 * 11 * estimate.seconds_per_step)


def test_the_selector_adds_one_qubit() -> None:
    stacked = _r1()
    plain = estimate_gqsp(stacked, 1)
    assert estimate_gqsp(stacked, 1, selector=True).n_qubits == plain.n_qubits + 1


def test_a_run_over_its_memory_budget_is_refused_without_timing_a_step() -> None:
    tight = Budget(1e6, 3600.0, "override")
    estimate = estimate_gqsp(_r1(), 10, budget=tight)
    assert math.isnan(estimate.seconds_per_step)
    with pytest.raises(InfeasibleError, match="memory") as refused:
        require(estimate, tight)
    assert refused.value.estimate is estimate
    assert refused.value.budget is tight


def test_a_run_over_its_time_budget_is_refused() -> None:
    estimate = estimate_gqsp(_r1(), 10**9)
    with pytest.raises(InfeasibleError, match="time"):
        require(estimate, Budget(1e15, 3600.0, "override"))


def test_a_run_within_its_budget_passes() -> None:
    require(estimate_gqsp(_r1(), 10), _ROOMY)


@pytest.mark.parametrize(
    ("environ", "memory_gib", "hours", "source"),
    [
        ({"PIHM_MEMORY_BUDGET_GB": "2", "PIHM_TIME_BUDGET_HOURS": "0.5"}, 2.0, 0.5, "override"),
        ({"SLURM_MEM_PER_NODE": "1024", "PIHM_TIME_BUDGET_HOURS": "1"}, 1.0, 1.0, "slurm+override"),
        (
            {"SLURM_MEM_PER_CPU": "512", "SLURM_CPUS_PER_TASK": "4"},
            2.0,
            DEFAULT_HOURS,
            "slurm+machine",
        ),
    ],
)
def test_the_budget_comes_from_the_environment(
    environ: dict[str, str], memory_gib: float, hours: float, source: str
) -> None:
    budget = Budget.from_environment(environ)
    assert budget.memory_bytes == pytest.approx(memory_gib * _GIB)
    assert budget.seconds == pytest.approx(hours * 3600.0)
    assert budget.source == source


def test_the_budget_takes_what_is_left_of_a_slurm_job() -> None:
    environ = {"SLURM_MEM_PER_NODE": "1024", "SLURM_JOB_END_TIME": str(time.time() + 7200.0)}
    budget = Budget.from_environment(environ)
    assert budget.seconds == pytest.approx(7200.0, abs=60.0)
    assert budget.source == "slurm"


def test_without_an_allocation_the_budget_is_the_machine() -> None:
    budget = Budget.from_environment({})
    physical = os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
    assert budget.memory_bytes == pytest.approx(MACHINE_FRACTION * physical)
    assert budget.seconds == pytest.approx(DEFAULT_HOURS * 3600.0)
    assert budget.source == "machine"


def test_the_memory_model_matches_a_fresh_process() -> None:
    """The model's peak is within a few per cent above what 22-qubit steps actually reach.

    A fresh interpreter, so that the peak resident set is this measurement's alone.
    """
    script = textwrap.dedent(
        """
        import json, platform, resource
        import numpy as np
        from qiskit import QuantumCircuit
        from qiskit.circuit.library import UnitaryGate
        from qiskit.quantum_info import random_unitary
        from pihm.circuits.primitives import compile_circuit, evolve

        def peak():
            r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            return r if platform.system() == "Darwin" else r * 1024

        q = 22
        qc = QuantumCircuit(q)
        for j in range(0, q - 5, 5):
            qc.append(UnitaryGate(random_unitary(32, seed=j).data), range(j, j + 5))
        qc.h(range(q))
        step = compile_circuit(qc)
        state = np.zeros(2**q, complex)
        state[0] = 1.0
        for _ in range(3):
            state = evolve(step, state)
        print(json.dumps({"q": q, "peak": peak()}))
        """
    )
    completed = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, check=True
    )
    measured = json.loads(completed.stdout.strip().splitlines()[-1])
    predicted = BASELINE_BYTES + STATEVECTOR_COPIES * 16.0 * 2.0 ** measured["q"]
    assert 0.85 * predicted <= measured["peak"] <= 1.05 * predicted


def test_the_step_is_timed_warm_and_by_its_median(monkeypatch: pytest.MonkeyPatch) -> None:
    # One untimed warm-up, then TIMED_STEPS timed simulations; a slow outlier (a neighbour's
    # start-up on a shared node) does not move the median.
    calls: list[int] = []
    monkeypatch.setattr(feasibility, "evolve", lambda step, state: calls.append(1))
    ticks = iter(value for duration in (100.0, 1.0, 1.0, 2.0, 1.0) for value in (0.0, duration))
    monkeypatch.setattr(
        "pihm.qsp.feasibility.time", SimpleNamespace(perf_counter=lambda: next(ticks))
    )
    assert feasibility._time_one_step(QuantumCircuit(2)) == 1.0
    assert len(calls) == 1 + feasibility.TIMED_STEPS


def test_every_estimate_is_logged_against_its_budget(caplog: pytest.LogCaptureFixture) -> None:
    # A run that passes its estimate still logs it, so a slow node shows against what the run
    # expected.
    with caplog.at_level("INFO", logger="pihm.qsp.feasibility"):
        require(estimate_gqsp(_r1(), 10, simulations=2), _ROOMY)
    (message,) = caplog.messages
    assert message.startswith("estimate: ")
    assert "2 x 11 steps at" in message
    assert "(override)" in message
