r"""Resource estimates made before a circuit simulation runs.

GQSP is simulated one controlled walk at a time (:func:`pihm.circuits.gqsp.evolve_gqsp`), so what
a run needs is known before it starts:

* **Memory** follows the statevector: :math:`16\cdot2^q` bytes for :math:`q` qubits, of which a
  simulation step holds :data:`STATEVECTOR_COPIES` at its peak, plus the step's own dense gate
  matrices (qiskit keeps one per instruction, :data:`GATE_COPIES` of them alive at once) and a
  fixed :data:`BASELINE_BYTES` for the interpreter and its libraries. The constants are measured
  in a fresh process, and pinned by ``tests/unit/test_feasibility.py``.
* **Time** is the degree times one step. Once the memory is known to fit, the real circuit's step
  is simulated once untimed, to warm its caches and libraries, then :data:`TIMED_STEPS` times; the
  estimate is the median of those times, times the steps of one simulation (the degree, plus its
  opening rotation), times the simulations the run makes. One cold step is not enough: tasks that
  start together on a shared node time their step while their neighbours set up, and on a cluster a
  single step read up to 26 times its steady-state cost.

:func:`estimate_gqsp` returns a :class:`CostEstimate`. A :class:`Budget` is what a run may spend,
by default the environment's (:meth:`Budget.from_environment`). :func:`require` logs the estimate
against the budget, whatever the verdict, so a run's log shows what it expected before it starts,
and raises :class:`InfeasibleError`, carrying the estimate, when a run would exceed its budget; the
pipeline then records the run as ``infeasible`` with those numbers instead of
attempting it.
"""

from __future__ import annotations

import logging
import math
import os
import time
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Final

import numpy as np
from qiskit import QuantumCircuit

from pihm.circuits.compose import controlled_walk
from pihm.circuits.primitives import BlockEncoding, compile_circuit, evolve

__all__ = [
    "BASELINE_BYTES",
    "DEFAULT_HOURS",
    "GATE_COPIES",
    "MACHINE_FRACTION",
    "STATEVECTOR_COPIES",
    "TIMED_STEPS",
    "Budget",
    "CostEstimate",
    "InfeasibleError",
    "describe",
    "estimate_gqsp",
    "require",
]

#: Timed simulations of one step, after one untimed warm-up; the estimate takes their median.
TIMED_STEPS: Final[int] = 5

#: Statevector-sized buffers alive at a simulation step's peak (8.0 measured from 20 to 24 qubits).
STATEVECTOR_COPIES: Final[int] = 8

#: Copies of each of a step's dense gate matrices alive at once (the compiled step, its per-step
#: copy, the simulated circuit and the simulator's own).
GATE_COPIES: Final[int] = 4

#: Resident memory before any statevector exists, for the interpreter, numpy, qiskit and Aer
#: (133 MiB measured) and the simulator's fixed working set at small sizes.
BASELINE_BYTES: Final[float] = 160 * 2.0**20

#: The share of the machine's physical memory a run may use when no allocation says otherwise.
MACHINE_FRACTION: Final[float] = 0.8

#: The time a run may take when neither Slurm nor an override says otherwise.
DEFAULT_HOURS: Final[float] = 24.0

_GIB: Final[float] = 2.0**30

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Budget:
    """What a run may spend.

    Attributes
    ----------
    memory_bytes
        Resident memory.
    seconds
        Wall time.
    source
        Where the budget came from: ``"override"`` (the ``PIHM_*`` variables), ``"slurm"`` (the
        job's allocation) or ``"machine"`` (this machine's memory, :data:`DEFAULT_HOURS`).
    """

    memory_bytes: float
    seconds: float
    source: str

    @classmethod
    def from_environment(cls, environ: Mapping[str, str] | None = None) -> Budget:
        r"""Return the budget the environment gives a run.

        Memory: ``PIHM_MEMORY_BUDGET_GB`` (gibibytes, like Slurm's ``G``) if set; else the Slurm
        job's allocation (``SLURM_MEM_PER_NODE``, or ``SLURM_MEM_PER_CPU`` times
        ``SLURM_CPUS_PER_TASK``, both in mebibytes); else :data:`MACHINE_FRACTION` of physical
        memory. Time: ``PIHM_TIME_BUDGET_HOURS`` if set; else what is left of the Slurm job
        (``SLURM_JOB_END_TIME``); else :data:`DEFAULT_HOURS`.

        Parameters
        ----------
        environ
            The environment to read; ``None`` reads :data:`os.environ`.
        """
        env = os.environ if environ is None else environ
        sources = []
        if "PIHM_MEMORY_BUDGET_GB" in env:
            memory = float(env["PIHM_MEMORY_BUDGET_GB"]) * _GIB
            sources.append("override")
        elif "SLURM_MEM_PER_NODE" in env:
            memory = float(env["SLURM_MEM_PER_NODE"]) * 2.0**20
            sources.append("slurm")
        elif "SLURM_MEM_PER_CPU" in env:
            cpus = float(env.get("SLURM_CPUS_PER_TASK", "1"))
            memory = float(env["SLURM_MEM_PER_CPU"]) * 2.0**20 * cpus
            sources.append("slurm")
        else:
            pages = os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
            memory = MACHINE_FRACTION * float(pages)
            sources.append("machine")
        if "PIHM_TIME_BUDGET_HOURS" in env:
            seconds = float(env["PIHM_TIME_BUDGET_HOURS"]) * 3600.0
            sources.append("override")
        elif "SLURM_JOB_END_TIME" in env:
            seconds = max(0.0, float(env["SLURM_JOB_END_TIME"]) - time.time())
            sources.append("slurm")
        else:
            seconds = DEFAULT_HOURS * 3600.0
            sources.append("machine")
        source = sources[0] if sources[0] == sources[1] else "+".join(sources)
        return cls(memory, seconds, source)


@dataclass(frozen=True)
class CostEstimate:
    r"""A GQSP simulation's estimated memory and time.

    Attributes
    ----------
    n_qubits
        Qubits simulated: the walk's own, the signal, and the selector if there is one.
    degree
        Controlled walks per simulation.
    simulations
        Full simulations the run makes (one per initial-state candidate, for T1).
    statevector_bytes
        :math:`16\cdot2^{\text{n\_qubits}}`.
    gate_bytes
        The dense gate matrices of one step's circuit, one per instruction.
    peak_bytes
        The estimated peak resident memory.
    seconds_per_step
        The median measured step, or ``nan`` when it was not measured (the memory did not fit).
    seconds
        The estimated wall time of every simulation, or ``nan`` when not measured.
    """

    n_qubits: int
    degree: int
    simulations: int
    statevector_bytes: float
    gate_bytes: float
    peak_bytes: float
    seconds_per_step: float
    seconds: float


class InfeasibleError(Exception):
    """A run whose estimated cost exceeds its budget; it carries both, and why."""

    def __init__(self, estimate: CostEstimate, budget: Budget, reasons: tuple[str, ...]) -> None:
        super().__init__("; ".join(reasons))
        self.estimate = estimate
        self.budget = budget
        self.reasons = reasons


def _dense_bytes(circuit: QuantumCircuit) -> float:
    """Return the bytes of the dense matrices ``circuit`` holds, counted per instruction."""
    return float(
        sum(
            16 * 4**inst.operation.num_qubits
            for inst in circuit.data
            if inst.operation.name == "unitary"
        )
    )


def _time_one_step(step: QuantumCircuit) -> float:
    """Return the median wall time of :data:`TIMED_STEPS` simulations of ``step``, warmed up."""
    dim = 1 << step.num_qubits
    state = np.full(dim, 1.0 / math.sqrt(dim), dtype=np.complex128)
    evolve(step, state)
    times = []
    for _ in range(TIMED_STEPS):
        started = time.perf_counter()
        evolve(step, state)
        times.append(time.perf_counter() - started)
    return float(np.median(times))


def estimate_gqsp(
    stacked: BlockEncoding,
    degree: int,
    *,
    simulations: int = 1,
    selector: bool = False,
    budget: Budget | None = None,
) -> CostEstimate:
    """Estimate ``simulations`` GQSP simulations of ``degree`` controlled walks each.

    The memory estimate needs only the circuit's structure. The time estimate simulates one step
    of the real circuit, so it is taken only when the memory fits ``budget`` (always, when no
    budget is given).

    Parameters
    ----------
    stacked
        :math:`U_R` (:func:`pihm.circuits.compose.stack_rows`).
    degree
        Controlled walks per simulation.
    simulations
        Full simulations the run makes.
    selector
        Whether the simulation carries the readout's selector qubit
        (:func:`pihm.circuits.gqsp.evolve_gqsp`'s ``selector``).
    budget
        If given, and the memory estimate exceeds it, the step is not timed.

    Returns
    -------
    CostEstimate
    """
    if selector:
        walk_step = controlled_walk(stacked, n_controls=2, ctrl_state=0b10)
    else:
        walk_step = controlled_walk(stacked)
    step = compile_circuit(walk_step)
    n_qubits = step.num_qubits
    statevector = 16.0 * 2.0**n_qubits
    gate_bytes = _dense_bytes(step)
    peak = BASELINE_BYTES + STATEVECTOR_COPIES * statevector + GATE_COPIES * gate_bytes
    if budget is not None and peak > budget.memory_bytes:
        per_step = total = float("nan")
    else:
        per_step = _time_one_step(step)
        total = simulations * (degree + 1) * per_step
    return CostEstimate(
        n_qubits, degree, simulations, statevector, gate_bytes, peak, per_step, total
    )


def describe(estimate: CostEstimate, budget: Budget) -> str:
    """Return one line setting ``estimate`` against ``budget``, as :func:`require` logs it."""
    if math.isnan(estimate.seconds):
        time_part = "time not measured"
    else:
        time_part = (
            f"{estimate.simulations} x {estimate.degree + 1} steps at "
            f"{estimate.seconds_per_step:.3g} s a step, {estimate.seconds / 3600:.3g} h of "
            f"{budget.seconds / 3600:.3g} h"
        )
    return (
        f"estimate: {estimate.n_qubits} qubits, {time_part}; peak "
        f"{estimate.peak_bytes / _GIB:.3g} GiB of {budget.memory_bytes / _GIB:.3g} GiB "
        f"({budget.source})"
    )


def require(estimate: CostEstimate, budget: Budget) -> None:
    """Log ``estimate`` against ``budget``, and raise :class:`InfeasibleError` if it exceeds it.

    An estimate whose time was not measured exceeds the budget on memory already, so it is only
    ever refused, never passed.
    """
    logger.info("%s", describe(estimate, budget))
    reasons = []
    if estimate.peak_bytes > budget.memory_bytes:
        reasons.append(
            f"memory: {estimate.peak_bytes / _GIB:.3g} GiB estimated, "
            f"{budget.memory_bytes / _GIB:.3g} GiB budget ({budget.source})"
        )
    if not estimate.seconds <= budget.seconds:
        if math.isnan(estimate.seconds):
            reasons.append("time: not measured, the memory does not fit")
        else:
            reasons.append(
                f"time: {estimate.seconds / 3600:.3g} h estimated, "
                f"{budget.seconds / 3600:.3g} h budget ({budget.source})"
            )
    if reasons:
        raise InfeasibleError(estimate, budget, tuple(reasons))
