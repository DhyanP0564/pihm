r"""Pipeline stages: a run specification in, a record out.

:func:`run` takes a :class:`~pihm.pipeline.spec.RunSpec`, runs its stage, and returns (and writes)
the record. This module holds the classical and Pillar 1 stages; Pillar 2's are in
:mod:`pihm.pipeline.pillar2`.

The ``"build"`` stage assembles the
residual, solves it classically and reads the ground state out as the paper does
(:func:`pihm.classical.solve`), recording what every run records:

* **classical:** the field error against the reference (L∞ and relative L²), the spectrum
  (:math:`\lambda_0`, :math:`\lambda_1`, :math:`\Delta`, :math:`\|H\|`, the relative gap), the
  kernel (its dimension, its gap ratio, whether it is trusted, and a full count) and
  :math:`\eta_e` (read out, ideal, its spread, and the printed value for a paper panel);
* **readout:** the field-block weight (1 in the standard form of a single field, whose state is
  one block; a system's weight on its own fields' blocks) and the readout amplification;
* **dimensions and time:** the state dimension, its blocks (with the padding and the field
  register's qubits), the rows and non-zeros of :math:`R`, and wall times per step;
* **the descriptor's block rescaling:** whether it is applied and its :math:`\lambda` per axis;
* **the nonlinearity:** its treatment and a Carleman lift's order (0 without one);
* **a first-order problem's layout** in the descriptor: its mass form, or the chain ablation.

A regime runs only the stages built for it (:data:`REGIME_STAGES`); any other pairing is a failed
run whose error says so, never a run of another regime's stage.

The ``"encode"`` stage builds the
regime's circuit blocks (:func:`pihm.circuits.assemble.encode_standard_linear`, the structured
standard form; :func:`~pihm.circuits.assemble.encode_descriptor_linear`, the descriptor's term-IR
LCU, block-rescaled as its residual is), compares their stacked reflection against the
classically assembled :math:`H` (:attr:`~pihm.pipeline.spec.RunSpec.verify` controls how:
``"exact"``, ``"probe"``, or ``"none"`` for a construction-only, resource-only run) and records
:math:`\alpha_R`, the register split and (:attr:`~pihm.pipeline.spec.RunSpec.resources`
permitting) the transpiled gate counts; a descriptor record also keeps its rescaling and
:math:`\lambda`.
"""

from __future__ import annotations

import hashlib
import io
import logging
import resource
import sys
import time
import traceback
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Final, get_args

import numpy as np
import scipy.sparse as sp
from numpy.typing import NDArray

from pihm.assemble import Regime, assemble
from pihm.circuits.assemble import encode_linear
from pihm.circuits.primitives import BlockEncoding
from pihm.classical import ClassicalSolution, lift_errors, solve
from pihm.pipeline.lift import run_lift
from pihm.pipeline.pillar2 import (
    infeasible_metrics,
    run_filters,
    run_protocol,
    run_readout,
    run_stateprep,
)
from pihm.pipeline.records import (
    JSONValue,
    Provenance,
    Record,
    Status,
    write_record,
)
from pihm.pipeline.spec import RunSpec, Stage, record_directory
from pihm.problems import ladder
from pihm.problems.ladder import PANELS
from pihm.qsp.feasibility import InfeasibleError
from pihm.resources import ResourceCounts, count_resources
from pihm.verify import VerificationResult, verify_exact, verify_probe

__all__ = ["REGIME_STAGES", "build_metrics", "encode_metrics", "run"]

logger = logging.getLogger(__name__)

#: The stages built for each regime.
REGIME_STAGES: Final[Mapping[Regime, frozenset[Stage]]] = {
    "standard": frozenset(get_args(Stage)),
    "descriptor": frozenset({"build", "encode", "stateprep", "filters"}),
}


def build_metrics(spec: RunSpec, solution: ClassicalSolution) -> dict[str, JSONValue]:
    """Return the scalar metrics of a ``"build"`` run (strict JSON, flat, for ``summary.csv``)."""
    residual = solution.residual
    state = solution.state
    problem = spec.problem
    panel = PANELS.get(problem.name)
    R = residual.matrix()
    blocks = len(residual.derivatives)
    size = residual.dimension // blocks
    fields = [solution.vector[i * size : (i + 1) * size] for i in residual.field_blocks]
    lift, axis = (
        lift_errors(residual, solution.vector)
        if residual.nonlinearity == "carleman"
        else (float("nan"), float("nan"))
    )
    return {
        "problem": problem.name,
        "variant": problem.variant,
        "rung": ladder.rung(problem.name),
        "figure": panel.figure if panel else "",
        "label": spec.label,
        "regime": spec.regime,
        "n": int(max(spec.n)),
        "qubits": list(spec.n),
        "dimension": residual.dimension,
        "blocks": blocks,
        "padding_blocks": sum(derivative is None for derivative in residual.derivatives),
        "field_qubits": residual.field_qubits,
        "rescale": residual.rescale,
        "scales": list(residual.scales),
        "rows": int(R.shape[0]),
        "nonzeros": int(R.nnz),
        "lifted": any(residual.lifted),
        "nonlinearity": residual.nonlinearity,
        "carleman": spec.carleman or 0,
        "layout": spec.layout,
        "mult": residual.mult,
        "truncation": residual.truncation,
        "sigma_min": float(state.singular_values[0]),
        "lambda0": state.lambda0,
        "lambda1": state.lambda1,
        "gap": state.gap,
        "hamiltonian_norm": state.hamiltonian_norm,
        "relative_gap": state.relative_gap,
        "kernel_dimension": state.kernel_dimension,
        "kernel_count": state.kernel_count(),
        "gap_ratio": state.gap_ratio,
        "trusted": state.trusted,
        "backward_error": state.backward_error,
        "vector_error_bound": state.vector_error_bound,
        "fidelity_bound": state.fidelity_bound,
        "driver": state.driver,
        "selection": solution.selection,
        "regular_point": list(problem.regular.at),
        "regular_value": residual.regular_value,
        "scale": solution.scale,
        "eta": solution.eta,
        "eta_ideal": solution.eta_ideal,
        "eta_min": solution.eta_range[0],
        "eta_max": solution.eta_range[1],
        "eta_printed": panel.eta if panel else float("nan"),
        "printed_n": panel.n if panel else 0,
        "infidelity": solution.infidelity,
        "target_residual": solution.target_residual,
        "field_err_linf": solution.field_error_linf,
        "field_err_l2": solution.field_error_l2,
        "lift_err_linf": lift,
        "axis_err_linf": axis,
        "field_weight": 1.0 if blocks == 1 else float(sum(np.vdot(f, f).real for f in fields)),
        "readout_amplification": solution.readout_amplification,
    }


def encode_metrics(
    spec: RunSpec,
    stacked: BlockEncoding,
    reflection: BlockEncoding,
    verification: VerificationResult | None,
    resources: ResourceCounts | None,
) -> dict[str, JSONValue]:
    r"""Return the scalar metrics of an ``"encode"`` run (strict JSON, flat, for ``summary.csv``).

    Parameters
    ----------
    spec
        The run.
    stacked
        :func:`~pihm.circuits.compose.stack_rows`'s own result, for :math:`\alpha_R`.
    reflection
        :func:`~pihm.circuits.compose.stacked_reflection`'s result, for the register split.
    verification
        The result of :func:`pihm.verify.verify_exact` or :func:`~pihm.verify.verify_probe`, or
        ``None`` when ``spec.verify == "none"``.
    resources
        :func:`pihm.resources.count_resources`'s result, or ``None`` when ``spec.resources`` is
        false.
    """
    problem = spec.problem
    return {
        "problem": problem.name,
        "variant": problem.variant,
        "rung": ladder.rung(problem.name),
        "label": spec.label,
        "regime": spec.regime,
        "n": int(max(spec.n)),
        "qubits": list(spec.n),
        "carleman": spec.carleman or 0,
        "alpha_R": stacked.alpha,
        "n_sys_qubits": reflection.n_sys_qubits,
        "n_anc_qubits": reflection.n_anc_qubits,
        "n_qubits_total": reflection.n_qubits,
        "verify_mode": spec.verify,
        "verify_passed": verification.passed if verification else None,
        "verify_max_error": verification.max_error if verification else None,
        "verify_tolerance": verification.tolerance if verification else None,
        "resources_measured": resources is not None,
        "cx_count": resources.cx_count if resources else None,
        "one_qubit_count": resources.one_qubit_count if resources else None,
        "cited_count": resources.cited_count if resources else None,
        "depth": resources.depth if resources else None,
    }


def _arrays(solution: ClassicalSolution) -> dict[str, NDArray[Any]]:
    """Return the array results of a ``"build"`` run: spectrum, state, and fields on the grid."""
    return {
        "singular_values": solution.state.singular_values,
        "state": solution.vector,
        "coefficients": solution.coefficients,
        **{f"grid_{index}": axis for index, axis in enumerate(solution.axes)},
        "field": solution.field,
        "reference": solution.reference,
    }


def _run_encode(
    spec: RunSpec, timings: dict[str, float]
) -> tuple[Mapping[str, JSONValue], dict[str, NDArray[Any]]]:
    """Run the ``"encode"`` stage: build circuit blocks, verify, count resources."""
    started = time.perf_counter()
    stacked, reflection = encode_linear(
        spec.problem, spec.n, regime=spec.regime, rescale=spec.rescale, carleman=spec.carleman
    )
    timings["encode"] = time.perf_counter() - started

    started = time.perf_counter()
    residual = assemble(
        spec.problem, spec.n, regime=spec.regime, rescale=spec.rescale, carleman=spec.carleman
    )
    timings["classical_reference"] = time.perf_counter() - started

    started = time.perf_counter()
    verification: VerificationResult | None = None
    if spec.verify != "none":
        H = residual.hamiltonian()
        target = sp.csr_array(2.0 * H / stacked.alpha**2 - sp.eye_array(H.shape[0]))
        if spec.verify == "exact":
            verification = verify_exact(reflection, target)
        else:
            digest = hashlib.sha256(spec.config_hash.encode()).digest()
            rng = np.random.default_rng(int.from_bytes(digest[:8], "little"))
            verification = verify_probe(reflection, target, rng)
    timings["verify"] = time.perf_counter() - started

    started = time.perf_counter()
    resources = count_resources(reflection) if spec.resources else None
    timings["resources"] = time.perf_counter() - started

    metrics = encode_metrics(spec, stacked, reflection, verification, resources)
    if spec.regime == "descriptor":
        metrics |= {"rescale": residual.rescale, "scales": list(residual.scales)}
    return metrics, {}


def _peak_memory_gib() -> float:
    """Return this process's peak resident memory so far, in GiB (bytes on macOS, KiB on Linux)."""
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak / (2**30 if sys.platform == "darwin" else 2**20)


class _Capture(logging.Handler):
    """Collect log messages emitted during a run, for the record's ``log.txt``."""

    def __init__(self) -> None:
        super().__init__(logging.INFO)
        self.stream = io.StringIO()
        self.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))

    def emit(self, record: logging.LogRecord) -> None:
        self.stream.write(self.format(record) + "\n")


def run(spec: RunSpec, root: Path | None = None) -> Record:
    """Run one specification and return its record, writing it under ``root`` if given.

    A run that raises is not lost: it returns (and writes) a ``"failed"`` record with the
    traceback, so that a campaign continues and the failure can be inspected and rerun.

    Parameters
    ----------
    spec
        The run.
    root
        The results directory; the record goes to
        :func:`~pihm.pipeline.spec.record_directory`. ``None`` returns the record without writing
        it.

    Returns
    -------
    Record
        The run's record.
    """
    capture = _Capture()
    package_logger = logging.getLogger("pihm")
    level = package_logger.level
    package_logger.addHandler(capture)
    package_logger.setLevel(min(level or logging.INFO, logging.INFO))
    timings: dict[str, float] = {}
    started = time.perf_counter()
    try:
        logger.info(
            "run %s stage=%s n=%s (%s)",
            spec.problem.key,
            spec.stage,
            list(spec.n),
            spec.config_hash,
        )
        if spec.stage not in REGIME_STAGES[spec.regime]:
            raise NotImplementedError(
                f"the {spec.stage!r} stage is not built for the {spec.regime} regime"
            )
        metrics: Mapping[str, JSONValue]
        if spec.stage == "encode":
            metrics, arrays = _run_encode(spec, timings)
        elif spec.stage == "stateprep":
            metrics, arrays = run_stateprep(spec, timings)
        elif spec.stage == "readout":
            metrics, arrays = run_readout(spec, timings)
        elif spec.stage == "protocol":
            metrics, arrays = run_protocol(spec, timings)
        elif spec.stage == "filters":
            metrics, arrays = run_filters(spec, timings)
        elif spec.stage == "lift":
            metrics, arrays = run_lift(spec, timings)
        else:
            residual = assemble(
                spec.problem,
                spec.n,
                regime=spec.regime,
                mult=spec.mult,
                truncation=spec.truncation,
                rescale=spec.rescale,
                carleman=spec.carleman,
                layout=spec.layout,
            )
            timings["assemble"] = time.perf_counter() - started
            solution = solve(residual)
            timings.update(solution.timings)
            metrics = build_metrics(spec, solution)
            arrays = _arrays(solution)
        status: Status = "ok"
        error = None
    except InfeasibleError as refusal:
        metrics, arrays, status, error = infeasible_metrics(spec, refusal), {}, "infeasible", None
        logger.info("run %s infeasible: %s", spec.config_hash, refusal)
    except Exception:
        metrics, arrays, status = {}, {}, "failed"
        error = traceback.format_exc()
        logger.exception("run %s failed", spec.config_hash)
    finally:
        package_logger.removeHandler(capture)
        package_logger.setLevel(level)
    timings["total"] = time.perf_counter() - started
    roots = [c.at for c in spec.problem.constraints if c.root is not None]
    if roots:
        # The problem's identity holds each root's recipe (pihm.problems.spec.Root); the record
        # keeps the positions this machine resolved them to.
        metrics = {**metrics, "resolved_roots": roots}
    record = Record(
        spec=spec.to_dict(),
        status=status,
        metrics=metrics,
        provenance=Provenance.capture(),
        timings=timings,
        peak_memory_gib=_peak_memory_gib(),
        arrays=arrays,
        log=capture.stream.getvalue(),
        error=error,
    )
    if root is not None:
        write_record(record, record_directory(root, spec))
    return record
