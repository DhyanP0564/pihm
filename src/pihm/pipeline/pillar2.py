r"""Pillar 2's pipeline stages: state preparation and the readout.

The ``"stateprep"`` stage runs Pillar 2 at :attr:`~pihm.pipeline.spec.RunSpec.tier`:
T2 (:func:`pihm.qsp.prepare.t2`, exact polynomial emulation) or T1 (:func:`pihm.qsp.prepare.t1`,
the GQSP circuit simulated): the QITE filter at the circuit's own :math:`\alpha_R`, its GQSP
phases, the prepared state and its infidelity against the classical reference, and for T1 its
agreement with T2. :attr:`~pihm.pipeline.spec.RunSpec.kind` and
:attr:`~pihm.pipeline.spec.RunSpec.epsilon` control the initial state and the filter's own target
accuracy. Its arrays are the prepared state's decoded field beside the ground state's and the
reference, which the solution viewer draws (:mod:`pihm.pipeline.solutions`).

The ``"readout"`` stage reads the prepared field out as the paper does
(:func:`pihm.qsp.readout.readout`): T1 simulates the interferometric circuits,
and with :attr:`~pihm.pipeline.spec.RunSpec.shots` also samples their probabilities; T2 takes the
exact overlap. It records :math:`f_Q` at the probe points against the analytic solution and
against T2, and :math:`\eta_e`.

The ``"filters"`` stage compares imaginary time with
the minimax edge filter at like error on the problem's own spectrum and initial state
(:mod:`pihm.qsp.filter_comparison`): at each infidelity of
:data:`~pihm.qsp.filter_comparison.INFIDELITIES`, each filter's least certified and measured
degree, its success probability and the preparations amplitude amplification needs; and at
:attr:`~pihm.pipeline.spec.RunSpec.epsilon`, both measured filters applied through the T2 engine
and their GQSP phases found.

A T1 run of either stage first estimates its memory and time (:mod:`pihm.qsp.feasibility`)
against the environment's budget. One that would exceed it raises
:class:`~pihm.qsp.feasibility.InfeasibleError`, which :func:`pihm.pipeline.stages.run` records as
``infeasible`` (:func:`infeasible_metrics`), with the estimate, instead of attempting it.
"""

from __future__ import annotations

import hashlib
import math
import time
from collections.abc import Mapping
from typing import Any, Final

import numpy as np
from numpy.typing import NDArray

from pihm.assemble import Residual, assemble
from pihm.circuits.assemble import encode_linear
from pihm.classical import (
    ReadoutBound,
    decode,
    physical_reference,
    readout_bound,
    reference_target,
)
from pihm.pipeline.records import JSONValue
from pihm.pipeline.spec import RunSpec
from pihm.problems import ladder
from pihm.qsp.budget import state_budget
from pihm.qsp.feasibility import Budget, InfeasibleError
from pihm.qsp.filter_comparison import (
    INFIDELITIES,
    StateComparison,
    compare_state,
    preparations,
    qite_state_coefficients,
    spectrum,
)
from pihm.qsp.filters import MAX_DEGREE, QiteFilter
from pihm.qsp.initial import initial_states
from pihm.qsp.minimax import minimax_filter
from pihm.qsp.phases import find_phases
from pihm.qsp.prepare import (
    T1Result,
    T2Result,
    T3Result,
    apply_qite_filter,
    filter_target,
    paper_imaginary_time,
    t1,
    t2,
    t3,
)
from pihm.qsp.readout import ReadoutResult, readout

__all__ = [
    "budget_metrics",
    "infeasible_metrics",
    "readout_metrics",
    "run_filters",
    "run_readout",
    "run_stateprep",
    "state_distance",
    "stateprep_metrics",
]


def _seeded(spec: RunSpec) -> np.random.Generator:
    """Return a generator seeded by the run's configuration hash, so its record reproduces."""
    digest = hashlib.sha256(spec.config_hash.encode()).digest()
    return np.random.default_rng(int.from_bytes(digest[:8], "little"))


def _identity(spec: RunSpec) -> dict[str, JSONValue]:
    """Return the columns that name a Pillar 2 run in ``summary.csv``."""
    problem = spec.problem
    return {
        "problem": problem.name,
        "variant": problem.variant,
        "rung": ladder.rung(problem.name),
        "label": spec.label,
        "regime": spec.regime,
        "n": int(max(spec.n)),
        "qubits": list(spec.n),
        "tier": spec.tier,
        "geometric_r": list(spec.geometric_r),
        **({"filter": spec.filter} if spec.filter != "qite" else {}),
        **({"encoder": spec.encoder} if spec.encoder != "structured" else {}),
        **({"carleman": spec.carleman} if spec.carleman is not None else {}),
    }


def _residual(spec: RunSpec) -> Residual:
    """Return the residual the run's state is prepared for."""
    return assemble(
        spec.problem,
        spec.n,
        regime=spec.regime,
        truncation=spec.truncation,
        rescale=spec.rescale,
        carleman=spec.carleman,
    )


def _regime_metrics(residual: Residual) -> dict[str, JSONValue]:
    """Return a descriptor run's block rescaling and its λ (nothing for the standard form)."""
    if residual.regime != "descriptor":
        return {}
    return {"rescale": residual.rescale, "scales": list(residual.scales)}


def state_distance(state: NDArray[Any], reference: NDArray[Any]) -> float:
    r"""Return the distance between two states, the global phase chosen to minimise it.

    Both are normalised first. It is measured as a vector norm, so it resolves distances down to
    double precision, where :math:`\sqrt{2(1-|\langle a|b\rangle|^2)}` loses everything below
    about ``1e-8``.

    Examples
    --------
    >>> import numpy as np
    >>> a = np.array([1.0, 0.0])
    >>> round(state_distance(np.array([1.0, 1e-9j]), -a), 12)
    1e-09
    """
    a = state / np.linalg.norm(state)
    b = reference / np.linalg.norm(reference)
    overlap = complex(np.vdot(b, a))
    phase = overlap / abs(overlap) if overlap != 0 else 1.0
    return float(np.linalg.norm(a - phase * b))


def field_metrics(
    spec: RunSpec, result: T1Result | T2Result
) -> tuple[dict[str, JSONValue], dict[str, NDArray[Any]]]:
    r"""Return the "field" check and the error budget's terms for a prepared state, and its fields.

    The prepared state is decoded as the classical ground state, the preparation's own reference,
    is (:func:`pihm.classical.decode`) and compared with the analytic solution. By the
    triangle inequality its error is at most the classical ground state's own,
    :math:`\varepsilon_{\rm disc}`, plus how far the two states' fields differ, which
    :class:`pihm.classical.ReadoutBound` bounds exactly for the states' distance :math:`d`, the
    phase chosen to minimise it, where :math:`Kd<|u_s|`; past that, the bound does not apply
    and the run is shown, not judged. The distance is measured directly: through the
    infidelity, :math:`d\le\sqrt{2\varepsilon_{\rm prep}}`, it would round to zero once
    :math:`\varepsilon_{\rm prep}` falls below double precision.

    Returns
    -------
    metrics
        The check and the terms.
    arrays
        The prepared state's field (``field``), the ground state's (``ground``) and the reference
        (``reference``) on the comparison grid, whose coordinates are ``grid_0`` (and
        ``grid_1``), as a build record keeps its own.
    """
    residual = result.residual
    ground = result.reference.vector
    classical = decode(residual, ground)
    eps_disc = classical.errors[0]
    decoded = decode(residual, result.state)
    linf, l2 = decoded.errors
    terms = readout_bound(residual, ground)
    deviation = state_distance(result.state, ground)
    bound = eps_disc + terms.bound(deviation)
    applicable = math.isfinite(bound)
    metrics: dict[str, JSONValue] = {
        "eps_disc": eps_disc,
        "kappa_read": terms.amplification,
        "field_max": terms.field_max,
        "state_deviation": deviation,
        "field_err_linf": linf,
        "field_err_l2": l2,
        "field_bound": bound if applicable else None,
        "field_within_bound": (linf <= bound) if applicable else None,
    }
    if isinstance(result, T2Result):
        metrics.update(budget_metrics(result, eps_disc, terms, linf))
    arrays = {
        "field": decoded.field,
        "ground": classical.field,
        "reference": decoded.reference,
        **{f"grid_{index}": axis for index, axis in enumerate(decoded.axes)},
    }
    return metrics, arrays


def budget_metrics(
    result: T2Result, eps_disc: float, terms: ReadoutBound, linf: float
) -> dict[str, JSONValue]:
    r"""Return the error budget, term by term (:mod:`pihm.qsp.budget`).

    The state terms are distances along the chain target → untruncated filter → T2 → the phases'
    realised polynomial; each is carried to the field by the exact readout bound. T2's field is
    judged against :math:`\varepsilon_{\rm disc}` plus the bound at the filter and truncation
    terms' sum; the phases term is what the circuit adds. :math:`d\,\varepsilon_{\rm BE}` and
    :math:`\varepsilon_{\rm sample}` are exactly zero (exact encodings, classical readout).
    """

    def finite(value: float) -> float | None:
        return value if math.isfinite(value) else None

    budget = state_budget(result)
    field_budget = eps_disc + terms.bound(budget.t2)
    return {
        "budget_filter": finite(budget.filter),
        "budget_truncation": finite(budget.truncation),
        "budget_phases": finite(budget.phases),
        "budget_measured": finite(budget.measured),
        "field_filter": finite(terms.bound(budget.filter)),
        "field_truncation": finite(terms.bound(budget.truncation)),
        "field_phases": finite(terms.bound(budget.phases)),
        "field_budget": finite(field_budget),
        "field_within_budget": (linf <= field_budget) if math.isfinite(field_budget) else None,
    }


def stateprep_metrics(
    spec: RunSpec, result: T1Result | T2Result
) -> tuple[dict[str, JSONValue], dict[str, NDArray[Any]]]:
    """Return the metrics of a ``"stateprep"`` run (strict JSON, flat, for summary.csv), and arrays.

    A T1 run adds the circuit's width, its agreement with T2 (normalised and before normalising)
    and the resource estimate it ran under, for comparison with its measured wall time. A run
    whose field is decoded, every one but a doubled space's, keeps ``field_metrics``'s
    arrays.
    """
    arrays: dict[str, NDArray[Any]] = {}
    metrics: dict[str, JSONValue] = {
        **_identity(spec),
        **_regime_metrics(result.residual),
        "truncation": spec.truncation,
        "alpha_R": result.alpha_r,
        "gap": result.gap,
        "kernel_count": result.reference.kernel_count(),
        "kernel_dimension": result.reference.kernel_dimension,
        "gap_ratio": result.reference.gap_ratio,
        "trusted": result.reference.trusted,
        "epsilon": spec.epsilon,
        "tau": result.filter.tau if isinstance(result.filter, QiteFilter) else None,
        "degree": result.filter.degree,
        "filter_weight": result.filter.weight,
        "phases_max_error": result.phases.max_error(),
        "initial_kind": result.initial_kind,
        "success_probability": result.success_probability,
        "infidelity": result.infidelity,
    }
    if isinstance(result, T2Result):
        bound = result.infidelity_bound
        metrics["gamma2"] = result.gamma**2
        metrics["infidelity_bound"] = bound if math.isfinite(bound) else None
    if spec.problem.is_linear or spec.carleman is not None:
        fields, arrays = field_metrics(spec, result)
        metrics.update(fields)
    else:
        # The doubled space's kernel is degenerate, and imaginary time prepares
        # the initial state's projection onto it (the infidelity above); against the physical
        # product state the prepared state is expected to be far off.
        residual = result.residual
        product = reference_target(residual)
        R = residual.matrix()
        sigma_max = float(result.reference.singular_values[-1])
        metrics["product_fidelity"] = float(abs(np.vdot(product, result.state)) ** 2)
        metrics["target_residual"] = float(np.linalg.norm(R @ product) / sigma_max)
    if isinstance(result, T1Result):
        cost = result.cost
        metrics.update(
            {
                "n_qubits": result.n_qubits,
                "t2_agreement": result.t2_agreement,
                "filtered_agreement": result.filtered_agreement,
                "predicted_peak_gib": cost.peak_bytes / 2**30 if cost else None,
                "predicted_seconds_per_step": cost.seconds_per_step if cost else None,
                "predicted_seconds": cost.seconds if cost else None,
            }
        )
    return metrics, arrays


def estimate_metrics(spec: RunSpec, result: T3Result) -> dict[str, JSONValue]:
    """Return the scalar metrics of a tier-T3 ``"stateprep"`` run.

    ``beyond_ceiling`` records whether the degree passes the filter's own ceiling, past which
    tier T2 refuses the run (:func:`pihm.qsp.filters.qite_coefficients`'s ``max_degree``).
    """
    return {
        **_identity(spec),
        **_regime_metrics(_residual(spec)),
        "truncation": spec.truncation,
        "epsilon": spec.epsilon,
        "method": result.method,
        "alpha_R": result.alpha_r,
        "gap": result.gap,
        "tau": result.tau,
        "degree": result.degree,
        "norm_R": result.norm_r,
        "ideal_degree": result.ideal_degree,
        "beyond_ceiling": result.degree > MAX_DEGREE,
        "predicted_p": result.predicted_p,
        "amplification_rounds": result.amplification_rounds,
        "preparations": result.preparations,
        "gate_count": result.gate_count,
        "cited_count": result.cited_count,
        "n_qubits": result.n_qubits,
        "statevector_gib": 16.0 * 2.0**result.n_qubits / 2**30,
    }


def infeasible_metrics(spec: RunSpec, refusal: InfeasibleError) -> dict[str, JSONValue]:
    """Return the metrics of a run its resource estimate refused: the estimate, the budget, why."""
    estimate, budget = refusal.estimate, refusal.budget
    return {
        **_identity(spec),
        "n_qubits": estimate.n_qubits,
        "degree": estimate.degree,
        "simulations": estimate.simulations,
        "statevector_gib": estimate.statevector_bytes / 2**30,
        "predicted_peak_gib": estimate.peak_bytes / 2**30,
        "predicted_seconds_per_step": estimate.seconds_per_step,
        "predicted_seconds": estimate.seconds,
        "budget_gib": budget.memory_bytes / 2**30,
        "budget_seconds": budget.seconds,
        "budget_source": budget.source,
        "infeasible_reasons": "; ".join(refusal.reasons),
    }


def run_stateprep(
    spec: RunSpec, timings: dict[str, float]
) -> tuple[Mapping[str, JSONValue], dict[str, NDArray[Any]]]:
    """Run the ``"stateprep"`` stage at the spec's tier.

    Tier T1 estimates its cost against :meth:`~pihm.qsp.feasibility.Budget.from_environment`
    first, and raises :class:`~pihm.qsp.feasibility.InfeasibleError` rather than exceed it. Tier
    T3 estimates the cost at the spec's resolution directly (:func:`pihm.qsp.prepare.t3`). A
    random initial state is drawn from a generator seeded by the configuration hash, so a record
    reproduces.
    """
    n = spec.n
    rng = _seeded(spec) if spec.kind == "random" else None
    started = time.perf_counter()
    if spec.tier == "T3":
        estimate = t3(
            spec.problem,
            n,
            kind=spec.kind,
            geometric_r=spec.geometric_r,
            epsilon=spec.epsilon,
            truncation=spec.truncation,
            regime=spec.regime,
            rescale=spec.rescale,
            carleman=spec.carleman,
            filter=spec.filter,
            encoder=spec.encoder,
        )
        timings["stateprep"] = time.perf_counter() - started
        return estimate_metrics(spec, estimate), {}
    result: T1Result | T2Result
    if spec.tier == "T1":
        result = t1(
            spec.problem,
            n,
            kind=spec.kind,
            geometric_r=spec.geometric_r,
            epsilon=spec.epsilon,
            truncation=spec.truncation,
            regime=spec.regime,
            rescale=spec.rescale,
            carleman=spec.carleman,
            filter=spec.filter,
            budget=Budget.from_environment(),
            rng=rng,
        )
    else:
        result = t2(
            spec.problem,
            n,
            kind=spec.kind,
            geometric_r=spec.geometric_r,
            epsilon=spec.epsilon,
            truncation=spec.truncation,
            regime=spec.regime,
            rescale=spec.rescale,
            carleman=spec.carleman,
            filter=spec.filter,
            rng=rng,
        )
    timings["stateprep"] = time.perf_counter() - started
    return stateprep_metrics(spec, result)


#: The variants of the filter's imaginary time, each a factor on the gap's own τ
#: (the gap underestimated ×2 and ×10); the paper's rule is added by :func:`run_protocol`.
PROTOCOL_VARIANTS: Final[dict[str, float]] = {"gap": 1.0, "gap_half": 2.0, "gap_tenth": 10.0}


def run_protocol(
    spec: RunSpec, timings: dict[str, float]
) -> tuple[Mapping[str, JSONValue], dict[str, NDArray[Any]]]:
    """Run the ``"protocol"`` stage: tier T2 under the paper's variants of the time rule.

    One record holds every variant: the imaginary time the gap gives, the gap underestimated
    ×2 and ×10 (τ doubled and multiplied by ten), and the paper's own rule
    (:func:`pihm.qsp.prepare.paper_imaginary_time`). Each reports τ, the degree, the
    success probability, the infidelity and whether it reaches ε; a degree past the filter's
    ceiling is recorded as such.
    """
    started = time.perf_counter()
    baseline = t2(
        spec.problem,
        spec.n,
        kind=spec.kind,
        geometric_r=spec.geometric_r,
        epsilon=spec.epsilon,
        truncation=spec.truncation,
    )
    assert isinstance(baseline.filter, QiteFilter)  # the protocol varies QITE's time rule
    taus = {name: baseline.filter.tau * factor for name, factor in PROTOCOL_VARIANTS.items()}
    taus["paper"] = paper_imaginary_time(spec.problem, spec.n)
    metrics: dict[str, JSONValue] = {
        **_identity(spec),
        "truncation": spec.truncation,
        "epsilon": spec.epsilon,
        "trusted": baseline.reference.trusted,
        "gap": baseline.gap,
    }
    for name, tau in taus.items():
        metrics[f"{name}_tau"] = tau
        try:
            result = t2(
                spec.problem,
                spec.n,
                kind=spec.kind,
                geometric_r=spec.geometric_r,
                epsilon=spec.epsilon,
                truncation=spec.truncation,
                tau=tau,
            )
        except ValueError:  # the filter's degree is past its ceiling
            metrics[f"{name}_beyond_ceiling"] = True
            continue
        metrics[f"{name}_beyond_ceiling"] = False
        metrics[f"{name}_degree"] = result.filter.degree
        metrics[f"{name}_success_probability"] = result.success_probability
        metrics[f"{name}_infidelity"] = result.infidelity
        metrics[f"{name}_reaches_epsilon"] = result.infidelity <= spec.epsilon
    timings["protocol"] = time.perf_counter() - started
    return metrics, {}


def _cost(degree: int | None, probability: float | None) -> dict[str, JSONValue]:
    """Return the preparations and the walk calls, preparations times degree, for one filter."""
    if degree is None or probability is None or not probability > 0.0:
        return {"preparations": None, "walk_calls": None}
    count = preparations(probability)
    return {"preparations": count, "walk_calls": count * degree}


def _comparison_metrics(result: StateComparison) -> dict[str, JSONValue]:
    """Return one infidelity's columns, unprefixed."""
    metrics: dict[str, JSONValue] = {
        "minimax_certified": result.minimax_certified,
        "minimax_degree": result.minimax_degree,
        "minimax_infidelity": result.minimax_infidelity,
        "minimax_probability": result.minimax_probability,
        "minimax_ground": result.minimax_ground,
        "qite_worst": result.qite_worst,
        "qite_certified": result.qite_certified,
        "qite_certified_tau": result.qite_certified_tau,
        "qite_degree": result.qite_degree,
        "qite_tau": result.qite_tau,
        "qite_infidelity": result.qite_infidelity,
        "qite_probability": result.qite_probability,
        "qite_ground": result.qite_ground,
        "production_degree": result.production_degree,
        "production_tau": result.production_tau,
        "production_infidelity": result.production_infidelity,
        "production_probability": result.production_probability,
        "production_reaches": (
            None
            if result.production_infidelity is None
            else result.production_infidelity <= result.epsilon
        ),
    }
    for name in ("minimax", "qite", "production"):
        cost = _cost(metrics[f"{name}_degree"], metrics[f"{name}_probability"])
        metrics.update({f"{name}_{key}": value for key, value in cost.items()})
    return metrics


def run_filters(
    spec: RunSpec, timings: dict[str, float]
) -> tuple[Mapping[str, JSONValue], dict[str, NDArray[Any]]]:
    r"""Run the ``"filters"`` stage: QITE against the minimax filter at like error.

    The initial state is the geometric candidate with the larger overlap with the target, the
    same for both filters, which the overlap alone decides (the ground weight, not either
    filter's own probability). The spectrum is the physical subspace's, where the initial state
    lies (:func:`pihm.qsp.initial.initial_states`): the descriptor's padding decouples exactly,
    so neither filter meets it, and ``dimension`` counts the physical columns. Columns for each
    infidelity :math:`\varepsilon` carry the prefix ``e<k>_``, :math:`\varepsilon=10^{-k}`.
    """
    started = time.perf_counter()
    residual = _residual(spec)
    R = residual.matrix()
    stacked, _reflection = encode_linear(
        spec.problem, spec.n, regime=spec.regime, rescale=spec.rescale, carleman=spec.carleman
    )
    reference = physical_reference(residual)
    target = reference.kernel_count() if residual.nonlinearity == "doubled" else 1
    columns = residual.physical_columns
    physical = R[:, columns].toarray()
    placed = initial_states("geometric", residual, spec.geometric_r)
    states = dict(zip(spec.geometric_r, placed, strict=True))
    candidates = {
        r: spectrum(physical, stacked.alpha, state[columns], target) for r, state in states.items()
    }
    r_best = max(candidates, key=lambda r: candidates[r].gamma)
    spectral = candidates[r_best]
    metrics: dict[str, JSONValue] = {
        **_identity(spec),
        **_regime_metrics(residual),
        "truncation": spec.truncation,
        "epsilon": spec.epsilon,
        "trusted": reference.trusted,
        "dimension": int(columns.size),
        "target": target,
        "alpha_R": float(stacked.alpha),
        "initial_r": r_best,
        "gamma2": spectral.gamma**2,
        "x_ground": spectral.x_ground,
        "x_excited": spectral.x_excited,
        "delta": spectral.x_excited - spectral.x_ground,
        "offset": spectral.offset,
    }
    timings["spectrum"] = time.perf_counter() - started
    chosen: StateComparison | None = None
    for epsilon in INFIDELITIES:
        step = time.perf_counter()
        result = compare_state(spectral, epsilon)
        prefix = f"e{round(-math.log10(epsilon))}_"
        metrics.update({prefix + key: value for key, value in _comparison_metrics(result).items()})
        timings[prefix + "compare"] = time.perf_counter() - step
        if math.isclose(epsilon, spec.epsilon):
            chosen = result

    if chosen is not None:
        step = time.perf_counter()
        phi0 = states[r_best]
        _gap, project = filter_target(residual, reference)
        goal = project(phi0)

        def through_t2(coefficients: NDArray[np.float64]) -> tuple[float, float]:
            state = apply_qite_filter(R, stacked.alpha, coefficients, phi0)
            infidelity = 1.0 - abs(np.vdot(goal, state)) ** 2 / float(np.vdot(state, state).real)
            phases = find_phases(coefficients.astype(np.complex128))
            return float(infidelity), phases.max_error()

        minimax = minimax_filter(chosen.minimax_degree, spectral.x_excited).coefficients
        metrics["t2_minimax_infidelity"], metrics["t2_minimax_phases_error"] = through_t2(minimax)
        if chosen.qite_degree is not None and chosen.qite_tau is not None:
            qite = qite_state_coefficients(
                spectral, spec.epsilon, chosen.qite_tau, chosen.qite_degree
            )
            metrics["t2_qite_infidelity"], metrics["t2_qite_phases_error"] = through_t2(qite)
        timings["t2"] = time.perf_counter() - step
    timings["filters"] = time.perf_counter() - started
    return metrics, {}


def readout_metrics(spec: RunSpec, result: ReadoutResult) -> dict[str, JSONValue]:
    """Return the scalar metrics of a ``"readout"`` run (strict JSON, flat, for summary.csv).

    Field errors are against the analytic solution at the probe points; a T1 run adds its
    agreement with T2 (relative to T2's largest value), the circuits' probabilities, the sampled
    field's departure from the exact one, and the resource estimate it ran under.
    """
    reference = result.reference
    metrics: dict[str, JSONValue] = {
        **_identity(spec),
        "truncation": spec.truncation,
        "epsilon": spec.epsilon,
        "degree": result.stateprep.filter.degree,
        "success_probability": result.stateprep.success_probability,
        "infidelity": result.stateprep.infidelity,
        "points": int(result.points.shape[0]),
        "regular_point": [float(x) for x in np.atleast_1d(result.points[-1])],
        "eta": result.eta,
        "field_err_linf": float(np.max(np.abs(result.field - reference))),
        "field_err_l2": float(np.linalg.norm(result.field - reference) / np.linalg.norm(reference)),
    }
    if result.tier == "T1":
        scale = float(np.max(np.abs(result.exact_field)))
        cost = result.cost
        metrics.update(
            {
                "n_qubits": result.n_qubits,
                "prepared_probability": result.prepared_probability,
                "combined_probability": result.combined_probability,
                "t2_agreement": float(np.max(np.abs(result.field - result.exact_field)) / scale),
                "eta_agreement": abs(result.eta - result.exact_eta) / result.exact_eta,
                "sampled_departure": (
                    float(np.max(np.abs(result.sampled_field - result.field)) / scale)
                    if result.shots
                    else None
                ),
                "sampled_eta": result.sampled_eta if result.shots else None,
                "predicted_peak_gib": cost.peak_bytes / 2**30 if cost else None,
                "predicted_seconds_per_step": cost.seconds_per_step if cost else None,
                "predicted_seconds": cost.seconds if cost else None,
            }
        )
    return metrics


def run_readout(
    spec: RunSpec, timings: dict[str, float]
) -> tuple[Mapping[str, JSONValue], dict[str, NDArray[Any]]]:
    """Run the ``"readout"`` stage at the spec's tier (the paper's protocol).

    Shot sampling draws from a generator seeded by the configuration hash, so a record reproduces.
    """
    if spec.kind != "geometric":
        raise ValueError("the readout reads out the geometric preparation only")
    if spec.tier == "T3":
        raise ValueError("the readout runs at tier T1 or T2; T3 estimates a stateprep run")
    rng = _seeded(spec) if spec.shots else None
    started = time.perf_counter()
    result = readout(
        spec.problem,
        spec.n,
        tier=spec.tier,
        geometric_r=spec.geometric_r,
        epsilon=spec.epsilon,
        truncation=spec.truncation,
        shots=spec.shots,
        rng=rng,
        budget=Budget.from_environment() if spec.tier == "T1" else None,
    )
    timings["readout"] = time.perf_counter() - started
    arrays = {
        "points": result.points,
        "field": result.field,
        "exact_field": result.exact_field,
        "reference": result.reference,
        "sampled_field": result.sampled_field,
    }
    return readout_metrics(spec, result), arrays
