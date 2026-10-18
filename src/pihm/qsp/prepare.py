r"""State preparation in three tiers.

Circuit (T1), exact polynomial emulation (T2) and resource estimation (T3).

:func:`t2` runs the whole of Pillar 2 for one rung at one resolution without simulating any
circuit: build the QITE filter (:mod:`pihm.qsp.filters`) at the *circuit's own*
:math:`\alpha_R` (the regime's Pillar-1-verified encoder,
:func:`pihm.circuits.assemble.encode_standard_linear` or
:func:`~pihm.circuits.assemble.encode_descriptor_linear`), find
its GQSP phases as a realised-polynomial certificate (:mod:`pihm.qsp.phases`), then apply the
filter's coefficients directly to an initial state via
:math:`H'v=2R^\dagger(Rv)/\alpha_R^2-v` — two sparse matvecs, never forming
:math:`H`. By the GQSP theorem this is an *exact* stand-in for a real circuit's post-selected
output, since that output depends on the encoding only through :math:`\Pi_{\rm in}V\Pi_{\rm in}`.

The recurrence is the three-term Chebyshev one, applied to :math:`H'` directly, since
:math:`H'=2H/\alpha_R^2-I` already spans :math:`[-1,1]`.

:func:`t1` runs the *actual circuit* (:mod:`pihm.circuits.gqsp`): the same filter and phases as
:func:`t2`, applied by simulating the GQSP circuit on the Aer statevector one controlled walk at a
time (:func:`~pihm.circuits.gqsp.apply_gqsp`) instead of by the sparse Chebyshev recurrence. Its
memory is the statevector's, :math:`2^q` amplitudes for the circuit's :math:`q` qubits, and its
time is the degree times one walk's simulation, so it is practical at the
smallest n of each rung; T2 is the tier every larger run and campaign uses.

:func:`t3` estimates resources at an n beyond what T1 or T2 can reach. Directly,
it computes every quantity at the target n itself, :math:`\alpha_R` and the walk's gate count from
the circuit and the gap and :math:`\gamma` from the classical SVD, leaving only the degree to its
closed-form estimate (:func:`~pihm.qsp.filters.asymptotic_degree`): that reaches the paper's
printed resolutions, where the filter's degree passes :func:`t2`'s ceiling. Fitted, for
resolutions beyond the classical reference too, it measures :math:`\alpha_R`, the gap and the gate
count at cheap sample resolutions and extrapolates each as a power law in :math:`N=2^n`
(:func:`pihm.resources.fit_power_law`); :math:`\gamma=\sqrt p` is bounded and saturates, so it
is taken from T2 at the *largest feasible* n, never extrapolated.

**Both forms.** Every tier takes the regime and the block rescaling as
:func:`pihm.assemble.assemble` does. The initial state lies on the field block, zero on every
other block (:func:`pihm.qsp.initial.initial_states`), and the reference, the
gap and so the filter are the physical subspace's (:func:`pihm.classical.physical_reference`):
the descriptor's padding decouples exactly, so the state never meets it. The standard form's state
is one block, the field, on which all of this is the identity.

**The filter.** Every tier takes it as ``filter``: ``"qite"``, imaginary time under the rule of
the method (:func:`~pihm.qsp.filters.qite_filter`), or ``"minimax"``, the minimax edge filter
at the least degree whose suppression of the excited spectrum, relative to the ground state, is
:math:`\varepsilon` (:func:`~pihm.qsp.minimax.edge_degree`).
Both are a priori, from :math:`\alpha_R` and the spectrum's edge, and both give
the same guarantee; the minimax filter needs no tail and fewer degrees.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Final, Literal, TypeAlias

import numpy as np
from numpy.typing import NDArray

from pihm.assemble import Regime, Rescale, Residual, Truncation, assemble
from pihm.circuits.assemble import encode_linear, encode_standard_linear
from pihm.circuits.compose import walk
from pihm.circuits.gqsp import apply_gqsp
from pihm.circuits.standard import StandardEncoder
from pihm.classical import ReferenceState, physical_reference, reference_state
from pihm.operators.tensor import SparseArray
from pihm.problems.spec import Problem
from pihm.qsp.feasibility import Budget, CostEstimate, estimate_gqsp, require
from pihm.qsp.filters import (
    DEFAULT_EPSILON,
    QiteFilter,
    asymptotic_degree,
    imaginary_time,
    qite_coefficients,
    qite_filter,
)
from pihm.qsp.initial import GEOMETRIC_R, InitialKind, initial_states
from pihm.qsp.minimax import MinimaxFilter, edge_angles, edge_degree, minimax_filter
from pihm.qsp.phases import Phases, find_phases
from pihm.resources import count_resources, fit_power_law

__all__ = [
    "RESOLVABLE_PROBABILITY",
    "FilterKind",
    "PreparationFilter",
    "T1Result",
    "T2Result",
    "T3Result",
    "apply_qite_filter",
    "filter_target",
    "infidelity_bound",
    "paper_imaginary_time",
    "preparation_filter",
    "t1",
    "t2",
    "t3",
]

#: The filter a preparation applies (the module documentation's "The filter").
FilterKind: TypeAlias = Literal["qite", "minimax"]

#: Either filter as built, imaginary time's truncated series or the minimax polynomial.
PreparationFilter: TypeAlias = QiteFilter | MinimaxFilter

#: The smallest success probability at which a prepared state can be compared with
#: the T1 vs T2 bar (1e-10) in double precision. Normalising divides the output's rounding
#: (about 1e-16) by the square root of the probability, so below this the comparison measures
#: rounding, not the circuit; such a run is reported as not preparable, and its circuit is judged
#: before normalising (:attr:`T1Result.filtered_agreement`).
RESOLVABLE_PROBABILITY: Final[float] = 1e-12


def apply_qite_filter(
    R: SparseArray,
    alpha_r: float,
    coefficients: NDArray[np.floating],
    phi0: NDArray[np.complexfloating],
) -> NDArray[np.complex128]:
    r"""Return :math:`\sum_kp_kT_k(H')\varphi_0`, applied via two sparse matvecs.

    :math:`H'v=2R^\dagger(Rv)/\alpha_R^2-v` is never formed as a matrix; each Chebyshev step costs
    one multiply by :math:`R` and one by :math:`R^\dagger`.

    Parameters
    ----------
    R
        The stacked residual (:meth:`pihm.assemble.Residual.matrix`).
    alpha_r
        The block encoding's own subnormalisation (the circuit's, not a separately fitted bound).
    coefficients
        Low-to-high-degree Chebyshev coefficients, domain :math:`[-1,1]`
        (:attr:`~pihm.qsp.filters.QiteFilter.coefficients`).
    phi0
        The state to apply the filter to.

    Returns
    -------
    NDArray[np.complex128]
        The filtered state — **not renormalised**; the caller decides whether and when to.
    """
    adjoint = R.conj().T

    def h_prime(v: NDArray[np.complex128]) -> NDArray[np.complex128]:
        step = (2.0 / alpha_r**2) * (adjoint @ (R @ v)) - v
        return np.asarray(step, dtype=np.complex128)

    coeffs = np.asarray(coefficients, dtype=np.float64)
    state = np.asarray(phi0, dtype=np.complex128)
    if coeffs.size == 0:
        return np.zeros_like(state)

    T_prev = state
    result: NDArray[np.complex128] = coeffs[0] * T_prev
    if coeffs.size == 1:
        return result

    T_curr = h_prime(state)
    result = result + coeffs[1] * T_curr
    for k in range(2, coeffs.size):
        T_next = 2.0 * h_prime(T_curr) - T_prev
        result = result + coeffs[k] * T_next
        T_prev, T_curr = T_curr, T_next
    return result


@dataclass(frozen=True)
class T2Result:
    r"""Tier T2's output: the prepared state and every quantity its acceptance checks use.

    Attributes
    ----------
    state
        The prepared state, normalised (or the zero vector, if the filter fully underflowed).
    success_probability
        :math:`\lVert P_\Phi(H')\varphi_0\rVert^2` before normalising,
        and, as :math:`\tau\to\infty`, :math:`\to\gamma^2`.
    infidelity
        :math:`1-|\langle\psi_{\rm target}|\psi\rangle|^2`. The target is :attr:`reference`'s
        own vector; in a doubled space, whose kernel is degenerate, it is the initial state's
        projection onto the kernel, :math:`\Pi_{\ker}\varphi_0/\|\Pi_{\ker}\varphi_0\|`, which
        is what imaginary time prepares there.
    gap
        The gap the filter is built on: :math:`\lambda_1-\lambda_0`, or in a doubled space the
        gap from the kernel to the first eigenvalue above it.
    initial_kind, initial
        Which kind of initial state was used, and the actual initial state (after the geometric
        state's own :math:`r=\pm\tfrac12` selection by measured probability).
    filter
        The filter applied (the module documentation's "The filter").
    phases
        Its GQSP phases, with their own realised-polynomial certificate.
    reference
        The classical ground state (:func:`pihm.classical.physical_reference`) this was checked
        against.
    residual
        The residual the state was prepared for, whose readout decodes it.
    alpha_r
        The circuit's own subnormalisation (the regime's encoder).
    gamma
        :math:`\gamma=\lVert\Pi\varphi_0\rVert`, the initial state's overlap with the target.
    infidelity_bound
        :func:`infidelity_bound` for this filter and initial state, the a priori bound
        :attr:`infidelity` is judged against; ``inf`` where the tail reaches the
        target's filtered value, so nothing is certified.
    """

    state: NDArray[np.complex128]
    success_probability: float
    infidelity: float
    gap: float
    initial_kind: InitialKind
    initial: NDArray[np.complex128]
    filter: PreparationFilter
    phases: Phases
    reference: ReferenceState
    residual: Residual
    alpha_r: float
    gamma: float
    infidelity_bound: float


def _target_count(residual: Residual, reference: ReferenceState) -> int:
    """Return how many of the lowest singular vectors the filter keeps."""
    return reference.kernel_count() if residual.nonlinearity == "doubled" else 1


def infidelity_bound(
    singular: NDArray[np.floating],
    alpha_r: float,
    qf: PreparationFilter,
    gamma: float,
    target: int,
) -> float:
    r"""Return the a priori bound on the infidelity :func:`t2` prepares.

    The untruncated filter :math:`f(s)=e^{-2\tau s^2/\alpha_R^2}` falls with the singular value
    :math:`s`. On the initial state it leaves at least :math:`\gamma f(s_{t-1})` along the target
    (its :math:`t` lowest singular vectors, projected), and at most
    :math:`\sqrt{1-\gamma^2}\,f(s_t)` in the excited states plus
    :math:`\gamma\,(f(s_0)-f(s_{t-1}))` across the target's own singular values; truncation moves
    the state by at most the certified tail :math:`T`. So

    .. math::

        1-F \le \left(\frac{\sqrt{1-\gamma^2}\,f(s_t)+\gamma\,(f(s_0)-f(s_{t-1}))+T}
        {\gamma f(s_{t-1})-T}\right)^2,

    which needs only the spectrum's edge, :math:`\gamma`, :math:`\tau` and :math:`T`: what the
    filter's parameters were chosen from. It is ``inf`` where :math:`T\ge\gamma f(s_{t-1})`.

    The minimax filter is its own polynomial, :math:`T=0`, and :math:`|p_d|` is largest on the
    excited states at :math:`x_1`. Relative to its value at :math:`s_{t-1}` it is
    :math:`1/\cosh(d\theta_{t-1})` there and :math:`\cosh(d\theta_0)/\cosh(d\theta_{t-1})` at
    :math:`s_0` (:func:`~pihm.qsp.minimax.edge_angles`), which the same bound takes in place of
    :math:`f`'s.

    Parameters
    ----------
    singular
        The residual's singular values, ascending
        (:attr:`~pihm.classical.ReferenceState.singular_values`).
    alpha_r
        The block encoding's subnormalisation.
    qf
        The filter applied.
    gamma
        The initial state's overlap with the target.
    target
        How many of the lowest singular vectors the target spans (1, or the kernel's dimension).
    """
    if isinstance(qf, MinimaxFilter):
        if gamma <= 0.0:
            return math.inf
        heights = 2.0 * (np.asarray(singular[: target + 1], dtype=np.float64) / alpha_r) ** 2
        angles = qf.degree * edge_angles(heights[:target], float(heights[target]))
        first, last = float(angles[0]), float(angles[-1])
        excited = 2.0 * math.exp(-last) / (1.0 + math.exp(-2.0 * last))
        lowest = math.exp(first - last) * (1.0 + math.exp(-2.0 * first))
        lowest /= 1.0 + math.exp(-2.0 * last)
        off = math.sqrt(max(1.0 - gamma * gamma, 0.0)) * excited + gamma * (lowest - 1.0)
        return (off / gamma) ** 2
    f = np.exp(-2.0 * qf.tau * (np.asarray(singular, dtype=np.float64) / alpha_r) ** 2)
    along = gamma * float(f[target - 1]) - qf.tail_bound
    if along <= 0.0:
        return math.inf
    spread = gamma * float(f[0] - f[target - 1])
    off = math.sqrt(max(1.0 - gamma * gamma, 0.0)) * float(f[target]) + spread + qf.tail_bound
    return (off / along) ** 2


def filter_target(
    residual: Residual, reference: ReferenceState
) -> tuple[float, Callable[[NDArray[np.complex128]], NDArray[np.complex128]]]:
    r"""Return the gap the filter is built on, and the state it prepares from an initial state.

    For a nondegenerate ground state these are :math:`\lambda_1-\lambda_0` and the ground state
    itself. In the paper's doubled space the kernel is degenerate (:math:`N^2` minus the rank), so
    imaginary time projects onto it: the gap is from the kernel, the
    :meth:`~pihm.classical.ReferenceState.kernel_count` smallest singular values, to the next
    one, and the target is :math:`\Pi_{\ker}\varphi_0/\|\Pi_{\ker}\varphi_0\|`.
    """
    if residual.nonlinearity != "doubled":
        return reference.gap, lambda _phi0: reference.vector
    count = _target_count(residual, reference)
    values = reference.singular_values
    dense = residual.matrix().toarray()
    basis = np.linalg.svd(dense, full_matrices=True)[2][-count:].conj().T

    def project(phi0: NDArray[np.complex128]) -> NDArray[np.complex128]:
        kernel = basis @ (basis.conj().T @ phi0)
        return np.asarray(kernel / np.linalg.norm(kernel), dtype=np.complex128)

    return float(values[count] ** 2 - values[0] ** 2), project


def preparation_filter(
    filter: FilterKind,
    residual: Residual,
    reference: ReferenceState,
    alpha_r: float,
    epsilon: float,
) -> PreparationFilter:
    r"""Return the filter a preparation applies, from :math:`\alpha_R` and the spectrum's edge.

    QITE's :math:`\tau` comes from the gap (:func:`filter_target`); the minimax filter's degree
    from the heights :math:`2s^2/\alpha_R^2` of the target's highest singular value and the next,
    :math:`1+x_0` and :math:`1+x_1` (:func:`~pihm.qsp.minimax.edge_degree`).

    Raises
    ------
    ValueError
        For an unknown ``filter``, or a degree past the filter's ceiling.
    """
    if filter == "qite":
        gap, _target = filter_target(residual, reference)
        return qite_filter(alpha_r, gap, epsilon=epsilon)
    if filter != "minimax":
        raise ValueError(f"unknown filter {filter!r}, expected 'qite' or 'minimax'")
    count = _target_count(residual, reference)
    heights = 2.0 * (reference.singular_values[: count + 1] / alpha_r) ** 2
    ground, excited = float(heights[count - 1]), float(heights[count])
    return minimax_filter(edge_degree(ground, excited, epsilon), excited - 1.0)


def t2(
    problem: Problem,
    n: int | Sequence[int],
    *,
    kind: InitialKind = "geometric",
    geometric_r: tuple[float, ...] = GEOMETRIC_R,
    epsilon: float = DEFAULT_EPSILON,
    truncation: Truncation = "none",
    regime: Regime = "standard",
    rescale: Rescale | None = None,
    carleman: int | None = None,
    filter: FilterKind = "qite",
    tau: float | None = None,
    rng: np.random.Generator | None = None,
) -> T2Result:
    r"""Run tier T2.

    Parameters
    ----------
    problem
        A problem the regime's encoder covers.
    n
        Qubits per axis: one number for every axis, or one per axis.
    kind
        The initial state. ``"geometric"`` (the default) tries every ``geometric_r``
        and keeps the one with the larger *measured* :attr:`~T2Result.success_probability`
        (observable on hardware); ``"uniform"`` and ``"random"`` are its controls.
    geometric_r
        The :math:`r` values tried when ``kind == "geometric"``.
    epsilon
        The suppression the filter holds the excited spectrum to (:func:`preparation_filter`):
        :func:`~pihm.qsp.filters.qite_filter`'s :math:`\varepsilon_f`.
    truncation
        Forwarded to :func:`pihm.assemble.assemble`'s classical reference only (never to the
        circuit, which the paper's own untruncated residual verifies). ``"none"`` matches the
        circuit exactly and is right for every rung checked so far except R2's stiffest panel
        (``{1, 5, 400}``), whose classical reference is only cleanly 1-dimensional
        (:attr:`~pihm.classical.ReferenceState.trusted`) under ``"tau"`` — the
        tau-truncation ablation. Always check :attr:`T2Result.reference`'s own ``trusted`` before
        treating :attr:`~T2Result.infidelity` as meaningful.
    regime, rescale, carleman
        As :func:`pihm.assemble.assemble` takes them (the module documentation's "Both forms"):
        ``carleman`` is a Carleman lift's order.
    filter
        ``"qite"`` or ``"minimax"`` (the module documentation's "The filter").
    tau
        QITE's imaginary time, in place of the one the gap gives: the variants of
        the method (the gap underestimated, the paper's own time rule,
        :func:`paper_imaginary_time`).
    rng
        Used only when ``kind == "random"``; defaults to an unseeded generator.

    Returns
    -------
    T2Result

    Raises
    ------
    ValueError
        If ``tau`` is given for the minimax filter.
    """
    if tau is not None and filter != "qite":
        raise ValueError("an imaginary time applies to the QITE filter only")
    residual = assemble(
        problem, n, regime=regime, truncation=truncation, rescale=rescale, carleman=carleman
    )
    candidates = initial_states(kind, residual, geometric_r, rng)

    stacked, _reflection = encode_linear(
        problem, n, regime=regime, rescale=rescale, carleman=carleman
    )
    R = residual.matrix()
    reference = physical_reference(residual)
    gap, target = filter_target(residual, reference)
    qf: PreparationFilter
    if tau is None:
        qf = preparation_filter(filter, residual, reference, stacked.alpha, epsilon)
    else:
        qf = qite_coefficients(tau, tolerance=epsilon / 10.0)
    phases = find_phases(qf.coefficients.astype(np.complex128))

    best_raw: NDArray[np.complex128] | None = None
    best_p = -1.0
    best_phi0 = candidates[0]
    for phi0 in candidates:
        raw = apply_qite_filter(R, stacked.alpha, qf.coefficients, phi0)
        p = float(np.vdot(raw, raw).real)
        if p > best_p:
            best_raw, best_p, best_phi0 = raw, p, phi0

    assert best_raw is not None  # candidates is never empty
    norm = np.sqrt(best_p)
    prepared = best_raw / norm if norm > 0.0 else best_raw
    goal = target(best_phi0)
    infidelity = 1.0 - abs(np.vdot(goal, prepared)) ** 2 if norm > 0.0 else 1.0
    gamma = min(float(abs(np.vdot(goal, best_phi0))), 1.0)
    bound = infidelity_bound(
        reference.singular_values, stacked.alpha, qf, gamma, _target_count(residual, reference)
    )

    return T2Result(
        state=prepared,
        success_probability=best_p,
        infidelity=float(infidelity),
        gap=gap,
        initial_kind=kind,
        initial=best_phi0,
        filter=qf,
        phases=phases,
        reference=reference,
        residual=residual,
        alpha_r=float(stacked.alpha),
        gamma=gamma,
        infidelity_bound=bound,
    )


def paper_imaginary_time(problem: Problem, n: int | Sequence[int]) -> float:
    r"""Return the imaginary time the paper's own rule gives, as this filter's :math:`\tau`.

    The paper evolves :math:`e^{-t\tilde H}` with :math:`\tilde H=H/\|H\|_F` for
    :math:`t\ge\tilde\lambda_{\max}/(\tilde\lambda_2(n-1))`,
    :math:`\tilde\lambda_2` the first excited eigenvalue and :math:`n` the register's qubits. The
    filter here is :math:`e^{-\tau(1+H')}=e^{-2\tau H/\alpha_R^2}`, so the same evolution is
    :math:`\tau=t\,\alpha_R^2/(2\|H\|_F)`, :math:`\alpha_R` the circuit's.
    """
    residual = assemble(problem, n)
    values = reference_state(residual.matrix()).singular_values ** 2
    stacked, _ = encode_standard_linear(problem, n)
    qubits = sum(residual.qubits)
    t = float(values[-1] / (values[1] * (qubits - 1)))
    return t * float(stacked.alpha) ** 2 / (2.0 * float(np.sqrt(np.sum(values**2))))


@dataclass(frozen=True)
class T1Result:
    r"""Tier T1's output: the circuit-simulated prepared state, and its agreement with T2.

    Attributes
    ----------
    state, success_probability, infidelity, gap, initial_kind, initial, filter, phases,
    reference, residual, alpha_r
        As :class:`T2Result`, computed from the *circuit's* own output instead of the sparse
        Chebyshev recurrence.
    t2_agreement
        :math:`\max|\psi_{T1}-\psi_{T2}|` between this result's own normalised ``state`` and
        :func:`apply_qite_filter`'s, applied to the *same* filter, phases and initial state this
        result used: the "T1 vs T2" acceptance metric
        (:math:`\le10^{-10}`). ``nan`` if either side's success probability underflowed to zero.
    filtered_agreement
        :math:`\max|\alpha\,\Pi_{\rm out}U\Pi_{\rm in}\varphi_0 - P(H')\varphi_0|`: the same
        comparison before either vector is normalised. Normalising divides both by
        :math:`\sqrt p`, so :attr:`t2_agreement` cannot fall below about this value over
        :math:`\sqrt p`; where :math:`p` is tiny, rounding at this level dominates it, and this is
        the comparison that still measures the circuit.
    n_qubits
        The circuit's own total qubit count (system, the walk's own ancilla, and the signal
        qubit): the simulated statevector has :math:`2^{\text{n\_qubits}}` amplitudes.
    cost
        The resource estimate the run was checked against before simulating, when it was given
        a budget (:mod:`pihm.qsp.feasibility`).
    """

    state: NDArray[np.complex128]
    success_probability: float
    infidelity: float
    gap: float
    t2_agreement: float
    filtered_agreement: float
    initial_kind: InitialKind
    initial: NDArray[np.complex128]
    filter: PreparationFilter
    phases: Phases
    reference: ReferenceState
    residual: Residual
    alpha_r: float
    n_qubits: int
    cost: CostEstimate | None


def t1(
    problem: Problem,
    n: int | Sequence[int],
    *,
    kind: InitialKind = "geometric",
    geometric_r: tuple[float, ...] = GEOMETRIC_R,
    epsilon: float = DEFAULT_EPSILON,
    truncation: Truncation = "none",
    regime: Regime = "standard",
    rescale: Rescale | None = None,
    carleman: int | None = None,
    filter: FilterKind = "qite",
    rng: np.random.Generator | None = None,
    budget: Budget | None = None,
) -> T1Result:
    r"""Run tier T1 by simulating the actual GQSP circuit.

    Same parameters, same filter and phases, and the same initial-state protocol as :func:`t2`
    — only *how* the filter is applied differs: the GQSP circuit is simulated on the
    statevector one controlled walk at a time (:func:`~pihm.circuits.gqsp.apply_gqsp`), once per
    initial-state candidate, instead of by the sparse Chebyshev recurrence.

    Parameters
    ----------
    problem, n, kind, geometric_r, epsilon, truncation, regime, rescale, carleman, filter, rng
        As :func:`t2`.
    budget
        If given, the run's memory and time are estimated as soon as the filter's degree is known
        (:func:`~pihm.qsp.feasibility.estimate_gqsp`), and a run that would exceed the budget is
        refused before anything is simulated.

    Returns
    -------
    T1Result

    Raises
    ------
    pihm.qsp.feasibility.InfeasibleError
        If ``budget`` is given and the estimate exceeds it.
    """
    residual = assemble(
        problem, n, regime=regime, truncation=truncation, rescale=rescale, carleman=carleman
    )
    candidates = initial_states(kind, residual, geometric_r, rng)

    stacked, _reflection = encode_linear(
        problem, n, regime=regime, rescale=rescale, carleman=carleman
    )
    R = residual.matrix()
    reference = physical_reference(residual)
    gap, target = filter_target(residual, reference)
    qf = preparation_filter(filter, residual, reference, stacked.alpha, epsilon)
    cost = None
    if budget is not None:
        cost = estimate_gqsp(stacked, qf.degree, simulations=len(candidates), budget=budget)
        require(cost, budget)
    phases = find_phases(qf.coefficients.astype(np.complex128))

    best_raw: NDArray[np.complex128] | None = None
    best_p = -1.0
    best_phi0 = candidates[0]
    for phi0 in candidates:
        raw = apply_gqsp(
            phases.theta, phases.phi, phases.lam, stacked, phi0, rescale=phases.rescale
        )
        p = float(np.vdot(raw, raw).real)
        if p > best_p:
            best_raw, best_p, best_phi0 = raw, p, phi0

    assert best_raw is not None  # candidates is never empty
    norm = np.sqrt(best_p)
    prepared = best_raw / norm if norm > 0.0 else best_raw
    infidelity = 1.0 - abs(np.vdot(target(best_phi0), prepared)) ** 2 if norm > 0.0 else 1.0

    t2_raw = apply_qite_filter(R, stacked.alpha, qf.coefficients, best_phi0)
    t2_norm = float(np.sqrt(np.vdot(t2_raw, t2_raw).real))
    if norm > 0.0 and t2_norm > 0.0:
        t2_agreement = float(np.max(np.abs(prepared - t2_raw / t2_norm)))
    else:
        t2_agreement = float("nan")

    return T1Result(
        state=prepared,
        success_probability=best_p,
        infidelity=float(infidelity),
        gap=gap,
        t2_agreement=t2_agreement,
        filtered_agreement=float(np.max(np.abs(best_raw - t2_raw))),
        initial_kind=kind,
        initial=best_phi0,
        filter=qf,
        phases=phases,
        reference=reference,
        residual=residual,
        alpha_r=float(stacked.alpha),
        n_qubits=stacked.n_qubits + 1,
        cost=cost,
    )


@dataclass(frozen=True)
class T3Result:
    r"""Tier T3's output: resource estimates at ``n``.

    Attributes
    ----------
    n
        The target resolution, qubits per axis.
    method
        ``"direct"``: every quantity computed at ``n`` itself, the degree from its asymptotic
        formula; ``"fitted"``: :math:`\alpha_R`, the gap and the gate count extrapolated from
        :func:`t2` samples.
    alpha_r, gap
        :math:`\alpha_R` and the gap the filter is built on (:attr:`T2Result.gap`): measured at
        ``n``, or the value :attr:`fits`' power law gives there.
    gamma
        :math:`\gamma=\sqrt p` as :math:`\tau\to\infty`: directly, the overlap of the best
        initial state with the filter's target (the ground state, or in a doubled space the
        kernel); fitted, :math:`\sqrt p` measured by :func:`t2` at the *largest* sample, never
        extrapolated (a bounded, saturating quantity).
    tau, degree
        QITE's :func:`~pihm.qsp.filters.imaginary_time` and
        :func:`~pihm.qsp.filters.asymptotic_degree` at ``alpha_r`` and ``gap``; for the minimax
        filter ``None`` and its exact degree (:func:`~pihm.qsp.minimax.edge_degree`), from the
        ground and first excited heights (fitted: the ground state at the spectrum's edge, where a
        converged resolution's sits).
    norm_r, ideal_degree
        :math:`\|R\|` on the physical subspace and the degree the same filter needs at
        :math:`\alpha_R = \|R\|`, the least any block encoding of :math:`R` has: the cost with
        the encoding's looseness taken out. Fitted, :math:`\|R\|` is the fitted
        :math:`\alpha_R` over the looseness :math:`\alpha_R/\|R\|` at the *largest* sample,
        bounded as :math:`\gamma` is, never extrapolated on its own.
    amplification_rounds
        :math:`k=\lfloor\pi/(4\arcsin\gamma)\rfloor`, the amplitude-amplification iterations
        that raise :attr:`predicted_p` towards 1 (about :math:`\pi/(4\gamma)` for small
        :math:`\gamma`); zero when one preparation already succeeds with
        :math:`\gamma\ge\sin(\pi/4)`.
    preparations
        :math:`2k+1`, the preparations (the GQSP circuit or its inverse) a run applies.
    gate_count
        :math:`\text{preparations}\times\text{degree}\times g(W)`, :math:`g(W)` the
        transpiled cx count of one walk application (measured at ``n``, or fitted).
    cited_count
        The same product for the walk's dense gates, the cited primitives
        (:attr:`pihm.resources.ResourceCounts.cited_count`); zero outside the doubled space.
    n_qubits
        The GQSP circuit's total qubit count at ``n``, from the circuit itself.
    predicted_p
        :math:`\gamma^2`.
    fits
        Fitted only: ``{"alpha_r": (coefficient, exponent), "gap": ..., "gamma": ...,
        "gate_count": ...}``, :func:`~pihm.resources.fit_power_law` in :math:`N=2^n`;
        ``fits["gamma"]`` is diagnostic, never evaluated at ``n``. Empty when direct.
    """

    n: tuple[int, ...]
    method: Literal["direct", "fitted"]
    alpha_r: float
    gap: float
    gamma: float
    tau: float | None
    degree: int
    norm_r: float
    ideal_degree: int
    amplification_rounds: float
    preparations: float
    gate_count: float
    cited_count: float
    n_qubits: int
    predicted_p: float
    fits: dict[str, tuple[float, float]]


def t3(
    problem: Problem,
    n: int | Sequence[int],
    *,
    sample_n: tuple[int, ...] | None = None,
    kind: InitialKind = "geometric",
    geometric_r: tuple[float, ...] = GEOMETRIC_R,
    epsilon: float = DEFAULT_EPSILON,
    truncation: Truncation = "none",
    regime: Regime = "standard",
    rescale: Rescale | None = None,
    carleman: int | None = None,
    filter: FilterKind = "qite",
    encoder: StandardEncoder = "structured",
) -> T3Result:
    r"""Estimate tiers T1 and T2's resource cost at ``n``.

    Directly (``sample_n`` omitted), everything but the degree is computed at ``n`` itself:
    :math:`\alpha_R` and the walk's gate count from the circuit, the gap and :math:`\gamma` from
    the classical SVD. That reaches every resolution the classical reference does, including the
    paper's printed ones where the filter's degree passes :func:`t2`'s ceiling. Fitted (with
    ``sample_n``), :func:`t2` runs at each sample, :math:`\alpha_R`, the gap and the gate count
    are fitted as power laws in :math:`N=2^n` and evaluated at ``n``, for resolutions beyond the
    classical reference too; :math:`\gamma` is :func:`t2`'s at the largest sample.

    Parameters
    ----------
    problem
        A problem the regime's encoder covers.
    n
        The target resolution, qubits per axis (one number for every axis, or one per axis).
    sample_n
        Resolutions (every axis alike) to fit from, at least two; omit for the direct estimate.
    kind, geometric_r, epsilon, truncation, regime, rescale, carleman, filter
        As :func:`t2`.
    encoder
        The encoding of :math:`\mathbb{G}` (:data:`~pihm.circuits.standard.DERIVATIVE_ENCODERS`);
        the published one is approximate, so only T3 runs it.

    Returns
    -------
    T3Result

    Raises
    ------
    ValueError
        If ``sample_n`` has fewer than two resolutions.
    """
    axes = len(problem.fields[0].axes)
    ns = (n,) * axes if isinstance(n, int) else tuple(n)
    _stacked, reflection = encode_linear(
        problem, ns, regime=regime, rescale=rescale, encoder=encoder, carleman=carleman
    )
    W = walk(reflection)
    n_qubits = W.n_qubits + 1  # the GQSP circuit's signal qubit
    fits: dict[str, tuple[float, float]] = {}
    method: Literal["direct", "fitted"]
    if sample_n is None:
        method = "direct"
        counts = count_resources(W)
        residual = assemble(
            problem, ns, regime=regime, truncation=truncation, rescale=rescale, carleman=carleman
        )
        reference = physical_reference(residual)
        gap, target = filter_target(residual, reference)
        candidates = initial_states(kind, residual, geometric_r, None)
        overlaps = [abs(np.vdot(target(phi0), phi0)) for phi0 in candidates]
        alpha_r = float(_stacked.alpha)
        gamma = min(float(max(overlaps)), 1.0)
        count = _target_count(residual, reference)
        norm_r = float(np.max(reference.singular_values))
        low, high = (float(v) for v in reference.singular_values[count - 1 : count + 1])
        ground, excited = 2.0 * (low / alpha_r) ** 2, 2.0 * (high / alpha_r) ** 2
        ideal = (2.0 * (low / norm_r) ** 2, 2.0 * (high / norm_r) ** 2)
        gate_per_walk = float(counts.cx_count)
        cited_per_walk = float(counts.cited_count)
    else:
        if len(sample_n) < 2:
            raise ValueError("t3 needs at least two sample_n resolutions to fit a scaling trend")
        method = "fitted"
        alpha_samples, gap_samples, gamma_samples, gate_samples, norm_samples = [], [], [], [], []
        cited_per_walk = 0.0
        for sample in sample_n:
            sample_stacked, sample_reflection = encode_linear(
                problem,
                (sample,) * axes,
                regime=regime,
                rescale=rescale,
                encoder=encoder,
                carleman=carleman,
            )
            counts = count_resources(walk(sample_reflection))
            result = t2(
                problem,
                (sample,) * axes,
                kind=kind,
                geometric_r=geometric_r,
                epsilon=epsilon,
                truncation=truncation,
                regime=regime,
                rescale=rescale,
                carleman=carleman,
                filter=filter,
            )
            alpha_samples.append(float(sample_stacked.alpha))
            gap_samples.append(result.gap)
            gamma_samples.append(float(np.sqrt(result.success_probability)))
            gate_samples.append(float(counts.cx_count))
            norm_samples.append(float(np.max(result.reference.singular_values)))
            cited_per_walk = float(counts.cited_count)
        Ns = [2.0**sample for sample in sample_n]
        fits = {
            "alpha_r": fit_power_law(Ns, alpha_samples),
            "gap": fit_power_law(Ns, gap_samples),
            "gamma": fit_power_law(Ns, gamma_samples),
            "gate_count": fit_power_law(Ns, gate_samples),
        }

        def evaluate(name: str) -> float:
            coefficient, exponent = fits[name]
            return float(coefficient * (2.0 ** max(ns)) ** exponent)

        alpha_r, gap, gate_per_walk = evaluate("alpha_r"), evaluate("gap"), evaluate("gate_count")
        largest = int(np.argmax(sample_n))
        gamma = min(gamma_samples[largest], 1.0)
        norm_r = alpha_r * norm_samples[largest] / alpha_samples[largest]
        ground, excited = 0.0, 2.0 * gap / alpha_r**2
        ideal = (0.0, 2.0 * gap / norm_r**2)

    tau: float | None
    if filter == "minimax":
        tau, degree = None, edge_degree(ground, excited, epsilon)
        ideal_degree = edge_degree(*ideal, epsilon)
    else:
        tau = imaginary_time(alpha_r, gap, epsilon)
        degree = asymptotic_degree(tau, epsilon / 10.0)
        ideal_degree = asymptotic_degree(imaginary_time(norm_r, gap, epsilon), epsilon / 10.0)
    theta = float(np.arcsin(gamma))
    amplification_rounds = float(np.floor(np.pi / (4.0 * theta))) if theta > 0.0 else float("inf")
    preparations = 2.0 * amplification_rounds + 1.0
    return T3Result(
        n=ns,
        method=method,
        alpha_r=alpha_r,
        gap=gap,
        gamma=gamma,
        tau=tau,
        degree=degree,
        norm_r=norm_r,
        ideal_degree=ideal_degree,
        amplification_rounds=amplification_rounds,
        preparations=preparations,
        gate_count=preparations * degree * gate_per_walk,
        cited_count=preparations * degree * cited_per_walk,
        n_qubits=n_qubits,
        predicted_p=gamma**2,
        fits=fits,
    )
