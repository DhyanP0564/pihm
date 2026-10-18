r"""The paper's interferometric readout: the field from overlap measurements.

Wu et al. (arXiv:2504.13174, Eqs. 36–39, after their refs. Kyriienko, *npj Quantum Information*
**6**, 7 (2020), and Scali, Umeano & Kyriienko, arXiv:2312.07115) read the prepared state
:math:`\psi` out at a point :math:`x` without any :math:`x`-keyed controlled unitary: they run the
preparation :math:`\hat U_{\rm prep}` followed by the feature map's inverse
:math:`\hat U_\tau^\dagger(x)` (:func:`pihm.circuits.primitives.chebyshev_feature_map`) and
measure the all-zero outcome, twice — once on the prepared state G, and once on the *combined*
state C of an LCU combiner: a selector qubit in :math:`|+\rangle`, :math:`\hat U_{\rm prep}`
controlled on it, and a closing Hadamard. The all-zero probabilities :math:`P_G(x)` and
:math:`P_C(x)` combine into the signed overlap

.. math::

    f^*(x) = 4P_C(x) - P_G(x) - \frac1{2^dN} = \frac2{2^d}\sqrt{\frac{p_G}N}\,\tau(x)\cdot\psi,

with :math:`p_G` the preparation's own success probability, :math:`d` the number of axes and
:math:`N` the field's dimension. The all-zero amplitude after :math:`\hat U_\tau^\dagger(x)` (one
feature map per axis) is :math:`\langle u(x)|v\rangle`, where :math:`u(x)=\tau(x)/2^{d/2}` is the
feature maps' own leading column, :math:`\tau(x)` the Kronecker product of the axes' features
(:func:`feature_amplitudes`), and :math:`v` the state's all-ancilla-zero branch:
:math:`g=\sqrt{p_G}\,\psi` for G, and :math:`(e_0+g)/2` for C, with :math:`u_0=1/\sqrt{2^dN}`.
For one axis and a deterministic preparation (:math:`p_G=1`) the identity is the paper's Eq. 36;
a GQSP preparation succeeds with :math:`p_G<1`, and the raw probabilities carry it exactly.
The field follows from the regular datum :math:`f(x_s)`, as the paper recovers :math:`\eta_e`:

.. math::

    f_Q(x) = f(x_s)\,\frac{f^*(x)}{f^*(x_s)}, \qquad
    \eta_e = \Bigl(\frac2{2^d}\Bigr)^2\frac{f(x_s)^2\,p_G}{N\,f^*(x_s)^2}.

* **Tier T1** simulates the circuits on T1's engine: the combiner
  (:func:`combined_statevectors`) controls the geometric preparation's rotations, the GQSP
  rotations, the walk's two reflections and GQSP's global phase on the selector
  (:func:`pihm.circuits.gqsp.evolve_gqsp` with ``selector=True``), never :math:`U_R`. Before its
  closing Hadamard, the combiner's selector-one branch is :math:`\hat U_{\rm prep}|0\rangle`
  itself, so one evolution gives both G and C. Each measured probability can be shot-sampled,
  drawn from its binomial distribution.
* **Tier T2** is the exact overlap, :math:`f_Q(x)=f(x_s)\,\tau(x)\cdot\psi/\tau(x_s)\cdot\psi`, of
  T2's prepared state.

Both tiers read out T2's preparation: of the geometric initial states, the one prepared with the
larger success probability. T1 reproduces T2's probabilities to rounding, so
measuring every candidate on the circuit would choose the same one, at that many times the cost.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Literal, TypeAlias

import numpy as np
from numpy.typing import NDArray
from qiskit import QuantumCircuit

from pihm.assemble import Truncation
from pihm.basis import tau
from pihm.circuits.assemble import encode_standard_linear
from pihm.circuits.gqsp import evolve_gqsp
from pihm.circuits.primitives import (
    BlockEncoding,
    chebyshev_feature_map,
    compile_circuit,
    evolve,
)
from pihm.problems.reference import reference_solution
from pihm.problems.spec import Problem
from pihm.qsp.feasibility import Budget, CostEstimate, estimate_gqsp, require
from pihm.qsp.filters import DEFAULT_EPSILON
from pihm.qsp.initial import GEOMETRIC_R
from pihm.qsp.prepare import T2Result, t2

__all__ = [
    "READOUT_POINTS",
    "ReadoutResult",
    "ReadoutTier",
    "combined_statevectors",
    "feature_amplitudes",
    "readout",
]

#: The probe coordinates on each axis; :math:`f_Q` is read out on their grid over the axes, and at
#: the regular point.
READOUT_POINTS: NDArray[np.float64] = np.linspace(-0.9, 0.9, 7)

#: How a readout is computed, the circuits simulated (T1) or the exact overlap (T2).
ReadoutTier: TypeAlias = Literal["T1", "T2"]


@dataclass(frozen=True)
class ReadoutResult:
    r"""A readout of :math:`f_Q` at probe points, and what it was measured from.

    Attributes
    ----------
    tier
        ``"T1"`` or ``"T2"``.
    points
        The probe points, the regular point :math:`x_s` last: shape ``(P,)`` on one axis,
        ``(P, d)`` on :math:`d`.
    field
        :math:`f_Q` at ``points``: for T1, from the circuits' exact probabilities.
    exact_field
        T2's :math:`f_Q` at ``points``: ``field`` itself for T2, its ground truth for T1.
    reference
        The analytic solution at ``points``.
    eta
        :math:`\eta_e`, recovered from the regular datum.
    exact_eta
        T2's :math:`\eta_e`.
    prepared_probability
        T1: :math:`p_G`, the preparation circuit's success probability; T2: ``nan``.
    combined_probability
        T1: the combiner's success probability, :math:`N_C^2/4`; T2: ``nan``.
    shots
        T1: the shots each probability is sampled with; ``0`` when it is not.
    sampled_field, sampled_eta
        T1 with ``shots``: :attr:`field` and :attr:`eta` from shot-sampled probabilities, drawn
        from the same simulation; otherwise ``nan``.
    stateprep
        T2's preparation, whose initial state and filter both tiers read out.
    n_qubits
        T1: the qubits simulated (the preparation's and the selector); T2: ``0``.
    cost
        T1's resource estimate, when one was made against a budget.
    """

    tier: ReadoutTier
    points: NDArray[np.float64]
    field: NDArray[np.float64]
    exact_field: NDArray[np.float64]
    reference: NDArray[np.float64]
    eta: float
    exact_eta: float
    prepared_probability: float
    combined_probability: float
    shots: int
    sampled_field: NDArray[np.float64]
    sampled_eta: float
    stateprep: T2Result
    n_qubits: int
    cost: CostEstimate | None


def feature_amplitudes(x: float, n: int) -> NDArray[np.complex128]:
    r"""Return :math:`u(x)`, the branch of the feature map that reads the all-zero outcome.

    :math:`\langle0|\hat U_\tau^\dagger(x)|v\rangle=\langle u(x)|v\rangle` for any state whose
    feature-map ancilla is :math:`|0\rangle`, with :math:`u(x)` the ancilla-zero branch of
    :math:`\hat U_\tau(x)|0\rangle`: one evolution of the :math:`(n+1)`-qubit feature map, equal to
    :math:`\tau(x)/\sqrt2`.
    """
    circuit = chebyshev_feature_map(x, n).circuit
    zero = np.zeros(1 << circuit.num_qubits, dtype=np.complex128)
    zero[0] = 1.0
    amplitudes: NDArray[np.complex128] = evolve(compile_circuit(circuit), zero)[: 1 << n]
    return amplitudes


def _on_axes(
    part: Callable[[float, int], NDArray[Any]], point: NDArray[np.float64], ns: tuple[int, ...]
) -> NDArray[Any]:
    """Return the Kronecker product of ``part`` over the axes, the first axis most significant."""
    result: NDArray[Any] = np.ones(1)
    for coordinate, n in zip(point, ns, strict=True):
        result = np.kron(result, part(float(coordinate), n))
    return result


def _geometric_angles(initial: NDArray[np.complexfloating], n: int) -> NDArray[np.float64]:
    r"""Return the :math:`R_Y` angles that prepare the geometric product state ``initial``.

    Qubit :math:`j` carries :math:`|0\rangle+r^{2^j}|1\rangle`, so its angle is
    :math:`2\arctan(r^{2^j})`, with :math:`r` the state's own ratio ``initial[1] / initial[0]``.
    """
    r = float(np.real(initial[1] / initial[0]))
    return np.array([2.0 * np.arctan(r ** (2**j)) for j in range(n)])


def combined_statevectors(
    stateprep: T2Result, stacked: BlockEncoding, n: int
) -> tuple[NDArray[np.complex128], NDArray[np.complex128]]:
    r"""Return the statevectors of the readout's G and C circuits, from one evolution.

    The combiner acts on the preparation's register plus a selector qubit, the highest: a Hadamard
    on the selector, the geometric state's :math:`R_Y` rotations controlled on it, the GQSP circuit
    controlled on it (:func:`~pihm.circuits.gqsp.evolve_gqsp` with ``selector=True``), then a
    closing Hadamard. Before that Hadamard, the selector-one half times :math:`\sqrt2` is
    :math:`\hat U_{\rm prep}|0\rangle`, the G state.

    Parameters
    ----------
    stateprep
        T2's preparation: its geometric initial state and its phases.
    stacked
        :math:`U_R` for the same problem and resolution.
    n
        Qubits in the field's register, every axis together.

    Returns
    -------
    g_state, c_state
        :math:`\hat U_{\rm prep}|0\rangle`, and the combiner's output.
    """
    phases = stateprep.phases
    width = stacked.n_qubits + 2
    selector = width - 1
    opening = QuantumCircuit(width)
    opening.h(selector)
    for qubit, angle in enumerate(_geometric_angles(stateprep.initial, n)):
        opening.cry(float(angle), selector, qubit)
    state = np.zeros(1 << width, dtype=np.complex128)
    state[0] = 1.0
    state = evolve(compile_circuit(opening), state)
    state = evolve_gqsp(phases.theta, phases.phi, phases.lam, stacked, state, selector=True)
    g_state: NDArray[np.complex128] = np.sqrt(2.0) * state[1 << (width - 1) :]
    closing = QuantumCircuit(width)
    closing.h(selector)
    return g_state, evolve(compile_circuit(closing), state)


def _sample(
    probabilities: NDArray[np.float64], shots: int, rng: np.random.Generator
) -> NDArray[np.float64]:
    """Return each probability's estimate from ``shots`` binomial draws."""
    clipped = np.clip(probabilities, 0.0, 1.0)
    sampled: NDArray[np.float64] = rng.binomial(shots, clipped) / shots
    return sampled


def _field(
    p_g: float,
    prob_g: NDArray[np.float64],
    prob_c: NDArray[np.float64],
    f_s: float,
    N: int,
    axes: int,
) -> tuple[NDArray[np.float64], float]:
    r"""Return :math:`f_Q` and :math:`\eta_e` from the measured probabilities (module docstring)."""
    f_star = 4.0 * prob_c - prob_g - 1.0 / (2**axes * N)
    field: NDArray[np.float64] = f_s * f_star / f_star[-1]
    scale = 2.0 / 2**axes
    return field, float((scale * f_s) ** 2 * p_g / (N * f_star[-1] ** 2))


def readout(
    problem: Problem,
    n: int | Sequence[int],
    *,
    tier: ReadoutTier = "T1",
    points: NDArray[np.floating] | None = None,
    geometric_r: tuple[float, ...] = GEOMETRIC_R,
    epsilon: float = DEFAULT_EPSILON,
    truncation: Truncation = "none",
    shots: int = 0,
    rng: np.random.Generator | None = None,
    budget: Budget | None = None,
) -> ReadoutResult:
    r"""Read :math:`f_Q` out at ``points`` and the regular point.

    Parameters
    ----------
    problem, n
        A problem :func:`~pihm.circuits.assemble.encode_standard_linear` covers, and its qubits
        per axis.
    tier
        ``"T1"``, the interferometric circuits simulated, or ``"T2"``, the exact overlap.
    points
        Probe points in :math:`[-1, 1]^d`, shape ``(P,)`` on one axis or ``(P, d)``; by default
        the grid of :data:`READOUT_POINTS` over the axes.
    geometric_r, epsilon, truncation
        As :func:`~pihm.qsp.prepare.t2`, whose preparation both tiers read out.
    shots
        T1 only: also sample every probability with this many shots (``0``: exact only).
    rng
        The generator shot sampling draws from; needed when ``shots`` is positive.
    budget
        T1 only: if given, the combiner's simulation is estimated first
        (:func:`~pihm.qsp.feasibility.estimate_gqsp`) and refused
        (:class:`~pihm.qsp.feasibility.InfeasibleError`) if it would exceed it.

    Returns
    -------
    ReadoutResult
    """
    if shots < 0:
        raise ValueError(f"shots must be non-negative, got {shots}")
    if shots and tier == "T2":
        raise ValueError("shots apply to tier T1, the circuits, only")
    if shots and rng is None:
        raise ValueError("shot sampling needs a random generator")
    stateprep = t2(
        problem,
        n,
        kind="geometric",
        geometric_r=geometric_r,
        epsilon=epsilon,
        truncation=truncation,
    )
    axes = len(problem.fields[0].axes)
    ns = (n,) * axes if isinstance(n, int) else tuple(n)
    N = stateprep.state.shape[0]
    if points is None:
        points = np.stack(np.meshgrid(*[READOUT_POINTS] * axes, indexing="ij"), -1)
    probes = np.asarray(points, dtype=np.float64).reshape(-1, axes)
    grid = np.vstack([probes, np.asarray(problem.regular.at, dtype=np.float64)])
    physical = reference_solution(problem).physical
    reference = np.asarray(physical(*grid.T), dtype=np.float64)
    f_s = float(reference[-1])

    overlaps = np.array([float(np.real(_on_axes(tau, x, ns) @ stateprep.state)) for x in grid])
    if axes == 1:
        grid = grid[:, 0]
    exact_field = f_s * overlaps / overlaps[-1]
    exact_eta = float((f_s / overlaps[-1]) ** 2)
    nan = float("nan")
    unsampled = np.full(grid.shape, nan)
    if tier == "T2":
        return ReadoutResult(
            tier="T2",
            points=grid,
            field=exact_field,
            exact_field=exact_field,
            reference=reference,
            eta=exact_eta,
            exact_eta=exact_eta,
            prepared_probability=nan,
            combined_probability=nan,
            shots=0,
            sampled_field=unsampled,
            sampled_eta=nan,
            stateprep=stateprep,
            n_qubits=0,
            cost=None,
        )

    stacked, _reflection = encode_standard_linear(problem, n)
    cost = None
    if budget is not None:
        cost = estimate_gqsp(stacked, stateprep.filter.degree, selector=True, budget=budget)
        require(cost, budget)
    g_state, c_state = combined_statevectors(stateprep, stacked, sum(ns))
    g, c = g_state[:N], c_state[:N]
    p_g, p_c = float(np.vdot(g, g).real), float(np.vdot(c, c).real)
    amplitudes = np.array(
        [_on_axes(feature_amplitudes, np.atleast_1d(x), ns) for x in grid], dtype=np.complex128
    )
    prob_g = np.abs(amplitudes.conj() @ g) ** 2
    prob_c = np.abs(amplitudes.conj() @ c) ** 2
    field, eta = _field(p_g, prob_g, prob_c, f_s, N, axes)
    sampled_field, sampled_eta = unsampled, nan
    if shots:
        assert rng is not None  # checked above
        sampled_p_g = float(_sample(np.array([p_g]), shots, rng)[0])
        sampled_field, sampled_eta = _field(
            sampled_p_g, _sample(prob_g, shots, rng), _sample(prob_c, shots, rng), f_s, N, axes
        )
    return ReadoutResult(
        tier="T1",
        points=grid,
        field=field,
        exact_field=exact_field,
        reference=reference,
        eta=eta,
        exact_eta=exact_eta,
        prepared_probability=p_g,
        combined_probability=p_c,
        shots=shots,
        sampled_field=sampled_field,
        sampled_eta=sampled_eta,
        stateprep=stateprep,
        n_qubits=stacked.n_qubits + 2,
        cost=cost,
    )
