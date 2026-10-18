r"""Classical reference solutions: the SVD ground state, its spectrum and whether to trust it.

The ground state of :math:`H = R^\dagger R` is the smallest right singular vector of the stacked
residual :math:`R` (:mod:`pihm.classical.reference`, the dense SVD with its safeguards). Past
the dense SVD's ceiling it is computed iteratively, from sparse operations alone, with an a
posteriori bound; :func:`physical_reference` chooses, on the physical subspace's size.

:func:`solve` then evaluates the ground state of an assembled
:class:`~pihm.assemble.Residual` the way the paper reads it out: the model
function :math:`f_q`, its scale :math:`\eta_e` from the regular constraint
(:math:`\sqrt{\eta_e} = f(x_s)/f_q^\star(x_s)`, Fig. 1d), the decoded field against the
reference, and the state against the reference's own coefficients. :func:`decode` reads any
state out the same way, a prepared one included.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, replace
from typing import Any, Final, Literal

import numpy as np
import scipy.sparse as sp
from numpy.typing import NDArray
from scipy import optimize
from scipy.sparse.linalg import expm_multiply

from pihm.assemble import Residual, represent, system_state
from pihm.assemble.systems import components, coupling, ratio_data
from pihm.basis import tau
from pihm.classical.iterative import DENSE_COLUMNS, iterative_reference_state
from pihm.classical.reference import (
    DEFAULT_MIN_GAP_RATIO,
    ReferenceState,
    SvdDriver,
    reference_state,
)
from pihm.operators.tensor import SparseArray
from pihm.problems.expressions import compile_expression
from pihm.problems.reference import reference_solution

__all__ = [
    "DEFAULT_MIN_GAP_RATIO",
    "DENSE_COLUMNS",
    "ETA_POINTS",
    "GRID_POINTS",
    "ClassicalSolution",
    "DecodedField",
    "ReadoutBound",
    "ReferenceState",
    "SvdDriver",
    "decode",
    "evaluate_model",
    "field_error",
    "iterative_reference_state",
    "lift_errors",
    "physical_reference",
    "readout_bound",
    "reference_state",
    "reference_target",
    "solve",
]

#: Points per axis of the uniform grid on which decoded fields are compared with the reference
#: (1-D and 2-D).
GRID_POINTS: Final[dict[int, int]] = {1: 1001, 2: 101}

#: Points per axis in :math:`[-0.9, 0.9]` at which :math:`\eta` is read out to measure its
#: spread; points where the solution is below 1% of
#: its largest value there are skipped.
ETA_POINTS: Final[int] = 7


def evaluate_model(
    coefficients: NDArray[Any], qubits: tuple[int, ...], *coordinates: NDArray[Any]
) -> NDArray[Any]:
    r"""Evaluate :math:`\langle\tau(x)|c\rangle` at scattered points, on one or two axes.

    Parameters
    ----------
    coefficients
        Normalised coefficients in Kronecker order, on ``qubits`` per axis.
    qubits
        The register size per axis.
    coordinates
        One array of points per axis, all of the same shape.
    """
    if len(qubits) == 1:
        return np.asarray(tau(coordinates[0], qubits[0]) @ coefficients)
    if len(qubits) == 2:
        tensor = coefficients.reshape(1 << qubits[0], 1 << qubits[1])
        left, right = (tau(c, k) for c, k in zip(coordinates, qubits, strict=True))
        return np.asarray(np.einsum("...i,ij,...j->...", left, tensor, right))
    raise NotImplementedError("model evaluation is implemented on one and two axes")


@dataclass(frozen=True, eq=False)
class ClassicalSolution:
    r"""The classical ground state of a residual, read out and compared with the reference.

    Attributes
    ----------
    residual
        The assembled residual.
    state
        The SVD ground state and spectrum (:func:`reference_state`), of the physical columns:
        in the descriptor form the padding blocks, which decouple exactly, are left out.
    selection
        Which state is read out: ``"svd"``, the smallest right singular vector; or, in a doubled
        space, whose kernel is degenerate, ``"product"``, the product state
        :math:`\psi\otimes\psi` of least residual, which is the paper's own selection of "a
        degenerate state that closely matches the analytic solution" (its Section V).
    vector
        The state read out.
    target_residual
        :math:`\|R\hat b\|/\sigma_{\max}` for the normalised reference state :math:`\hat b`
        (:math:`\hat b \otimes \hat b` in a doubled space): how nearly the truncated solution
        itself satisfies the discrete equations.
    coefficients
        The model's coefficients, :attr:`~pihm.assemble.Readout.decoder` applied to the state.
    scale
        :math:`s = u(x_s)/f_q^\star(x_s)`, so that :math:`f_Q = s f_q`.
    eta
        The paper's scale factor, :math:`\eta_e = s^p` with :math:`p` the readout's power.
    eta_ideal
        The :math:`\eta` the reference's own state reads out: :math:`\|b\|^2` for its
        coefficients :math:`b`, with :math:`b` represented in the residual's
        state space (:func:`pihm.assemble.represent`; a system's every component,
        :func:`pihm.assemble.system_state`), whose derivative blocks the descriptor's
        unit state shares its norm with.
    eta_range
        The smallest and largest :math:`\eta` read out over the :data:`ETA_POINTS` set, which
        measures how far the ground state is from proportional to the solution.
    infidelity
        :math:`1 - |\langle\hat b|\psi\rangle|^2` against the normalised reference
        coefficients (the product :math:`\hat b \otimes \hat b` in a doubled space).
    field_error_linf
        :math:`\max|f_Q - f|` over the uniform :data:`GRID_POINTS` grid (the discretisation
        error :math:`\varepsilon_{\rm disc}`).
    field_error_l2
        The relative :math:`L^2` error over the same grid.
    readout_amplification
        :math:`|s|\max_x\|D^\top\tau(x)\|` over the grid, for the decoder :math:`D`: the
        largest change in :math:`f_Q(x)` per unit change of the state.
    grid
        The comparison grid, one array per axis.
    field
        The decoded physical field :math:`w\,f_Q` on the grid; a system's fields stacked along a
        first axis, in field order.
    reference
        The reference solution on the grid, stacked as :attr:`field` is.
    timings
        Seconds spent in the SVD and in the readout.
    """

    residual: Residual
    state: ReferenceState
    selection: Literal["svd", "product"]
    vector: NDArray[Any]
    target_residual: float
    coefficients: NDArray[Any]
    scale: float
    eta: float
    eta_ideal: float
    eta_range: tuple[float, float]
    infidelity: float
    field_error_linf: float
    field_error_l2: float
    readout_amplification: float
    grid: tuple[NDArray[np.float64], ...]
    field: NDArray[Any]
    reference: NDArray[Any]
    timings: dict[str, float]

    @property
    def axes(self) -> tuple[NDArray[np.float64], ...]:
        """The comparison grid's coordinates, one vector per axis."""
        return _axes(self.grid)


def _grid(d: int) -> tuple[NDArray[np.float64], ...]:
    """Return the uniform comparison grid on ``d`` axes (meshgrid, ``ij`` indexing)."""
    axis = np.linspace(-1.0, 1.0, GRID_POINTS[d])
    return tuple(np.meshgrid(*([axis] * d), indexing="ij"))


def _axes(grid: tuple[NDArray[np.float64], ...]) -> tuple[NDArray[np.float64], ...]:
    """Return a meshgrid's (``ij``) coordinates, one vector per axis."""
    return tuple(
        np.moveaxis(g, i, 0)[(slice(None),) + (0,) * (g.ndim - 1)] for i, g in enumerate(grid)
    )


def _ideal_state(residual: Residual) -> NDArray[Any]:
    r"""Return the reference solution's state in the residual's space, unnormalised.

    :func:`pihm.assemble.represent` of its coefficients :math:`b` (:math:`b` itself in the
    standard form, :math:`b` with its derivatives in the descriptor form), or a system's
    components (:func:`pihm.assemble.system_state`). Its norm squared is :math:`\eta_e`.
    """
    if residual.is_system:
        return system_state(residual)
    b = reference_solution(residual.problem).coefficients(list(residual.qubits))
    return represent(residual, b)


def reference_target(residual: Residual) -> NDArray[Any]:
    r"""Return the normalised reference state in the residual's state space.

    The state that represents the reference solution's coefficients :math:`b`
    (:func:`pihm.assemble.represent`: :math:`b` itself in the standard form, :math:`b` with its
    derivatives in the descriptor form; a system's components,
    :func:`pihm.assemble.system_state`), normalised; or in a doubled space the physical product
    state :math:`\hat b\otimes\hat b`.
    """
    if residual.nonlinearity == "doubled":
        b = reference_solution(residual.problem).coefficients(list(residual.qubits))
        b = b / np.linalg.norm(b)
        return np.kron(b, b)
    state = _ideal_state(residual)
    return np.asarray(state / np.linalg.norm(state))


def _field_functionals(residual: Residual, rows: NDArray[Any]) -> list[NDArray[Any]]:
    r"""Return, per decoded field, the functionals :math:`D_i^\top\tau(x)` for each row of ``rows``.

    ``rows`` holds :math:`\tau(x)` on the readout's registers, one point per row; the decoder's
    output holds the problem's fields one after another (:class:`pihm.assemble.Readout`).
    """
    decoder = residual.readout.decoder
    count = len(residual.problem.fields)
    size = decoder.shape[0] // count
    return [np.asarray(decoder[i * size : (i + 1) * size].T @ rows.T) for i in range(count)]


def _regular_index(residual: Residual) -> int:
    """Return the position of the regular point's field among the decoded fields."""
    return [f.name for f in residual.problem.fields].index(residual.problem.regular.field)


def _largest_readout_norm(residual: Residual) -> float:
    r"""Return :math:`\max_x\|D^\top\tau(x)\|` for the decoder :math:`D`, sampled uniformly.

    The largest values sit at the corners of the domain, which every sample grid includes.
    """
    readout = residual.readout
    d = len(readout.qubits)
    axis = np.linspace(-1.0, 1.0, 1001 if d == 1 else 33)
    mesh = [m.ravel() for m in np.meshgrid(*([axis] * d), indexing="ij")]
    rows = tau(mesh[0], readout.qubits[0])
    for coordinates, k in zip(mesh[1:], readout.qubits[1:], strict=True):
        rows = (rows[:, :, None] * tau(coordinates, k)[:, None, :]).reshape(rows.shape[0], -1)
    return max(
        float(np.max(np.linalg.norm(functionals, axis=0)))
        for functionals in _field_functionals(residual, rows)
    )


def _product_state(R: SparseArray, target: NDArray[Any], sigma_max: float) -> NDArray[Any]:
    r"""Return the product state :math:`\psi\otimes\psi` of least residual, normalised.

    The residual :math:`\|R(\psi\otimes\psi)\|^2/\|\psi\|^4` is minimised by BFGS with its
    exact gradient, from the reference's own coefficients (the paper selects the degenerate
    state "that closely matches the analytic solution") and from two answer-independent
    starts, the uniform and the constant states; the best result is kept.
    """
    size = math.isqrt(target.size)
    matrix = sp.csr_array(R) / sigma_max

    def energy(psi: NDArray[Any]) -> tuple[float, NDArray[Any]]:
        norm2 = float(psi @ psi)
        r = matrix @ np.kron(psi, psi)
        value = float(r @ r) / norm2**2
        back = np.asarray(matrix.T @ r).reshape(size, size)
        gradient = 2 * (back @ psi + back.T @ psi) / norm2**2 - 4 * value * psi / norm2
        return value, gradient

    reference = target.reshape(size, size)[:, 0] / np.linalg.norm(target.reshape(size, size)[:, 0])
    starts = [reference, np.full(size, size**-0.5), np.eye(size)[0]]
    results = [
        optimize.minimize(
            energy, x0, jac=True, method="BFGS", options={"gtol": 1e-15, "maxiter": 20000}
        )
        for x0 in starts
    ]
    best = min(results, key=lambda result: float(result.fun))
    psi = best.x / np.linalg.norm(best.x)
    return np.asarray(np.kron(psi, psi))


def _decoded_coefficients(residual: Residual, vector: NDArray[Any]) -> list[NDArray[Any]]:
    """Return the coefficients of each of the problem's fields that ``vector`` decodes to."""
    coefficients = np.asarray(residual.readout.decoder @ vector)
    return list(np.split(coefficients, len(residual.problem.fields)))


@dataclass(frozen=True, eq=False)
class DecodedField:
    r"""A state's physical field on the comparison grid, beside the reference solution.

    Attributes
    ----------
    grid
        The uniform comparison grid, one meshgrid array per axis (:data:`GRID_POINTS` points
        per axis).
    field
        The decoded physical field :math:`w\,f_Q`; a system's fields stacked along a first axis,
        in field order.
    reference
        The reference solution on the grid, stacked as :attr:`field` is.
    """

    grid: tuple[NDArray[np.float64], ...]
    field: NDArray[Any]
    reference: NDArray[Any]

    @property
    def axes(self) -> tuple[NDArray[np.float64], ...]:
        """The comparison grid's coordinates, one vector per axis."""
        return _axes(self.grid)

    @property
    def errors(self) -> tuple[float, float]:
        """The field's L-infinity and relative L2 errors against the reference."""
        error = self.field - self.reference
        linf = float(np.max(np.abs(error)))
        l2 = float(np.sqrt(np.sum(np.abs(error) ** 2) / np.sum(np.abs(self.reference) ** 2)))
        return linf, l2


def decode(residual: Residual, vector: NDArray[Any]) -> DecodedField:
    """Return ``vector``'s physical field on the comparison grid, beside the reference.

    ``vector`` is read out as :func:`solve` reads out the ground state: the residual's decoder,
    then the scale that meets the regular datum, so a global phase or norm of ``vector`` cancels.
    A system's fields share that scale.
    """
    problem = residual.problem
    readout = residual.readout
    fields = _decoded_coefficients(residual, vector)
    at = tuple(np.asarray(c) for c in problem.regular.at)
    regular = fields[_regular_index(residual)]
    scale = residual.regular_value / evaluate_model(regular, readout.qubits, *at)
    grid = _grid(len(problem.fields[0].axes))
    decoded, exact = [], []
    for field, coefficients in zip(problem.fields, fields, strict=True):
        weight = None if field.weight is None else compile_expression(field.weight, field.axes)
        model = scale * evaluate_model(coefficients, readout.qubits, *grid)
        decoded.append(model if weight is None else model * weight(*grid))
        exact.append(np.asarray(reference_solution(problem, field.name).physical(*grid)))
    if len(decoded) == 1:
        return DecodedField(grid, decoded[0], exact[0])
    return DecodedField(grid, np.stack(decoded), np.stack(exact))


def field_error(residual: Residual, vector: NDArray[Any]) -> tuple[float, float]:
    r"""Return the L∞ and relative L² errors of ``vector``'s field against the reference.

    ``vector`` is decoded as :func:`solve` decodes its ground state (the residual's readout,
    scaled by the regular datum) on the uniform :data:`GRID_POINTS` grid, so a prepared state is
    compared with the analytic solution on the same footing as the classical one.
    """
    return decode(residual, vector).errors


def lift_errors(residual: Residual, vector: NDArray[Any]) -> tuple[float, float]:
    r"""Return a Carleman build's error in two parts: the lift's own, and the time axis's.

    The exact lifted system :math:`z' = \mathbb{A}_Kz`, exponentiated on :func:`field_error`'s
    grid from its data's moments, is the lift with no time discretisation. Its first level's L∞
    distance from the reference is the lift's
    truncation error, the Carleman floor; ``vector``'s decoded field's L∞ distance from it is the
    time axis's. Their sum bounds the field's error.

    Raises
    ------
    ValueError
        For a residual that is not a Carleman lift.
    """
    if residual.nonlinearity != "carleman":
        raise ValueError("only a Carleman lift has an exact lifted system")
    problem = residual.problem
    order = residual.carleman_order
    keys = components(problem, order)
    (constraint,) = [c for c in problem.constraints if c.kind == "ratio"]
    start, _anchor = ratio_data(problem, constraint, keys, len(keys))
    decoded = decode(residual, vector)
    (times,) = decoded.grid
    lifted = expm_multiply(
        coupling(problem, order).tocsc(),
        start,
        start=times[0] - constraint.at,
        stop=times[-1] - constraint.at,
        num=times.size,
        endpoint=True,
    )
    level = residual.lift_scale * np.asarray(lifted[:, : len(problem.fields)].T)
    level = level.reshape(np.shape(decoded.reference))
    return (
        float(np.max(np.abs(level - decoded.reference))),
        float(np.max(np.abs(decoded.field - level))),
    )


@dataclass(frozen=True)
class ReadoutBound:
    r"""The terms that bound how far a state's decoded field can move from a reference's.

    For states :math:`\psi` and :math:`\psi_{\rm ref}` (unit, :math:`\psi`'s phase chosen to
    minimise :math:`d=\|\psi-\psi_{\rm ref}\|`) decoded as :func:`field_error` decodes them,
    :math:`f=w\,u` with :math:`u(x)=u_s\,D^\top\tau(x)\cdot\psi/D^\top\tau(x_s)\cdot\psi`,
    exactly

    .. math::

       u(x)-u_{\rm ref}(x) = s\,\frac{D^\top\tau(x)\cdot\delta-(u_{\rm ref}(x)/u_s)\,
       D^\top\tau(x_s)\cdot\delta}{1+D^\top\tau(x_s)\cdot\delta/D^\top\tau(x_s)\cdot
       \psi_{\rm ref}},\qquad s=\frac{u_s}{D^\top\tau(x_s)\cdot\psi_{\rm ref}},

    with :math:`\delta=\psi-\psi_{\rm ref}`, so on the comparison grid

    .. math::

       \max|f-f_{\rm ref}| \le \frac{W K (1+U/|u_s|)\,d}{1-K d/|u_s|}\quad (Kd<|u_s|),

    :math:`K=|s|\max\|D^\top\tau(x)\|`, :math:`U=\max|u_{\rm ref}|`, :math:`W=\max|w|`, every
    maximum over the grid. Not a first-order estimate: an inequality wherever it applies.

    The proof needs :math:`\|D^\top\tau(x_s)\|` at most the grid maximum in :math:`K`. That holds
    where the regular point is a grid point, and, for a single field, where it is not, since the
    functional's norm is largest at the grid's corners;
    a decoder that mixes fields could break it.

    Attributes
    ----------
    amplification
        :math:`K`.
    field_max
        :math:`U`.
    regular
        :math:`|u_s|`, the regular datum's magnitude.
    weight_max
        :math:`W` (1 without a weight).
    """

    amplification: float
    field_max: float
    regular: float
    weight_max: float

    def bound(self, deviation: float) -> float:
        """Return the bound at a state distance ``deviation``; infinite past its reach."""
        denominator = 1.0 - self.amplification * deviation / self.regular
        if denominator <= 0.0:
            return float("inf")
        scale = self.weight_max * self.amplification * (1.0 + self.field_max / self.regular)
        return scale * deviation / denominator


def readout_bound(residual: Residual, reference: NDArray[Any]) -> ReadoutBound:
    """Return :class:`ReadoutBound`'s terms for ``reference``, on :func:`field_error`'s grid."""
    problem = residual.problem
    readout = residual.readout
    grid = _grid(len(problem.fields[0].axes))
    mesh = [np.ravel(g) for g in grid]
    rows = tau(mesh[0], readout.qubits[0])
    for coordinates, k in zip(mesh[1:], readout.qubits[1:], strict=True):
        rows = (rows[:, :, None] * tau(coordinates, k)[:, None, :]).reshape(rows.shape[0], -1)
    functionals = _field_functionals(residual, rows)
    at = [np.atleast_1d(np.asarray(c, dtype=np.float64)) for c in problem.regular.at]
    anchor = tau(at[0], readout.qubits[0])
    for coordinates, k in zip(at[1:], readout.qubits[1:], strict=True):
        anchor = (anchor[:, :, None] * tau(coordinates, k)[:, None, :]).reshape(1, -1)
    anchor_functional = _field_functionals(residual, anchor)[_regular_index(residual)][:, 0]
    regular = abs(residual.regular_value)
    s = residual.regular_value / complex(anchor_functional @ reference)
    values = [s * (field.T @ reference) for field in functionals]
    weight = 1.0
    for field in problem.fields:
        if field.weight is not None:
            weight = float(np.max(np.abs(compile_expression(field.weight, field.axes)(*grid))))
    return ReadoutBound(
        amplification=abs(s) * max(float(np.max(np.linalg.norm(f, axis=0))) for f in functionals),
        field_max=max(float(np.max(np.abs(v))) for v in values),
        regular=regular,
        weight_max=weight,
    )


def physical_reference(residual: Residual, **options: Any) -> ReferenceState:
    """Return the reference state of the physical subspace, its vector on the whole state.

    The padding blocks of a descriptor residual decouple exactly: the physical
    columns hold the spectrum the filter sees, and the ground state's padding is zero. Without
    padding this is the whole residual's. Up to
    :data:`~pihm.classical.iterative.DENSE_COLUMNS` physical columns the
    reference is the dense SVD (:func:`reference_state`), and past them the iterative one
    (:func:`iterative_reference_state`); the state's ``driver`` says which.

    Parameters
    ----------
    residual
        An assembled residual.
    options
        Passed to the reference (``min_gap_ratio``, ``max_kernel``).
    """
    R = residual.matrix()
    columns = residual.physical_columns
    physical = R if columns.size == residual.dimension else sp.csr_array(R[:, columns])
    compute = reference_state if columns.size <= DENSE_COLUMNS else iterative_reference_state
    state = compute(physical, **options)
    if columns.size == residual.dimension:
        return state
    vector = np.zeros(residual.dimension, dtype=state.vector.dtype)
    vector[columns] = state.vector
    return replace(state, vector=vector)


def solve(residual: Residual, **options: Any) -> ClassicalSolution:
    r"""Solve a residual classically and read the ground state out as the paper does.

    Parameters
    ----------
    residual
        An assembled residual (:func:`pihm.assemble.assemble`).
    options
        Passed to :func:`reference_state` (``min_gap_ratio``, ``max_kernel``).

    Returns
    -------
    ClassicalSolution
        The ground state, its scale and its errors against the reference.
    """
    start = time.perf_counter()
    R = residual.matrix()
    state = physical_reference(residual, **options)
    ground = state.vector
    solved = time.perf_counter()
    problem = residual.problem
    readout = residual.readout
    target = reference_target(residual)
    sigma_max = float(state.singular_values[-1])
    if residual.nonlinearity == "doubled":
        vector = _product_state(R, target, sigma_max)
        selection: Literal["svd", "product"] = "product"
    else:
        vector, selection = ground, "svd"
    fields = _decoded_coefficients(residual, vector)
    at = tuple(np.asarray(c) for c in problem.regular.at)
    regular = fields[_regular_index(residual)]
    scale = float(residual.regular_value / evaluate_model(regular, readout.qubits, *at).real)
    decoded = decode(residual, vector)
    linf, l2 = decoded.errors

    points = np.linspace(-0.9, 0.9, ETA_POINTS)
    mesh = [
        m.ravel() for m in np.meshgrid(*([points] * len(problem.fields[0].axes)), indexing="ij")
    ]
    etas = []
    for field, field_coefficients in zip(problem.fields, fields, strict=True):
        values = np.asarray(reference_solution(problem, field.name).field(*mesh))
        keep = np.abs(values) >= 1e-2 * np.max(np.abs(values))
        model = evaluate_model(field_coefficients, readout.qubits, *(m[keep] for m in mesh))
        etas.append((values[keep] / model).real ** readout.power)
    eta_values = np.concatenate(etas)

    if residual.nonlinearity == "doubled":
        b_ideal = reference_solution(problem).coefficients(list(residual.qubits))
    else:
        b_ideal = _ideal_state(residual)
    finished = time.perf_counter()
    return ClassicalSolution(
        residual=residual,
        state=state,
        selection=selection,
        vector=vector,
        target_residual=float(np.linalg.norm(R @ target)) / sigma_max,
        coefficients=np.asarray(readout.decoder @ vector),
        scale=scale,
        eta=float(scale**readout.power),
        eta_ideal=float(np.vdot(b_ideal, b_ideal).real),
        eta_range=(float(np.min(eta_values)), float(np.max(eta_values))),
        infidelity=float(max(0.0, 1.0 - abs(np.vdot(target, vector)) ** 2)),
        field_error_linf=linf,
        field_error_l2=l2,
        readout_amplification=abs(scale) * _largest_readout_norm(residual),
        grid=decoded.grid,
        field=decoded.field,
        reference=decoded.reference,
        timings={"svd": solved - start, "readout": finished - solved},
    )
