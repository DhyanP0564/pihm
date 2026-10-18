r"""Reference solutions: values, derivatives, coefficients, and checks against the problem.

A problem's :class:`~pihm.problems.spec.Reference` is realised by :func:`reference_solution`
as a :class:`ReferenceSolution`. Closed forms are evaluated exactly; a ``"bvp"`` reference is
solved by :func:`scipy.integrate.solve_bvp` from the problem's own equation, and an ``"ivp"``
reference, a first-order system's initial-value problem, by :func:`scipy.integrate.solve_ivp`.
Derivatives come from a high-resolution Chebyshev series of the solution (:data:`SERIES_QUBITS`
qubits per axis), which is accurate to about :math:`10^{-10}` for the smooth solutions of the
ladder.

Every reference is checked *numerically* against its problem (:func:`check_reference`): the residual
of
each equation, the stated data and the constraints, all evaluated pointwise from the spectral
derivatives. This needs no extra dependency, and it catches a wrong transcription on either
side, which is what the symbolic check was for (the paper's Laplace typo, fails it).

Notation: a field :math:`u` with a readout weight :math:`w` describes the *physical* solution
:math:`f = w\,u` (:class:`~pihm.problems.spec.Field`). References and data are stated for
:math:`f`; equations, constraints and the regular value are for :math:`u`. A problem with several
fields (a first-order system) has one reference per field, :func:`reference_solution`'s
``field``.
"""

from __future__ import annotations

import functools
import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Final

import numpy as np
from numpy.polynomial import chebyshev as npcheb
from numpy.typing import ArrayLike, NDArray
from scipy import integrate, optimize

from pihm import basis
from pihm.problems.expressions import compile_expression
from pihm.problems.functions import term_values
from pihm.problems.polynomial import jacobian, slope, vector_field
from pihm.problems.spec import Constraint, Factor, Field, Problem, Term

__all__ = [
    "SERIES_QUBITS",
    "ReferenceCheck",
    "ReferenceSolution",
    "check_reference",
    "exact_root",
    "reference_solution",
    "regular_value",
    "slice_data",
]

#: Qubits per axis of the high-resolution series behind derivatives, by number of axes.
SERIES_QUBITS: Final[dict[int, int]] = {1: 10, 2: 7, 3: 5}

#: A series has converged when its last quarter of coefficients is below this, relative to its
#: largest; it is then cut where the coefficients reach :data:`_CHOP` (noise), which keeps
#: derivatives accurate. Solutions with a branch point (the associated Legendre functions)
#: never converge, and are differentiated by finite differences instead.
_CONVERGED: Final[float] = 1.0e-13
_CHOP: Final[float] = 1.0e-15

#: The step of the finite differences used where the series has not converged.
_STEP: Final[float] = 1.0e-3


def _to_standard(coefficients: NDArray[Any], ns: list[int]) -> NDArray[Any]:
    """Reshape normalised coefficients to a standard Chebyshev tensor, one index per axis."""
    tensor = coefficients.reshape([1 << n for n in ns])
    for axis, n in enumerate(ns):
        shape = [1] * len(ns)
        shape[axis] = 1 << n
        tensor = tensor / basis.chebyshev_scale(n).reshape(shape)
    return tensor


def _chop(series: NDArray[Any]) -> tuple[NDArray[Any], bool]:
    """Cut a Chebyshev tensor where it reaches rounding; report whether it had converged."""
    magnitude = np.abs(series)
    peak = float(magnitude.max()) or 1.0
    converged = True
    for axis in range(series.ndim):
        others = tuple(a for a in range(series.ndim) if a != axis)
        envelope = magnitude.max(axis=others) if others else magnitude
        tail = envelope[3 * envelope.size // 4 :]
        converged = converged and bool(tail.max(initial=0.0) < _CONVERGED * peak)
        significant = np.flatnonzero(envelope > _CHOP * peak)
        keep = min(envelope.size, int(significant[-1]) + 3) if significant.size else 1
        series = np.take(series, np.arange(keep), axis=axis)
        magnitude = np.take(magnitude, np.arange(keep), axis=axis)
    return series, converged


def _evaluate_series(series: NDArray[Any], coordinates: tuple[NDArray[Any], ...]) -> NDArray[Any]:
    """Evaluate a standard Chebyshev tensor at scattered points (one array per axis)."""
    if len(coordinates) == 1:
        return np.asarray(npcheb.chebval(coordinates[0], series))
    if len(coordinates) == 2:
        return np.asarray(npcheb.chebval2d(coordinates[0], coordinates[1], series))
    return np.asarray(npcheb.chebval3d(coordinates[0], coordinates[1], coordinates[2], series))


def _finite_difference(f: Callable[..., NDArray[Any]], order: int, x: NDArray[Any]) -> NDArray[Any]:
    r"""Differentiate a one-axis function by Richardson-extrapolated central differences.

    The step shrinks near :math:`\pm1` so that every sample stays in the interval; the error is
    :math:`O(h^4)` with :math:`h \le` :data:`_STEP`.
    """
    h = np.minimum(_STEP, (1.0 - np.abs(x)) / 2.5)

    def central(step: NDArray[Any]) -> NDArray[Any]:
        if order == 1:
            return np.asarray((f(x + step) - f(x - step)) / (2 * step))
        return np.asarray((f(x + step) - 2 * f(x) + f(x - step)) / step**2)

    if order not in (1, 2):
        raise ValueError(f"finite differences are implemented for orders 1 and 2, got {order}")
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.asarray((4 * central(h / 2) - central(h)) / 3)


@dataclass(frozen=True, eq=False)
class ReferenceSolution:
    r"""A realised reference solution of one field of a problem.

    Attributes
    ----------
    problem
        The problem.
    physical
        The physical solution :math:`f`, vectorised over one array per axis.
    field
        The solved field :math:`u = f/w` (equal to :attr:`physical` without a weight).
    series
        Standard Chebyshev coefficients of :math:`u`, one index per axis, cut where they reach
        rounding.
    physical_series
        The same for :math:`f`.
    smooth
        Whether :attr:`series` converged. If not (a branch point), derivatives of :math:`u`
        come from finite differences of the exact solution, on one axis only.
    physical_smooth
        The same for :attr:`physical_series`.
    """

    problem: Problem
    physical: Callable[..., NDArray[Any]]
    field: Callable[..., NDArray[Any]]
    series: NDArray[np.float64]
    physical_series: NDArray[np.float64]
    smooth: bool
    physical_smooth: bool

    def derivative(self, orders: tuple[int, ...], *, physical: bool = False) -> Callable[..., Any]:
        r"""Return the derivative :math:`\partial^\alpha u` (or of :math:`f`) as a function."""
        exact = self.physical if physical else self.field
        if not any(orders):
            return exact
        if not (self.physical_smooth if physical else self.smooth):
            if len(orders) != 1:
                raise ValueError("finite-difference derivatives are implemented on one axis")
            return lambda x: _finite_difference(exact, orders[0], np.asarray(x, dtype=np.float64))
        tensor: NDArray[Any] = self.physical_series if physical else self.series
        for axis, order in enumerate(orders):
            if order:
                tensor = npcheb.chebder(tensor, m=order, axis=axis)

        def evaluate(*coordinates: ArrayLike) -> NDArray[Any]:
            arrays = tuple(np.asarray(c, dtype=np.float64) for c in coordinates)
            return _evaluate_series(tensor, tuple(np.broadcast_arrays(*arrays)))

        return evaluate

    def coefficients(self, ns: int | list[int] | tuple[int, ...]) -> NDArray[Any]:
        r"""Return the normalised coefficients of :math:`u`, truncated to registers ``ns``.

        This is the vector :math:`b` of the paper's ideal scale factor
        :math:`\eta_e = \|b\|^2` (:func:`pihm.basis.scale_factor`).
        """
        counts = [ns] * len(self.problem.fields[0].axes) if isinstance(ns, int) else list(ns)
        if len(counts) > 1:
            points: list[int] | None = [max(8 << n, 512) for n in counts]
        else:
            # A branch point (4c, 4d) aliases at the default 64N nodes: use 2^17.
            points = None if self.smooth else [max(1 << 17, 64 << counts[0])]
        return basis.fit(self.field, counts, points=points)


def _weight(field: Field) -> Callable[..., NDArray[Any]] | None:
    """Return the field's readout weight as a function, or ``None``."""
    return None if field.weight is None else compile_expression(field.weight, field.axes)


def _constant(expression: str) -> float:
    """Evaluate a datum's constant expression, such as ``"-1/8"``."""
    return float(compile_expression(expression, [])())


def _bvp_solution(problem: Problem) -> Callable[..., NDArray[Any]]:
    """Solve the problem's second-order boundary-value problem with ``solve_bvp``.

    The equation must contain :math:`f''` in exactly one term, alone and with a constant
    coefficient; the data must be two values, at :math:`x = -1` and :math:`x = 1`.
    """
    (equation,) = problem.equations
    top = [t for t in equation.terms if any(f.derivative == (2,) for f in t.factors)]
    if len(top) != 1 or top[0].degree != 1 or top[0].multiplier is not None:
        raise ValueError("a boundary-value reference needs f'' alone in exactly one term")
    if problem.order != 2:
        raise ValueError("a boundary-value reference is implemented for second order only")
    rest = [t for t in equation.terms if t is not top[0]]
    ends = {d.at: _constant(d.value) for d in problem.data if d.derivative == 0}
    if set(ends) != {-1.0, 1.0}:
        raise ValueError("a boundary-value reference needs the values at x = -1 and x = 1")
    axis = problem.axes[0].name

    def rhs(x: NDArray[Any], y: NDArray[Any]) -> NDArray[Any]:
        values = {0: y[0], 1: y[1]}

        def factor(item: Factor) -> NDArray[Any]:
            return np.asarray(values[item.derivative[0]])

        others = sum((term_values(t, factor, {axis: x}) for t in rest), np.zeros_like(x))
        return np.vstack([y[1], -others / top[0].coefficient])

    def boundary(left: NDArray[Any], right: NDArray[Any]) -> NDArray[Any]:
        return np.array([left[0] - ends[-1.0], right[0] - ends[1.0]])

    mesh = np.linspace(-1.0, 1.0, 201)
    guess = np.vstack([np.interp(mesh, [-1.0, 1.0], [ends[-1.0], ends[1.0]]), np.zeros_like(mesh)])
    guess[1] = (ends[1.0] - ends[-1.0]) / 2
    solution = integrate.solve_bvp(
        rhs, boundary, mesh, guess, tol=problem.reference.tolerance, max_nodes=1_000_000
    )
    if solution.status != 0:
        raise RuntimeError(f"solve_bvp failed for {problem.key}: {solution.message}")
    # The solver's interpolant is a C1 spline, accurate to the tolerance. Its Chebyshev
    # interpolant, cut where the coefficients fall to that tolerance, is the smooth solution to
    # the same accuracy, without the spline's kinks, and so it differentiates spectrally.
    standard = npcheb.chebinterpolate(lambda x: solution.sol(x)[0], 127)
    significant = np.flatnonzero(
        np.abs(standard) > problem.reference.tolerance * np.abs(standard).max()
    )
    standard = standard[: int(significant[-1]) + 1]

    def evaluate(x: ArrayLike) -> NDArray[Any]:
        return np.asarray(npcheb.chebval(np.asarray(x, dtype=np.float64), standard))

    return evaluate


@functools.lru_cache(maxsize=64)
def _ivp_solutions(problem: Problem) -> dict[str, Callable[..., NDArray[Any]]]:
    r"""Solve a first-order system's initial-value problem (``solve_ivp``, Radau), per field.

    The implicit Radau method, since a Galerkin system's decay rates span its modes' (R9's
    :math:`\nu k^2` to :math:`\nu M^2`): an explicit method's step control leaves the stiff modes
    orders of magnitude above the tolerance. The absolute tolerance is relative to the data's size,
    so that a small amplitude is held to the same relative accuracy.

    Equation :math:`i` must hold its field's first derivative once, with a constant coefficient,
    and every field needs its value stated at one end of the axis, where the solve starts. Each
    field's solution is then fitted by its Chebyshev interpolant, of a degree raised until its
    last quarter falls below the tolerance relative to the largest field, and cut there, as a
    boundary-value reference is (:func:`_bvp_solution`).

    A polynomial system (:func:`~pihm.problems.polynomial.vector_field`), every Galerkin system
    among them, is integrated from its sparse vector field with its exact Jacobian
    (:func:`~pihm.problems.polynomial.slope`, :func:`~pihm.problems.polynomial.jacobian`):
    R10's largest truncation has 56,264 terms, which one Python loop per evaluation, and a
    Jacobian by differences, would take hours to integrate. Any other system's terms are
    evaluated one by one.
    """
    (axis,) = [a.name for a in problem.axes]
    names = [field.name for field in problem.fields]
    stated = {(d.field, d.at): _constant(d.value) for d in problem.data if not d.derivative}
    points = {at for _, at in stated}
    if len(points) != 1 or points - {-1.0, 1.0}:
        raise ValueError("an initial-value reference needs every field's value at one end")
    (start,) = points
    if {field for field, _ in stated} != set(names):
        raise ValueError("an initial-value reference needs every field's value stated")
    tops: list[tuple[int, float, list[Term]]] = []
    for equation in problem.equations:
        top = [t for t in equation.terms if any(any(f.derivative) for f in t.factors)]
        if len(top) != 1 or top[0].degree != 1 or top[0].factors[0].derivative != (1,):
            raise ValueError("an initial-value reference needs each equation's first derivative")
        leading = top[0].coefficient
        if top[0].multiplier is not None:
            if not top[0].multiplier.is_constant:
                raise ValueError("an initial-value reference needs constant leading coefficients")
            leading *= top[0].multiplier.coefficients[0]
        rest = [t for t in equation.terms if t is not top[0]]
        tops.append((names.index(top[0].factors[0].field), leading, rest))

    def by_term(x: float, y: NDArray[Any]) -> NDArray[Any]:
        def factor(item: Factor) -> NDArray[Any]:
            return np.asarray(y[names.index(item.field)])

        slopes = np.zeros(len(names))
        for i, leading, rest in tops:
            coordinates = {axis: np.asarray(x)}
            slopes[i] = -sum(float(term_values(t, factor, coordinates)) for t in rest) / leading
        return slopes

    rhs: Callable[[float, NDArray[Any]], NDArray[Any]] = by_term
    jac: Callable[[float, NDArray[Any]], Any] | None = None
    try:
        field = vector_field(problem)
    except NotImplementedError:
        pass
    else:
        # A polynomial system is autonomous: time enters only through its fields.
        def by_field(_x: float, y: NDArray[Any]) -> NDArray[Any]:
            return slope(field, y)

        def by_field_jacobian(_x: float, y: NDArray[Any]) -> Any:
            return jacobian(field, y)

        rhs, jac = by_field, by_field_jacobian

    y0 = np.array([stated[(name, start)] for name in names])
    tolerance = problem.reference.tolerance
    solution = integrate.solve_ivp(
        rhs,
        (start, -start),
        y0,
        method="Radau",
        jac=jac,
        rtol=tolerance,
        atol=tolerance * (float(np.max(np.abs(y0))) or 1.0),
        dense_output=True,
    )
    if not solution.success:
        raise RuntimeError(f"solve_ivp failed for {problem.key}: {solution.message}")
    dense = solution.sol
    assert dense is not None  # dense_output=True
    fitted: dict[str, Callable[..., NDArray[Any]]] = {}
    peak = float(np.max(np.abs(dense(np.linspace(-1.0, 1.0, 257)))))
    for i, name in enumerate(names):

        def component(x: Any, i: int = i) -> Any:
            return dense(x)[i]

        for degree in (127, 255, 511, 1023, 2047):
            standard = npcheb.chebinterpolate(component, degree)
            if np.max(np.abs(standard[-(degree // 4) :])) <= tolerance * peak:
                break
        significant = np.flatnonzero(np.abs(standard) > tolerance * peak)
        series = standard[: int(significant[-1]) + 1] if significant.size else standard[:1]

        def evaluate(x: ArrayLike, series: NDArray[Any] = series) -> NDArray[Any]:
            return np.asarray(npcheb.chebval(np.asarray(x, dtype=np.float64), series))

        fitted[name] = evaluate
    return fitted


@functools.lru_cache(maxsize=256)
def reference_solution(problem: Problem, field: str | None = None) -> ReferenceSolution:
    """Realise the reference solution of one field of ``problem`` (cached; problems are hashable).

    Parameters
    ----------
    problem
        The problem.
    field
        The field's name; by default the problem's first (its only one, but for a system).

    Raises
    ------
    ValueError
        For a field the problem does not have.
    """
    names = [axis.name for axis in problem.axes]
    fields = [entry.name for entry in problem.fields]
    chosen = fields[0] if field is None else field
    if chosen not in fields:
        raise ValueError(f"{problem.key} has no field {chosen!r}")
    index = fields.index(chosen)
    entry = problem.fields[index]
    if problem.reference.kind == "expression":
        expressions = problem.reference.expressions or (problem.reference.expression,)
        expression = expressions[index]
        assert expression is not None
        compiled = compile_expression(expression, names)
        indices = [names.index(axis) for axis in entry.axes]

        def physical(*coordinates: ArrayLike) -> NDArray[Any]:
            full: list[ArrayLike] = [0.0] * len(names)
            for index, value in zip(indices, coordinates, strict=True):
                full[index] = value
            return compiled(*full)

    elif problem.reference.kind == "ivp":
        physical = _ivp_solutions(problem)[chosen]
    else:
        physical = _bvp_solution(problem)
    weight = _weight(entry)

    def solved(*coordinates: ArrayLike) -> NDArray[Any]:
        values = physical(*coordinates)
        return values if weight is None else values / weight(*coordinates)

    d = len(entry.axes)
    ns = [SERIES_QUBITS[d]] * d
    points = None if d == 1 else [4 << n for n in ns]
    series, smooth = _chop(_to_standard(basis.fit(solved, ns, points=points), ns))
    if weight is None:
        physical_series, physical_smooth = series, smooth
    else:
        physical_series, physical_smooth = _chop(
            _to_standard(basis.fit(physical, ns, points=points), ns)
        )
    return ReferenceSolution(
        problem, physical, solved, series, physical_series, smooth, physical_smooth
    )


def regular_value(problem: Problem) -> float:
    r"""Return the solved field's value :math:`u(x_s)` at the regular point.

    A datum whose slice contains the regular point is used when there is one (it is the known
    value the paper's :math:`\mathcal{DC}_R` assumes); otherwise the reference solution is
    evaluated there.

    Raises
    ------
    ValueError
        If the value is zero, which a regular point must not be.
    """
    regular = problem.regular
    field = next(entry for entry in problem.fields if entry.name == regular.field)
    axes = field.axes
    value: float | None = None
    for datum in problem.data:
        if datum.field != regular.field:
            continue
        index = axes.index(datum.axis)
        if not datum.derivative and datum.at == regular.at[index]:
            others = [a for a in axes if a != datum.axis]
            rest = [c for i, c in enumerate(regular.at) if i != index]
            value = float(compile_expression(datum.value, others)(*rest))
            weight = _weight(field)
            if weight is not None:
                value /= float(weight(*regular.at))
            break
    if value is None:
        value = float(reference_solution(problem, regular.field).field(*regular.at))
    if value == 0.0 or not math.isfinite(value):
        raise ValueError(f"the regular point {regular.at} of {problem.key} has value {value}")
    return value


def slice_data(problem: Problem, axis: str, at: float) -> NDArray[np.float64]:
    r"""Return every field's stated value at the point ``axis = at``, in the problem's field order.

    This is the data vector :math:`y` of a ``"ratio"`` constraint (one axis, no readout weight).

    Raises
    ------
    ValueError
        If a field has no stated value there, or a field has several axes.
    """
    values = []
    for field in problem.fields:
        if field.axes != (axis,):
            raise ValueError("ratio data are values at a point: one axis per field")
        found = [
            d
            for d in problem.data
            if d.field == field.name and d.axis == axis and d.at == at and not d.derivative
        ]
        if not found:
            raise ValueError(f"{problem.key} states no value of {field.name!r} at {axis} = {at}")
        values.append(_constant(found[0].value))
    return np.asarray(values)


def exact_root(problem: Problem, near: float, derivative: int = 0, width: float = 0.05) -> float:
    """Return the root of :math:`u^{(j)}` nearest ``near`` (one axis), for corrected variants.

    The root is bracketed within ``near ± width`` and refined by Brent's method to machine
    precision, from the high-resolution series of the reference.

    Raises
    ------
    ValueError
        If there is no sign change in the bracket.
    """
    if len(problem.fields[0].axes) != 1:
        raise ValueError("exact roots are defined for one axis")
    g = reference_solution(problem).derivative((derivative,))
    grid = np.linspace(max(-1.0, near - width), min(1.0, near + width), 2001)
    values = g(grid)
    changes = np.flatnonzero(np.sign(values[:-1]) * np.sign(values[1:]) <= 0)
    if changes.size == 0:
        raise ValueError(f"no root of the derivative of order {derivative} near {near}")
    best = changes[np.argmin(np.abs(grid[changes] - near))]
    root = optimize.brentq(lambda x: float(g(x)), grid[best], grid[best + 1], xtol=1e-15)
    return float(root)


@dataclass(frozen=True)
class ReferenceCheck:
    """How well the reference solution satisfies its problem (all relative, see below).

    Attributes
    ----------
    equation
        The largest pointwise equation residual, divided by the largest term magnitude.
    data
        The largest error in the stated data, divided by the solution's maximum.
    constraints
        The largest residual of each constraint row's condition, divided by the solution's
        maximum, in the order of the problem's constraints.
    """

    equation: float
    data: float
    constraints: tuple[float, ...]


def _points(count: int, d: int) -> tuple[NDArray[np.float64], ...]:
    """Return Chebyshev–Gauss points (all interior) on ``d`` axes, flattened as a grid."""
    nodes = np.cos(np.pi * (2 * np.arange(count) + 1) / (2 * count))
    grids = np.meshgrid(*([nodes] * d), indexing="ij")
    return tuple(grid.ravel() for grid in grids)


def check_reference(problem: Problem, points: int = 257) -> ReferenceCheck:
    """Check the reference against the problem's equation, data and constraints.

    Parameters
    ----------
    problem
        A problem; a system's references are checked together, each equation with every field's
        reference.
    points
        Chebyshev–Gauss points per axis for the equation residual (fewer are used on several
        axes).

    Returns
    -------
    ReferenceCheck
        Relative residuals. A correct transcription gives about :math:`10^{-9}` or less for the
        equation, rounding for the data, and rounding for every constraint whose position is an
        exact root (the rounded positions of the faithful variants do not vanish).
        The sizes they are relative to are the largest over the fields, and a system's equations
        are relative to their largest term, so that a field far smaller than the others is not
        held to its own size.
    """
    references = {field.name: reference_solution(problem, field.name) for field in problem.fields}
    axes = problem.fields[0].axes
    d = len(axes)
    grid = _points(points if d == 1 else 65, d)
    coordinates = dict(zip(axes, grid, strict=True))

    def factor_values(item: Factor) -> NDArray[Any]:
        return np.asarray(references[item.field].derivative(item.derivative)(*grid))

    residuals, scales = [], []
    for entry in problem.equations:
        values = [term_values(term, factor_values, coordinates) for term in entry.terms]
        scales.append(max(float(np.max(np.abs(v))) for v in values))
        residuals.append(float(np.max(np.abs(np.sum(values, axis=0)))))
    equation = max(residuals) / max(scales)

    size = max(float(np.max(np.abs(r.field(*grid)))) for r in references.values())
    physical_size = max(float(np.max(np.abs(r.physical(*grid)))) for r in references.values())
    data_error = max(
        (
            _slice_residual(references[d.field], d.axis, d.at, d.derivative, d.value, physical=True)
            for d in problem.data
        ),
        default=0.0,
    )
    residuals = [
        _ratio_residual(problem, c, references)
        if c.kind == "ratio"
        else _slice_residual(
            references[c.field], c.axis, c.at, c.derivative, c.datum, physical=False
        )
        for c in problem.constraints
    ]
    data_error /= physical_size
    return ReferenceCheck(equation, data_error, tuple(r / size for r in residuals))


def _ratio_residual(
    problem: Problem, constraint: Constraint, references: dict[str, ReferenceSolution]
) -> float:
    r"""Return :math:`\max_c|y_a u_c(x^*) - y_c u_a(x^*)|/|y_a|` for a ratio constraint."""
    data = slice_data(problem, constraint.axis, constraint.at)
    values = np.array([float(references[f.name].field(constraint.at)) for f in problem.fields])
    a = [f.name for f in problem.fields].index(constraint.field)
    return float(np.max(np.abs(data[a] * values - data * values[a]))) / abs(float(data[a]))


def _slice_residual(
    reference: ReferenceSolution,
    axis: str,
    at: float,
    derivative: int,
    function: str | None,
    *,
    physical: bool,
) -> float:
    r"""Return the largest :math:`|\partial_a^j u - g|` on the slice :math:`x_a = x^*`."""
    axes = reference.problem.fields[0].axes
    index = axes.index(axis)
    grids = [np.linspace(-1.0, 1.0, 129) for _ in axes]
    grids[index] = np.array([at])
    mesh = [m.ravel() for m in np.meshgrid(*grids, indexing="ij")]
    if derivative == 0:
        exact = reference.physical if physical else reference.field
        values = np.asarray(exact(*mesh))
    else:
        orders = tuple(derivative if i == index else 0 for i in range(len(axes)))
        values = np.asarray(reference.derivative(orders, physical=physical)(*mesh))
    if function is not None:
        others = [a for a in axes if a != axis]
        values = values - compile_expression(function, others)(
            *[mesh[axes.index(o)] for o in others]
        )
    return float(np.max(np.abs(values)))
