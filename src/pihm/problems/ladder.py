r"""The ladder of differential equations, R1 to R10, with the paper's panels as data.

Rungs R2 to R6 are the panels of Wu et al.'s Figs 3 to 7 :cite:`wu2025pihm`,
transcribed from the rendered pages, never from extracted text. A panel is the
paper's problem with its defects corrected; the reproduction of the paper also runs the panel as
printed, its faithful variant:

``"faithful"``
    The paper's own choices, exactly as printed: its rounded constraint points (Fig. 5's
    :math:`x_m = -0.2`, :math:`x_z = -0.73`, :math:`0.91`, :math:`x_m = -0.75`; Fig. 7c's
    :math:`x_z = 0.02615`), its hand-picked PDE invariants, and its Maclaurin sources
    (:math:`\bar p = 3, 5, 7`).
``"corrected"``, the default
    Our corrections: exact roots of the reference solution, Chebyshev series for the sources,
    datum-slice rows for the PDEs' initial and boundary *functions* (the heat equation's at
    :math:`t = -1`, where it runs forward over the whole interval), and for the
    associated-Legendre panels the substitution :math:`f = \sqrt{1-x^2}\,v`, whose equation
    :math:`(1-x^2)v'' - 4xv' + [l(l+1) - 2]v = 0` has polynomial coefficients and a polynomial
    solution. Where the paper's choice is already exact (all of Fig. 3, Figs 4a and 4b,
    Fig. 7b) the corrected variant equals the faithful one.

R1 and R7 to R10 are not the paper's, and have no faithful variant: R7 is the
coupled oscillator, R8 a logistic-type equation for the Carleman lift, R9 viscous Burgers on
its sine modes and R10 two-dimensional Navier–Stokes on its Fourier modes,
all first-order systems whose data enter as ratio rows. R9's problems are named for their
nonlinearity strength and number of modes, ``"R9c-M8"`` for :math:`a/\nu = 1` on eight modes
(:data:`R9_STRENGTHS`, :data:`R9_MODES`), and R10's likewise, ``"R10a"`` the Taylor–Green vortex
and ``"R10b"`` to ``"R10d"`` the vortex pair at rising strength, with ``"-W8"`` for the window
:math:`T = 1/8` (:data:`R10_STRENGTHS`, :data:`R10_MODES`, :data:`R10_WINDOWS`). What the paper
printed for each panel (:math:`n` and :math:`\eta_e`) is in :data:`PANELS`, apart from the
mathematics.

**Choices the paper leaves open**:

* the regular point :math:`x_s` is not printed for Figs 3 to 6. We use the point of the stated
  data where the solution is non-zero (:math:`x = 0` for Figs 3b, 3d and 5; :math:`x = 1` for
  Figs 4a and 4b), otherwise :math:`x = 0.5` (Figs 3c, 4c, 4d), and for the PDEs a point on the
  initial or top-boundary slice where the stated function equals 1 (for the corrected heat
  equation, :math:`(t, x) = (-1, 1/4)` on its datum slice);
* the paper's :math:`\mathcal{T}(\mathbb{B}_n(\pm1) \otimes \mathbb{1}_n)` is read as two rows,
  one at each sign.

Examples
--------
>>> p = problem("R4b", "faithful")
>>> p.description
"Fig. 5b: f'' + 4f' + 4f = e^{-2x}, BC {f(0), f'(0)} = {-1, 1}, DC_I f(x_z = -0.73) = 0, p = 3"
>>> [round(c.at, 12) for c in problem("R4b").constraints]  # the root 1 - sqrt(3)
[-0.732050807569]
"""

from __future__ import annotations

import dataclasses
import functools
import math
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Final

from pihm.problems.fluids import (
    Galerkin,
    burgers,
    burgers_galerkin,
    taylor_green,
    vortex_pair,
    vorticity_galerkin,
)
from pihm.problems.spec import (
    Axis,
    Constraint,
    Datum,
    Equation,
    Factor,
    Field,
    Polynomial,
    Problem,
    Reference,
    Regular,
    Root,
    Source,
    Term,
    Variant,
)

__all__ = [
    "OMEGA",
    "PANELS",
    "R8_TIME",
    "R9_MODES",
    "R9_STRENGTHS",
    "R10_FIELD_SCALE",
    "R10_MODES",
    "R10_PANEL_MODES",
    "R10_STRENGTHS",
    "R10_WINDOWS",
    "RUNGS",
    "Panel",
    "burgers_panel",
    "galerkin",
    "names",
    "navier_stokes_panel",
    "problem",
    "rung",
]


@dataclass(frozen=True)
class Panel:
    r"""What the paper printed for one panel.

    Attributes
    ----------
    rung
        The rung identifier, such as ``"R2a"``.
    figure
        The paper's figure and panel, such as ``"3b"``.
    n
        The printed number of qubits (per axis).
    eta
        The printed scale factor :math:`\eta_e`.
    decimals
        How many decimals :attr:`eta` is printed to.
    """

    rung: str
    figure: str
    n: int
    eta: float
    decimals: int


#: The paper's printed values, by rung (Figs 3 to 7, read from the rendered pages).
PANELS: Final[Mapping[str, Panel]] = {
    panel.rung: panel
    for panel in (
        Panel("R2a", "3b", 3, 1.29, 2),
        Panel("R2b", "3c", 3, 32.47, 2),
        Panel("R2c", "3d", 5, 451.71, 2),
        Panel("R3a", "4a", 3, 1.75, 2),
        Panel("R3b", "4b", 3, 1.49, 2),
        Panel("R3c", "4c", 5, 108.51, 2),
        Panel("R3d", "4d", 5, 135.38, 2),
        Panel("R4a", "5a", 3, 1.35, 2),
        Panel("R4b", "5b", 3, 14.74, 2),
        Panel("R4c", "5c", 4, 0.82, 2),
        Panel("R4d", "5d", 4, 77.51, 2),
        Panel("R5a", "6b", 3, 5.21559, 5),
        Panel("R5b", "6d", 4, 603.863, 3),
        Panel("R5c", "6f", 5, 216.469, 3),
        Panel("R6a", "7b", 2, 3.75367, 5),
        Panel("R6b", "7c", 2, 0.07984, 5),
    )
}

# ---------------------------------------------------------------------------- shorthands

_X: Final = (Axis("x"),)


def _d(order: int) -> Factor:
    """Return the factor :math:`f^{(j)}` of the one-axis field ``f``."""
    return Factor("f", (order,))


def _linear(coefficient: float, order: int, multiplier: tuple[float, ...] | None = None) -> Term:
    r"""Return :math:`c\,m(x) f^{(j)}`, with ``multiplier`` as monomial coefficients."""
    polynomial = None if multiplier is None else Polynomial("x", multiplier)
    return Term(coefficient, (_d(order),), polynomial)


def _zero(at: float, derivative: int = 0, axis: str = "x") -> Constraint:
    r"""Return the invariant constraint :math:`\partial^j f = 0` on the slice ``axis = at``."""
    return Constraint("zero", "f", axis, at, derivative)


def _data(*conditions: tuple[float, str, int]) -> tuple[Datum, ...]:
    """Return one-axis data from ``(at, value, derivative)`` triples, values as printed."""
    return tuple(Datum("f", "x", at, value, order) for at, value, order in conditions)


def _ode(
    name: str,
    terms: tuple[Term, ...],
    constraints: tuple[Constraint, ...],
    data: tuple[Datum, ...],
    regular: float,
    reference: Reference,
    description: str,
    field: Field | None = None,
    variant: Variant = "corrected",
) -> Problem:
    """Return a one-field problem on the axis ``x``."""
    return Problem(
        name=name,
        variant=variant,
        axes=_X,
        fields=(field or Field("f", ("x",)),),
        equations=(Equation(terms),),
        constraints=constraints,
        data=data,
        regular=Regular("f", (regular,)),
        reference=reference,
        description=description,
    )


def _relabel(problem: Problem) -> Problem:
    """Return the corrected variant of a panel whose printed choices are already exact."""
    return dataclasses.replace(problem, variant="corrected")


def _move_constraint(problem: Problem, near: float) -> Problem:
    """Return the corrected variant: the single invariant moved to the exact root nearby.

    The constraint carries the root's recipe, so the problem's identity does not depend on the
    last bits the root finder returns on a given machine (:class:`pihm.problems.spec.Root`).
    """
    (constraint,) = problem.constraints
    moved = dataclasses.replace(constraint, at=math.nan, root=Root(near))
    return dataclasses.replace(problem, variant="corrected", constraints=(moved,))


# ---------------------------------------------------------------------------- R1


def _r1() -> Problem:
    """R1: :math:`f' + 2f = 0`, :math:`f(-1) = 1`; no invariant constraint (first order)."""
    return _ode(
        "R1",
        (_linear(1.0, 1), _linear(2.0, 0)),
        (),
        _data((-1.0, "1", 0)),
        -1.0,
        Reference("expression", "exp(-2*(x + 1))"),
        "f' + 2f = 0, f(-1) = 1",
    )


# ---------------------------------------------------------------------------- R2 (Fig. 3)

#: Fig. 3: (figure, {a, b, c}, BC {f(0), f'(0)}, invariant (at, order), x_s, solution).
_FIG3: Final = {
    "R2a": ("3b", (1.0, 4.0, 4.0), ("0.5", "-0.5"), (-1.0, 0), 0.0, "0.5*(1 + x)*exp(-2*x)"),
    "R2b": ("3c", (1.0, -2.0, -3.0), ("0", "1"), (0.0, 0), 0.5, "0.25*(exp(3*x) - exp(-x))"),
    "R2c": (
        "3d",
        (1.0, 5.0, 400.0),
        ("1", "0"),
        (0.0, 1),
        0.0,
        "exp(-5*x/2)*(cos(15*sqrt(7)*x/2) + sqrt(7)/21*sin(15*sqrt(7)*x/2))",
    ),
}


def _fig3(name: str, variant: Variant) -> Problem:
    """Fig. 3: :math:`af'' + bf' + cf = 0` with one invariant (Eq. 15)."""
    figure, (a, b, c), (f0, f1), (at, order), regular, solution = _FIG3[name]
    kind = "f'" if order else "f"
    faithful = _ode(
        name,
        (_linear(a, 2), _linear(b, 1), _linear(c, 0)),
        (_zero(at, order),),
        _data((0.0, f0, 0), (0.0, f1, 1)),
        regular,
        Reference("expression", solution),
        f"Fig. {figure}: C = {{{a:g}, {b:g}, {c:g}}}, BC {{f(0), f'(0)}} = {{{f0}, {f1}}}, "
        f"DC_I {kind}({at:g}) = 0",
        variant="faithful",
    )
    return faithful if variant == "faithful" else _relabel(faithful)


# ---------------------------------------------------------------------------- R3 (Fig. 4)

#: Fig. 4: (figure, l, m, invariant order at x = 0, BC {f(-1), f(1)}, x_s).
_FIG4: Final = {
    "R3a": ("4a", 4, 0, 1, ("1", "1"), 1.0),
    "R3b": ("4b", 5, 0, 0, ("-1", "1"), 1.0),
    "R3c": ("4c", 5, 1, 1, ("0", "0"), 0.5),
    "R3d": ("4d", 6, 1, 0, ("0", "0"), 0.5),
}


def _fig4(name: str, variant: Variant) -> Problem:
    """Fig. 4: Legendre (m = 0) and associated Legendre (m = 1) with the paper's lifts (Eq. 17)."""
    figure, degree, m, order, (left, right), regular = _FIG4[name]
    ll = float(degree * (degree + 1))
    if m == 0:
        terms = (_linear(1.0, 2, (1.0, 0.0, -1.0)), _linear(1.0, 1, (0.0, -2.0)), _linear(ll, 0))
        solution = f"legendre({degree}, x)"
    else:  # the paper's operator for m = 1: Eq. 17 multiplied by (1 - x^2)
        terms = (
            _linear(1.0, 2, (1.0, 0.0, -2.0, 0.0, 1.0)),
            _linear(1.0, 1, (0.0, -2.0, 0.0, 2.0)),
            _linear(1.0, 0, (ll - 1.0, 0.0, -ll)),
        )
        solution = f"assoc_legendre(1, {degree}, x)"
    kind = "f'" if order else "f"
    faithful = _ode(
        name,
        terms,
        (_zero(0.0, order),),
        _data((-1.0, left, 0), (1.0, right, 0)),
        regular,
        Reference("expression", solution),
        f"Fig. {figure}: Legendre l = {degree}, m = {m}, BC {{f(-1), f(1)}} = {{{left}, {right}}}, "
        f"DC_I {kind}(0) = 0",
        variant="faithful",
    )
    if variant == "faithful":
        return faithful
    if m == 0:
        return _relabel(faithful)
    # f = sqrt(1 - x^2) v with v = -P_l'(x), a polynomial; f(0) = v(0) and f'(0) = v'(0). The
    # reference is written through that identity, so that dividing by the weight is exact.
    return dataclasses.replace(
        faithful,
        variant="corrected",
        fields=(Field("f", ("x",), weight="sqrt(1 - x**2)"),),
        reference=Reference("expression", f"-sqrt(1 - x**2)*dlegendre({degree}, x)"),
        equations=(
            Equation(
                (
                    _linear(1.0, 2, (1.0, 0.0, -1.0)),
                    _linear(1.0, 1, (0.0, -4.0)),
                    _linear(ll - 2.0, 0),
                )
            ),
        ),
        description=faithful.description + "; solved for v = f/sqrt(1 - x^2)",
    )


# ---------------------------------------------------------------------------- R4 (Fig. 5)


def _fig5(name: str, variant: Variant) -> Problem:
    r"""Fig. 5: inhomogeneous equations, sources injected through :math:`\mathbb{D}^{(0)}(0)`."""
    exact = variant == "corrected"
    if name == "R4a":
        faithful = _ode(
            name,
            (
                _linear(1.0, 2, (-1.0, 1.0)),
                _linear(1.0, 1, (0.0, -1.0)),
                _linear(1.0, 0),
                Term(-1.0, (), Polynomial("x", (1.0, -2.0, 1.0))),
            ),
            (_zero(-0.2, 1),),
            _data((0.0, "1/2", 0), (0.0, "-1/8", 1)),
            0.0,
            Reference("expression", "1.5*exp(x) - 0.125*x*(8*x + 13) - 1"),
            "Fig. 5a: (x - 1)f'' - xf' + f = (x - 1)^2, BC {f(0), f'(0)} = {1/2, -1/8}, "
            "DC_I f'(x_m = -0.2) = 0",
            variant="faithful",
        )
        return _move_constraint(faithful, -0.2) if exact else faithful
    # (coefficients {1, b, c}, source, Maclaurin order, BC {f(0), f'(0)}, invariant, solution)
    table: dict[str, tuple[tuple[float, float], str, int, tuple[str, str], tuple[float, int], str]]
    table = {
        "R4b": (
            (4.0, 4.0),
            "exp(-2*x)",
            3,
            ("-1", "1"),
            (-0.73, 0),
            "0.5*exp(-2*x)*(x**2 - 2*x - 2)",
        ),
        "R4c": (
            (-5.0, 6.0),
            "x*exp(2*x)",
            5,
            ("1/3", "1/3"),
            (0.91, 0),
            "(4*exp(3*x) - exp(2*x)*(3*x**2 + 6*x + 2))/6",
        ),
        "R4d": (
            (4.0, 20.0),
            "x*exp(-2*x)*sin(4*x)",
            7,
            ("-1", "-1"),
            (-0.75, 1),
            "exp(-2*x)*(-4*(x**2 + 16)*cos(4*x) + (x - 48)*sin(4*x))/64",
        ),
    }
    descriptions = {
        "R4b": "Fig. 5b: f'' + 4f' + 4f = e^{-2x}, BC {f(0), f'(0)} = {-1, 1}, "
        "DC_I f(x_z = -0.73) = 0, p = 3",
        "R4c": "Fig. 5c: f'' - 5f' + 6f = xe^{2x}, BC {f(0), f'(0)} = {1/3, 1/3}, "
        "DC_I f(x_z = 0.91) = 0, p = 5",
        "R4d": "Fig. 5d: f'' + 4f' + 20f = xe^{-2x} sin(4x), BC {f(0), f'(0)} = {-1, -1}, "
        "DC_I f'(x_m = -0.75) = 0, p = 7",
    }
    (b, c), source, degree, (f0, f1), (at, order), solution = table[name]
    terms = (
        _linear(1.0, 2),
        _linear(b, 1),
        _linear(c, 0),
        Term(-1.0, (), source=Source(source, "maclaurin", degree)),
    )
    faithful = _ode(
        name,
        terms,
        (_zero(at, order),),
        _data((0.0, f0, 0), (0.0, f1, 1)),
        0.0,
        Reference("expression", solution),
        descriptions[name],
        variant="faithful",
    )
    if not exact:
        return faithful
    moved = _move_constraint(faithful, at)
    chebyshev = Term(-1.0, (), source=Source(source, "chebyshev"))
    (equation,) = moved.equations
    return dataclasses.replace(moved, equations=(Equation((*equation.terms[:3], chebyshev)),))


# ---------------------------------------------------------------------------- R5 (Fig. 6)


def _fig6(name: str, variant: Variant) -> Problem:
    """Fig. 6: Laplace, heat and wave equations, square in both axes (Eqs 23, 27 and 28)."""
    faithful_rows: tuple[Constraint, ...]
    corrected_rows: tuple[Constraint, ...]
    if name == "R5a":
        axes = (Axis("x"), Axis("y"))
        corrected_regular: tuple[float, ...]
        note = "the top-boundary datum f(x, 1)"
        terms = (Term(1.0, (Factor("f", (2, 0)),)), Term(1.0, (Factor("f", (0, 2)),)))
        data = (
            Datum("f", "x", 1.0, "0"),
            Datum("f", "x", -1.0, "0"),
            Datum("f", "y", -1.0, "0"),
            Datum("f", "y", 1.0, "sin(pi*(x + 1)/2)"),
        )
        faithful_rows = (_zero(1.0), _zero(-1.0), _zero(-1.0, axis="y"))
        corrected_rows = (*faithful_rows, Constraint("datum", "f", "y", 1.0, datum=data[3].value))
        regular = corrected_regular = (0.0, 1.0)
        solution = "cos(pi*x/2)*sinh(pi*(y + 1)/2)/sinh(pi)"
        description = (
            "Fig. 6b: Laplace, BC {f(±1, y) = 0, f(x, -1) = 0, f(x, 1) = sin(pi(x + 1)/2)}, "
            "DC_I {f(±1, y) = 0, f(x, -1) = 0}"
        )
    else:
        axes = (Axis("t", "time"), Axis("x"))
        heat = name == "R5b"
        note = (
            "the solution imposed at t = -1, forward"
            if heat
            else "the initial data f(0, x) and f_t(0, x) = 0"
        )
        terms = (
            Term(1.0, (Factor("f", (1, 0) if heat else (2, 0)),)),
            Term(-1.0 / 25.0 if heat else -4.0, (Factor("f", (0, 2)),)),
        )
        initial = Datum("f", "t", 0.0, "sin(2*pi*x)")
        edges = (Datum("f", "x", 1.0, "0"), Datum("f", "x", -1.0, "0"))
        velocity = () if heat else (Datum("f", "t", 0.0, "0", 1),)
        data = (initial, *velocity, *edges)
        walls = (_zero(1.0), _zero(-1.0))
        if heat:
            faithful_rows = (*walls, _zero(0.75, 1), _zero(-0.75, 1))
        else:
            faithful_rows = (_zero(0.125, axis="t"), _zero(0.875, axis="t"), *walls)
        if heat:
            # The printed data at t = 0 would make [-1, 0] a backward heat
            # problem, which is ill posed. The corrected variant imposes the same solution at
            # t = -1 (the reference's value there), where it runs forward over the whole interval.
            datum_row = Constraint("datum", "f", "t", -1.0, datum="exp(4*pi**2/25)*sin(2*pi*x)")
        else:
            datum_row = Constraint("datum", "f", "t", 0.0, datum=initial.value)
        rest_row = () if heat else (_zero(0.0, 1, axis="t"),)
        corrected_rows = (*walls, datum_row, *rest_row)
        regular = (0.0, 0.25)
        corrected_regular = (-1.0, 0.25) if heat else regular
        solution = "exp(-4*pi**2*t/25)*sin(2*pi*x)" if heat else "cos(4*pi*t)*sin(2*pi*x)"
        description = (
            "Fig. 6d: heat, k = 1/25, IBC {f(0, x) = sin(2 pi x), f(t, ±1) = 0}, "
            "DC_I {f(t, ±1) = 0, f_x(t, ±3/4) = 0}"
            if heat
            else "Fig. 6f: wave, c = 2, IBC {f(0, x) = sin(2 pi x), f_t(0, x) = 0, f(t, ±1) = 0}, "
            "DC_I {f(1/8, x) = 0, f(7/8, x) = 0, f(t, ±1) = 0}"
        )
    faithful = variant == "faithful"
    return Problem(
        name=name,
        variant=variant,
        axes=axes,
        fields=(Field("f", tuple(axis.name for axis in axes)),),
        equations=(Equation(terms),),
        constraints=faithful_rows if faithful else corrected_rows,
        data=data,
        regular=Regular("f", regular if faithful else corrected_regular),
        reference=Reference("expression", solution),
        description=description + ("" if faithful else f"; corrected: {note}"),
    )


# ---------------------------------------------------------------------------- R6 (Fig. 7)


def _fig7(name: str, variant: Variant) -> Problem:
    """Fig. 7: quadratic equations, solved in the paper's doubled space (Eqs 29 and 34)."""
    if name == "R6a":
        faithful = _ode(
            name,
            (_linear(4.0, 2), Term(2.0, (_d(1), _d(1))), _linear(1.0, 0)),
            (_zero(0.0, 1),),
            _data((0.0, "1", 0), (0.0, "0", 1)),
            0.0,
            Reference("expression", "1 - x**2/8"),
            "Fig. 7b: 4f'' + 2(f')^2 + f = 0, BC {f(0) = 1, f'(0) = 0}, DC_I f'(0) = 0, x_s = 0",
            variant="faithful",
        )
        return faithful if variant == "faithful" else _relabel(faithful)
    faithful = _ode(
        name,
        (_linear(1.0, 2), Term(-2.0, (_d(0), _d(0))), Term(1.0, (), Polynomial("x", (0.0, 1.0)))),
        (_zero(0.02615),),
        _data((-1.0, "-0.1", 0), (1.0, "0.1", 0)),
        0.5,
        Reference("bvp"),
        "Fig. 7c: f'' - 2f^2 + x = 0, BC {f(-1) = -0.1, f(1) = 0.1}, DC_I f(x_z = 0.02615) = 0, "
        "x_s = 0.5",
        variant="faithful",
    )
    return faithful if variant == "faithful" else _move_constraint(faithful, 0.02615)


# ---------------------------------------------------------------------------- R7 and R8

#: R7's angular frequency :math:`\omega`.
OMEGA: Final[float] = math.pi

#: R8's time interval :math:`[0, T]`, carried on the Chebyshev axis :math:`t = 2s/T - 1`.
R8_TIME: Final[float] = 3.0


def _r7() -> Problem:
    r"""R7, the coupled oscillator: :math:`u' = v`, :math:`v' = -\omega^2 u`.

    The data are :math:`(u, v)(-1) = (0, \omega)`, and the solution is
    :math:`u = \sin(\omega(x + 1))`, :math:`v = \omega\cos(\omega(x + 1))`.
    """
    u, v = Factor("u", (0,)), Factor("v", (0,))
    return Problem(
        name="R7",
        axes=_X,
        fields=(Field("u", ("x",)), Field("v", ("x",))),
        equations=(
            Equation((Term(1.0, (Factor("u", (1,)),)), Term(-1.0, (v,)))),
            Equation((Term(1.0, (Factor("v", (1,)),)), Term(OMEGA**2, (u,)))),
        ),
        constraints=(Constraint("ratio", "v", "x", -1.0),),
        data=(Datum("u", "x", -1.0, "0"), Datum("v", "x", -1.0, "pi")),
        regular=Regular("v", (-1.0,)),
        reference=Reference("expression", expressions=("sin(pi*(x + 1))", "pi*cos(pi*(x + 1))")),
        description="u' = v, v' = -pi^2 u, (u, v)(-1) = (0, pi)",
    )


def _r8() -> Problem:
    r"""R8, for the Carleman lift: :math:`y' = -y + y^2`, :math:`y(0) = 1/2`, :math:`s \in [0, 3]`.

    Time is carried on :math:`t = 2s/3 - 1`, where :math:`dy/dt = (T/2)(-y + y^2)`, and the
    solution is :math:`1/(1 + e^s)`. The nonlinearity-to-dissipation ratio is :math:`R = 1/2`.
    """
    y = Factor("y", (0,))
    half = R8_TIME / 2
    return Problem(
        name="R8",
        axes=(Axis("t", "time"),),
        fields=(Field("y", ("t",)),),
        equations=(
            Equation((Term(1.0, (Factor("y", (1,)),)), Term(half, (y,)), Term(-half, (y, y)))),
        ),
        constraints=(Constraint("ratio", "y", "t", -1.0),),
        data=(Datum("y", "t", -1.0, "1/2"),),
        regular=Regular("y", (-1.0,)),
        reference=Reference("expression", f"1/(1 + exp({half:g}*(t + 1)))"),
        description="y' = -y + y^2 on [0, 3], y(0) = 1/2, time on t = 2s/3 - 1",
    )


# ---------------------------------------------------------------------------- R9

#: R9's nonlinearity strengths :math:`a/\nu`, one panel each. The lift's error falls
#: by about :math:`a/5\nu` per order, so they run from inside the convergence theorem at every
#: number of modes (a), through the headline (c), to where the lift diverges (e).
R9_STRENGTHS: Final[Mapping[str, float]] = {"a": 0.03, "b": 0.3, "c": 1.0, "d": 3.0, "e": 10.0}

#: R9's numbers of sine modes :math:`M`.
R9_MODES: Final[tuple[int, ...]] = (4, 8, 16)


def burgers_panel(name: str) -> tuple[float, int]:
    r"""Return an R9 problem's :math:`a/\nu` and number of modes, from its name (``"R9c-M8"``)."""
    rung, _, modes = name.partition("-M")
    if rung[:2] != "R9" or rung[2:] not in R9_STRENGTHS or not modes.isdigit():
        raise ValueError(f"{name!r} is not an R9 problem")
    return R9_STRENGTHS[rung[2:]], int(modes)


def _r9(name: str) -> Problem:
    r"""R9, viscous Burgers from :math:`a\sin x` on :math:`M` sine modes, :math:`\nu = T = 1`.

    :func:`pihm.problems.fluids.burgers` builds it; only :math:`a/\nu` and :math:`\nu T` matter.
    """
    strength, modes = burgers_panel(name)
    return burgers(modes, strength, 1.0, 1.0, name=name)


_R9_NAMES: Final[tuple[str, ...]] = tuple(
    f"R9{panel}-M{modes}" for panel in R9_STRENGTHS for modes in R9_MODES
)


# ---------------------------------------------------------------------------- R10

#: R10's vortex-pair strengths :math:`a/\nu`, the norm of the initial vorticity's coefficients
#:: the lift's error falls by about 0.03-0.1 an order at 3, 0.1-0.4 at
#: 10, and at 30 it stops falling on the larger truncations. ``"R10a"``, the Taylor–Green vortex,
#: has no strength: its lift is exact.
R10_STRENGTHS: Final[Mapping[str, float]] = {"b": 3.0, "c": 10.0, "d": 30.0}

#: R10's numbers of modes :math:`M`, each with the cutoff :math:`K_c` on :math:`|k|^2` that keeps
#: them. Every panel runs on 12 to 48 modes; the Taylor–Green vortex and the pair at
#: a/ν = 10 on 80 as well, the largest the classical build resolves; and the pair at a/ν = 10 on
#: 148 and 316, whose lift only the tensor layout's circuit carries (:data:`R10_FIELD_SCALE`).
R10_MODES: Final[Mapping[int, int]] = {12: 4, 20: 5, 28: 9, 48: 16, 80: 25, 148: 49, 316: 100}

#: The truncations every R10 panel runs on.
R10_PANEL_MODES: Final[tuple[int, ...]] = (12, 20, 28, 48)

#: R10's field-scale truncations, past every classical build, which only the tensor layout's
#: circuit reaches.
R10_FIELD_SCALE: Final[tuple[int, ...]] = (148, 316)

#: R10's sub-windows, each name's suffix with its :math:`T`, for the pair at
#: the boundary of the lift's convergence.
R10_WINDOWS: Final[Mapping[str, float]] = {"W8": 0.125}


def navier_stokes_panel(name: str) -> tuple[float | None, int, float]:
    r"""Return an R10 problem's :math:`a/\nu` (``None``: Taylor–Green), :math:`K_c` and :math:`T`.

    From its name: ``"R10c-M20"`` is the pair at :math:`a/\nu = 10` on 20 modes, over
    :math:`T = 1`, and ``"R10d-M12-W8"`` the pair at 30 on 12 modes over :math:`T = 1/8`.
    """
    rung, _, rest = name.partition("-M")
    modes, _, window = rest.partition("-")
    panel = rung[3:]
    if (
        rung[:3] != "R10"
        or panel not in ("a", *R10_STRENGTHS)
        or not modes.isdigit()
        or int(modes) not in R10_MODES
        or (window and window not in R10_WINDOWS)
    ):
        raise ValueError(f"{name!r} is not an R10 problem")
    time = R10_WINDOWS[window] if window else 1.0
    return R10_STRENGTHS.get(panel), R10_MODES[int(modes)], time


def _r10(name: str) -> Problem:
    r"""R10, two-dimensional Navier-Stokes on :math:`M` Fourier modes, :math:`\nu = 1`."""
    strength, cutoff, time = navier_stokes_panel(name)
    if strength is None:
        return taylor_green(cutoff, 1.0, time, name=name)
    return vortex_pair(cutoff, strength, 1.0, time, name=name)


def galerkin(name: str) -> Galerkin:
    r"""Return an R9 or R10 problem's structured form (:class:`~pihm.problems.fluids.Galerkin`).

    Its Fourier symbols, from which a circuit builds the lift's :math:`F_2`;
    :math:`\nu = 1`, as the problems have.

    Raises
    ------
    ValueError
        For any other problem.
    """
    if rung(name) == "R9":
        return burgers_galerkin(burgers_panel(name)[1], 1.0, 1.0)
    if rung(name) == "R10":
        _, cutoff, time = navier_stokes_panel(name)
        return vorticity_galerkin(cutoff, 1.0, time)
    raise ValueError(f"{name!r} is not a Fourier-Galerkin problem")


_R10_NAMES: Final[tuple[str, ...]] = (
    *(f"R10{panel}-M{modes}" for panel in ("a", *R10_STRENGTHS) for modes in R10_PANEL_MODES),
    *(f"R10d-M{modes}-{window}" for window in R10_WINDOWS for modes in R10_PANEL_MODES),
    "R10a-M80",
    "R10c-M80",
    *(f"R10c-M{modes}" for modes in R10_FIELD_SCALE),
)


# ---------------------------------------------------------------------------- registry

#: The paper's panels, each built at a variant.
_PANEL_BUILDERS: Final[Mapping[str, Callable[[str, Variant], Problem]]] = {
    **dict.fromkeys(("R2a", "R2b", "R2c"), _fig3),
    **dict.fromkeys(("R3a", "R3b", "R3c", "R3d"), _fig4),
    **dict.fromkeys(("R4a", "R4b", "R4c", "R4d"), _fig5),
    **dict.fromkeys(("R5a", "R5b", "R5c"), _fig6),
    **dict.fromkeys(("R6a", "R6b"), _fig7),
}

#: The problems that are not the paper's.
_BUILDERS: Final[Mapping[str, Callable[[str], Problem]]] = {
    "R1": lambda name: _r1(),
    "R7": lambda name: _r7(),
    "R8": lambda name: _r8(),
    **dict.fromkeys(_R9_NAMES, _r9),
    **dict.fromkeys(_R10_NAMES, _r10),
}

#: The rungs of this phase and their problems, in ladder order.
RUNGS: Final[Mapping[str, tuple[str, ...]]] = {
    "R1": ("R1",),
    "R2": ("R2a", "R2b", "R2c"),
    "R3": ("R3a", "R3b", "R3c", "R3d"),
    "R4": ("R4a", "R4b", "R4c", "R4d"),
    "R5": ("R5a", "R5b", "R5c"),
    "R6": ("R6a", "R6b"),
    "R7": ("R7",),
    "R8": ("R8",),
    "R9": _R9_NAMES,
    "R10": _R10_NAMES,
}


def rung(name: str) -> str:
    """Return the rung of a problem's name: ``"R10"`` for ``"R10c-M20"``, ``"R2"`` for ``"R2a"``.

    Raises
    ------
    ValueError
        For a name that does not start with a rung.
    """
    found = re.match(r"R\d+", name)
    if found is None:
        raise ValueError(f"{name!r} names no rung")
    return found.group()


def names(rung: str | None = None) -> tuple[str, ...]:
    """Return the problem names of one rung, or of every rung in ladder order."""
    if rung is None:
        return tuple(name for group in RUNGS.values() for name in group)
    if rung not in RUNGS:
        raise ValueError(f"unknown rung {rung!r}; known: {sorted(RUNGS)}")
    return RUNGS[rung]


@functools.cache
def problem(name: str, variant: Variant = "corrected") -> Problem:
    """Return a problem of the ladder (built once, then cached).

    Parameters
    ----------
    name
        The rung identifier, such as ``"R4d"``.
    variant
        ``"faithful"`` for a paper panel's choices as printed (one of :data:`PANELS`), which only
        the reproduction of the paper runs; by default the problem itself.

    Raises
    ------
    ValueError
        For an unknown name, or the faithful variant of a problem that is not the paper's.
    """
    if name in _PANEL_BUILDERS:
        return _PANEL_BUILDERS[name](name, variant)
    if name not in _BUILDERS:
        raise ValueError(f"unknown problem {name!r}")
    if variant == "faithful":
        raise ValueError(f"{name} is not a panel of the paper: it has no faithful variant")
    return _BUILDERS[name](name)
