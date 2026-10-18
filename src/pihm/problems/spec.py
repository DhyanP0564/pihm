r"""Declarative problem specifications: axes, fields, equations, constraints and references.

A :class:`Problem` states a differential equation the way the paper does, as data: every class
here is a frozen, hashable dataclass of numbers, strings and tuples. The
known functions it mentions are text in the language of :mod:`pihm.problems.expressions`. A
problem therefore

* serialises to plain data and YAML (:meth:`Problem.to_dict`, :meth:`Problem.to_yaml`) and
  back (:meth:`Problem.from_dict`, :meth:`Problem.from_yaml`);
* has a stable :attr:`Problem.digest`, which the run records carry, so that editing a panel's
  data invalidates the results computed from it.

**Equations.** An :class:`Equation` states :math:`\sum_t \text{term}_t = 0`. A :class:`Term` is

.. math::

   c \; m(x) \; \prod_i \partial^{\alpha_i} u_{k_i},

a coefficient :math:`c`, an optional polynomial multiplier :math:`m` on one axis and a product
of field derivatives (its :class:`Factor` s). A term without factors is a known function (a
source), given exactly by a :class:`Polynomial` or approximately by a :class:`Source`. So the
paper's Fig. 5a, :math:`(x-1)f'' - xf' + f = (x-1)^2`, is four terms, and its Eq. 29,
:math:`4f'' + 2f'^2 + f = 0`, has one quadratic term.

**Constraints** (:class:`Constraint`) are the shared rows placed on a field:
``"zero"`` is the paper's invariant constraint :math:`\partial^j u = 0` on the slice
:math:`x_a = x^*` (a point, for one axis), ``"datum"`` imposes a known function on such a
slice, and ``"ratio"`` makes a first-order system's values at a point parallel to its data
there. The regular constraint :math:`\mathcal{DC}_R`, the point :math:`x_s` at which the solution
is known and non-zero, is the problem's :class:`Regular` point. **Data** (:class:`Datum`) are the
conditions stated with the problem, such as the paper's :math:`\mathcal{BC}`; they determine the
reference solution but are not rows of the residual unless a constraint says so.

Examples
--------
Fig. 3b of the paper, :math:`f'' + 4f' + 4f = 0` with the invariant :math:`f(-1) = 0`:

>>> x = Axis("x")
>>> f = Field("f", ("x",))
>>> problem = Problem(
...     name="R2a",
...     axes=(x,),
...     fields=(f,),
...     equations=(
...         Equation(
...             (
...                 Term(1.0, (Factor("f", (2,)),)),
...                 Term(4.0, (Factor("f", (1,)),)),
...                 Term(4.0, (Factor("f", (0,)),)),
...             )
...         ),
...     ),
...     constraints=(Constraint("zero", "f", "x", -1.0),),
...     data=(Datum("f", "x", 0.0, "1/2"), Datum("f", "x", 0.0, "-1/2", 1)),
...     regular=Regular("f", (0.0,)),
...     reference=Reference("expression", "0.5*(1 + x)*exp(-2*x)"),
... )
>>> problem.order, problem.degree
(2, 1)
>>> Problem.from_yaml(problem.to_yaml()) == problem
True
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
from dataclasses import dataclass
from functools import cached_property
from typing import Any, Literal, TypeAlias, get_args

import yaml

from pihm.problems.expressions import CONSTANTS, FUNCTIONS, compile_expression
from pihm.problems.plain import from_plain, to_plain

__all__ = [
    "Axis",
    "Constraint",
    "ConstraintKind",
    "Datum",
    "Equation",
    "Factor",
    "Field",
    "Polynomial",
    "Problem",
    "Reference",
    "ReferenceKind",
    "Regular",
    "Representation",
    "Root",
    "Source",
    "Term",
    "Variant",
]

#: The two variants of a paper panel, the paper's own choices as printed and
#: our corrections of them. The corrected variant is the problem itself, so a problem names its
#: variant only when it is the paper's.
Variant: TypeAlias = Literal["faithful", "corrected"]

#: The constraint rows a problem can place (an invariant, a datum function on a slice, or the
#: ratio rows that pin the direction of a first-order system's data at a point).
ConstraintKind: TypeAlias = Literal["zero", "datum", "ratio"]

#: How a non-polynomial source enters the residual, as the paper's truncated Maclaurin series
#: (Eq. 14) or as the Chebyshev series of the function itself (the corrected variant).
Representation: TypeAlias = Literal["maclaurin", "chebyshev"]

#: A closed form, or a numerical solution of the problem's own boundary-value or initial-value
#: problem.
ReferenceKind: TypeAlias = Literal["expression", "bvp", "ivp"]


def _check_point(values: tuple[float, ...], what: str) -> None:
    """Refuse coordinates outside the Chebyshev interval :math:`[-1, 1]`."""
    if any(not -1.0 <= value <= 1.0 for value in values):
        raise ValueError(f"{what} must lie in [-1, 1], got {values}")


@dataclass(frozen=True)
class Axis:
    """One independent variable, on the Chebyshev interval :math:`[-1, 1]`.

    Attributes
    ----------
    name
        The variable's name in expressions, such as ``"x"`` or ``"t"``.
    label
        A description for plots and reports, such as ``"time"``.
    """

    name: str
    label: str = ""

    def __post_init__(self) -> None:
        """Validate the name."""
        if not self.name.isidentifier() or self.name in FUNCTIONS or self.name in CONSTANTS:
            raise ValueError(f"{self.name!r} cannot name an axis")


@dataclass(frozen=True)
class Field:
    r"""An unknown function of some of the problem's axes.

    Attributes
    ----------
    name
        The field's name, such as ``"f"``.
    axes
        The axes it depends on, in Kronecker order (axis 0 most significant).
    weight
        An optional readout weight :math:`w`: the physical solution is :math:`w\,u` for the
        solved field :math:`u`. The corrected associated-Legendre rungs solve for
        :math:`v = P_l^1/\sqrt{1-x^2}`, which is a polynomial, with ``weight="sqrt(1 - x**2)"``.
    """

    name: str
    axes: tuple[str, ...]
    weight: str | None = None


@dataclass(frozen=True)
class Polynomial:
    r"""A polynomial in one axis, :math:`m(x) = \sum_j a_j x^j`, given by monomial coefficients."""

    axis: str
    coefficients: tuple[float, ...]

    def __post_init__(self) -> None:
        """Refuse an empty coefficient tuple."""
        if not self.coefficients:
            raise ValueError("a polynomial needs at least one coefficient")

    @property
    def degree(self) -> int:
        """The degree, ignoring trailing zero coefficients (0 for the zero polynomial)."""
        nonzero = [index for index, value in enumerate(self.coefficients) if value != 0]
        return nonzero[-1] if nonzero else 0

    @property
    def is_constant(self) -> bool:
        """Whether the polynomial is a constant."""
        return self.degree == 0


@dataclass(frozen=True)
class Source:
    r"""A known non-polynomial function, and how the residual represents it.

    Attributes
    ----------
    expression
        The function, in the problem's axis variables.
    representation
        ``"maclaurin"``, the paper's truncated Maclaurin series (Eq. 14) of degree
        :attr:`degree`, or ``"chebyshev"``, its Chebyshev series on the residual's register.
    degree
        The Maclaurin order :math:`\bar p` (``"maclaurin"`` only).
    """

    expression: str
    representation: Representation = "chebyshev"
    degree: int | None = None

    def __post_init__(self) -> None:
        """Check that the degree is given exactly when it is needed."""
        if self.representation not in get_args(Representation):
            raise ValueError(f"unknown representation {self.representation!r}")
        if (self.representation == "maclaurin") != (self.degree is not None):
            raise ValueError("a Maclaurin source needs a degree, and only a Maclaurin source")
        if self.degree is not None and self.degree < 0:
            raise ValueError(f"the Maclaurin degree must be non-negative, got {self.degree}")


@dataclass(frozen=True)
class Factor:
    r"""One factor of a term: a derivative :math:`\partial^\alpha u` of a field.

    Attributes
    ----------
    field
        The field's name.
    derivative
        The derivative order along each of the field's axes, such as ``(2,)`` for
        :math:`f''` or ``(1, 0)`` for :math:`\partial_t f(t, x)`.
    """

    field: str
    derivative: tuple[int, ...]

    def __post_init__(self) -> None:
        """Refuse negative orders."""
        if any(order < 0 for order in self.derivative):
            raise ValueError(f"derivative orders must be non-negative, got {self.derivative}")


@dataclass(frozen=True)
class Term:
    r"""One term :math:`c\,m\,\prod_i \partial^{\alpha_i} u_{k_i}` of an equation.

    Attributes
    ----------
    coefficient
        The constant :math:`c`.
    factors
        The field derivatives multiplied together: one for a linear term, two for a quadratic
        one, none for a source.
    multiplier
        An optional polynomial multiplier on one axis (the paper's :math:`\mathbb{M}_{x^p}`).
    source
        For a term without factors only: a non-polynomial known function. A factorless term
        is ``multiplier``, ``source`` or the constant ``coefficient`` alone.
    """

    coefficient: float = 1.0
    factors: tuple[Factor, ...] = ()
    multiplier: Polynomial | None = None
    source: Source | None = None

    def __post_init__(self) -> None:
        """Check that a source stands alone and a multiplier is not combined with one."""
        if self.source is not None and self.factors:
            raise ValueError("a source term cannot have factors")
        if self.source is not None and self.multiplier is not None:
            raise ValueError("a term has a polynomial multiplier or a source, not both")

    @property
    def degree(self) -> int:
        """The number of field factors, 0 for a source, 1 linear and more nonlinear."""
        return len(self.factors)


@dataclass(frozen=True)
class Equation:
    r"""An equation :math:`\sum_t \text{term}_t = 0`."""

    terms: tuple[Term, ...]
    label: str = ""

    def __post_init__(self) -> None:
        """Refuse an empty equation."""
        if not self.terms:
            raise ValueError("an equation needs at least one term")


@dataclass(frozen=True)
class Root:
    r"""A slice position given by what it is: the reference solution's root nearest a point.

    A corrected variant moves the paper's invariant constraint to the exact root of
    :math:`\partial^j u` nearby, :math:`j` the constraint's own derivative.
    The root is found numerically, and its last bits depend on the machine's floating point, so a
    constraint defined by a root carries this recipe: only the recipe enters the problem's plain
    form and digest, and the root is resolved when the problem is built
    (:func:`pihm.problems.reference.exact_root`).

    Attributes
    ----------
    near
        The point the root is sought nearest.
    """

    near: float

    def __post_init__(self) -> None:
        """Check the point."""
        _check_point((self.near,), "a root's starting point")


@dataclass(frozen=True)
class Constraint:
    r"""A constraint row on a field, on the slice :math:`x_a = x^*` of one of its axes.

    Attributes
    ----------
    kind
        ``"zero"``: :math:`\partial_a^j u = 0` on the slice (the paper's invariant constraint;
        on one axis the slice is a point). ``"datum"``: :math:`u = g` on the slice, for the
        known function :attr:`datum` of the other axes. ``"ratio"``: the
        vector of every field's value at the point is parallel to the problem's data there,
        :math:`y_a u_c - y_c u_a = 0` for each field :math:`c`, which imposes the data's ratios
        and leaves its scale free (the rows :math:`\mathbb{F}`); in a
        Carleman lift it pins every component to the data's moments.
    field
        The constrained field; for ``"ratio"``, the anchor :math:`a`, whose datum must be
        non-zero.
    axis
        The axis :math:`a` that the slice fixes.
    at
        The slice position :math:`x^*`. For a constraint defined by a :attr:`root`, it is resolved
        when the problem is built, and left unset (``nan``) until then.
    derivative
        The order :math:`j` of the derivative along ``axis`` (``"zero"`` only).
    datum
        The slice function :math:`g`, in the other axes' variables (``"datum"`` only).
    root
        The recipe that defines :attr:`at`, if a root of the reference defines it (``"zero"``
        only). Only the recipe enters the plain form and the digest (:attr:`plain_omits`).
    """

    kind: ConstraintKind
    field: str
    axis: str
    at: float = math.nan
    derivative: int = 0
    datum: str | None = None
    root: Root | None = None

    def __post_init__(self) -> None:
        """Check the kind-specific parameters."""
        if self.kind not in get_args(ConstraintKind):
            raise ValueError(f"unknown constraint kind {self.kind!r}")
        if self.root is not None and self.kind != "zero":
            raise ValueError("only a zero constraint can sit at a root")
        if self.root is None or not math.isnan(self.at):
            _check_point((self.at,), "a constraint position")
        if self.derivative < 0:
            raise ValueError(f"the derivative order must be non-negative, got {self.derivative}")
        if (self.kind == "datum") != (self.datum is not None):
            raise ValueError("a datum constraint needs a datum function, and only it")
        if self.kind == "datum" and self.derivative:
            raise ValueError("datum constraints impose values; use a zero constraint for slopes")
        if self.kind == "ratio" and self.derivative:
            raise ValueError("ratio constraints relate the fields' values, not their slopes")

    @property
    def plain_omits(self) -> tuple[str, ...]:
        """Name the fields the plain form leaves out (the position, where a root defines it)."""
        return ("at",) if self.root is not None else ()


@dataclass(frozen=True)
class Datum:
    r"""A stated condition :math:`\partial_a^j f = g` on the slice :math:`x_a = x^*`.

    Data are the boundary and initial conditions printed with a problem, for the *physical*
    solution :math:`f`. On one axis the slice is a point and :math:`g` a constant, written as
    printed (for example ``"-1/8"``); on several axes :math:`g` is a function of the others,
    such as the heat equation's ``"sin(2*pi*x)"`` at :math:`t = 0`.

    Attributes
    ----------
    field
        The field.
    axis
        The axis :math:`a` that the slice fixes.
    at
        The slice position :math:`x^*`.
    value
        The expression :math:`g`, in the other axes' variables.
    derivative
        The order :math:`j` of the derivative along ``axis``.
    """

    field: str
    axis: str
    at: float
    value: str
    derivative: int = 0

    def __post_init__(self) -> None:
        """Check the position and the order."""
        _check_point((self.at,), "a datum position")
        if self.derivative < 0:
            raise ValueError(f"the derivative order must be non-negative, got {self.derivative}")


@dataclass(frozen=True)
class Regular:
    r"""The regular constraint :math:`\mathcal{DC}_R`: a point :math:`x_s` where :math:`u \ne 0`.

    The value there comes from the problem's data when a datum sits at the point, and from the
    reference solution otherwise. It fixes the scale :math:`\eta_e` of the prepared state, and
    it is the collapse point of :math:`\mathbb{D}^{(0)}(x_s)` wherever the residual needs one
    (sources, and linear terms in a doubled space).
    """

    field: str
    at: tuple[float, ...]

    def __post_init__(self) -> None:
        """Check the point."""
        _check_point(self.at, "the regular point")


@dataclass(frozen=True)
class Reference:
    """The solution the method is compared against.

    Attributes
    ----------
    kind
        ``"expression"``, a closed form; ``"bvp"``, the numerical solution of the problem's
        own equation with its two point data as boundary conditions (``scipy.integrate``'s
        ``solve_bvp``, to :attr:`tolerance`); or ``"ivp"``, the numerical solution of a
        first-order system's initial-value problem, every field's value stated at one point
        (``solve_ivp``, to :attr:`tolerance`).
    expression
        The closed form of the *physical* solution of a problem with one field
        (``"expression"`` only).
    expressions
        The closed forms of a problem with several fields, one per field in the problem's order
        (``"expression"`` only, in place of :attr:`expression`).
    tolerance
        The numerical solver's tolerance (``"bvp"`` and ``"ivp"``).
    """

    kind: ReferenceKind
    expression: str | None = None
    expressions: tuple[str, ...] = ()
    tolerance: float = 1.0e-12

    def __post_init__(self) -> None:
        """Check that closed forms are given exactly when they are needed, in one place."""
        if self.kind not in get_args(ReferenceKind):
            raise ValueError(f"unknown reference kind {self.kind!r}")
        given = (self.expression is not None) + bool(self.expressions)
        if (self.kind == "expression") != (given == 1) or given > 1:
            raise ValueError(
                "an expression reference needs an expression or one per field, and only it"
            )


@dataclass(frozen=True)
class Problem:
    """A differential equation with its constraints, data and reference, as pure data.

    Attributes
    ----------
    name
        The rung identifier, such as ``"R2a"``.
    axes
        The independent variables.
    fields
        The unknown functions: one for every rung up to R6; several for a first-order system
        (R7), which share their axes and have no readout weight.
    equations
        One equation per field.
    constraints
        The constraint rows.
    data
        The conditions stated with the problem.
    regular
        The regular point :math:`x_s`.
    reference
        The solution to compare against.
    description
        Free text for reports, such as the paper's figure and panel.
    variant
        ``"faithful"`` for a paper panel's choices as printed, which only the reproduction of the
        paper runs; ``"corrected"``, the default, otherwise.
    """

    name: str
    axes: tuple[Axis, ...]
    fields: tuple[Field, ...]
    equations: tuple[Equation, ...]
    constraints: tuple[Constraint, ...]
    data: tuple[Datum, ...]
    regular: Regular
    reference: Reference
    description: str = ""
    variant: Variant = "corrected"

    def __post_init__(self) -> None:
        """Validate every cross-reference, and compile every expression once."""
        if self.variant not in get_args(Variant):
            raise ValueError(f"unknown variant {self.variant!r}")
        axis_names = [axis.name for axis in self.axes]
        if len(set(axis_names)) != len(axis_names) or not axis_names:
            raise ValueError(f"axis names must be distinct and non-empty, got {axis_names}")
        fields = {entry.name: entry for entry in self.fields}
        if len(fields) != len(self.fields) or not fields:
            raise ValueError("field names must be distinct and non-empty")
        for entry in self.fields:
            positions = [axis_names.index(a) for a in entry.axes if a in axis_names]
            if len(positions) != len(entry.axes) or positions != sorted(set(positions)):
                raise ValueError(f"field {entry.name!r} must use known axes in the problem's order")
            if entry.weight is not None:
                compile_expression(entry.weight, entry.axes)
        if len(self.equations) != len(self.fields):
            raise ValueError("a problem needs exactly one equation per field")
        for equation in self.equations:
            for term in equation.terms:
                self._check_term(term, fields, axis_names)
        items: tuple[Constraint | Datum, ...] = (*self.constraints, *self.data)
        for item in items:
            owner = self._field(item.field, fields)
            if item.axis not in owner.axes:
                raise ValueError(f"axis {item.axis!r} of {item!r} is not an axis of the field")
            function = item.datum if isinstance(item, Constraint) else item.value
            if function is not None:
                compile_expression(function, [a for a in owner.axes if a != item.axis])
        owner = self._field(self.regular.field, fields)
        if len(self.regular.at) != len(owner.axes):
            raise ValueError(f"the regular point needs one coordinate per axis of {owner.name!r}")
        if len(self.fields) > 1:
            self._check_system()
        elif self.reference.expressions:
            raise ValueError("a problem with one field takes one reference expression")
        for expression in (self.reference.expression, *self.reference.expressions):
            if expression is not None:
                compile_expression(expression, axis_names)
        if self.reference.kind == "bvp" and (len(self.fields) != 1 or len(self.axes) != 1):
            raise ValueError("a boundary-value reference needs one field on one axis")
        if self.reference.kind == "ivp" and len(self.axes) != 1:
            raise ValueError("an initial-value reference needs one axis")
        for constraint in self.constraints:
            if constraint.kind == "ratio":
                self._check_ratio(constraint)
        if any(c.root is not None and math.isnan(c.at) for c in self.constraints):
            object.__setattr__(self, "constraints", self._resolve_roots())

    def _check_system(self) -> None:
        """Check a problem with several fields: shared axes, no weights, a closed form each."""
        if len({entry.axes for entry in self.fields}) != 1:
            raise ValueError("the fields of a system must share their axes")
        if any(entry.weight is not None for entry in self.fields):
            raise ValueError("the fields of a system take no readout weight")
        if self.reference.kind == "expression" and len(self.reference.expressions) != len(
            self.fields
        ):
            raise ValueError("a system's reference needs one closed form per field")

    def _check_ratio(self, constraint: Constraint) -> None:
        """Check that a ratio constraint has every field's value on its point."""
        for entry in self.fields:
            if entry.axes != (constraint.axis,):
                raise ValueError("a ratio constraint relates values at a point: one axis only")
            if not any(
                d.field == entry.name
                and d.axis == constraint.axis
                and d.at == constraint.at
                and not d.derivative
                for d in self.data
            ):
                raise ValueError(
                    f"a ratio constraint at {constraint.axis} = {constraint.at} needs a datum "
                    f"of {entry.name!r} there"
                )

    def _resolve_roots(self) -> tuple[Constraint, ...]:
        """Return the constraints with every root recipe resolved to its position."""
        from pihm.problems.reference import exact_root  # the reference module builds on this one

        resolved = []
        for constraint in self.constraints:
            if constraint.root is not None and math.isnan(constraint.at):
                at = exact_root(self, constraint.root.near, constraint.derivative)
                constraint = dataclasses.replace(constraint, at=at)
            resolved.append(constraint)
        return tuple(resolved)

    @staticmethod
    def _field(name: str, fields: dict[str, Field]) -> Field:
        """Return the field called ``name``."""
        if name not in fields:
            raise ValueError(f"unknown field {name!r}")
        return fields[name]

    @classmethod
    def _check_term(cls, term: Term, fields: dict[str, Field], axis_names: list[str]) -> None:
        """Validate one term against the problem's fields and axes."""
        for factor in term.factors:
            owner = cls._field(factor.field, fields)
            if len(factor.derivative) != len(owner.axes):
                raise ValueError(f"{factor!r} needs one derivative order per axis of the field")
        if term.multiplier is not None and term.multiplier.axis not in axis_names:
            raise ValueError(f"unknown multiplier axis {term.multiplier.axis!r}")
        if term.source is not None:
            compile_expression(term.source.expression, axis_names)

    # ------------------------------------------------------------------ derived properties

    @property
    def order(self) -> int:
        """The highest derivative order along any single axis."""
        return max(
            (
                max(factor.derivative, default=0)
                for eq in self.equations
                for term in eq.terms
                for factor in term.factors
            ),
            default=0,
        )

    @property
    def degree(self) -> int:
        """The polynomial degree of the equations in the fields (1 linear, 2 quadratic)."""
        return max(term.degree for equation in self.equations for term in equation.terms)

    @property
    def is_linear(self) -> bool:
        """Whether every term has at most one field factor."""
        return self.degree <= 1

    @property
    def key(self) -> str:
        """The problem's identity in records and reports, ``"R2a"`` or ``"R2a/faithful"``."""
        return f"{self.name}/faithful" if self.variant == "faithful" else self.name

    # ------------------------------------------------------------------ serialisation

    def to_dict(self) -> dict[str, Any]:
        """Return the problem as plain data (dicts, lists, numbers and strings).

        Fields equal to their default are left out, so that adding a new optional field later
        does not change the digest of existing problems.
        """
        plain = to_plain(self)
        assert isinstance(plain, dict)
        return plain

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Problem:
        """Rebuild a problem from :meth:`to_dict` output (or equivalent YAML)."""
        problem = from_plain(cls, data)
        assert isinstance(problem, Problem)
        return problem

    def to_yaml(self) -> str:
        """Return the problem as YAML text."""
        return yaml.safe_dump(self.to_dict(), sort_keys=False, allow_unicode=True)

    @classmethod
    def from_yaml(cls, text: str) -> Problem:
        """Rebuild a problem from :meth:`to_yaml` output."""
        data = yaml.safe_load(text)
        if not isinstance(data, dict):
            raise ValueError("a problem's YAML must be a mapping")
        return cls.from_dict(data)

    @cached_property
    def digest(self) -> str:
        """Sixteen hexadecimal digits of the SHA-256 of the canonical JSON of :meth:`to_dict`."""
        canonical = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode()).hexdigest()[:16]
