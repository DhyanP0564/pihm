---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
mystnb:
  execution_mode: force
---

# Problems

A problem in pihm is plain data: a frozen {class}`~pihm.problems.spec.Problem` of axes, fields,
equations, constraints, conditions and an exact solution. It serialises to YAML and carries a
stable digest, which every run record stores, so a result always says exactly which problem it
solved.

## The ladder

{mod}`pihm.problems.ladder` holds the problems the published results are computed on, ordered
from the simplest to the hardest. Each *rung* is a family; {func}`~pihm.problems.ladder.problem`
loads one problem by name and {func}`~pihm.problems.ladder.names` lists them.

| Rung | Problems | Equation |
| --- | --- | --- |
| R1 | `R1` | $f' + 2f = 0$, a single first-order ODE |
| R2 | `R2a` to `R2c` | Second-order ODEs with constant coefficients (the paper's Fig. 3); `R2c` is stiff |
| R3 | `R3a` to `R3d` | Associated Legendre equations (Fig. 4) |
| R4 | `R4a` to `R4d` | Second-order ODEs with variable coefficients and sources (Fig. 5) |
| R5 | `R5a` to `R5c` | Laplace, heat and wave equations on two axes (Fig. 6) |
| R6 | `R6a`, `R6b` | Quadratic nonlinear ODEs (Fig. 7) |
| R7 | `R7` | A coupled oscillator, a first-order system of two fields |
| R8 | `R8` | $y' = -y + y^2$, a logistic-type equation for the Carleman lift |
| R9 | `R9a-M4` to `R9e-M16` | Viscous Burgers on 4, 8 or 16 sine modes, at five nonlinearity strengths |
| R10 | `R10a-M12` to `R10c-M316` | Two-dimensional Navier–Stokes on its Fourier modes |

R2 to R6 are the panels of the paper {cite}`wu2025pihm`. Where the paper's printed choices are
inexact (rounded constraint points, truncated Maclaurin sources, hand-picked PDE invariants) the
ladder corrects them; the panels exactly as printed are available as the `"faithful"` variant,
for reproducing the paper.

```{code-cell} python
from pihm.problems import ladder

for name in ladder.names("R4"):
    print(f"{name}: {ladder.problem(name).description}")
```

```{code-cell} python
faithful = ladder.problem("R4b", "faithful")
corrected = ladder.problem("R4b")
print("printed root:", faithful.constraints[0].at)
print("exact root:  ", corrected.constraints[0].at)
```

## Stating your own problem

A problem is built from a few small classes in {mod}`pihm.problems.spec`. As an example, take
$f'' + 4f' + 4f = 0$ on $[-1, 1]$ with $f(0) = 1/2$ and $f'(0) = -1/2$, whose solution is
$f(x) = \tfrac12 (1 + x) e^{-2x}$:

```{code-cell} python
from pihm.problems.spec import (
    Axis, Constraint, Datum, Equation, Factor, Field, Problem, Reference, Regular, Term,
)

problem = Problem(
    name="damped",
    axes=(Axis("x"),),
    fields=(Field("f", ("x",)),),
    # f'' + 4 f' + 4 f = 0: one term per derivative, Factor(field, (order,))
    equations=(
        Equation((
            Term(1.0, (Factor("f", (2,)),)),
            Term(4.0, (Factor("f", (1,)),)),
            Term(4.0, (Factor("f", (0,)),)),
        )),
    ),
    # an invariant constraint, a row of the residual: f(-1) = 0
    constraints=(Constraint("zero", "f", "x", -1.0),),
    # the conditions as stated: f(0) = 1/2 and f'(0) = -1/2
    data=(Datum("f", "x", 0.0, "1/2"), Datum("f", "x", 0.0, "-1/2", 1)),
    # the regular point, where the solution is known and non-zero; it fixes the scale
    regular=Regular("f", (0.0,)),
    reference=Reference("expression", "0.5*(1 + x)*exp(-2*x)"),
)
problem.order, problem.degree
```

The pieces:

`Term(coefficient, factors, multiplier=None, source=None)`
: A constant times a product of field derivatives. `Factor("f", (2,))` is $f''$, with one
  derivative order per axis of the field. One factor makes a linear term and two a quadratic
  one. An optional {class}`~pihm.problems.spec.Polynomial` multiplier makes the coefficient a
  polynomial in $x$. A term with no factors is a known function: a constant, a polynomial, or a
  non-polynomial {class}`~pihm.problems.spec.Source`.

`Constraint(kind, field, axis, at, ...)`
: A row block of the residual. `"zero"` is an invariant constraint $\partial^j f(x^*) = 0$,
  `"datum"` imposes a known function on a slice of a PDE's domain, and `"ratio"` ties a
  system's values at a point to its data.

`Datum` and `Regular`
: The conditions as stated with the problem. They define the exact solution and the scale; they
  enter the residual only through a constraint or the regular point.

`Reference`
: The exact solution, as an expression in the language of {mod}`pihm.problems.expressions`, used
  to measure the error.

The new problem goes through everything else unchanged:

```{code-cell} python
from pihm.assemble import assemble
from pihm.classical import decode, physical_reference

for form in ("standard", "descriptor"):
    residual = assemble(problem, 6, regime=form)
    state = physical_reference(residual)
    linf, _ = decode(residual, state.vector).errors
    print(f"{form:10}  trusted {state.trusted}  max error {linf:.1e}")
```

## Saving and loading

A problem round-trips through YAML, and its digest identifies it:

```{code-cell} python
text = problem.to_yaml()
print(text[:400], "...")
assert Problem.from_yaml(text) == problem
problem.digest
```
