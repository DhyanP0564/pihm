---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
mystnb:
  execution_mode: force
---

# Quick start

This page solves one equation end to end in a few lines. It is executed each time the site is
built, so every output below is what this version of pihm prints.

## Load a problem

A *problem* is data: its fields and axes, its equations, its constraints and its exact solution.
{mod}`pihm.problems.ladder` holds a ladder of them, from a single first-order ODE (`R1`) to
two-dimensional Navier–Stokes (`R10`). Start with the first:

```{code-cell} python
from pihm.problems import ladder

problem = ladder.problem("R1")
problem.description
```

## Assemble and solve

{func}`~pihm.assemble.assemble` turns the problem into its stacked residual $R$ on $n$ qubits per
axis, that is $N = 2^n$ Chebyshev coefficients. The solution is the ground state of
$H = R^\dagger R$, which is the smallest right singular vector of $R$.
{func}`~pihm.classical.physical_reference` finds it classically.

```{code-cell} python
from pihm.assemble import assemble
from pihm.classical import physical_reference

residual = assemble(problem, 4)  # n = 4: 16 Chebyshev coefficients
state = physical_reference(residual)

print(residual.matrix().shape)  # R: rows x coefficients
print(state.trusted)  # the ground state is unique and well separated
```

`state.trusted` is the check to look at first. It is `True` when the kernel of $H$ is
one-dimensional and the gap above it is wide enough for the ground state to be well defined.

## Read the field back

The ground state holds Chebyshev coefficients. {func}`~pihm.classical.decode` evaluates them on a
uniform grid, scales the result by the problem's regular datum, and sets it beside the exact
solution:

```{code-cell} python
from pihm.classical import decode

field = decode(residual, state.vector)
linf, rel_l2 = field.errors
print(f"max error {linf:.1e}, relative L2 error {rel_l2:.1e}")
```

```{code-cell} python
:tags: [hide-input]

import matplotlib.pyplot as plt

(x,) = field.axes
fig, ax = plt.subplots(figsize=(6, 3.2))
ax.plot(x, field.reference, color="0.75", lw=5, label="exact")
ax.plot(x, field.field, color="C0", lw=1.5, label="decoded ground state")
ax.set(xlabel="$x$", ylabel="$f(x)$", title="$f' + 2f = 0$, $f(-1) = 1$")
ax.legend(frameon=False)
fig.tight_layout()
```

## Both forms

Every problem can be assembled in two forms: the paper's **standard form** and the banded
**descriptor form** ({doc}`../guide/forms`). Pass `regime` to choose; everything downstream is the
same.

```{code-cell} python
for form in ("standard", "descriptor"):
    residual = assemble(problem, 4, regime=form)
    state = physical_reference(residual)
    linf, rel_l2 = decode(residual, state.vector).errors
    print(f"{form:10}  R {residual.matrix().shape}  trusted {state.trusted}  error {linf:.1e}")
```

## Next steps

- {doc}`../guide/method` explains what $R$ is made of and why its ground state is the solution.
- {doc}`../guide/circuits` and {doc}`../guide/stateprep` take the same residual to circuits and
  prepare its ground state with quantum signal processing.
- {doc}`../guide/problems` lists the ladder and shows how to state a problem of your own.
- The {doc}`../examples/index` hold the published results, figure by figure.
