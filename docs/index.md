---
html_theme.sidebar_secondary.remove: true
---

# pihm

:::{rst-class} lead
Solve differential equations as the ground state of an effective Hamiltonian, and check every
step from the operator to the prepared quantum state.
:::

pihm implements the physics-informed effective Hamiltonian method of Wu et al. {cite}`wu2025pihm`.
A differential equation and its conditions are turned into a residual matrix $R$; the solution's
Chebyshev coefficients are the ground state of $H = R^\dagger R$. pihm builds $R$, finds its
ground state classically, compiles it to gate-level circuits, prepares the ground state with
quantum signal processing, and verifies each stage against the one before.

It does this in two formulations: the paper's **standard form**, and a **descriptor form** whose
residual is banded. The banded residual keeps the block encoding's subnormalisation small, and
with it the number of circuit queries that preparing the ground state takes.

```python
from pihm.assemble import assemble
from pihm.classical import decode, physical_reference
from pihm.problems import ladder

residual = assemble(ladder.problem("R2a"), 5, regime="descriptor")
state = physical_reference(residual)
print(decode(residual, state.vector).errors)  # L-infinity and relative L2 error of the field
```

::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item-card} {fas}`rocket` Getting started
:link: getting-started/index
:link-type: doc

Install pihm and solve a first equation in both forms in a few lines.
:::

:::{grid-item-card} {fas}`book-open` User guide
:link: guide/index
:link-type: doc

How the method works, the problems it solves, the two forms, circuits, state preparation and
running campaigns.
:::

:::{grid-item-card} {fas}`chart-line` Examples
:link: examples/index
:link-type: doc

Nineteen notebooks, each answering one question from the published results, with their figures.
:::

:::{grid-item-card} {fas}`code` API reference
:link: reference/index
:link-type: doc

Every module, class and function, with the notation and terms used throughout.
:::
::::

```{toctree}
:hidden:

getting-started/index
guide/index
examples/index
reference/index
```
