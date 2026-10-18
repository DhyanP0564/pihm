# User guide

The guide follows a problem through pihm, from its statement to a prepared quantum state and
the records of a campaign. Each page can be read on its own; the pages with code are executed
when the site is built.

```{toctree}
:hidden:
:maxdepth: 2

method
problems
forms
systems
circuits
stateprep
campaigns
```

::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item-card} How the method works
:link: method
:link-type: doc

The residual $R$, the effective Hamiltonian $H = R^\dagger R$, the Chebyshev basis, and how a
ground state becomes a field.
:::

:::{grid-item-card} Problems
:link: problems
:link-type: doc

The ladder of problems from R1 to R10, and how to state an equation of your own.
:::

:::{grid-item-card} The two forms
:link: forms
:link-type: doc

The paper's standard form and the banded descriptor form, and why the second is cheaper to
encode.
:::

:::{grid-item-card} Systems and nonlinear equations
:link: systems
:link-type: doc

First-order systems, the Carleman lift of a polynomial nonlinearity and the doubled space.
:::

:::{grid-item-card} Circuits and verification
:link: circuits
:link-type: doc

Block-encoding the residual as gates, checking the circuit against the matrix, counting
resources.
:::

:::{grid-item-card} State preparation
:link: stateprep
:link-type: doc

Preparing the ground state with GQSP: filters, initial states and the three tiers T1, T2 and T3.
:::

:::{grid-item-card} Campaigns and results
:link: campaigns
:link-type: doc

Running sets of runs from YAML, on a laptop or a cluster, and reading their records back.
:::
::::
