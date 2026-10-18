---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
mystnb:
  execution_mode: force
---

# Systems and nonlinear equations

## First-order systems

A first-order system $z' = \mathbb{A}z$ of several fields on a Chebyshev time axis assembles into
a Kronecker-structured residual in either form ({mod}`pihm.assemble.systems`). Its initial data
enter as ratio rows, which make the fields' values at the initial time parallel to the data;
the regular datum fixes the scale. R7, a coupled oscillator $u' = v$, $v' = -\pi^2 u$, is the
smallest example:

```{code-cell} python
from pihm.assemble import assemble
from pihm.classical import decode, physical_reference
from pihm.problems import ladder

problem = ladder.problem("R7")
residual = assemble(problem, 5, regime="descriptor")
field = decode(residual, physical_reference(residual).vector)
print(problem.description)
print("decoded fields:", field.field.shape)  # (fields, grid points)
print(f"max error {field.errors[0]:.1e}")
```

In the descriptor form a first-order system needs no derivative blocks: it is carried in *mass
form*, the standard form's rows multiplied by the banded $I \otimes \tilde{\mathbb{B}}$.

## The Carleman lift

A polynomial nonlinear system $u' = F_1 u + F_2\, u \otimes u + \dots$ has no linear residual of
its own. {mod}`pihm.carleman` lifts it to the linear system of its monomials
$z_\alpha = \prod_i u_i^{\alpha_i}$ up to degree $K$, dropping every monomial above $K$:

$$
\frac{dz}{dt} = \mathbb{A}_K z .
$$

The truncated system is linear with a unique solution, so its residual has a one-dimensional
kernel in either form, and everything downstream (circuits, state preparation) applies
unchanged. Its error falls as $O(\mathsf{R}^K)$ with
$\mathsf{R} = \lVert u_0\rVert\,\lVert F_2\rVert / |\mathrm{Re}\,\lambda_1(F_1)|$
({func}`~pihm.carleman.nonlinearity_ratio`), so the lift converges when $\mathsf{R} < 1$.

Pass the order as `carleman=K` to {func}`~pihm.assemble.assemble`. On R8, $y' = -y + y^2$,
the error falls geometrically with the order:

```{code-cell} python
problem = ladder.problem("R8")
orders = range(1, 7)
errors = []
for K in orders:
    residual = assemble(problem, 5, regime="descriptor", carleman=K)
    errors.append(decode(residual, physical_reference(residual).vector).errors[0])
    print(f"K = {K}: R is {residual.matrix().shape}, max error {errors[-1]:.1e}")
```

```{code-cell} python
:tags: [hide-input]

import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(5, 3))
ax.semilogy(list(orders), errors, "o-")
ax.set(xlabel="Carleman order $K$", ylabel="max error of $y$", title="R8, descriptor form, $n = 5$")
fig.tight_layout()
```

Two layouts of the lifted state are available. The *symmetric* layout carries each monomial once,
$\binom{M+k-1}{k}$ components at level $k$; the *tensor* layout carries $u^{\otimes k}$ in full,
$M^k$ components in $k$ slots, whose matrix is a short sum of structured terms that suits a
register of slots ({func}`~pihm.carleman.tensor_terms`). Burgers (R9) and Navier–Stokes (R10)
are lifted the same way; see {doc}`../examples/62_burgers` and
{doc}`../examples/70_navier_stokes`.

## The doubled space

For a single quadratic equation, the paper instead works in the doubled space
$\Psi = \psi \otimes \psi$, where the quadratic term becomes linear. pihm builds it in the standard
form with `nonlinearity="doubled"`. Its kernel is degenerate, so state preparation there targets
the initial state's projection onto the kernel rather than a single ground state.
{doc}`../examples/12_paper_quantum_claims` tests the paper's claims about it, and
{doc}`../examples/61_carleman_lift` compares it with the Carleman lift.
