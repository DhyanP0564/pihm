# The two forms

Every problem can be assembled in two forms. Both give the same solution; they differ in how the
residual carries derivatives, and so in what its block encoding costs.

```python
assemble(problem, n, regime="standard")  # the paper's formulation (the default)
assemble(problem, n, regime="descriptor")  # the banded reformulation
```

## Standard form

The paper's formulation. A derivative is the dense matrix
$\mathbb{G} = \tilde{\mathbb{B}}^{-1}\tilde{\mathbb{D}}$ on the Chebyshev coefficients, and for
$\sum_j a_j(x) f^{(j)} = r(x)$ the equation block is

$$
\sum_j \mathbb{M}_{a_j}\mathbb{G}^j - \mathbb{M}_r\mathbb{D}^{(0)}(x_s),
$$

where $\mathbb{M}_a$ is multiplication by $a(x)$ and $\mathbb{D}^{(0)}(x_s)$ is the rank-one
regular datum. The state is the field's coefficients alone.

## Descriptor form

The descriptor form never forms $\mathbb{G}$. It keeps the two banded factors
$\tilde{\mathbb{B}}$ and $\tilde{\mathbb{D}}$ instead, so the residual is banded.

Higher-order equations
: The state carries the derivatives as blocks of their own, $w = (p_0; \dots; p_k)$, tied by
  the recurrences $-\tilde{\mathbb{D}}p_j + \tilde{\mathbb{B}}p_{j+1} = 0$. $\tilde{\mathbb{B}}$ is
  invertible, so the kernels of the two forms correspond exactly.

Block rescaling
: The block of order $m$ along an axis is carried as $\lambda^{-m}p$, with $\lambda$ chosen a
  priori from the equation's coefficients. The kernel is unchanged, and the blocks' weights are
  balanced. Pass `rescale="none"` to switch it off.

Padding
: The number of blocks is rounded up to a power of two; the extra blocks are pinned to zero and
  decouple exactly.

First-order equations and systems
: These need no derivative blocks. They are carried in *mass form*, the standard form's rows
  multiplied by the banded $I \otimes \tilde{\mathbb{B}}$.

## What the difference costs

Both forms block-encode with polynomially many gates per query. The difference is in the
subnormalisation $\alpha_R$, the factor by which the circuit scales $R$ down. In the standard
form the dense powers of $\mathbb{G}$ make $\alpha_R$ grow as a high power of $N$; in the
descriptor form the bands keep it small. The filter that prepares the ground state needs a
degree, a number of queries, that grows with $\alpha_R^2$ over the gap of $H$, so it is the
number of queries rather than the cost of each that separates the forms. The
{doc}`circuits` page measures both for one problem, and the examples
{doc}`../examples/40_where_the_cost_lies` and {doc}`../examples/41_forms_head_to_head` compare
them across the ladder.

## Constraints

Both forms share the same constraint rows: a zero constraint
$\mathbb{B}(x) = \sqrt{N}\,|0\rangle\langle\tau(x)|$ for $f(x) = 0$ or a derivative's, the regular
datum, ratio rows for vector data, and datum slices for an initial or boundary function. In the
descriptor form a constraint on $f^{(j)}$ reads the block that carries it, rather than applying
$\mathbb{G}^j$.

## Choosing a form

| | Standard form | Descriptor form |
| --- | --- | --- |
| State | The field's coefficients | The field and its derivatives, padded to a power of two |
| Residual | Dense in the derivative | Banded |
| Subnormalisation $\alpha_R$ | Grows quickly with $N$ | Grows slowly with $N$ |
| Reproduces the paper | Yes, exactly | No; same solution, different operator |
| Doubled space for quadratic equations | Yes | No; use the Carleman lift |

Use the standard form to reproduce the paper's construction, and the descriptor form to prepare
states at resolutions where the standard form's filter degree is out of reach.
