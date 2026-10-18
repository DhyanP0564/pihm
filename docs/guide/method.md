# How the method works

The method of Wu et al. {cite}`wu2025pihm` turns a differential equation into a ground-state
problem. This page sets out the pieces and names the module that implements each one; the
{doc}`../reference/glossary` collects the notation.

## The pipeline at a glance

| Step | What happens | Where |
| --- | --- | --- |
| 1. State the problem | Fields, axes, equations, constraints and the exact solution, as data | {mod}`pihm.problems` |
| 2. Assemble | Stack the equation and constraint rows into the residual $R$ | {func}`pihm.assemble.assemble` |
| 3. Solve classically | The ground state of $H = R^\dagger R$ by SVD, and whether to trust it | {func}`pihm.classical.physical_reference` |
| 4. Compile to circuits | Block-encode $R$ and the reflection $V$ as gates; verify against step 2 | {mod}`pihm.circuits`, {mod}`pihm.verify` |
| 5. Prepare the state | A GQSP filter of $V$ applied to an initial state; checked against step 3 | {mod}`pihm.qsp.prepare` |
| 6. Read out | Decode the state into the field and compare with the exact solution | {func}`pihm.classical.decode` |

Steps 4 and 5 are each checked against the classical result of the step before, so a number
reported at the end rests on a chain of verified identities.

## The Chebyshev model

A field on one axis $x \in [-1, 1]$ is represented by $N = 2^n$ Chebyshev coefficients, which
an $n$-qubit register holds as the amplitudes of a state $\psi$. The model is

$$
f_q(x) = \langle \tau(x) | \psi \rangle, \qquad
\tau(x)_k = \phi_k(x) = T_k(x) / s_k,
$$

with $s_0 = 2^{n/2}$ and $s_k = 2^{(n-1)/2}$ for $k \ge 1$, the normalised Chebyshev basis of the
paper ({mod}`pihm.basis`). A field on several axes takes $n$ qubits per axis, flattened in
Kronecker order.

## The residual and the Hamiltonian

Every operation on the field is a matrix on its coefficients: the derivative $\mathbb{G}$,
multiplication by a known function $\mathbb{M}_a$, and evaluation at a point
$\langle \tau(x)|$ ({mod}`pihm.operators`). An equation $\sum_j a_j(x) f^{(j)} = r(x)$ becomes a
block of rows, and each condition on the solution becomes a block of its own. Stacked, they form
the **residual**

$$
R = \begin{bmatrix} R_{\text{equation}} \\ R_{\text{constraint}\,1} \\ \vdots \end{bmatrix},
\qquad H = R^\dagger R .
$$

$H$ is positive semi-definite, and the coefficients that satisfy every block at once are its
ground state: the right singular vector of $R$ with the smallest singular value. A well-posed
problem has a one-dimensional kernel, so the ground state is unique up to a scale.

The constraints come in a few kinds ({mod}`pihm.operators.constraints`):

- an **invariant** (zero) constraint, $f(x^*) = 0$ or $f^{(j)}(x^*) = 0$ at a constraint point
  $x^*$, which is linear and homogeneous;
- the **regular point** $x_s$, where the solution is known and non-zero, which fixes the scale;
- **ratio rows**, which make a system's values at a point parallel to its data there;
- **datum slices**, which impose a known function along a slice of a PDE's domain, such as an
  initial condition.

## From the ground state to the field

A state is normalised, so the ground state fixes the field only up to a factor. The regular
datum supplies it: with $f(x_s)$ known, the physical field is $f_Q = \eta^{1/p} f_q$, with the
scale $\eta$ chosen so that $f_Q(x_s) = f(x_s)$, as the paper does. {func}`~pihm.classical.decode`
does this for any state, the classical ground state or a prepared one, and evaluates the result
on a uniform grid beside the exact solution.

## The classical reference

{func}`~pihm.classical.physical_reference` computes the ground state of the physical subspace by
a dense SVD up to {data}`~pihm.classical.iterative.DENSE_COLUMNS` columns, and by a shift-invert Lanczos
iteration beyond ({mod}`pihm.classical.iterative`). It reports the singular values, the
dimension of the kernel and whether the result is **trusted**: the kernel is one-dimensional and
the ratio of the singular values around it clears {data}`~pihm.classical.reference.DEFAULT_MIN_GAP_RATIO`.
Every circuit-level result in the package is checked against this reference.

## Where the quantum computer comes in

On a quantum computer the residual is **block-encoded**: a circuit $U_R$ whose top-left block is
$R / \alpha_R$, with $\alpha_R \ge \lVert R \rVert$ its subnormalisation. A reflection $V$ built
from it encodes $H' = 2H/\alpha_R^2 - I$, whose spectrum lies in $[-1, 1]$ and whose lowest
eigenvector is the ground state. Generalised quantum signal processing (GQSP) applies a
polynomial filter of $V$ to an initial state, suppressing every excited component; the
{doc}`stateprep` page describes the filters and what they cost.

The cost of each application of $V$ depends on how $R$ is built. In the paper's standard form
$\mathbb{G}$ is dense and its encoding exponentially deep; the descriptor form keeps $R$ banded.
That trade is the subject of {doc}`forms`.
