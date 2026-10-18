# Glossary

## Notation

| Symbol | Meaning |
| --- | --- |
| $n$ | Qubits per axis of a field |
| $N = 2^n$ | Chebyshev coefficients per axis |
| $T_k$, $\phi_k = T_k / s_k$ | Chebyshev polynomials, and the normalised basis the state holds ({mod}`pihm.basis`) |
| $\tau(x)$ | The feature vector $(\phi_0(x), \dots, \phi_{N-1}(x))$; the model is $f_q(x) = \langle\tau(x)\vert\psi\rangle$ |
| $\psi$, $w$ | The state: the field's coefficients, or in the descriptor form the field and its derivative blocks |
| $f_q$, $f_Q$ | The model field from the state, and the physical field after scaling by the regular datum |
| $\eta$ | The scale between them, $f_Q = \eta^{1/p} f_q$, fixed at the regular point |
| $\mathbb{G}$ | The dense derivative matrix on Chebyshev coefficients, $\tilde{\mathbb{B}}^{-1}\tilde{\mathbb{D}}$ |
| $\tilde{\mathbb{B}}$, $\tilde{\mathbb{D}}$ | Its two banded factors, which the descriptor form uses directly |
| $\mathbb{M}_a$ | Multiplication by a known function $a(x)$ |
| $\mathbb{B}(x)$, $\mathbb{D}^{(0)}(x_s)$ | The rank-one constraint row at $x$, and the regular datum at $x_s$ |
| $R$ | The stacked residual: the equation's rows and the constraints' |
| $H = R^\dagger R$ | The effective Hamiltonian; its ground state is the solution |
| $\alpha_R$ | The subnormalisation of $R$'s block encoding, $\alpha_R \ge \lVert R\rVert$ |
| $H' = 2H/\alpha_R^2 - I$ | The operator the reflection $V$ encodes, with spectrum in $[-1, 1]$ |
| $\Delta$ | The gap of $H$ above its ground state |
| $\varepsilon$ | The target suppression of the excited spectrum (`epsilon`) |
| $\gamma$ | The initial state's overlap with the ground state |
| $\lambda$ | The descriptor form's block-rescaling factor |
| $K$ | The order of a Carleman lift |
| $\mathsf{R}$ | A Carleman lift's nonlinearity ratio; the lift converges when it is below 1 |

## Terms

```{glossary}
block encoding
  A unitary circuit with a scaled copy of a matrix, $A/\alpha$, in its top-left block, the
  ancillas starting and ending in $|0\rangle$ ({class}`~pihm.circuits.primitives.BlockEncoding`).

campaign
  A YAML file listing a grid of runs, executed by `examples/run_campaign.py`
  ({mod}`pihm.pipeline.campaign`).

Carleman lift
  The linear system of a polynomial system's monomials up to order $K$, which replaces a
  nonlinear equation by a linear one ({mod}`pihm.carleman`).

constraint point
  The point $x^*$ at which an invariant constraint sets the field or a derivative to zero.

corrected variant
  A ladder problem with the paper's inexact printed choices replaced by exact ones. It is the
  default; see *faithful variant*.

datum slice
  A constraint that imposes a known function along a slice of a PDE's domain, such as its
  initial condition.

descriptor form
  The banded reformulation of the residual, carrying derivatives as blocks of the state
  ({doc}`../guide/forms`).

doubled space
  The paper's treatment of a quadratic equation, on the state $\psi \otimes \psi$, in which the
  quadratic term is linear.

faithful variant
  A ladder problem exactly as the paper printed it, used to reproduce the paper
  (`ladder.problem(name, "faithful")`).

GQSP
  Generalised quantum signal processing: a polynomial of a unitary applied through one ancilla
  and a sequence of phases ({mod}`pihm.circuits.gqsp`, {mod}`pihm.qsp.phases`).

invariant constraint
  A homogeneous constraint, $\partial^j f(x^*) = 0$, which the solution satisfies whatever its
  scale.

ladder
  The problems R1 to R10, from a single ODE to Navier–Stokes ({mod}`pihm.problems.ladder`). Each
  family is a *rung*.

mass form
  How the descriptor form carries a first-order equation or system: the standard form's rows
  multiplied by the banded $I \otimes \tilde{\mathbb{B}}$.

minimax filter
  The Chebyshev polynomial of least deviation on the suppressed part of the spectrum
  ({mod}`pihm.qsp.minimax`).

Pillar 1
  The circuit layer: block-encoding the residual and verifying the circuit against the classical
  operator. The `encode` stage and the `p3_pillar1` campaign.

Pillar 2
  The state-preparation layer: preparing the ground state with GQSP, and its readout
  ({mod}`pihm.pipeline.pillar2`).

QITE
  Quantum imaginary-time evolution, $e^{-\tau(1 + H')}$, applied as a Chebyshev series; the
  default filter ({mod}`pihm.qsp.filters`).

ratio rows
  The constraint rows that make a system's values at a point parallel to its data there.

record
  The output of one run: its specification, metrics, timings and provenance, as JSON with NPZ
  arrays ({mod}`pihm.pipeline.records`).

reflection
  The block encoding $V$ of $H' = 2H/\alpha_R^2 - I$ that state preparation queries.

regular point
  The point $x_s$ where the solution is known and non-zero; it fixes the field's scale.

residual
  The stacked matrix $R$ of a problem's equation and constraint rows
  ({class}`~pihm.assemble.residual.Residual`).

source digest
  A hash of every module a run imports, stored in each record, so that a record says which
  source made it ({func}`~pihm.pipeline.records.source_digest`).

standard form
  The paper's formulation, with the dense derivative matrix $\mathbb{G}$.

T1, T2, T3
  The three tiers of state preparation: the circuit simulated, the exact polynomial emulation,
  and the resource estimate ({mod}`pihm.qsp.prepare`).

trusted
  Said of a classical ground state whose kernel is one-dimensional and well separated from the
  rest of the spectrum ({func}`~pihm.classical.physical_reference`).
```
