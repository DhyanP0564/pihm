---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
mystnb:
  execution_mode: force
---

# State preparation

The ground state is prepared by generalised quantum signal processing (GQSP): a polynomial
$p(H')$ of the reflection $V$ from {doc}`circuits` is applied to an initial state $\varphi_0$,
and the outcome with every ancilla in $|0\rangle$ is kept. A polynomial that is large at the
bottom of the spectrum and small elsewhere suppresses every excited component, leaving the ground
state.

## The filter

Two filters are available, chosen with `filter=`. Both are fixed a priori from $\alpha_R$ and the
spectrum's edge, and both meet the same target: excited components suppressed to $\varepsilon$
relative to the ground state (`epsilon`, default $10^{-6}$).

`"qite"` (the default)
: Imaginary time, $e^{-\tau(1 + H')}$ as a Chebyshev series, with $\tau$ from the method's a priori
  rule ({mod}`pihm.qsp.filters`).

`"minimax"`
: The Chebyshev polynomial of least deviation on the suppressed part of the spectrum
  ({mod}`pihm.qsp.minimax`). It reaches the same suppression at a lower degree and needs no
  truncated tail; the published results use it.

The degree of either grows with $\alpha_R^2$ over the gap of $H$. That is why the two forms'
values of $\alpha_R$ matter so much: the degree is the number of queries to $V$.

## The initial state

The initial state never reads the solution. The default, `kind="geometric"`, is a product state
tried at $r = \pm\tfrac12$, keeping the one with the larger success probability; `"uniform"` and
`"random"` are its controls ({mod}`pihm.qsp.initial`). The state sits on the field's block of the
descriptor's state and is zero on the derivative blocks and the padding.

## Three tiers

{mod}`pihm.qsp.prepare` offers the same preparation at three levels of fidelity to the circuit:

| Tier | Function | What it does | Reach |
| --- | --- | --- | --- |
| T1 | {func}`~pihm.qsp.prepare.t1` | Simulates the GQSP circuit on a statevector, one controlled query at a time | The smallest resolutions |
| T2 | {func}`~pihm.qsp.prepare.t2` | Applies the same polynomial exactly with sparse matrix products, never forming $H$ | Any resolution with a classical reference |
| T3 | {func}`~pihm.qsp.prepare.t3` | Estimates the degree, $\alpha_R$, the gap and the gates without preparing anything | Beyond the classical reference |

T2 stands in for T1 exactly: by the GQSP theorem the circuit's kept output depends on the
encoding only through its top-left block, which is what T2 applies. Where T1 is affordable the two
agree to rounding ({doc}`../examples/31_circuit_vs_emulation`).

## Preparing a state

T2 on R2a at $n = 3$, in both forms:

```{code-cell} python
from pihm.classical import decode
from pihm.problems import ladder
from pihm.qsp import prepare

problem = ladder.problem("R2a")
for form in ("standard", "descriptor"):
    result = prepare.t2(problem, 3, regime=form, filter="minimax")
    linf, _ = decode(result.residual, result.state).errors
    print(
        f"{form:10}  infidelity {result.infidelity:.1e} (bound {result.infidelity_bound:.1e})"
        f"  success probability {result.success_probability:.2f}  field error {linf:.1e}"
    )
```

`infidelity` is measured against the classical ground state; `infidelity_bound` is the a priori
bound it is judged against. `success_probability` is the chance of keeping the outcome, which
amplitude amplification would raise. The field error is the decoded state's against the exact
solution, so it includes the discretisation error at this $n$ as well as the preparation's.

## Estimating resources

T3 estimates what the same preparation would cost at a resolution too large to prepare. At
$n = 8$ the two forms' filter degrees are orders of magnitude apart:

```{code-cell} python
for form in ("standard", "descriptor"):
    estimate = prepare.t3(problem, 8, regime=form, filter="minimax")
    print(f"{form:10}  degree {estimate.degree:.2e}  alpha_R {estimate.alpha_r:.2e}")
```

`t3` computes every quantity at the target $n$ itself where the classical reference reaches
(`method="direct"`), and otherwise fits power laws to cheaper resolutions (`sample_n=...`,
`method="fitted"`).

## Simulating the circuit

T1 takes the same arguments as T2 and returns, besides the state, its agreement with T2. Its
memory is the statevector's, $2^q$ amplitudes for the circuit's $q$ qubits, and its time is the
degree times one query, so it is practical only at the smallest resolutions:

```python
result = prepare.t1(problem, 2, regime="descriptor", filter="minimax")  # minutes, even at n = 2
```

Inside a campaign, a T1 run first estimates its memory and time ({mod}`pihm.qsp.feasibility`) and
records itself as `infeasible` rather than starting a run that cannot finish.

## The error budget

{mod}`pihm.qsp.budget` splits the distance between the prepared state and the ground state into
its sources: the filter, its truncation, and the GQSP phases. The encodings are exact and the field
is read out classically, so these are the only terms; {doc}`../examples/32_field_and_budget`
follows them through to the field.

## Readout

The field is read out classically from the prepared state's amplitudes
({func}`~pihm.classical.decode`). The paper's interferometric readout, which estimates the field
from overlap measurements, is implemented in {mod}`pihm.qsp.readout` and the `"readout"` stage of
the pipeline, but none of the example campaigns runs it.
