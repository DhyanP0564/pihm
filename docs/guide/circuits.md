---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
mystnb:
  execution_mode: force
---

# Circuits and verification

pihm compiles a residual to gate-level Qiskit circuits, checks each circuit against the matrix
assembled classically, and counts its resources. This page does all three for one problem in
both forms: `R2a`, $f'' + 4f' + 4f = 0$.

## Block encodings

A **block encoding** of a matrix $A$ is a unitary circuit $U$ with $A/\alpha$ in its top-left
block, where the ancilla qubits start and end in $|0\rangle$; $\alpha \ge \lVert A\rVert$ is its
subnormalisation. pihm builds them bottom-up ({mod}`pihm.circuits`): primitive atoms such as the
Chebyshev feature map and the zero projector, linear combinations of unitaries, products sharing
one ancilla register, and finally the stacked residual and its reflection.

{func}`~pihm.circuits.assemble.encode_linear` returns two encodings:

- the **stacked residual** $U_R$, encoding $R/\alpha_R$;
- the **reflection** $V$, encoding $H' = 2H/\alpha_R^2 - I$ with $H = R^\dagger R$. This is the
  operator state preparation queries.

```{code-cell} python
from pihm.circuits.assemble import encode_linear
from pihm.problems import ladder

problem = ladder.problem("R2a")
stacked, reflection = encode_linear(problem, 3, regime="descriptor")
print(f"alpha_R = {stacked.alpha:.1f}")
print(f"V: {reflection.n_sys_qubits} system + {reflection.n_anc_qubits} ancilla qubits")
```

The circuit itself is an ordinary {class}`qiskit.circuit.QuantumCircuit`, `reflection.circuit`.

## Verification

{mod}`pihm.verify` compares a block encoding with a target built independently, from
{mod}`pihm.operators`, never from the circuit. It has three modes:

| Mode | How | When |
| --- | --- | --- |
| `exact` | The full top-left block, one statevector run per basis state | Up to about 20 qubits in all |
| `probe` | A handful of random and structured vectors | Well past `exact`'s ceiling |
| `approximation` | The measured error, reported against a stated bound | Encoders that are approximate by design, such as the paper's own $\mathbb{G}$ |

Here the reflection is checked exactly against $2H/\alpha_R^2 - I$ built from the classically
assembled residual:

```{code-cell} python
import scipy.sparse as sp

from pihm.assemble import assemble
from pihm.verify import verify_exact

residual = assemble(problem, 3, regime="descriptor")
H = residual.hamiltonian()
target = sp.csr_array(2 * H / stacked.alpha**2 - sp.eye_array(H.shape[0]))

check = verify_exact(reflection, target)
print(check.passed, f"max error {check.max_error:.1e}")
```

## Resources

{func}`~pihm.resources.count_resources` reads the qubit counts off the circuit and transpiles it
onto CNOT and single-qubit gates. Compared across resolutions, the two forms differ less in gates
per query than in $\alpha_R$:

```{code-cell} python
from pihm.resources import count_resources

print(f"{'n':>2}  {'form':10}  {'qubits':>6}  {'CNOTs':>7}  {'alpha_R':>10}")
for n in (3, 4, 5):
    for form in ("standard", "descriptor"):
        stacked, reflection = encode_linear(problem, n, regime=form)
        counts = count_resources(reflection)
        print(f"{n:>2}  {form:10}  {counts.n_qubits:>6}  {counts.cx_count:>7}  {stacked.alpha:>10.3g}")
```

The number of queries to $V$ that state preparation needs grows with $\alpha_R^2$, so the
standard form's fast-growing $\alpha_R$ is what makes it expensive; {doc}`stateprep` shows the
consequence. {func}`~pihm.resources.fit_power_law` turns such a sequence into a scaling exponent.

## Which problems compile

Every linear problem of the ladder compiles in both forms: single fields on one or two axes with
any of the constraint kinds, first-order systems, and Carleman lifts. In the standard form the
quadratic R6 compiles in the paper's doubled space. Pass `encoder="published"` to use the paper's
own approximate circuit for $\mathbb{G}$ ({func}`~pihm.circuits.standard.published_derivative_encoding`)
instead of the exact structured one.
