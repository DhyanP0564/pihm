# Examples

Each notebook answers one question from the records of the example campaigns. It reads the
records and draws them; it never reruns a campaign. The notebooks are shown here as saved, with
their figures, so they can be read without installing anything.

```{toctree}
:hidden:
:maxdepth: 1

01_solutions
10_reproduction
11_faithful_vs_corrected
12_paper_quantum_claims
20_circuits_exact
21_encoding_cost
30_stateprep_accuracy
31_circuit_vs_emulation
32_field_and_budget
33_protocol
34_filters
40_where_the_cost_lies
41_forms_head_to_head
42_block_rescaling
43_representative_problems
60_coupled_fields
61_carleman_lift
62_burgers
70_navier_stokes
```

## Solutions

Any case's solution, in both forms, with an interactive viewer.

| Notebook | Question | Campaigns |
| --- | --- | --- |
| {doc}`01_solutions` | What does a case's solution look like: its decoded field in both forms against the exact solution, the matrices behind it, and its metrics from every stage? It holds an interactive viewer of every case. | every build, encode, state-preparation and lift campaign |

## Reproducing the paper

The standard form built as the paper prints it, and what its printed choices cost.

| Notebook | Question | Campaigns |
| --- | --- | --- |
| {doc}`10_reproduction` | Does the standard form, built as printed, reproduce Wu et al. (arXiv:2504.13174): the printed scale factors η_e and the figures at the printed resolution? | `p2_reproduction` |
| {doc}`11_faithful_vs_corrected` | What do the paper's printed choices cost (rounded constraint points, Maclaurin sources, hand-picked PDE invariants), and do the corrected variants remove it? | `p2_reproduction` |
| {doc}`12_paper_quantum_claims` | Do the paper's claims about its Hamiltonians and their preparation hold: H positive definite, the doubled space's kernel, its imaginary-time rule, and preparing the nonlinear solution? | `p2_reproduction`, `p4_stateprep`, `p4_protocol` |

## Circuits

Whether each form's circuit is the operator assembled classically, and what one query costs.

| Notebook | Question | Campaigns |
| --- | --- | --- |
| {doc}`20_circuits_exact` | Is each form's circuit the operator that was assembled classically: does the stacked reflection V encode 2H/α_R² − I exactly? | `p3_pillar1`, `p5_encode` |
| {doc}`21_encoding_cost` | What does one query of the walk cost in each form: its looseness α_R/‖R‖, its ancilla and its gates, and how do they grow with n? | `p3_pillar1`, `p5_encode`, `p2_reproduction`, `p5_build` |

## State preparation

How accurately the ground state is prepared, by circuit and by emulation, and what the protocol's choices cost.

| Notebook | Question | Campaigns |
| --- | --- | --- |
| {doc}`30_stateprep_accuracy` | Does the prepared state reach the target infidelity ε, within its a priori bound, in both forms? | `p5_stateprep` |
| {doc}`31_circuit_vs_emulation` | Does the GQSP circuit, simulated (T1), prepare what the exact emulation (T2) predicts, so that T2 can stand in for it where the circuit is too large to simulate? | `p4_t1`, `p5_t1`, `p4_stateprep`, `p5_stateprep` |
| {doc}`32_field_and_budget` | How accurate is the field decoded from the prepared state, and where does its error come from? | `p5_stateprep` |
| {doc}`33_protocol` | What do the protocol's answer-independent choices cost: the geometric initial state against its controls, and a gap underestimated when the imaginary time is chosen? | `p5_stateprep`, `p4_protocol` |
| {doc}`34_filters` | Which filter should prepare the state: imaginary time (QITE) or the minimax edge filter? | `p4_filters`, `p5_filters` |

## The two forms compared

Where the standard form's cost lies, and where the descriptor form saves.

| Notebook | Question | Campaigns |
| --- | --- | --- |
| {doc}`40_where_the_cost_lies` | Is the standard form's cost in its encoder or in its Hamiltonian, and what does the descriptor form change? | `p5_encoders`, `p5_t3`, `p2_reproduction`, `p5_build`, `p5_stateprep` |
| {doc}`41_forms_head_to_head` | Case by case, where does the descriptor form cost less than the standard form, and by how much? | `p5_t3`, `p2_reproduction`, `p5_build` |
| {doc}`42_block_rescaling` | What does the descriptor's block rescaling buy, in its gap and in the field block's weight, and where does it stop? | `p5_build`, `p2_reproduction` |
| {doc}`43_representative_problems` | How do the two forms compare on the three problems that test them hardest: a stiff equation (R2c), an exact kernel (R3a) and two axes (R5b)? | `p2_reproduction`, `p5_build`, `p5_t3` |

## Systems and nonlinear equations

Coupled fields, the Carleman lift and viscous Burgers.

| Notebook | Question | Campaigns |
| --- | --- | --- |
| {doc}`60_coupled_fields` | Do both forms carry a coupled system's fields, and what does the descriptor's mass form buy over carrying each field's derivative as a block of its own? | `p6_build`, `p6_stateprep`, `p6_t3` |
| {doc}`61_carleman_lift` | Does the Carleman lift give a nonlinear equation one ground state in both forms, how fast does it converge in its order, and how does its kernel compare with the paper's doubled space? | `p6_build`, `p6_stateprep`, `p6_t3`, `p2_reproduction` |
| {doc}`62_burgers` | Where does the Carleman lift converge on viscous Burgers, does the ratio R predict it, and what does a lift that fails look like? | `p6_build`, `p6_encode`, `p6_t3` |

## Navier–Stokes

How far the method reaches on two-dimensional Navier–Stokes.

| Notebook | Question | Campaigns |
| --- | --- | --- |
| {doc}`70_navier_stokes` | How far does the method reach on two-dimensional Navier–Stokes: does the Carleman lift converge, what does the descriptor's time axis save at one order, and what does the lift's coupling cost on a circuit as the modes grow? | `p7_build`, `p7_t3`, `p7_lift` |

## Running the notebooks yourself

The notebooks need the records of the campaigns they read. Either download the archive of every
campaign's records, [doi:10.5281/zenodo.XXXXXXX](https://doi.org/10.5281/zenodo.XXXXXXX)
(`pihm-results.tar.gz`, about 250 MB), and unpack it at the repository root, where it creates
`results/`, or run the campaigns yourself ({doc}`../guide/campaigns`).

```bash
pip install -e ".[notebooks]"
tar -xzf pihm-results.tar.gz
```

Then open a notebook in Jupyter or VS Code with this environment's kernel. The first cell of each
notebook lists the campaigns it reads and how many records each holds; a campaign with no records
is reported as missing.
