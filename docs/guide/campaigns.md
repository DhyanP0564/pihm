# Campaigns and results

A **run** is one problem at one resolution through one stage of the pipeline. A **campaign** is a
YAML file listing a grid of runs. Each run writes a record, and a campaign's records are
collected into a summary table that the example notebooks read.

## Stages

| Stage | What a run does | Records |
| --- | --- | --- |
| `build` | Assembles the residual and solves it classically | Spectrum, kernel, field errors, the state and fields |
| `encode` | Compiles the circuits, verifies them, counts resources | $\alpha_R$, qubits, gates, the verification |
| `stateprep` | Prepares the ground state at tier T2 or T1 | Infidelity, its bound, success probability, field error |
| `protocol` | Prepares at T2 under variants of QITE's time rule: the gap underestimated, and the paper's own rule | Each variant's infidelity and degree |
| `filters` | Compares the QITE and minimax filters at like error | Each filter's degree and success probability |
| `lift` | Builds a Carleman lift's circuit in the symmetric or tensor layout | Its resources and checks |
| `readout` | The paper's interferometric readout | The field at probe points |

A single run can be made directly with {func}`pihm.pipeline.stages.run` from a
{class}`~pihm.pipeline.spec.RunSpec`; campaigns make many.

## A campaign file

A campaign is a name, a description and a list of groups. Each group is a grid: every listed
problem, variant and resolution, with the options they share.

```yaml
name: my_campaign
description: R2 in both forms, built and prepared.
groups:
  - problems: [R2]               # a rung, or problem names such as R2a
    n: {from: 3, to: 8}          # or a list, such as [3, 4]
    regime: descriptor
    stage: build
  - problems: [R2a]
    n: [3, 4, 5]
    regime: descriptor
    stage: stateprep
    filter: minimax
    epsilon: 1.0e-6
```

The full set of options, with what each one does, is documented in {mod}`pihm.pipeline.campaign`.

## Running a campaign

`examples/run_campaign.py` runs a campaign file:

```bash
# run every run, then collect the summary
python examples/run_campaign.py my_campaign.yaml --out results/my_campaign

# inspect before running
python examples/run_campaign.py my_campaign.yaml --count       # how many runs
python examples/run_campaign.py my_campaign.yaml --list        # number them

# run one run, as a cluster array task would
python examples/run_campaign.py my_campaign.yaml --out results/my_campaign --task 3

# gather the records into summary.csv and audit them
python examples/run_campaign.py my_campaign.yaml --out results/my_campaign --collect
```

Each run writes its own record (JSON, with NPZ arrays where a run keeps fields) to a directory
named by its configuration's hash, so an interrupted campaign resumes where it stopped and a
finished run is never repeated. `--rerun` replaces existing records and `--prune` removes records
the campaign no longer contains.

### On a cluster

`--task i` runs a single run, so a job array can run one task per run. A group may state the
memory and hours its runs need (`memory: 32`, `hours: 24`); `--classes` prints the task numbers
of each allocation, one job array per allocation. The largest example campaigns (the
Navier–Stokes and Carleman builds and the T1 circuit simulations) need tens of gigabytes of memory
and hours per run.

### Records and their audit

A record holds the run's specification, its metrics, its timings and its provenance: the commit,
the installed versions of the dependencies and a **source digest** of every module the run
imports ({func}`pihm.pipeline.records.source_digest`). `--collect` audits the records as it
gathers them, reporting refusals, runs that never started and records made by a different source.

## Reading results back

`examples/campaigns.yaml` names the directory of each campaign's records.
{class}`~pihm.pipeline.results.Results` reads it, and returns each campaign's summary rows:

```python
from pathlib import Path

from pihm.pipeline.results import Results

results = Results.at(Path("."))  # the repository root
print(results.describe(["p2_reproduction"]))
rows = results.rows("p2_reproduction", problem="R1")
```

{mod}`pihm.pipeline.results` also holds the selections the notebooks share, such as each case's
reference build or the cost comparison between the forms.

## The example campaigns

`examples/campaigns/` holds the campaigns behind the published results. Their names are grouped
by topic:

| Campaigns | Contents |
| --- | --- |
| `p2_reproduction` | The paper's panels (R1 to R6) in the standard form, as printed and corrected |
| `p3_pillar1` | Standard-form circuit encodings, verified against the assembled operators |
| `p4_*` | Standard-form state preparation: T2, T1 circuit simulation, T3 estimates, the protocol's choices, filters |
| `p5_*` | The same in the descriptor form, and the two standard-form encoders compared |
| `p6_*` | First-order systems and the Carleman lift |
| `p7_*` | Burgers and Navier–Stokes |

Their records are published as an archive; to use them instead of rerunning, see
{doc}`../examples/index`.
