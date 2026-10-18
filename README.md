# pihm

Physics-informed effective Hamiltonians: a Python library for solving differential equations with
the method of Wu et al. ([arXiv:2504.13174](https://arxiv.org/abs/2504.13174)). The solution is
encoded as the ground state of an effective Hamiltonian built from the equations' residual.

**Documentation:** <https://dhyanp0564.github.io/pihm/>

Two formulations are implemented, each verified at the level of operators, circuits and prepared
states:

- **Standard form**, the formulation of the paper.
- **Descriptor form**, a reformulation whose residual is banded, so its block encoding's
  subnormalisation, and with it the number of circuit queries that state preparation takes,
  grows far more slowly with resolution.

## Install

Requires Python 3.11.

```bash
git clone https://github.com/DhyanP0564/pihm.git
cd pihm
pip install -e .          # library
pip install -e ".[dev]"   # plus the test and lint tools
pip install -e ".[notebooks]"  # plus matplotlib and Jupyter widgets, for the notebooks
```

## Layout

| Package | Contents |
| --- | --- |
| `pihm.problems` | Problem specifications and reference solutions |
| `pihm.basis`, `pihm.operators` | Chebyshev basis and the operator library |
| `pihm.assemble` | Residual and effective-Hamiltonian assembly (standard and descriptor) |
| `pihm.circuits` | Block-encoding and GQSP circuits (Qiskit) |
| `pihm.qsp` | Filters, phase finding and state preparation |
| `pihm.classical` | Classical reference solvers |
| `pihm.carleman` | Carleman lift for polynomial nonlinearities |
| `pihm.verify`, `pihm.resources` | Verification modes and resource counts |
| `pihm.pipeline` | Campaigns, run records and their analysis |
| `pihm.plotting` | The notebooks' figures and solution viewer (the `notebooks` extra) |

## Reproducing the reported numbers

`examples/campaigns/` holds one YAML file per set of runs behind the reported numbers: the
reproduction of the paper's panels, the circuit encodings, state preparation, filters,
resource estimates and the Carleman-lifted nonlinear cases. `examples/campaigns.yaml` lists where
each campaign's records go.

```bash
python examples/run_campaign.py examples/campaigns/p2_reproduction.yaml --out results/p2
python examples/run_campaign.py examples/campaigns/p4_t1.yaml --list    # number the runs
python examples/run_campaign.py examples/campaigns/p4_t1.yaml --out results/p4_t1 --task 3
```

Each run writes its own record, so an interrupted campaign resumes where it stopped, and `--task i`
runs a single run, which lets a cluster job array run one task per run. `--collect` gathers the
records into a `summary.csv` and audits them. Some campaigns are large (tens of GB of memory, many
hours); `--classes` prints the allocation each group of runs asks for. Collected records are read
back with `pihm.pipeline.results.Results`.

## Notebooks and results

`notebooks/` holds one notebook per question the results answer, from the reproduction of the
paper to two-dimensional Navier–Stokes; [`notebooks/README.md`](notebooks/README.md) lists them.
They are saved with their outputs, so they can be read as they are.

To run them, download the records of every campaign from Zenodo,
[doi:10.5281/zenodo.XXXXXXX](https://doi.org/10.5281/zenodo.XXXXXXX) (`pihm-results.tar.gz`,
about 250 MB), and unpack the archive at the repository root, where it creates `results/`:

```bash
pip install -e ".[notebooks]"
tar -xzf pihm-results.tar.gz
```

The records can also be made from scratch with the campaigns above, though the largest need a
cluster.

## Documentation

The documentation is published at <https://dhyanp0564.github.io/pihm/>: a quick start, a user
guide (the method, the problems, the two forms, systems and the Carleman lift, circuits, state
preparation and campaigns), the example notebooks with their figures, and the API reference with
a glossary. To build it locally:

```bash
pip install -e ".[docs]"
sphinx-build -n -W --keep-going docs build/docs
open build/docs/index.html
```

## Tests

```bash
pytest              # fast tests
pytest -m slow      # long-running tests
```

## Citing

If you use pihm, please cite it ([`CITATION.cff`](CITATION.cff); GitHub's "Cite this repository"
button gives the formats) and the method it implements, Wu et al., arXiv:2504.13174.

## Licence

MIT; see [`LICENSE`](LICENSE).
