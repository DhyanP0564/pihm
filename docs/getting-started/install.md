# Installation

pihm needs **Python 3.11**. Its dependencies (NumPy, SciPy, Qiskit, Qiskit Aer and PyYAML) are
pinned to exact versions in `pyproject.toml`, so a fresh environment reproduces the published
numbers.

## Install the library

From the repository root, in a new virtual environment:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e .
```

Check that it imports:

```bash
python -c "import pihm; print(pihm.__version__)"
```

## Optional extras

Each extra adds the tools for one job. Install several at once with a comma, as in
`pip install -e ".[dev,notebooks]"`.

| Extra | Adds | For |
| --- | --- | --- |
| `dev` | pytest, ruff, mypy | Running the tests and the linters |
| `notebooks` | matplotlib, ipykernel, ipywidgets | Running the example notebooks and {mod}`pihm.plotting` |
| `docs` | Sphinx and its extensions, plus `notebooks` | Building this site |

```bash
pip install -e ".[notebooks]"
```

:::{admonition} NumPy and SciPy from PyPI
:class: tip

Install NumPy and SciPy from PyPI wheels, as `pip` does above, rather than from conda. Even then
the platform's linear-algebra library differs between machines (Accelerate on macOS arm64,
OpenBLAS on Linux), so results agree to rounding, not bit for bit. The tests compare values near
the SVD's noise floor with absolute tolerances for this reason.
:::

## Run the tests

```bash
pip install -e ".[dev]"
pytest                                # the fast tests
pytest -m "not slow and not nightly"  # a quicker pass, skipping the slower ones too
pytest -m slow                        # the long-running ones
```

`pytest` also runs the examples in the docstrings, so every example in the
{doc}`API reference <../reference/api/index>` is checked.

## Build this site

```bash
pip install -e ".[docs]"
sphinx-build -n -W --keep-going docs build/docs
```

Then open `build/docs/index.html`. `-n` makes every cross-reference resolve or fail the build, and
`-W` turns warnings into errors. The inventories of Python, NumPy, SciPy and Qiskit are cached in
`docs/_inventories/`, so the build needs no network. The guide's code pages are executed during
the build, which takes about a minute; the example notebooks are shown as saved.
