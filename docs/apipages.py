"""Write the API reference's pages: one per group of modules, and one per module.

The groups below are the API reference's contents, in the package's dependency order. Each build
writes ``reference/api/<group>.rst`` (the group's modules in a table) and
``reference/api/generated/<module>.rst`` (the module's docstring, summary tables of what its
``__all__`` exports, then each export in full). Modules are imported by their full name, so a
package that re-exports a function under its submodule's name (``pihm.operators.multiplication``)
still gets the module's page.
"""

from __future__ import annotations

import importlib
import inspect
import re
from pathlib import Path
from types import ModuleType

from sphinx.pycode import ModuleAnalyzer

#: (page, title, introduction, modules) for each group, in dependency order.
GROUPS: list[tuple[str, str, str, list[str]]] = [
    (
        "core",
        "Spectral core",
        "The normalised Chebyshev basis and the operators built on it: derivatives, "
        "multiplication, folds, constraint rows and tensor products.",
        [
            "pihm.basis",
            "pihm.operators",
            "pihm.operators.derivative",
            "pihm.operators.multiplication",
            "pihm.operators.folds",
            "pihm.operators.constraints",
            "pihm.operators.tensor",
        ],
    ),
    (
        "problems",
        "Problems",
        "Problems as data: their specification, the expression language for known functions, "
        "exact solutions, the problem ladder R1 to R10 and the Carleman lift of a polynomial "
        "nonlinearity.",
        [
            "pihm.problems",
            "pihm.problems.spec",
            "pihm.problems.expressions",
            "pihm.problems.plain",
            "pihm.problems.functions",
            "pihm.problems.reference",
            "pihm.problems.ladder",
            "pihm.problems.fluids",
            "pihm.problems.polynomial",
            "pihm.problems.periodic",
            "pihm.problems.scales",
            "pihm.carleman",
        ],
    ),
    (
        "assemble",
        "Assembly",
        "The stacked residual :math:`R` of a problem in either form, and its structured term "
        "representation.",
        [
            "pihm.assemble",
            "pihm.assemble.residual",
            "pihm.assemble.standard",
            "pihm.assemble.descriptor",
            "pihm.assemble.systems",
            "pihm.ir",
        ],
    ),
    (
        "classical",
        "Classical reference",
        r"The ground state of :math:`H = R^\dagger R` computed classically, whether to trust it, "
        "and the field decoded from any state.",
        ["pihm.classical", "pihm.classical.reference", "pihm.classical.iterative"],
    ),
    (
        "circuits",
        "Circuits",
        "Gate-level block encodings of the residual in Qiskit, from primitive atoms to the "
        "stacked reflection and the GQSP circuit.",
        [
            "pihm.circuits",
            "pihm.circuits.primitives",
            "pihm.circuits.arithmetic",
            "pihm.circuits.lcu",
            "pihm.circuits.standard",
            "pihm.circuits.multiplication",
            "pihm.circuits.sources",
            "pihm.circuits.folds",
            "pihm.circuits.compose",
            "pihm.circuits.systems",
            "pihm.circuits.tensor",
            "pihm.circuits.assemble",
            "pihm.circuits.gqsp",
        ],
    ),
    (
        "verify",
        "Verification and resources",
        "Checking a block encoding against the operator assembled classically, and counting its "
        "qubits and gates.",
        ["pihm.verify", "pihm.resources"],
    ),
    (
        "qsp",
        "Signal processing",
        "Filters, GQSP phase finding, initial states, state preparation in three tiers, the "
        "readout and the error budget.",
        [
            "pihm.qsp",
            "pihm.qsp.filters",
            "pihm.qsp.minimax",
            "pihm.qsp.filter_comparison",
            "pihm.qsp.nlft",
            "pihm.qsp.phases",
            "pihm.qsp.feasibility",
            "pihm.qsp.initial",
            "pihm.qsp.prepare",
            "pihm.qsp.budget",
            "pihm.qsp.readout",
        ],
    ),
    (
        "pipeline",
        "Pipeline",
        "Run specifications, the stages that execute them, their records, campaigns of runs, "
        "and reading results back.",
        [
            "pihm.pipeline",
            "pihm.pipeline.spec",
            "pihm.pipeline.stages",
            "pihm.pipeline.pillar2",
            "pihm.pipeline.lift",
            "pihm.pipeline.records",
            "pihm.pipeline.campaign",
            "pihm.pipeline.summary",
            "pihm.pipeline.results",
            "pihm.pipeline.catalogue",
            "pihm.pipeline.solutions",
            "pihm.pipeline.audit",
        ],
    ),
    (
        "plotting",
        "Plotting",
        "The example notebooks' figures and the interactive solution viewer. They need the "
        "``notebooks`` extra.",
        [
            "pihm.plotting",
            "pihm.plotting.style",
            "pihm.plotting.reproduction",
            "pihm.plotting.scaling",
            "pihm.plotting.stateprep",
            "pihm.plotting.comparison",
            "pihm.plotting.checks",
            "pihm.plotting.lift",
            "pihm.plotting.solutions",
            "pihm.plotting.viewer",
        ],
    ),
]

#: Packages whose exports are all re-exported from their submodules: their pages show the
#: package docstring only, so that every name has one target.
DOCSTRING_ONLY = frozenset({"pihm.operators", "pihm.plotting"})


#: A cross-reference role other than ``:math:``, and its target.
_ROLE = re.compile(r":(?!math:)(?:py:)?[a-z]+:`~?(?:[^`<]*<)?([^`>]+)>?`")


def _summary(module: ModuleType) -> str:
    """Return the first paragraph of ``module``'s docstring, its cross-references as literals.

    A group's page sets no current module, so a relative reference there would not resolve.
    """
    doc = inspect.getdoc(module) or ""
    first = doc.split("\n\n", 1)[0].replace("\n", " ")
    return _ROLE.sub(lambda match: f"``{match[1].rsplit('.', 1)[-1]}``", first)


def _exports(module: ModuleType) -> dict[str, list[str]]:
    """Sort ``module.__all__`` into classes, functions, exceptions and data.

    Data are listed only where autodoc documents them: defined in the module with a ``#:``
    comment.
    """
    analyser = ModuleAnalyzer.for_module(module.__name__)
    commented = {name for scope, name in analyser.find_attr_docs() if not scope}
    kinds: dict[str, list[str]] = {"Classes": [], "Functions": [], "Exceptions": [], "Data": []}
    for name in module.__all__:
        value = getattr(module, name)
        if inspect.isclass(value) and issubclass(value, BaseException):
            kinds["Exceptions"].append(name)
        elif inspect.isclass(value):
            kinds["Classes"].append(name)
        elif inspect.isfunction(value):
            kinds["Functions"].append(name)
        elif name in commented:
            kinds["Data"].append(name)
    return kinds


def _module_page(name: str) -> str:
    """Return the reStructuredText of module ``name``'s page."""
    module = importlib.import_module(name)
    lines = [name, "=" * len(name), "", f".. automodule:: {name}"]
    if name in DOCSTRING_ONLY:
        return "\n".join([*lines, "   :no-members:", ""])
    kinds = _exports(module)
    titles = {"Data": "Constants and types"}
    for kind, names in kinds.items():
        if not names:
            continue
        lines += ["", f"   .. rubric:: {titles.get(kind, kind)}", "", "   .. autosummary::"]
        if kind != "Data":
            lines.append("      :nosignatures:")
        lines += ["", *(f"      {member}" for member in names)]
    if any(kinds.values()):
        lines += ["", "   .. rubric:: Reference"]
    return "\n".join([*lines, ""])


def _group_page(title: str, introduction: str, modules: list[str]) -> str:
    """Return the reStructuredText of a group's page: a table of its modules."""
    lines = [title, "=" * len(title), "", introduction, ""]
    lines += [".. list-table::", "   :widths: 35 65", ""]
    for name in modules:
        summary = _summary(importlib.import_module(name))
        lines += [f"   * - :doc:`{name} <generated/{name}>`", f"     - {summary}"]
    lines += ["", ".. toctree::", "   :hidden:", ""]
    lines += [f"   {name} <generated/{name}>" for name in modules]
    return "\n".join([*lines, ""])


def _write(path: Path, text: str) -> None:
    """Write ``text`` to ``path`` unless it already holds it, so unchanged pages stay fresh."""
    if not path.exists() or path.read_text() != text:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)


def write_pages(directory: Path) -> None:
    """Write every group's page and every module's page under ``directory``."""
    for page, title, introduction, modules in GROUPS:
        _write(directory / f"{page}.rst", _group_page(title, introduction, modules))
        for name in modules:
            _write(directory / "generated" / f"{name}.rst", _module_page(name))
