"""The source digest: what a record names as the code that made it."""

from __future__ import annotations

import shutil
from pathlib import Path

import pihm
from pihm.pipeline.records import RUN_MODULE, Provenance, run_modules, source_digest

PACKAGE = Path(pihm.__file__).resolve().parent


def test_a_run_imports_the_pipeline_and_not_the_analysis() -> None:
    modules = run_modules()
    assert RUN_MODULE in modules
    assert {"pihm", "pihm.pipeline", "pihm.qsp.prepare", "pihm.circuits.gqsp"} <= set(modules)
    analysis = {"pihm.pipeline.summary", "pihm.pipeline.catalogue", "pihm.pipeline.audit"}
    assert not analysis & set(modules)
    assert not any(name.startswith("pihm.plotting") for name in modules)


def test_the_digest_moves_with_what_a_run_executes_and_nothing_else(tmp_path: Path) -> None:
    copy = tmp_path / "pihm"
    shutil.copytree(PACKAGE, copy, ignore=shutil.ignore_patterns("__pycache__"))
    assert source_digest(copy) == source_digest()
    plotting = copy / "plotting" / "style.py"
    plotting.write_text(plotting.read_text() + "\n# an analysis change\n")
    assert source_digest(copy) == source_digest()
    prepare = copy / "qsp" / "prepare.py"
    prepare.write_text(prepare.read_text() + "\n# a change a record depends on\n")
    assert source_digest(copy) != source_digest()


def test_a_class_named_like_its_module_is_not_a_module(tmp_path: Path) -> None:
    """On a filesystem that ignores case, ``assemble/Residual.py`` opens ``residual.py``."""
    package = tmp_path / "pihm"
    for module, text in {
        "__init__.py": "",
        "pipeline/__init__.py": "",
        "pipeline/stages.py": "from pihm.assemble import Residual\n",
        "assemble/__init__.py": "from pihm.assemble.residual import Residual\n",
        "assemble/residual.py": "class Residual: ...\n",
    }.items():
        (package / module).parent.mkdir(parents=True, exist_ok=True)
        (package / module).write_text(text)
    assert set(run_modules(package)) == {
        "pihm",
        "pihm.pipeline",
        "pihm.pipeline.stages",
        "pihm.assemble",
        "pihm.assemble.residual",
    }
    assert "pihm.assemble.Residual" not in run_modules()


def test_every_record_names_its_source() -> None:
    provenance = Provenance.capture()
    assert provenance.source_digest == source_digest()
    assert Provenance.from_json(provenance.to_json()) == provenance
    old = provenance.to_json()
    del old["source_digest"]
    assert Provenance.from_json(old).source_digest is None
