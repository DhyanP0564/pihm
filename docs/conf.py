"""Sphinx configuration for the pihm documentation.

Build with ``sphinx-build -n -W --keep-going docs build/docs``. Intersphinx reads the inventories
cached in ``docs/_inventories/`` first, so the build needs no network.

Two sets of pages are generated at each build and never edited: the API reference's group and
module pages (``docs/apipages.py``), and the example notebooks, which live in ``notebooks/`` at the
repository root and are copied into ``docs/examples/``, then rendered with their saved outputs,
never executed. The guide's
own code pages (MyST notebooks in Markdown) are executed at each build, so their outputs are
always this tree's.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path
from typing import Any

from sphinx.ext.intersphinx import missing_reference

import pihm

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from apipages import write_pages  # noqa: E402

ROOT = HERE.parent
INVENTORIES = HERE / "_inventories"

project = "pihm"
author = "Dhyan Patel"
copyright = "2026, Dhyan Patel"
release = pihm.__version__
version = release

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.mathjax",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "sphinx.ext.doctest",
    "sphinxcontrib.bibtex",
    "sphinx_design",
    "sphinx_copybutton",
    "myst_nb",
]

source_suffix = {".rst": "restructuredtext", ".md": "myst-nb", ".ipynb": "myst-nb"}
exclude_patterns = ["_inventories", "**/.DS_Store", "**/.ipynb_checkpoints"]

# --------------------------------------------------------------------------------------- MyST
# Markdown pages use $...$ and $$...$$ mathematics.
myst_enable_extensions = ["dollarmath", "amsmath", "colon_fence", "deflist", "attrs_inline"]
myst_heading_anchors = 3
myst_dmath_allow_labels = True

# Notebooks: the examples are shown as saved; a guide page opts in to execution in its own front
# matter. An error in an executed page fails the build.
nb_execution_mode = "off"
nb_execution_raise_on_error = True
nb_execution_timeout = 300
nb_merge_streams = True

# --------------------------------------------------------------------------------------- Theme
html_theme = "pydata_sphinx_theme"
html_title = "pihm"
html_baseurl = "https://dhyanp0564.github.io/pihm/"
html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_theme_options = {
    "logo": {"text": "pihm"},
    "navbar_align": "left",
    "navbar_end": ["theme-switcher", "navbar-icon-links"],
    "header_links_before_dropdown": 5,
    "navigation_with_keys": True,
    "show_toc_level": 2,
    "navigation_depth": 3,
    "show_nav_level": 1,
    "collapse_navigation": False,
    "secondary_sidebar_items": ["page-toc"],
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/DhyanP0564/pihm",
            "icon": "fa-brands fa-github",
        }
    ],
    "footer_start": ["copyright"],
    "footer_end": ["sphinx-version"],
    "pygments_light_style": "a11y-light",
    "pygments_dark_style": "a11y-dark",
}
html_sidebars = {"index": [], "getting-started/*": []}
html_show_sourcelink = False
html_context = {"default_mode": "auto"}

# Copy buttons skip prompts and doctest output.
copybutton_prompt_text = r">>> |\.\.\. |\$ "
copybutton_prompt_is_regexp = True

# ------------------------------------------------------------------------------------ API pages
# Docstrings: NumPy style through napoleon; type hints in signatures are the source of truth.
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_use_rtype = False
napoleon_preprocess_types = False
autodoc_typehints = "signature"
autodoc_member_order = "bysource"
autodoc_default_options = {"members": True, "show-inheritance": True}

# One page per module, written by docs/apipages.py at each build; autosummary draws only their
# summary tables.
autosummary_generate = False

bibtex_bibfiles = [str(HERE / "references.bib")]
bibtex_default_style = "plain"
bibtex_reference_style = "author_year"

intersphinx_mapping = {
    "python": ("https://docs.python.org/3.11", (str(INVENTORIES / "python.inv"), None)),
    "numpy": ("https://numpy.org/doc/2.4", (str(INVENTORIES / "numpy.inv"), None)),
    "scipy": ("https://docs.scipy.org/doc/scipy-1.17.0", (str(INVENTORIES / "scipy.inv"), None)),
    "qiskit": (
        "https://quantum.cloud.ibm.com/docs/api/qiskit/2.1",
        (str(INVENTORIES / "qiskit.inv"), None),
    ),
}

# Every reference must resolve (-n); the few that cannot are listed here with a reason.
nitpicky = True
nitpick_ignore: list[tuple[str, str]] = []
nitpick_ignore_regex = [
    # numpy.typing.ArrayLike expands, in signatures, to private numpy._typing aliases that no
    # inventory lists. (autodoc_type_aliases would hide them but misrenders under Sphinx 9.)
    ("py:class", r"numpy\._typing\..*"),
    # pihm.qsp.nlft._LPoly is a private helper type (never in __all__), used only in other
    # methods' own signatures; it has no public Sphinx target by design.
    ("py:class", r"pihm\.qsp\.nlft\._LPoly"),
]


def _retry_as_any_object(app: Any, env: Any, node: Any, contnode: Any) -> Any:
    """Resolve a type hint that another project documents as data or an attribute.

    Autodoc links every annotation as a class, but numpy documents ``numpy.typing.ArrayLike``
    and ``numpy.typing.NDArray`` as data and ``numpy.float64`` as an attribute. Retry such
    references with the ``obj`` role, which matches any object type.
    """
    if node.get("refdomain") != "py" or node.get("reftype") != "class":
        return None
    node["reftype"] = "obj"
    try:
        return missing_reference(app, env, node, contnode)
    finally:
        node["reftype"] = "class"


def _copy_notebooks(app: Any) -> None:
    """Copy the example notebooks into ``docs/examples/``, replacing any earlier copy."""
    target = HERE / "examples"
    for stale in target.glob("*.ipynb"):
        stale.unlink()
    for notebook in sorted((ROOT / "notebooks").glob("*.ipynb")):
        shutil.copy2(notebook, target / notebook.name)


def _write_api_pages(app: Any) -> None:
    """Write the API reference's group and module pages."""
    write_pages(HERE / "reference" / "api")


def setup(app: Any) -> None:
    """Write the generated pages, and register the fallback resolver after intersphinx's own."""
    app.connect("builder-inited", _copy_notebooks)
    app.connect("builder-inited", _write_api_pages)
    app.connect("missing-reference", _retry_as_any_object, priority=900)
