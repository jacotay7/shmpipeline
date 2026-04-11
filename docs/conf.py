"""Sphinx configuration for the shmpipeline documentation site."""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

sys.path.insert(0, str(SRC))

project = "shmpipeline"
author = "Jacob Taylor"
release = "1.0.0"
version = release

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
]

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}
master_doc = "index"

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

autodoc_mock_imports = [
    "numba",
    "pyqtgraph",
    "pyshmem",
    "PySide6",
    "PySide6.QtCore",
    "PySide6.QtGui",
    "PySide6.QtWidgets",
    "torch",
]
autodoc_member_order = "bysource"
autoclass_content = "both"
autodoc_typehints = "description"

myst_enable_extensions = [
    "colon_fence",
    "deflist",
]

html_theme = "sphinx_rtd_theme"
html_title = "shmpipeline documentation"
html_static_path = ["_static"]
