import os
import sys

sys.path.insert(0, os.path.abspath('..'))

project = "Advanced AI Research 2026"
copyright = "2026, NullLabTests"
author = "NullLabTests"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx_rtd_theme",
    "myst_parser",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
