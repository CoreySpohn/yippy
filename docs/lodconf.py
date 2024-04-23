"""Sphinx configuration file."""

import lod_unit

project = "lod_unit"
copyright = "2024, Corey Spohn"
author = "Corey Spohn"
version = lod_unit.__version__
release = lod_unit.__version__

# -- General configuration ---------------------------------------------------

extensions = [
    "myst_parser",
    "autoapi.extension",
    "sphinx.ext.autodoc",
    "sphinx.ext.viewcode",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

language = "Python"

autoapi_dirs = ["../src"]
autoapi_ignore = ["**/*version.py"]
autodoc_typehints = "description"

# -- Options for HTML output -------------------------------------------------

html_theme = "sphinx_book_theme"
html_static_path = ["_static"]
master_doc = "index"
html_logo = "_static/logo.png"
html_favicon = "_static/favicon.ico"
html_title = "λ/D Unit and Equivalency Module"
html_theme_options = {
    "path_to_docs": "docs",
    "repository_url": "https://www.github.com/CoreySpohn/lod_unit",
    "repository_branch": "main",
    "use_repository_button": True,
}
