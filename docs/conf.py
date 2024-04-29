"""Sphinx configuration file."""

import yippy

project = "yippy"
copyright = "2024, Corey Spohn"
author = "Corey Spohn"
version = yippy.__version__
release = yippy.__version__

# -- General configuration ---------------------------------------------------

extensions = [
    "myst_nb",
    "autoapi.extension",
    "sphinx.ext.autodoc",
    "sphinx.ext.viewcode",
    "sphinx.ext.mathjax",
    "IPython.sphinxext.ipython_console_highlighting",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

language = "Python"

autoapi_dirs = ["../src"]
autoapi_ignore = ["**/*version.py"]
autodoc_typehints = "description"

myst_enable_extensions = ["amsmath", "dollarmath"]

# -- Options for HTML output -------------------------------------------------

html_theme = "sphinx_book_theme"
html_static_path = ["_static"]
master_doc = "index"
html_logo = "_static/logo.png"
html_favicon = "_static/favicon.ico"
html_title = "Yield Input Package Python Wrapper"
html_sidebars = {"posts/*": ["sbt-sidebar-nav.html"]}

html_theme_options = {
    "repository_url": "https://www.github.com/CoreySpohn/yippy",
    "repository_branch": "main",
    "use_repository_button": True,
    "show_toc_level": 2,
}
html_context = {
    "default_mode": "dark",
}
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "myst-nb",
}
nb_execution_mode = "off"
