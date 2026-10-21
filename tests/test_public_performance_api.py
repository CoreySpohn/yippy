"""Every function advertised by the performance documentation is exported."""

import ast
import json
from pathlib import Path

import yippy
from yippy import performance


def test_documented_performance_names_are_exported():
    """Notebook imports and the measurement kernels are supported package API."""
    names = {"compute_truncation_curves", "truncation_core", "circular_aperture"}
    for notebook in Path("docs/examples").glob("*.ipynb"):
        for cell in json.loads(notebook.read_text())["cells"]:
            if cell["cell_type"] != "code":
                continue
            source = "".join(cell["source"])
            if source.startswith("%"):
                continue
            for node in ast.walk(ast.parse(source)):
                if (
                    isinstance(node, ast.ImportFrom)
                    and node.module == "yippy.performance"
                ):
                    names.update(
                        alias.name
                        for alias in node.names
                        if not alias.name.startswith("_")
                    )
    for name in sorted(names):
        assert name in yippy.__all__, name
        assert getattr(yippy, name) is getattr(performance, name)
