"""Verify that the example notebooks are executable and have cleared outputs."""

import json
from pathlib import Path

import pytest

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


@pytest.mark.parametrize("path", sorted(EXAMPLES.glob("*.ipynb")), ids=lambda path: path.stem)
def test_example_notebook(path):
    notebook = json.loads(path.read_text())
    code_cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
    for cell in code_cells:
        assert cell["execution_count"] is None
        assert cell["outputs"] == []
    source = "\n".join("".join(cell["source"]) for cell in code_cells)
    exec(compile(source, str(path), "exec"), {"__name__": "__main__"})
