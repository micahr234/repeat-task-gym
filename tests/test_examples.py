"""Verify that the example notebooks are executable and have cleared outputs."""

import json
from copy import deepcopy
from pathlib import Path

import gymnasium as gym
import pytest

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


def execute_notebook(path):
    notebook = json.loads(path.read_text())
    code_cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
    for cell in code_cells:
        assert cell["execution_count"] is None
        assert cell["outputs"] == []
    source = "\n".join("".join(cell["source"]) for cell in code_cells)
    namespace = {"__name__": "__main__"}
    try:
        exec(compile(source, str(path), "exec"), namespace)
    except ModuleNotFoundError as error:
        if error.name == "procedural_frozenlake":
            pytest.skip(f"{path.name} requires procedural-frozenlake (Python 3.12+).")
        raise
    return namespace


@pytest.mark.filterwarnings("error:.*share an object.*:UserWarning")
@pytest.mark.parametrize("path", sorted(EXAMPLES.glob("*.ipynb")), ids=lambda path: path.stem)
def test_example_notebook(path):
    execute_notebook(path)


@pytest.mark.filterwarnings("error:.*share an object.*:UserWarning")
def test_frozenlake_info_snapshots_are_independent():
    namespace = execute_notebook(EXAMPLES / "03_rng_seeding_control.ipynb")
    env = gym.make(namespace["lake_spec"], map_seed=0, emit_map=True)
    try:
        _, reset_info = env.reset(seed=0)
        expected = deepcopy(reset_info)
        _, _, _, _, first_step_info = env.step(0)
        _, _, _, _, second_step_info = env.step(0)
        assert reset_info == first_step_info == second_step_info == expected

        # Editing a nested value must not alter saved frames or the live map.
        reset_info["map"]["rewards"].clear()
        assert first_step_info == second_step_info == expected
        first_step_info["map"]["rewards"].clear()
        _, next_reset_info = env.reset(seed=1)
        assert second_step_info == next_reset_info == expected

        env.reset(options={"regenerate_map": True})
        assert second_step_info == next_reset_info == expected
    finally:
        env.close()
