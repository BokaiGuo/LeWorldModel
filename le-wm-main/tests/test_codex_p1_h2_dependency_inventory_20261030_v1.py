import importlib.util
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "experiments/spectral_reachability/codex_p1_h2_dependency_inventory_20261030_v1.py"
spec = importlib.util.spec_from_file_location("h2_inventory", SCRIPT)
inventory = importlib.util.module_from_spec(spec)
spec.loader.exec_module(inventory)


def test_static_census_only_reachable_python_sources(tmp_path):
    package = tmp_path / "experiments"
    package.mkdir()
    (package / "entry.py").write_text("from experiments import helper\nimport numpy\n__import__('x')\n")
    (package / "helper.py").write_text("import json\n")
    (package / "unused.py").write_text("raise Exception('must not import')\n")
    (package / "outcome.json").write_text("not allowed to read")
    result = inventory.source_census(tmp_path, "experiments.entry")
    assert set(result["files_sha256"]) == {str(package / "entry.py"), str(package / "helper.py")}
    assert "numpy" in result["external_top_level_candidates"]
    assert result["dynamic_or_relative_imports"][0]["kind"] == "dynamic_import"


def test_symlink_source_not_followed(tmp_path):
    package = tmp_path / "experiments"
    package.mkdir()
    secret = tmp_path / "secret.py"
    secret.write_text("x = 1")
    (package / "entry.py").symlink_to(secret)
    assert inventory.source_census(tmp_path, "experiments.entry")["files_sha256"] == {}
