import ast
from pathlib import Path

from experiments.spectral_reachability import p1_engineering_failure_wiring_audit_20261102_exclusive as audit


def test_real_inventory_no_longer_finds_original_censor_wiring_pattern():
    result = audit.audit()
    assert result["scientific_negative_result"] is False
    assert result["engineering_failure_is_scientific_negative"] is False
    risks = {Path(item["path"]).name for item in result["targets"] if item["risks"]}
    assert "p1_dev305_contact_equal_scale_saved_20261102.py" not in risks
    assert "p1_dev305_contact_rgb_feedback_saved_20261102.py" not in risks


def test_updated_consumer_requires_fixed_horizon_status():
    item = audit.source_findings(audit.TARGETS[0])
    assert not item["signals"]["accepts_status_not_failed"]
    assert item["signals"]["computes_from_last_transition"]
    assert item["signals"]["requires_fixed_horizon"]
    assert not item["risks"]


def test_fixed_horizon_consumer_is_not_flagged():
    item = audit.source_findings(audit.TARGETS[3])
    assert item["syntax_ok"]
    assert item["signals"]["requires_fixed_horizon"]
    assert not item["risks"]


def test_audit_does_not_import_or_execute_experiment_runtime():
    text = Path(audit.__file__).read_text()
    tree = ast.parse(text)
    imports = [node for node in ast.walk(tree) if isinstance(node, (ast.Import, ast.ImportFrom))]
    assert all("torch" not in ast.unparse(node) and "gym" not in ast.unparse(node)
               for node in imports)
