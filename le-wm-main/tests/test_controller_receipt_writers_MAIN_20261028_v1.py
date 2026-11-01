"""Receipt durability tests only; fixtures are not model/physics evidence."""
import json
import sys
from types import SimpleNamespace

import pytest
from experiments.spectral_reachability import consume_common_controller_MAIN_20261028_v1 as model
from experiments.spectral_reachability import consume_controller_physical_MAIN_20261028_v1 as physics


@pytest.mark.parametrize("module", [model, physics])
def test_full_receipt_preserved_and_never_overwritten(tmp_path, monkeypatch, module):
    target = tmp_path / "receipt.json"
    result = dict(status="fixture", plans=[dict(commands=[[0., 1.]], private_values=[1, 2, 3])],
                  rows=[dict(status="fixture")], checkpoints=[dict(step=5)],
                  scientific_slots=6, learned_unknown_slots=2,
                  predictor_samples=dict(attempted=576, completed=576))
    owner = module.source if module is model else module.validator
    monkeypatch.setattr(owner, "consume", lambda *a: result)
    monkeypatch.setattr(module.shutil, "disk_usage", lambda p: SimpleNamespace(free=10**10))
    monkeypatch.setattr(sys, "argv", ["writer", "--output", "fixture", "--report-sha", "0"*64,
                                    "--receipt", str(target)])
    module.main()
    assert json.loads(target.read_bytes()) == result
    before = target.read_bytes()
    with pytest.raises((ValueError, FileExistsError)):
        module.main()
    assert target.read_bytes() == before


@pytest.mark.parametrize("module", [model, physics])
def test_validator_failure_does_not_create_receipt(tmp_path, monkeypatch, module):
    target = tmp_path / "receipt.json"
    owner = module.source if module is model else module.validator
    def fail(*args):
        raise ValueError("strict failed")
    monkeypatch.setattr(owner, "consume", fail)
    monkeypatch.setattr(sys, "argv", ["writer", "--output", "fixture", "--report-sha", "0"*64,
                                    "--receipt", str(target)])
    with pytest.raises(ValueError, match="strict failed"):
        module.main()
    assert not target.exists()
