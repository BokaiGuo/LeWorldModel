from copy import deepcopy

import pytest
import torch

from experiments.pc_input_permissions import RGB, STATE, freeze_audit, manifest, tensor_receipt
from experiments.pc_track_registry import eligible
from experiments.pc_track_registry_strict_20261107_codex import strict_eligible


@pytest.fixture(params=[RGB, STATE])
def binding(tmp_path, request):
    path = tmp_path / "source"
    path.write_text("local test")
    track = request.param
    family, fields = ("lewm", ()) if track == RGB else ("pointmaze_jepa", ("x", "y", "vx", "vy"))
    audit = freeze_audit(track, family, [path], preprocessing="fixture", clock="fixture",
                         proprio_fields=fields)
    op = "score" if track == RGB else "predict"
    names = (["pixels", "goal", "action", "candidate_actions"] if track == RGB else
             ["visual", "proprio", "candidate_actions"])
    receipt = {"track_id": track, "input_audit_hash": audit.fingerprint, "operation": op,
               "inputs": {name: tensor_receipt(torch.zeros(1, 2)) for name in names}}
    record = manifest(audit, algorithm_version="test", protocol_path=path,
                      task_manifest_path=path, receipts=[receipt])
    kwargs = dict(target_track=track, algorithm_version="test", protocol_path=path,
                  task_manifest_path=path, required_operations={op})
    return record, kwargs


def test_real_receipt_schema_accepted_and_other_track_rejected(binding):
    record, kwargs = binding
    assert strict_eligible(record, **kwargs)
    kwargs["target_track"] = STATE if kwargs["target_track"] == RGB else RGB
    with pytest.raises(ValueError):
        strict_eligible(record, **kwargs)


@pytest.mark.parametrize("bad", [{}, {"shape": [1], "dtype": "torch.float32", "sha256": "x" * 64},
                                  {"shape": [True], "dtype": "torch.float32", "sha256": "a" * 64},
                                  {"shape": [-1], "dtype": "torch.float32", "sha256": "a" * 64},
                                  {"shape": [1], "dtype": "forged", "sha256": "a" * 64}])
def test_malformed_tensor_receipts_rejected(binding, bad):
    record, kwargs = binding
    record = deepcopy(record)
    first = next(iter(record["input_receipts"][0]["inputs"]))
    record["input_receipts"][0]["inputs"][first] = bad
    # The historic checker only inspects channel names.
    assert eligible(record, target_track=kwargs["target_track"], algorithm_version="test",
                    protocol_hash=record["protocol_hash"], task_manifest_hash=record["task_manifest_hash"])
    with pytest.raises(ValueError):
        strict_eligible(record, **kwargs)


def test_requires_bound_files_and_runtime_operation(binding, tmp_path):
    record, kwargs = binding
    bad_operation = {**kwargs, "required_operations": {"not_executed"}}
    with pytest.raises(ValueError):
        strict_eligible(record, **bad_operation)
    altered = tmp_path / "altered"
    altered.write_text("different")
    with pytest.raises(ValueError):
        strict_eligible(record, **{**kwargs, "task_manifest_path": altered})
    with pytest.raises(ValueError):
        strict_eligible(record, **{**kwargs, "required_operations": set()})
