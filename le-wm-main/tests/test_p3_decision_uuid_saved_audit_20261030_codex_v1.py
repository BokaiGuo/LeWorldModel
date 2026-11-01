"""Synthetic saved bytes only; no CUDA, model or physical environment."""
import json

import pytest

from experiments.spectral_reachability import p3_decision_uuid_saved_audit_20261030_codex_v1 as m

GPU = "GPU-4e8973ff-0278-53f3-2bae-2615d461e441"


def fixture(tmp_path):
    paths = [tmp_path / (x + ".json") for x in ("plan", "base", "binding", "uuid")]
    base = {"gpu_uuid": GPU}
    base_raw = m.canonical(base)
    plan = {"schema": m.DRIVER + ".plan", "base_plan_path": str(paths[1]),
            "base_plan_file_sha256": m.digest(base_raw), "base_plan_sha256": m.digest(base_raw),
            "min_free_mib": 6144, "min_total_mib": 8192}
    ps = m.digest(m.canonical(plan))
    binding = {"schema": m.DRIVER + ".binding", "plan": plan, "plan_sha256": ps,
               "external_supervision_verified": False, "scientific_pass": False}
    observed = {"schema": m.DRIVER + ".uuid_binding", "plan_sha256": ps, "scientific_pass": False,
                "snapshot": {"uuid": GPU, "total_mib": 16311, "free_mib": 15763,
                             "compute_mode": "Default", "mig_mode": "[N/A]"},
                "binding": {"physical_uuid": GPU, "bytes_hex": "4e8973ff027853f32bae2615d461e441",
                            "runtime_type": "torch._C._CUuuid", "nvidia_smi_uuid": GPU,
                            "cuda_visible_devices": GPU, "visible_device_count": 1}}
    values = [plan, base, binding, observed]
    for path, value in zip(paths, values, strict=True):
        path.write_bytes(m.canonical(value))
    return paths, values, ps


def invoke(paths, ps):
    hashes = [m.digest(p.read_bytes()) for p in paths]
    return m.audit(paths[0], hashes[0], ps, paths[1], hashes[1], paths[2], hashes[2], paths[3], hashes[3])


def test_valid_saved_binding(tmp_path):
    paths, _, ps = fixture(tmp_path)
    result = invoke(paths, ps)
    assert result["gpu_uuid"] == GPU
    assert not result["hardware_attested"] and not result["execution_certified"]


@pytest.mark.parametrize("defect", ["bytes", "smi", "visibility", "type", "count", "memory", "mig", "plan"])
def test_rehashed_tamper_rejected(tmp_path, defect):
    paths, values, ps = fixture(tmp_path)
    value = values[3]
    if defect == "bytes":
        value["binding"]["bytes_hex"] = "00" + value["binding"]["bytes_hex"][2:]
    elif defect == "smi":
        value["binding"]["nvidia_smi_uuid"] = "other"
    elif defect == "visibility":
        value["binding"]["cuda_visible_devices"] = "0"
    elif defect == "type":
        value["binding"]["runtime_type"] = "str"
    elif defect == "count":
        value["binding"]["visible_device_count"] = True
    elif defect == "memory":
        value["snapshot"]["free_mib"] = 1
    elif defect == "mig":
        value["snapshot"]["mig_mode"] = "Enabled"
    else:
        value["plan_sha256"] = "0" * 64
    paths[3].write_bytes(m.canonical(value))
    with pytest.raises(ValueError):
        invoke(paths, ps)


def test_truncated_and_wrong_hash_fail(tmp_path):
    paths, _, ps = fixture(tmp_path)
    paths[3].write_text('{"schema":')
    with pytest.raises((ValueError, json.JSONDecodeError)):
        invoke(paths, ps)
    with pytest.raises(ValueError, match="SHA mismatch"):
        m.load(paths[0], "0" * 64)
