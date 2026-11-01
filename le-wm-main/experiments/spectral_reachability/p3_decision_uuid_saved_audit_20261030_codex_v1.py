"""Saved-only V2 UUID-binding consistency audit, not hardware attestation."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import uuid

NAME = "p3_decision_uuid_saved_audit_20261030_codex_v1"
DRIVER = "p3_decision_host_cuda_driver_20261030_codex_v2"


def need(condition, message):
    if not condition:
        raise ValueError(message)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(value).hexdigest()


def load(path, wanted):
    p = Path(path)
    need(type(wanted) is str and re.fullmatch(r"[0-9a-f]{64}", wanted) is not None, "expected SHA")
    need(p.is_absolute() and p.resolve() == p and p.is_file() and not p.is_symlink() and
         p.stat().st_size <= 1024 * 1024, "bounded regular JSON")
    raw = p.read_bytes()
    need(digest(raw) == wanted, "saved bytes SHA mismatch")
    def pairs(items):
        result = {}
        for k, v in items:
            need(k not in result, "duplicate key")
            result[k] = v
        return result
    return json.loads(raw, object_pairs_hook=pairs,
                      parse_constant=lambda x: (_ for _ in ()).throw(ValueError("invalid constant")))


def audit(plan_path, plan_file_sha, plan_sha, base_path, base_file_sha,
          binding_path, binding_sha, uuid_path, uuid_sha):
    plan = load(plan_path, plan_file_sha)
    base = load(base_path, base_file_sha)
    binding = load(binding_path, binding_sha)
    observed = load(uuid_path, uuid_sha)
    need(digest(canonical(plan)) == plan_sha and plan.get("schema") == DRIVER + ".plan", "plan identity")
    need(str(base_path) == plan.get("base_plan_path") and
         base_file_sha == plan.get("base_plan_file_sha256") and
         digest(canonical(base)) == plan.get("base_plan_sha256"), "base plan identity")
    need(binding == dict(schema=DRIVER + ".binding", plan=plan, plan_sha256=plan_sha,
                         external_supervision_verified=False, scientific_pass=False), "v2 binding")
    need(type(observed) is dict and set(observed) == {"schema", "snapshot", "binding", "plan_sha256",
                                                       "scientific_pass"} and
         observed["schema"] == DRIVER + ".uuid_binding" and observed["plan_sha256"] == plan_sha and
         observed["scientific_pass"] is False, "UUID record")
    proof, snap = observed["binding"], observed["snapshot"]
    need(type(proof) is dict and set(proof) == {"physical_uuid", "bytes_hex", "runtime_type",
         "nvidia_smi_uuid", "cuda_visible_devices", "visible_device_count"}, "UUID evidence fields")
    gpu = base["gpu_uuid"]
    need(type(gpu) is str and re.fullmatch(r"GPU-[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}", gpu),
         "canonical physical GPU")
    h = proof["bytes_hex"]
    need(type(h) is str and re.fullmatch(r"[0-9a-f]{32}", h) and
         "GPU-" + str(uuid.UUID(bytes=bytes.fromhex(h))) == gpu and
         proof["physical_uuid"] == proof["nvidia_smi_uuid"] == proof["cuda_visible_devices"] == gpu and
         proof["runtime_type"] == "torch._C._CUuuid" and type(proof["visible_device_count"]) is int and
         proof["visible_device_count"] == 1, "UUID byte identity")
    need(type(snap) is dict and set(snap) == {"uuid", "total_mib", "free_mib", "compute_mode", "mig_mode"}
         and snap["uuid"] == gpu and type(snap["free_mib"]) is int and type(snap["total_mib"]) is int and
         0 <= snap["free_mib"] <= snap["total_mib"] and snap["free_mib"] >= plan["min_free_mib"] and
         snap["total_mib"] >= plan["min_total_mib"] and snap["compute_mode"] in ("Default", "Exclusive_Process")
         and snap["mig_mode"] in ("Disabled", "N/A", "[N/A]"), "saved GPU admission")
    return dict(schema=NAME + ".receipt", plan_sha256=plan_sha, binding_sha256=binding_sha,
                uuid_record_sha256=uuid_sha, gpu_uuid=gpu, bytes_hex=h,
                status="saved_uuid_binding_consistent_only", hardware_attested=False,
                execution_certified=False, full_saved_strict=False, scientific_pass=False)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for item in ("plan", "plan-file-sha", "plan-sha", "base", "base-file-sha",
                 "binding", "binding-sha", "uuid", "uuid-sha"):
        p.add_argument("--" + item, required=True)
    a = p.parse_args()
    print(json.dumps(audit(a.plan, a.plan_file_sha, a.plan_sha, a.base, a.base_file_sha,
                           a.binding, a.binding_sha, a.uuid, a.uuid_sha), sort_keys=True))


if __name__ == "__main__":
    main()
