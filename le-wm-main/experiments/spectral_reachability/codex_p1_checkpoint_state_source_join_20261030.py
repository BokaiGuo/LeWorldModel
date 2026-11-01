"""Join fixed saved provenance projections without executing a model or selecting roots."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[2]
FILES = {
    "identity": ("results/codex_sra_checkpoint_identity_20261030_CPU02.json",
                 "ec42cfc0f32efc344fef05ccb7ac232209c5f32564f4ed8d8770fbc79c04a9cc"),
    "provenance": ("results/agent_p1_configuration_provenance_20261028_v1_MAIN01.json",
                   "43841acd6b2f0f293d041b2c2b871076b425cc3e090e1adfa21e96cfece7dd57"),
    "reset": ("results/codex_reset_ledger_recovery_map_20261030_LOCAL04.json",
              "d101bc2cad391b004998324ba31f8cc8bca3a02ef10c74a6b5f518851dce0582"),
    "checker": ("experiments/spectral_reachability/codex_sra_checkpoint_identity_20261030.py",
                "4fc0b8354d29922549af42717bdbc6933a41d5c081a2a59626536d9649c10171"),
}
WEIGHTS = "48938400ae3464c9680731287f583a9cb516f55a8ec64ea13a91be47fb15b607"
CONFIG = "2564086e961e7b5c7c04dffc451091115b389a590645ff19653c64fd0bc16e09"
PATTERN = re.compile(r"normal|scaler|mean|std|dataset|train|split", re.I)


def load_pinned(root=ROOT, files=FILES):
    blobs = {}
    for name, (relative, expected) in files.items():
        raw = (root / relative).read_bytes()
        if len(raw) > 8 * 1024 * 1024 or hashlib.sha256(raw).hexdigest() != expected:
            raise ValueError("source size or SHA mismatch: " + name)
        blobs[name] = raw
    return blobs


def project(blobs):
    identity = json.loads(blobs["identity"])
    provenance = json.loads(blobs["provenance"])
    reset = json.loads(blobs["reset"])
    cp = identity["checkpoint"]
    state = identity["reference"]["persistent_state"]
    if (identity["schema"] != "codex_sra_checkpoint_identity_v1" or identity["fixture"] is not False
            or identity["checker_sha256"] != hashlib.sha256(blobs["checker"]).hexdigest()
            or cp["weights_sha256"] != WEIGHTS or cp["config_sha256"] != CONFIG
            or provenance["checkpoint"]["weights"]["sha256"] != WEIGHTS
            or provenance["checkpoint"]["config"]["sha256"] != CONFIG
            or provenance["training_identity"]["verified"] is not False
            or identity["checkpoint_tensor_count"] != 303 or len(state) != 303
            or reset["schema"] != "codex_reset_ledger_recovery_map_20261030.v1"
            or reset["history_complete"] is not False or reset["ledger_completeness_certified"] is not False
            or reset["freshness_certified"] is not False or reset["p1_admitted"] is not False):
        raise ValueError("source identity or boundary mismatch")
    matches = {name: record for name, record in state.items() if PATTERN.search(name)}
    return {
        "schema": "codex_p1_checkpoint_state_source_join_20261030.v1",
        "sources": {name: {"path": rel, "sha256": digest} for name, (rel, digest) in FILES.items()},
        "checkpoint_sha256": WEIGHTS, "config_sha256": CONFIG,
        "state_projection": {
            "count": len(state), "matching_name_regex": PATTERN.pattern,
            "matches": matches,
            "scope": "saved CPU proof's remapped persistent tensor names; no tensor values inspected here",
            "limitations": "name search cannot rule out differently named or encoded data; proof is not independent execution attestation",
        },
        "source_boundaries": {
            "training_code_and_resolved_config": "unknown for checkpoint",
            "training_dataset_and_member_identity": "unknown",
            "normalizer_fitting_source_and_checkpoint_binding": "unknown",
            "historical_reset_ledger_complete": False,
            "development_history_scope_complete": "unknown",
            "training_disjoint": "unknown", "development_disjoint": "unknown",
            "source_certified": False, "P1_admitted": False,
        },
        "operations": {"new_weight_deserialization": False, "new_model_inference": False,
                       "new_gpu_execution": False, "new_reset_or_physics": False,
                       "candidate_or_outcome_used": False},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    target = args.output.resolve()
    if target.parent != ROOT / "results" or not target.name.startswith("codex_p1_checkpoint_state_source_join_20261030_") or target.suffix != ".json":
        raise ValueError("exclusive audit result path required")
    result = project(load_pinned())
    raw = (json.dumps(result, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    with target.open("xb") as f:
        f.write(raw)
        f.flush()
        os.fsync(f.fileno())
    print(hashlib.sha256(raw).hexdigest(), target)


if __name__ == "__main__":
    main()
