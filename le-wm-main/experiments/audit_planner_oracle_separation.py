"""Fail-closed oracle-separation audit for planner failure attribution.

The audit consumes *already completed* JSON artifacts.  It never runs a
planner, fits a readout, ranks methods, or searches test outcomes.  Instead a
predeclared spec names one metric for each oracle arm and asserts the shared
evaluation contract.  This keeps the diagnostic useful without turning a
post-hoc collection of results into a model selector.

The required causal ladder is:

``candidate oracle -> perfect dynamics -> real successor + learned geometry
-> learned dynamics + oracle readout``.

The first failed prerequisite is reported as candidate/horizon, ranking, or
information/support limited.  Missing or mismatched controls remain explicitly
unidentified.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


SCHEMA = "planner_oracle_separation.v2"
SPEC_SCHEMA = "planner_oracle_separation_spec.v2"
REQUIRED_CONTRACT_FIELDS = (
    "cohort_id",
    "candidate_bank_id",
    "horizon",
    "task_objective_id",
    "metric_direction",
    "minimum_success_rate",
    "outcome_root_ids_sha256",
    "candidate_action_hashes_sha256",
    "candidate_action_count",
    "execution_semantics",
)
ARM_ORDER = (
    "candidate_oracle",
    "perfect_dynamics_mpc",
    "real_successor_learned_geometry",
    "learned_dynamics_oracle_readout",
    "deployed_learned_planner",
)
ORACLE_ARMS = ARM_ORDER[:4]


def _identity_hash(values: list[str]) -> str:
    """Hash an ordered identity list using the exact representation in artifacts."""
    return hashlib.sha256(
        json.dumps(values, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    ).hexdigest()


def _validate_identity_contract(protocol: dict[str, Any]) -> None:
    """Require executable root and action identities, not descriptive IDs alone.

    Equal ``cohort_id`` or ``candidate_bank_id`` strings do not prove that
    two oracle arms used the same episodes or action sequences.  Without this
    check, oracle routing can attribute differences caused by a different bank
    to geometry or dynamics.
    """
    roots = protocol.get("outcome_root_ids")
    actions = protocol.get("candidate_action_hashes")
    if not isinstance(roots, list) or not roots or any(not isinstance(v, str) or not v for v in roots):
        raise ValueError("protocol requires a nonempty string outcome_root_ids list")
    if len(set(roots)) != len(roots):
        raise ValueError("protocol outcome_root_ids must be unique")
    if protocol.get("outcome_root_ids_sha256") != _identity_hash(roots):
        raise ValueError("protocol outcome_root_ids_sha256 does not bind outcome_root_ids")
    if not isinstance(actions, list) or len(actions) < 2 or any(not isinstance(v, str) or not v for v in actions):
        raise ValueError("protocol requires an ordered candidate_action_hashes list with at least two actions")
    if len(set(actions)) != len(actions):
        raise ValueError("protocol candidate_action_hashes must be unique")
    if protocol.get("candidate_action_hashes_sha256") != _identity_hash(actions):
        raise ValueError("protocol candidate_action_hashes_sha256 does not bind candidate_action_hashes")
    if int(protocol.get("candidate_action_count", -1)) != len(actions):
        raise ValueError("protocol candidate_action_count does not match candidate_action_hashes")
    if protocol.get("candidate_bank_mode") != "shared_fixed":
        raise ValueError("planner-oracle audit currently requires candidate_bank_mode='shared_fixed'")
    if not isinstance(protocol.get("execution_semantics"), str) or not protocol["execution_semantics"]:
        raise ValueError("protocol requires a nonempty execution_semantics declaration")


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"artifact {path} must contain a JSON object")
    return payload


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_spec(path: Path) -> dict[str, Any]:
    """Load a predeclared audit contract without evaluating any result."""

    spec = _load_json(path)
    if spec.get("schema") != SPEC_SCHEMA:
        raise ValueError(f"spec schema must be {SPEC_SCHEMA!r}")
    if not isinstance(spec.get("contract"), dict):
        raise ValueError("spec must contain an object-valued contract")
    missing = [key for key in REQUIRED_CONTRACT_FIELDS if key not in spec["contract"]]
    if missing:
        raise ValueError(f"spec contract missing required fields: {', '.join(missing)}")
    if spec["contract"]["metric_direction"] != "higher_is_better":
        raise ValueError("only higher_is_better success metrics are supported")
    threshold = spec["contract"]["minimum_success_rate"]
    if not isinstance(threshold, (int, float)) or not 0.0 <= float(threshold) <= 1.0:
        raise ValueError("minimum_success_rate must lie in [0, 1]")
    if not isinstance(spec["contract"]["outcome_root_ids_sha256"], str) or not spec["contract"]["outcome_root_ids_sha256"]:
        raise ValueError("contract requires outcome_root_ids_sha256")
    if not isinstance(spec["contract"]["candidate_action_hashes_sha256"], str) or not spec["contract"]["candidate_action_hashes_sha256"]:
        raise ValueError("contract requires candidate_action_hashes_sha256")
    if not isinstance(spec["contract"]["candidate_action_count"], int) or spec["contract"]["candidate_action_count"] < 2:
        raise ValueError("contract candidate_action_count must be an integer >= 2")
    if not isinstance(spec["contract"]["execution_semantics"], str) or not spec["contract"]["execution_semantics"]:
        raise ValueError("contract requires a nonempty execution_semantics declaration")
    arms = spec.get("arms")
    if not isinstance(arms, dict) or not arms:
        raise ValueError("spec must declare at least one audit arm")
    unknown = sorted(set(arms) - set(ARM_ORDER))
    if unknown:
        raise ValueError(f"unknown audit arms: {', '.join(unknown)}")
    for name, arm in arms.items():
        if not isinstance(arm, dict):
            raise ValueError(f"arm {name!r} must be an object")
        for field in ("artifact", "declared_method", "metric_path"):
            if not isinstance(arm.get(field), str) or not arm[field]:
                raise ValueError(f"arm {name!r} requires non-empty {field!r}")
        if "*" in arm["metric_path"] or "[" in arm["metric_path"]:
            raise ValueError("metric_path must be an exact dotted pointer without wildcards")
    spec["_spec_path"] = str(path.resolve())
    return spec


def _resolve_pointer(payload: dict[str, Any], pointer: str) -> float:
    value: Any = payload
    for segment in pointer.split("."):
        if not isinstance(value, dict) or segment not in value:
            raise ValueError(f"metric_path {pointer!r} does not resolve in artifact")
        value = value[segment]
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"metric_path {pointer!r} must resolve to a numeric scalar")
    if not 0.0 <= float(value) <= 1.0:
        raise ValueError(f"metric_path {pointer!r} must contain a success rate in [0, 1]")
    return float(value)


def _validate_declared_method(payload: dict[str, Any], method: str) -> None:
    """Reject a method name that the source artifact does not itself declare."""

    protocol = payload.get("protocol")
    if not isinstance(protocol, dict):
        raise ValueError("artifact must contain a protocol object")
    methods = protocol.get("methods")
    if methods is not None:
        if not isinstance(methods, list) or method not in methods:
            raise ValueError(
                f"declared_method {method!r} is not listed in artifact protocol.methods"
            )


def _read_arm(spec: dict[str, Any], name: str) -> dict[str, Any] | None:
    arm_spec = spec["arms"].get(name)
    if arm_spec is None:
        return None
    base = Path(spec["_spec_path"]).parent
    artifact = Path(arm_spec["artifact"])
    if not artifact.is_absolute():
        artifact = base / artifact
    artifact = artifact.resolve()
    if not artifact.is_file():
        raise ValueError(f"artifact for {name!r} does not exist: {artifact}")
    payload = _load_json(artifact)
    _validate_declared_method(payload, arm_spec["declared_method"])
    protocol = payload.get("protocol", {})
    if not isinstance(protocol, dict):
        raise ValueError("artifact must contain a protocol object")
    _validate_identity_contract(protocol)
    contract = spec["contract"]
    mismatches: dict[str, dict[str, Any]] = {}
    for field in (
        "cohort_id", "candidate_bank_id", "horizon", "task_objective_id",
        "outcome_root_ids_sha256", "candidate_action_hashes_sha256",
        "candidate_action_count", "execution_semantics",
    ):
        observed = protocol.get(field)
        if observed != contract[field]:
            mismatches[field] = {"expected": contract[field], "observed": observed}
    return {
        "role": name,
        "artifact": str(artifact),
        "artifact_sha256": _file_sha256(artifact),
        "declared_method": arm_spec["declared_method"],
        "metric_path": arm_spec["metric_path"],
        "success_rate": _resolve_pointer(payload, arm_spec["metric_path"]),
        "protocol_mismatches": mismatches,
    }


def _route(arms: dict[str, dict[str, Any] | None], threshold: float) -> dict[str, Any]:
    candidate = arms["candidate_oracle"]
    if candidate is None:
        return {
            "status": "unidentified",
            "label": "unidentified_missing_oracle_controls",
            "reason": "ground-truth candidate-bank oracle is absent",
            "required_controls_missing": ["candidate_oracle"],
        }
    if candidate["success_rate"] < threshold:
        return {
            "status": "identified",
            "label": "candidate_or_horizon_limited",
            "reason": "ground-truth candidate-bank ceiling is below the predeclared success threshold",
            "required_controls_missing": [],
        }
    perfect = arms["perfect_dynamics_mpc"]
    if perfect is None:
        return {
            "status": "unidentified",
            "label": "unidentified_missing_oracle_controls",
            "reason": "perfect-dynamics MPC is absent after a reachable candidate ceiling",
            "required_controls_missing": ["perfect_dynamics_mpc"],
        }
    if perfect["success_rate"] < threshold:
        return {
            "status": "identified",
            "label": "candidate_or_horizon_limited",
            "reason": "perfect dynamics cannot solve the declared horizon/objective despite candidate availability",
            "required_controls_missing": [],
        }
    geometry = arms["real_successor_learned_geometry"]
    if geometry is None:
        return {
            "status": "unidentified",
            "label": "unidentified_missing_oracle_controls",
            "reason": "real-successor plus learned-geometry control is absent",
            "required_controls_missing": [
                "real_successor_learned_geometry",
                "learned_dynamics_oracle_readout",
            ],
        }
    if geometry["success_rate"] < threshold:
        return {
            "status": "identified",
            "label": "ranking_limited",
            "reason": "learned task geometry fails even when candidate successors are real",
            "required_controls_missing": [],
        }
    dynamics = arms["learned_dynamics_oracle_readout"]
    if dynamics is None:
        return {
            "status": "unidentified",
            "label": "unidentified_missing_oracle_controls",
            "reason": "learned-dynamics plus oracle-readout control is absent",
            "required_controls_missing": ["learned_dynamics_oracle_readout"],
        }
    if dynamics["success_rate"] < threshold:
        return {
            "status": "identified",
            "label": "information_or_support_limited",
            "reason": "learned dynamics fails even under an oracle task readout",
            "required_controls_missing": [],
        }
    deployed = arms["deployed_learned_planner"]
    if deployed is not None and deployed["success_rate"] < threshold:
        return {
            "status": "identified",
            "label": "deployment_or_planner_integration_limited",
            "reason": "all oracle substitutions pass, but the declared deployed planner remains below threshold",
            "required_controls_missing": [],
        }
    return {
        "status": "not_a_failure",
        "label": "no_declared_failure_observed",
        "reason": "all supplied controls meet the predeclared success threshold",
        "required_controls_missing": [],
    }


def build_audit(spec: dict[str, Any]) -> dict[str, Any]:
    """Build a provenance-preserving route from an already validated spec."""

    if spec.get("schema") != SPEC_SCHEMA or "_spec_path" not in spec:
        raise ValueError("build_audit requires a spec returned by load_spec")
    arms = {name: _read_arm(spec, name) for name in ARM_ORDER}
    mismatched = {
        name: arm["protocol_mismatches"]
        for name, arm in arms.items()
        if arm is not None and arm["protocol_mismatches"]
    }
    if mismatched:
        route = {
            "status": "unidentified",
            "label": "protocol_mismatch",
            "reason": "one or more supplied artifacts do not match the predeclared evaluation contract",
            "required_controls_missing": [],
        }
    else:
        route = _route(arms, float(spec["contract"]["minimum_success_rate"]))
    return {
        "schema": SCHEMA,
        "status": "completed_artifact_audit",
        "input_spec": str(Path(spec["_spec_path"]).resolve()),
        "input_spec_sha256": _file_sha256(Path(spec["_spec_path"])),
        "contract": dict(spec["contract"]),
        "protocol_validation": {
            "valid": not bool(mismatched),
            "mismatches": mismatched,
            "required_match_fields": [
                "cohort_id", "candidate_bank_id", "horizon", "task_objective_id",
                "outcome_root_ids_sha256", "candidate_action_hashes_sha256",
                "candidate_action_count", "execution_semantics",
            ],
        },
        "arms": arms,
        "route": route,
        "selection_policy": {
            "test_outcome_model_selection": False,
            "method_selection": "spec-declared method names only",
            "metric_selection": "spec-declared exact metric paths only",
            "forbidden": ["argmin across methods", "argmax across methods", "wildcard metric pointers"],
        },
        "claim_boundary": {
            "supported": (
                "artifact-level failure routing under one predeclared, protocol-matched "
                "candidate/horizon/objective contract"
            ),
            "does_not_establish": [
                "a selected best learned model",
                "a causal property of a latent representation without the mixed controls",
                "closed-loop or cross-environment generality",
                "a universal world-model or transfer certificate",
            ],
            "promotion_requirement": (
                "Use matched, root-disjoint completed artifacts for all four oracle "
                "controls before promoting a route to a mechanism claim."
            ),
        },
    }


def write_audit(report: dict[str, Any], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True, help="predeclared audit contract JSON")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = build_audit(load_spec(args.spec))
    write_audit(report, args.output)
    print(json.dumps({"route": report["route"], "output": str(args.output)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
