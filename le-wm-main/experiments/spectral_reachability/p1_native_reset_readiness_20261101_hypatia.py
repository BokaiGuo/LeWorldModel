"""Read-only preseal for native reset source readiness; cannot dispatch reset."""
import argparse
import ast
import json
import os
from pathlib import Path

from experiments.spectral_reachability import p1_known_history_exclusion_20261101_hypatia as base

NAME = "p1_native_reset_readiness_20261101_hypatia"
ROOT = base.ROOT
FILES = {
    "native_env": (".venv/lib/python3.12/site-packages/stable_worldmodel/envs/pusht/env.py",
                   "c6d2104adc368ff1c352edc0e35ac74da9f07fda3a748ddbef0ba6b7a2d1daa2"),
    "calibration_runner": ("experiments/spectral_reachability/p1_conditional_procedural_source_20261027_v1.py",
                           "faddd18ae8d7e0a97c50348929cb4c860321c88a1aedfb755eedc8a0f717deb3"),
    "calibration_config": ("innerdocs/p1_conditional_procedural_source_20261027_v1_calibration2_config.json",
                           "86a6a95a0ef0e3bec36daf1537e6eaf91982a83f4473d5e46d85384a7617e8bc"),
    "fresh_runner": ("experiments/spectral_reachability/agent_p4_fresh_source_20261028_v2.py",
                     "5d2103a883b58db85e9877fb25697ef835e317c529771691a3bdec701a582d3a"),
    "known_fresh_manifest": ("results/chat_freshroots_20261027_root_manifest.json",
                             "3627dd2561a496ea2a0831c2aafa11337363ac1e9ca01285907c9f20e3a7b838"),
}


def inspect_reset(raw):
    tree = ast.parse(raw)
    cls = next((x for x in tree.body if isinstance(x, ast.ClassDef) and x.name == "PushT"), None)
    base.require(cls is not None, "PushT class missing")
    methods = {x.name: x for x in cls.body if isinstance(x, ast.FunctionDef)}
    base.require("reset" in methods and "_set_state" in methods, "reset methods missing")
    reset, setter = methods["reset"], methods["_set_state"]
    def named_call(node, name):
        return isinstance(node, ast.Call) and ast.unparse(node.func) == name
    count = sum(named_call(n, "self._set_state") for n in ast.walk(reset))
    top = sum(isinstance(n, ast.Expr) and named_call(n.value, "self._set_state") for n in reset.body)
    physics = sum(named_call(n, "self.space.step") for n in ast.walk(setter))
    renders = sum(named_call(n, "self.render") for n in ast.walk(reset))
    base.require(count == top == 2 and physics == 1 and renders == 1
                 and not any(isinstance(n, (ast.For, ast.While)) for n in ast.walk(setter)),
                 "unsupported reset call pattern")
    return {"direct_set_state_calls": count, "space_step_calls_per_set_state": physics,
            "direct_render_calls": renders, "inferred_space_steps_per_reset": count * physics,
            "basis": "static call pattern only, not instrumented runtime or complete native closure"}


def build(values, bindings, implementation_hashes):
    config = base.decode(values["calibration_config"])
    manifest = base.decode(values["known_fresh_manifest"])
    base.require(manifest["source_run_authorized"] is False and manifest["coverage_certified"] is False
                 and manifest["freshness_certified"] is False and manifest["certified_fresh_root_count"] == 0,
                 "saved freshness status changed")
    base.require(config["proposal_limit_per_seed"] == 2 and config["reset_attempts_per_proposal"] == 2
                 and config["placement_reset_seed"] == 0 and len(config["root_seeds"]) == 2,
                 "unsupported calibration config")
    return {
        "schema": NAME + ".v1", "sources": bindings, "implementation_sha256": implementation_hashes,
        "native_reset_static_inspection": inspect_reset(values["native_env"]),
        "existing_calibration": {
            "scope": "old calibration only; these seeds already used and must not be relabelled",
            "slot_count": len(config["root_seeds"]), "proposal_limit_per_seed": 2,
            "attempts_per_proposal": 2, "placement_reset_seed": 0,
            "history": "repeat current RGB; zero action padding; no observed warmup",
            "original_source_sha256": bindings["calibration_config"]["sha256"],
        },
        "existing_fresh_protocol_snapshot": {
            "sampling_denominator": manifest["sampling_denominator"], "source_run_authorized": False,
            "coverage_certified": False, "freshness_certified": False,
            "certified_fresh_root_count": 0,
            "scope": "fixed historical manifest, not a claim about any uninspected future authority",
        },
        "open_items": [
            "complete relevant historical reset/attempt and development-root identity coverage",
            "source-qualified candidate roster and actual state/history with source mapping",
            "separate confirmation contract and native runtime/source closure, including intended real RGB history",
            "existing fresh-source certificate and operational requirements where that contract applies",
            "actual checkpoint training run and dataset membership remain unknown",
        ],
        "candidate_roster": None, "candidate_states": None, "selected_candidates": 0,
        "history_complete": False, "development_disjoint_certified": False,
        "training_disjoint_certified": False, "freshness_certified": False,
        "source_run_authorized": False, "P1_admission": False,
        "neural_calls": 0, "physical_calls": 0, "reset_calls": 0,
        "status": "read_only_preseal_not_a_reset_prepare_or_authorization",
    }


def generate():
    bindings = {name: {"path": str(ROOT / rel), "sha256": digest}
                for name, (rel, digest) in FILES.items()}
    raw = {name: base.read_bound(Path(entry["path"]), entry["sha256"])
           for name, entry in bindings.items()}
    modules = [Path(__file__).resolve(), Path(base.__file__).resolve()]
    hashes = {str(p): base.sha(p.read_bytes()) for p in modules}
    value = build(raw, bindings, hashes)
    for entry in bindings.values():
        base.read_bound(Path(entry["path"]), entry["sha256"])
    base.require(all(base.sha(Path(path).read_bytes()) == digest for path, digest in hashes.items()),
                 "implementation changed")
    return value


def create(output):
    path = Path(output)
    base.require(path.is_absolute() and path.parent.resolve() == path.parent and
                 path.name.startswith(NAME + "_") and path.suffix == ".json", "exclusive output namespace")
    data = base.encode(generate())
    with path.open("xb") as stream:
        base.require(stream.write(data) == len(data), "short write")
        stream.flush()
        os.fsync(stream.fileno())
    fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
    return base.sha(data)


def verify(path, expected):
    data = base.read_bound(Path(path), expected)
    base.require(base.decode(data) == generate(), "preseal or source drift")
    return {"status": "verified_read_only_preseal", "sha256": expected,
            "source_run_authorized": False, "freshness_certified": False, "P1_admission": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    c = sub.add_parser("create")
    c.add_argument("--output", required=True)
    v = sub.add_parser("verify")
    v.add_argument("--preseal", required=True)
    v.add_argument("--sha256", required=True)
    args = parser.parse_args()
    if args.command == "create":
        print(json.dumps({"output": args.output, "sha256": create(args.output), "P1_admission": False}))
    else:
        print(json.dumps(verify(args.preseal, args.sha256)))


if __name__ == "__main__":
    main()
