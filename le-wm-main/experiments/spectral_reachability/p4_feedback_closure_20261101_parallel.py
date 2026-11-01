"""Saved-only P4 engineering closure; performs no CUDA, neural, or physical work."""
import argparse
import hashlib
import json
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
NAME = "p4_feedback_closure_20261101_parallel"
DEFAULT_OUTPUT = ROOT / "results" / f"{NAME}.json"
DEFAULT_INPUTS = {
    "live": {
        "path": ROOT / "results/p4_feedback_live_cuda_20261027_v1_cuda_main01/receipt.json",
        "sha256": "e3c91195540b7465e2fcbf8431d7f182c06bfdff9da6ab160489ae6fe23bbb47",
    },
    "audit": {
        "path": ROOT / "results/p4_feedback_live_saved_audit_20261027_v1_main01/report.json",
        "sha256": "6bdd30c1da733417147504050beed1b6a60ddd513abee45a083fd0955d741953",
    },
    "outcomes": {
        "path": ROOT / "results/p4_multiround_outcomes_20261027_v1_main_v1/report.json",
        "sha256": "d2302f8f89ed9616cb606a626e8eefe1f24d62769014d022fa31839d6ba34248",
    },
    "terminal": {
        "path": ROOT / "results/p4_terminal20_cuda_encoding_20261027_v1_cuda_main01/receipt.json",
        "sha256": "b181d413dec0ee4bb922e99c155542947c2daa43371f84e17ba24e8f17860cb3",
    },
}


def require(value, message):
    if not value:
        raise ValueError(message)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def strict_json(raw):
    def unique(items):
        result = {}
        for key, value in items:
            require(key not in result, "duplicate JSON key")
            result[key] = value
        return result
    return json.loads(raw, object_pairs_hook=unique,
                      parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))


def load_ref(ref):
    path = Path(ref["path"])
    require(path.is_file() and not path.is_symlink(), f"regular saved artifact: {path}")
    before = path.read_bytes()
    require(digest(before) == ref["sha256"], f"pinned SHA: {path}")
    value = strict_json(before)
    after = path.read_bytes()
    require(before == after and digest(after) == ref["sha256"], f"stable artifact: {path}")
    return value


def all_admissions_false(value):
    for key in ("P1_admission", "P2_admission", "P3_admission", "P4_admission"):
        if key in value:
            require(value[key] is False, f"{key} must remain false")


def build_report(refs):
    require(set(refs) == {"live", "audit", "outcomes", "terminal"}, "exact P4 inputs")
    values = {name: load_ref(ref) for name, ref in refs.items()}
    live, audit, outcomes, terminal = (values[k] for k in ("live", "audit", "outcomes", "terminal"))
    require(live["schema"] == "p4_feedback_live_cuda_20261027_v1.receipt" and
            live["status"] == "two_live_20step_3round_arms_completed" and
            live["known_roots_only"] is True and live["independent_confirmation"] is False and
            live["errors"] == [], "live feedback receipt contract")
    require({row["arm"] for row in live["rows"]} == {"actual", "imagined"} and
            len(live["rows"]) == 2 and
            all(row["status"] == "completed20" for row in live["rows"]),
            "two completed live feedback arms")
    require(live["model_hash_before"] == live["model_hash_after"], "live frozen model hash")
    require(audit["schema"] == "p4_feedback_live_saved_audit_20261027_v1.report" and
            audit["status"] == "independent_saved_live_audit_passed" and
            audit["independent_confirmation"] is False and
            audit["new_neural_calls"] == audit["new_physical_steps"] == 0,
            "independent saved audit contract")
    require({row["arm"] for row in audit["arms"]} == {"actual", "imagined"} and
            all(audit["gates"].values()), "saved audit gates")
    require(outcomes["schema"] == "p4_multiround_outcomes_20261027_v1" and
            outcomes["status"] == "saved_arithmetic_audit_passed", "multiround outcomes")
    require(terminal["schema"] == "p4_terminal20_cuda_encoding_20261027_v1.derived_receipt" and
            terminal["status"] == "two_terminal20_latents_saved", "terminal encoding")
    if "model_hash_before" in terminal or "model_hash_after" in terminal:
        require(terminal.get("model_hash_before") == terminal.get("model_hash_after"),
                "terminal frozen model hash")
    for value in values.values():
        all_admissions_false(value)
    return {
        "schema": NAME,
        "status": "p4_engineering_saved_closure_verified",
        "inputs": {name: {"path": str(Path(ref["path"])), "sha256": ref["sha256"]}
                   for name, ref in refs.items()},
        "engineering": {
            "real_feedback_closed_loop_saved": True,
            "three_feedback_rounds_per_arm": True,
            "terminal20_cuda_encoding_saved": True,
            "multiround_saved_arithmetic_verified": True,
            "independent_saved_validation": True,
            "frozen_model_identity_preserved": True,
        },
        "scientific_blockers": [
            "known_root_only",
            "upstream_stage_admission_false",
            "no_independent_confirmation",
            "saved engineering closure does not establish task benefit",
        ],
        "known_roots_only": True,
        "new_neural_calls": 0,
        "new_physical_steps": 0,
        "P1_admission": False,
        "P4_admission": False,
        "independent_confirmation": False,
        "evidence_level": "saved-artifact engineering validation",
    }


def write_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    fd = os.open(path, flags, 0o644)
    try:
        body = json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2).encode() + b"\n"
        os.write(fd, body)
        os.fsync(fd)
    finally:
        os.close(fd)
    return digest(path.read_bytes())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    require(args.output.name == f"{NAME}.json", "exclusive P4 output name")
    report = build_report(DEFAULT_INPUTS)
    sha256 = write_new(args.output, report)
    print(json.dumps({"status": report["status"], "output": str(args.output), "sha256": sha256},
                     sort_keys=True))


if __name__ == "__main__":
    main()
