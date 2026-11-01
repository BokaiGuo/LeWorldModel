"""Saved-only strong-baseline inventory with JSON-safe protocol separation.

Only sealed reports are read.  No model, simulator, network, or outcome-based
selection is performed.  Author values remain separate from local values.
"""
from __future__ import annotations
import argparse
import gzip
import hashlib
import json
from pathlib import Path

NAME = "agent_p2_strong_baseline_saved_summary_20261028_v2"
ROOT = Path(__file__).resolve().parents[2]
LOCAL = {
    "kinodynamic": (ROOT / "results/agent_kinodynamic_matched_domain_20261028_v1_cuda_MAIN01/report.json", "d4976aeca357b5324c74be8b2b7e3bfc254e82b834b3acfe1b0ab90a1ef4614c"),
    "no_composition": (ROOT / "results/p1_moving_no_composition_baseline_cuda_20261027_v1_consume_main01.json", "a05aa97f6c2688d7ae7a1fd8b961a23ccba1f4e45b14d95159cca63c5ba23167"),
    "sampling": (ROOT / "results/p1_sampling_physical_20261027_v1_main01/report.json", "c3eef328bee57ce5d42b25fcf5a9935af2d2282c9de3f8caf46f2b35b53a2ef6"),
}
FLAGS = {"P1_admission": False, "P2_admission": False, "scientific_admission": False}


def require(value, message):
    if not value:
        raise ValueError(message)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def read(path):
    path = Path(path)
    require(path.is_file() and not path.is_symlink(), "regular saved report")
    require(path.stat().st_size < 128 * 1024**2, "bounded report")
    return json.loads(path.read_bytes())


def write(path, value):
    path = Path(path)
    require(not path.exists(), "exclusive output")
    path.write_bytes(canonical(value) + b"\n")


def protocol(name, report):
    contract = report.get("contract") or {}
    resources = report.get("resources") or report.get("counts") or {}
    if name == "kinodynamic":
        return dict(family="kinodynamic", schema=report["schema"], roots=contract.get("roots"), horizon=contract.get("horizon"), history=contract.get("history"), action_support=contract.get("support_claim"), algorithm=contract.get("algorithm"), arms=contract.get("arms"), resources=resources, equal_total_compute=contract.get("equal_total_compute"), equal_search_domain=contract.get("equal_search_domain"), equal_FLOPs=contract.get("equal_FLOPs"), physics=report.get("new_physics"), admission=report.get("P1_admission"))
    if name == "no_composition":
        return dict(family="no_composition", schema=report["schema"], roots=len(report.get("rows", [])), horizon=2, history=next((row.get("history_semantics") for row in report.get("rows", []) if row.get("history_semantics")), None), action_support="spectral_packet_mapped", algorithm="SVD/VJP bounded solve", arms=sorted({row.get("arm") for row in report.get("rows", [])}), resources=resources, equal_total_compute=report.get("equal_total_compute_claim"), physics=False, admission=report.get("P1_admission"))
    return dict(family="sampling", schema=report["schema"], roots=contract.get("roots"), horizon="H2", history=contract.get("history"), action_support=None, algorithm="CEM/IID sampling", arms=contract.get("arms"), resources=resources, equal_total_compute=contract.get("equal_compute"), physics=True, admission=report.get("P1_admission"))


def local_metrics(name, report):
    fields = ("source_slot", "root_sha256", "arm", "status", "model_terminal_cost", "terminal_cost", "attempted_steps", "returned_steps", "full_horizon", "endpoint_rgb_fresh")
    rows = []
    for row in report.get("rows", []):
        out = {key: row[key] for key in fields if key in row}
        if name == "sampling":
            out["official_evaluation"] = row.get("official_evaluation")
            out["final_state"] = row.get("final_state")
            out["selection_sha256"] = row.get("selection_sha256")
        if name == "kinodynamic":
            out["result_file"] = row.get("result_file")
            out["task_id"] = row.get("task_id")
        rows.append(out)
    return rows


def load_local():
    out = {}
    for name, (path, expected) in LOCAL.items():
        require(sha(path) == expected, "sealed local report drift " + name)
        report = read(path)
        require(report.get("errors", []) == [], "local report errors " + name)
        require(report.get("P1_admission") is False, "local P1 flag " + name)
        out[name] = dict(path=str(path), report_sha256=expected, schema=report.get("schema"), status=report.get("status"), protocol=protocol(name, report), resources=report.get("resources") or report.get("counts"), local_values=local_metrics(name, report), raw_keys=sorted(report), report=report)
    report = out["kinodynamic"]["report"]
    for row in report.get("rows", []):
        if row.get("result_file"):
            archive = Path(out["kinodynamic"]["path"]).parent / row["result_file"]
            require(archive.is_file() and not archive.is_symlink(), "kinodynamic result archive exists beside report")
            if row.get("result_sha256") is not None:
                require(sha(archive) == row["result_sha256"], "kinodynamic result archive SHA")
            with gzip.open(archive, "rt") as stream:
                data = json.load(stream)
            require(data.get("source_slot") == row.get("source_slot"), "archive slot identity")
    return out


def author_section(author_path=None):
    if author_path is None:
        return dict(status="unavailable", values=None, source=None, reason="no author-supplied report was bound; local values remain local")
    path = Path(author_path)
    require(path.is_file(), "author report path")
    return dict(status="saved_unverified_until_protocol_check", values=read(path), source=str(path), report_sha256=sha(path))


def compare(a, b):
    pa, pb = a["protocol"], b["protocol"]
    if pa != pb:
        return dict(status="protocol_mismatch_no_direct_comparison", protocol_a=pa, protocol_b=pb, delta=None)
    return dict(status="protocol_match_direct_comparison_allowed", protocol_a=pa, protocol_b=pb, delta=dict(note="numeric delta intentionally not inferred from heterogeneous row schemas"))


def build(author_path=None):
    local = load_local()
    pairs = (("kinodynamic", "no_composition"), ("kinodynamic", "sampling"), ("no_composition", "sampling"))
    # JSON object keys are deliberately strings; tuple keys made the prior
    # version impossible to serialize before its mutation checks ran.
    comparisons = {f"{a}__vs__{b}": compare(local[a], local[b]) for a, b in pairs}
    return dict(schema=NAME + ".summary", status="saved_local_baselines_verified_protocol_separated", local=local, author=author_section(author_path), comparisons=comparisons, denominators={name: len(local[name]["local_values"]) for name in local}, limitations=["author report unavailable unless explicitly supplied", "protocol mismatch blocks direct numeric comparison; no common task/horizon/resource claim is synthesized", "P1/P2/scientific admission remain false; saved verification is not scientific evidence"], **FLAGS)


def verify(path, expected, author_path=None):
    require(sha(path) == expected, "summary SHA")
    saved = read(path)
    require(saved == build(author_path), "saved summary reconstruction and input closure")
    return saved


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    build_parser = sub.add_parser("build")
    build_parser.add_argument("--output", type=Path, required=True)
    build_parser.add_argument("--author-report", type=Path)
    verify_parser = sub.add_parser("verify")
    verify_parser.add_argument("--summary", type=Path, required=True)
    verify_parser.add_argument("--sha", required=True)
    verify_parser.add_argument("--author-report", type=Path)
    args = parser.parse_args()
    if args.command == "build":
        value = build(args.author_report)
        write(args.output, value)
        result = {"output": str(args.output.absolute()), "sha256": sha(args.output), "status": value["status"]}
    else:
        result = {"status": verify(args.summary, args.sha, args.author_report)["status"], "summary_sha256": args.sha}
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
