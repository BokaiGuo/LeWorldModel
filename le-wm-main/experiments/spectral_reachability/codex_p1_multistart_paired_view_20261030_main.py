"""Extend the sealed P1 paired view with the exploratory four-start suffix."""
import argparse
import json
from pathlib import Path

from . import codex_p1_complete_paired_view_20261030_main as original

ROOT = original.base.ROOT
VIEW = ROOT / "results/codex_p1_complete_paired_view_20261030_main_RUN01.json"
SAVED = ROOT / "results/codex_p1_multistart20_physical_20261030_main_SAVED_RUN01.json"
PLAN = ROOT / "results/codex_p1_multistart20_cp5_20261030_main_RUN01.json"
SHAS = (
    "096210929f78f2d8477a3705713dc4c0498e4bf826d89ae580b8652765d0b850",
    "c7cbdf2109d79cf29642659e2f751c8b20a75991cbb30e0427d2c4cdeea25454",
    "428f324d9fd78b5b2fd9461bc03556f263a6e1fefa8d6890c4897949826b4c3e",
)
NAME = "codex_p1_multistart_paired_view_20261030_main"


def build(view, saved, plan):
    if any(len(x["rows"]) != 24 for x in (view, saved, plan)):
        raise ValueError("incomplete roster")
    rows = []
    for i, (base, physical, candidate) in enumerate(zip(
            view["rows"], saved["rows"], plan["rows"], strict=True)):
        ident = base["identity"]
        if ident["index"] != i or physical["index"] != i or candidate["index"] != i:
            raise ValueError("index mismatch")
        for key in original.IDENTITY:
            if ident[key] != physical[key]:
                raise ValueError("physical identity mismatch: " + key)
        for key in ("logical_pair", "role"):
            if ident[key] != candidate[key]:
                raise ValueError("plan identity mismatch: " + key)
        if physical["raw10_sha256"] != candidate["selected_action_sha256"]:
            raise ValueError("selected action mismatch")
        status = physical["status"]
        endpoint = physical["endpoint_metrics"]
        if status == "probe_completed":
            if physical["returned_suffix"] != 10 or endpoint is None:
                raise ValueError("invalid completed endpoint")
        elif status == "native_stopped":
            if physical["returned_suffix"] >= 10 or endpoint is not None:
                raise ValueError("invalid stopped endpoint")
        else:
            raise ValueError("unexpected physical status")
        own_zero = base["common_suffixes"][0]["common_suffix_endpoint_native_metrics"]
        item = {
            "status": status, "accepted_suffix_steps": physical["returned_suffix"],
            "horizon_metrics": endpoint,
            "last_accepted_state_metrics": physical["last_accepted_state_metrics"],
            "raw10_sha256": physical["raw10_sha256"],
            "endpoint_minus_own_zero_suffix": original.delta(endpoint, own_zero),
            "planning_calls_explicit_code_path": candidate["calls"],
            "planning_wall_seconds_excluding_load": candidate["planning_wall_seconds"],
            "physical_counts": physical["counts"],
        }
        rows.append({**base, "multistart20_exploratory": item})
    for row in rows:
        item = row["multistart20_exploratory"]
        zero = rows[3 * row["identity"]["logical_pair"] + 1]
        if zero["identity"]["role"] != "zero":
            raise ValueError("zero prefix mismatch")
        item["endpoint_minus_zero_prefix_independent_endpoint"] = original.delta(
            item["horizon_metrics"], zero["multistart20_exploratory"]["horizon_metrics"])
    return {
        "schema": NAME, "status": "descriptive_exploratory_join", "rows": rows,
        "sources": {str(path): sha for path, sha in zip((VIEW, SAVED, PLAN), SHAS, strict=True)},
        "P1_admission": False, "equal_total_budget_certified": False,
        "limitations": [
            "two known roots and repeated trajectories",
            "common suffix intervention does not establish optimized capability",
            "multistart developed after observing previous outcomes",
            "planning calls are code-path counts, not independent instrumentation",
            "prefix generation and complete total costs not certified",
            "stopped horizon endpoints are null, not imputed",
            "physical wrapper source was not independently bound by original preparation manifest",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    out = Path(parser.parse_args().output)
    if (not out.is_absolute() or out.parent != ROOT / "results"
            or not out.name.startswith(NAME + "_") or out.exists() or out.is_symlink()):
        raise ValueError("exclusive output required")
    sources = [original.gap.read(path, sha)[0]
               for path, sha in zip((VIEW, SAVED, PLAN), SHAS, strict=True)]
    result = build(*sources)
    result["code_sha256"] = original.base.sha(Path(__file__).resolve())
    print("Joining fixed identities; stopped horizon endpoints remain null.", flush=True)
    print(json.dumps({"sha256": original.base.write(out, result), "rows": len(result["rows"])}))


if __name__ == "__main__":
    main()
