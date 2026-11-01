"""Join all fixed candidate interventions with sealed selected and zero-tail records."""
import argparse
import hashlib
import json
from pathlib import Path

from . import p1_selected_candidates_crossover_20261031 as experiment
from . import p1_uphill_all_candidates_midpoint_20261031 as previous

ROOT = experiment.ROOT
NAME = "p1_selected_candidates_crossover_analysis_20261031"
ZERO = ROOT / "results/p1_split_budget_crossover_20261031_consume_RUN01.json"
ZERO_SHA = "714b0336cba327519dd94cbabd29416c67b48d07cf3dcbc19451e3cac5c84b9a"
need, sha = experiment.need, experiment.sha


def analyze(prepared, prep_sha, consumed, consumed_sha):
    fixed = experiment.verify(prepared, prep_sha)
    need(sha(consumed) == consumed_sha and sha(ZERO) == ZERO_SHA, "source hashes")
    observed = json.loads(Path(consumed).read_bytes())
    zeros = json.loads(ZERO.read_bytes())
    need(observed["status"] == zeros["status"] == "saved_checks_passed"
         and observed["input_prepare_sha256"] == prep_sha and len(observed["rows"]) == 64
         and len(zeros["rows"]) == 96, "source roster")
    # This also verifies each old report, action hash, and midpoint RGB identity.
    entries, sources = previous.inputs()
    selected = {(e["index"] // 3, e["candidate"]): e for e in entries if e["index"] % 3 == 0}
    need(len(selected) == 32, "selected candidate coverage")
    index = {(r["logical_pair"], r["role"], r["candidate_index"]): r for r in observed["rows"]}
    zero_index = {(r["logical_pair"], r["role"]): r for r in zeros["rows"] if r["probe"] == "zero_suffix"}
    need(len(index) == 64 and len(zero_index) == 24, "control coverage")
    rows = []
    for pair in range(8):
        for candidate in range(4):
            e = selected[pair, candidate]
            expected = fixed["banks"][pair][candidate]["raw10_sha256"]
            need(hashlib.sha256(e["action"].tobytes()).hexdigest() == expected, "old action binding")
            for offset, role in enumerate(("selected", "zero", "greedy")):
                anchor = fixed["anchors"][3 * pair + offset]
                baseline = zero_index[pair, role]
                need(baseline["anchor"] == anchor["index"] and baseline["root_slot"] == anchor["root_slot"]
                     and baseline["condition"] == anchor["condition"] and baseline["arm"] == anchor["arm"]
                     and baseline["probe_sha256"] == hashlib.sha256(bytes(80)).hexdigest()
                     and baseline["prefix"] == experiment.metric.physical_metrics(
                         anchor["checkpoint"]["state"], anchor["goal"])
                     and baseline["root"] == experiment.metric.physical_metrics(anchor["root_state"], anchor["goal"]),
                     "zero-tail identity")
                if role == "selected":
                    name, report_hash, consume_hash = previous.REPORTS[e["kind"]]
                    suffix = "" if e["kind"] == 0 else f"_r{e['kind']-1}"
                    run = ROOT / f"results/{name}_run{suffix}_RUN01"
                    shard_report = json.loads((run / f"row_{anchor['index']:02d}/report.json").read_bytes())["row"]
                    checked = json.loads((ROOT / f"results/{name}_consume{suffix}_RUN01.json").read_bytes())["rows"][anchor["index"]]
                    responses = shard_report["responses"]
                    experiment.validation.validate_responses(checked, responses)
                    endpoint = (experiment.metric.physical_metrics(responses[-1]["state"], anchor["goal"])
                                if len(responses) == 10 else None)
                    current = {"status": checked["status"], "accepted_suffix_steps": len(responses),
                               "native_success": checked["last_accepted_state_metrics"]["native_success"],
                               "endpoint": endpoint}
                else:
                    current = index[pair, role, candidate]
                    need(current["probe_sha256"] == expected and current["anchor"] == anchor["index"]
                         and current["prefix"] == baseline["prefix"] and current["root"] == baseline["root"],
                         "new trajectory identity")
                endpoint, base_endpoint = current["endpoint"], baseline["endpoint"]
                fields = ("object_position_error", "object_angle_error_wrapped",
                          "pusher_position_error", "native_joint_xy_error")
                rows.append({"logical_pair": pair, "root_slot": anchor["root_slot"],
                             "condition": anchor["condition"], "arm": anchor["arm"], "role": role,
                             "candidate_index": candidate, "winner": fixed["banks"][pair][candidate]["winner"],
                             "action_sha256": expected, "status": current["status"],
                             "accepted_suffix_steps": current["accepted_suffix_steps"],
                             "native_success": current["native_success"], "root": baseline["root"],
                             "prefix": baseline["prefix"], "endpoint": endpoint,
                             "zero_suffix_endpoint": base_endpoint,
                             "endpoint_minus_zero": {k: endpoint[k] - base_endpoint[k] for k in fields}
                             if endpoint is not None and base_endpoint is not None else None})
    need(len(rows) == 96 and all(sha(p) == h for p, h in sources.items()), "final source drift")
    return {"schema": NAME, "status": "exploratory_paired_table", "rows": rows,
            "sources": {**sources, str(consumed): consumed_sha, str(ZERO): ZERO_SHA,
                        str(prepared): prep_sha, str(Path(__file__).resolve()): sha(__file__)},
            "P1_admission": False, "independent_confirmation": False, "fairness_proven": False}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for option in ("prepared", "prep-sha", "consumed", "consumed-sha", "output"):
        p.add_argument("--" + option, required=True)
    a = p.parse_args()
    out = Path(a.output)
    need(out.is_absolute() and out.parent == ROOT / "results" and out.name.startswith(NAME + "_")
         and not out.exists() and not out.is_symlink(), "output")
    value = analyze(a.prepared, a.prep_sha, a.consumed, a.consumed_sha)
    print(json.dumps({"sha256": experiment.physical.design.run._write_new(out, value),
                      "rows": len(value["rows"])}))


if __name__ == "__main__":
    main()
