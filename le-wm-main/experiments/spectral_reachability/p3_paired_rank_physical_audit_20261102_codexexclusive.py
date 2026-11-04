"""CPU-only audit of sealed P3 predictions and fixed-winner physical outcomes."""
import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from . import p3_venv_winner_physical_prepare_20261102_codexexclusive as selected
from . import p3_venv_winner_cpu_consume_20261102_codexexclusive as physical

ROOT = selected.ROOT
RUN = selected.RUN
PHYSICAL_RUN = ROOT / "results/p3_venv_winner_cpu_physical_20261102_codexexclusive_RUN_01"
PREP = ROOT / "results/p3_venv_winner_cpu_physical_20261102_codexexclusive_PREP_01.json"
PREP_SHA = "7d5cb06276b9c1fb2d86344a38f3baf4f67b08a6740a60bc321e3e033033df08"
PHYSICAL_REPORT_SHA = "672b020c82b289481f528c23722a41741b7fed72e7f7407e039492d8e5f4335f"
NAME = "p3_paired_rank_physical_audit_20261102_codexexclusive"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def action(value):
    raw = np.asarray(value, dtype=np.float32)
    require(raw.shape == (10, 2) and np.isfinite(raw).all()
            and np.array_equal(raw.astype(float), np.asarray(value))
            and (np.abs(raw) <= 1).all(), "invalid action bits")
    return raw


def analyze(checked, plans, report, events):
    require([r["arm"] for r in checked["rows"]] == ["signal", "reverse"], "arm order")
    require([r["allocation"] for r in checked["rows"]] == [[15, 5], [5, 15]], "allocation")
    require([(p["arm"], p["root_slot"]) for p in plans] ==
            [("signal", 0), ("signal", 1), ("reverse", 0), ("reverse", 1)], "roster")
    observed = {}
    for e in events:
        if e["kind"] != "candidate_complete":
            continue
        point, arm, phase, root = e["scope"]
        c = e["candidate"]
        require(point == 20 and arm in ("signal", "reverse") and phase == "search"
                and type(root) is int and root in (0, 1), "candidate scope")
        key = (arm, root, c["index"])
        require(key not in observed and math.isfinite(c["cost"]), "duplicate or invalid candidate")
        raw = action(c["commands"])
        observed[key] = {"index": c["index"], "cost": c["cost"],
                         "commands": c["commands"], "raw_float32_sha256": hashlib.sha256(raw.tobytes()).hexdigest()}
    for arm, sizes in (("signal", (15, 5)), ("reverse", (5, 15))):
        for root, size in enumerate(sizes):
            require(sorted(k[2] for k in observed if k[:2] == (arm, root)) == list(range(size)),
                    "missing or extra candidate")
    paired, roster = [], []
    for root in (0, 1):
        for idx in range(5):
            a, b = observed[("signal", root, idx)], observed[("reverse", root, idx)]
            require(a == b, "shared candidate action or cost drift")
        full_arm = "signal" if root == 0 else "reverse"
        candidates = [observed[(full_arm, root, i)] for i in range(15)]
        ranking = sorted(candidates, key=lambda c: (c["cost"], c["index"]))
        ranks = {c["index"]: i + 1 for i, c in enumerate(ranking)}
        for candidate in candidates:
            roster.append(dict(root_slot=root, **candidate, rank_in_15=ranks[candidate["index"]]))
        signal = next(p for p in plans if p["root_slot"] == root and p["arm"] == "signal")
        reverse = next(p for p in plans if p["root_slot"] == root and p["arm"] == "reverse")
        require(signal["root_sha256"] == reverse["root_sha256"] and
                signal["task_id"] == reverse["task_id"] and
                signal["task_export_sha256"] == reverse["task_export_sha256"], "paired task identity")
        planrows = {}
        for p in (signal, reverse):
            a = p["arm"]
            quota = 15 if a == full_arm else 5
            best = min((observed[(a, root, i)] for i in range(quota)),
                       key=lambda c: (c["cost"], c["index"]))
            require((p["candidate_index"], p["model_cost"], p["commands"], p["raw_float32_sha256"]) ==
                    (best["index"], best["cost"], best["commands"], best["raw_float32_sha256"]), "winner binding")
            row = next(r for r in report["rows"] if r["arm"] == a and r["root_slot"] == root)
            require(row["selection_digest"] == p["selection_sha256"], "physical selection")
            planrows[a] = {"candidate_index": p["candidate_index"], "predicted_cost": p["model_cost"],
                           "rank_in_own_quota": 1, "rank_in_union_15": ranks[p["candidate_index"]],
                           "raw_float32_sha256": p["raw_float32_sha256"],
                           "native_success": row["native_success"],
                           "initial_metrics": row["initial_metrics"], "final_metrics": row["final_metrics"]}
        sa, ra = action(signal["commands"]), action(reverse["commands"])
        same = bool(np.array_equal(sa, ra))
        paired.append({"root_slot": root, "root_sha256": signal["root_sha256"],
                       "task_id": signal["task_id"], "shared_candidates": 5,
                       "same_winner_actions_float32": same,
                       "same_step_actions": int(np.all(sa == ra, axis=1).sum()),
                       "signal_minus_reverse_predicted_cost": signal["model_cost"] - reverse["model_cost"],
                       "signal_minus_reverse_final_object_position_error":
                           planrows["signal"]["final_metrics"]["object_position_error"] -
                           planrows["reverse"]["final_metrics"]["object_position_error"],
                       "signal_minus_reverse_final_native_distance":
                           planrows["signal"]["final_metrics"]["native_distance"] -
                           planrows["reverse"]["final_metrics"]["native_distance"],
                       "arms": planrows})
    return paired, roster


def build():
    # The saved-host consumer authenticates the source, task/bank, journal chain,
    # recorded model outputs and winner semantics; no model inference occurs here.
    checked = selected.host.consume(RUN, selected.REPORT_SHA, selected.PREPARE_SHA)
    require(selected.journal.digest(checked) == selected.CONSUME_SHA, "host consumer identity")
    first, second = checked["rows"]
    require(first["attempted"] == first["returned"] == second["attempted"] == second["returned"],
            "recorded resource axes")
    require(selected.host.host.prior.sha(
            ROOT / "results/p3_venv_winner_physical_prepare_20261102_codexexclusive_01.json") ==
            "7f0d3698c324304551e4b9ab11bce7a871ba2463511dd1bf4aa733029c1ef889", "winner prepare SHA")
    source_path = ROOT / "results/p3_venv_winner_physical_prepare_20261102_codexexclusive_01.json"
    plans = json.loads(source_path.read_text())["plans"]
    physical_check = physical.consume(PREP, PREP_SHA, PHYSICAL_RUN)
    require(physical_check["report_sha256"] == PHYSICAL_REPORT_SHA, "physical report SHA")
    report = json.loads((PHYSICAL_RUN / "report.json").read_text())
    with (RUN / "events.jsonl").open() as f:
        events = [json.loads(line)["event"] for line in f]
    paired, roster = analyze(checked, plans, report, events)
    return {"schema": NAME + ".v1", "status": "sealed_known_root_pair_audited",
            "prediction_host_report_sha256": selected.REPORT_SHA,
            "prediction_consumer_sha256": selected.CONSUME_SHA,
            "physical_report_sha256": PHYSICAL_REPORT_SHA,
            "physical_prepare_sha256": PREP_SHA, "physical_consumer": physical_check,
            "prediction_note": "lower latent-model cost is preferred; cost is not physical task metric",
            "equal_recorded_resource_axes": first["attempted"],
            "observed_candidate_evaluations": 40, "shared_prefix_repeated_evaluations": 10,
            "unique_root_candidate_pairs": 30, "paired_roots": paired,
            "next_input": {"status": "development_input_only_no_execution_authority",
                           "policy": "all 15 candidates per known root, ordered root then index; no outcome filtering",
                           "horizon": 10, "warmup_steps_per_branch": 10,
                           "candidate_branches": 30, "max_planned_public_steps": 600,
                           "actions": roster,
                           "physical_outcomes_available_for_only_unique_candidates": 3,
                           "execution_prerequisites": ["freeze new execution contract and output before dispatch",
                               "reauthenticate environment, source, task, bank, raw commands, reset and warmup per branch",
                               "resource caps, terminal/truncation and failure retention",
                               "label known roots and previously opened winners; no independent inference",
                               "choose estimand and resource accounting before any future comparison"]},
            "equal_total_compute": False, "independent_outcome": False, "P3_admission": False}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    out = Path(args.output)
    require(out.is_absolute() and out.parent == ROOT / "results"
            and out.name.startswith(NAME + "_") and out.suffix == ".json", "output path")
    result = build()
    with out.open("x") as f:
        json.dump(result, f, indent=2, sort_keys=True, allow_nan=False)
        f.write("\n")
    print(json.dumps({"status": result["status"], "output": str(out),
                      "sha256": hashlib.sha256(out.read_bytes()).hexdigest(),
                      "pairs": len(result["paired_roots"]), "candidates": len(result["next_input"]["actions"])}))


if __name__ == "__main__":
    main()
