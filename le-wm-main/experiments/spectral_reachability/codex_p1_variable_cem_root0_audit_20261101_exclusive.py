"""Independent read-only audit of the saved variable-CEM runs."""
import collections
import json
import math
from pathlib import Path

import numpy as np

from .codex_p1_two_root_saved_audit_20261101_exclusive import (
    chain, digest, filehash, native, require,
)

RESULTS = Path(__file__).resolve().parents[2] / "results"
REPORT_SHAS = (
    "ec644cb8961d3e673edb2baa877bbe9faa530921fd9ce4f5889a197214333c68",
    "1c919ae38784d03822b948479593700f161d4ae7f71126d9e11daade7a4959d7",
)
BOUNDARIES = (0, 5, 10, 15)
HORIZONS = (3, 2, 1, 1)
SAMPLES = (16, 24, 48, 48)
COUNTS = (195, 194, 193, 193)


def load_sources(root=0):
    require(root in (0, 1), "root")
    path = RESULTS / f"codex_p1_variable_cem_host_20261101_exclusive_root{root}_RUN_MAIN01"
    report_path = path / "report.json"
    require(filehash(report_path) == REPORT_SHAS[root], "report SHA")
    report = json.loads(report_path.read_bytes())
    prepare_path = Path(report["prepared"])
    require(prepare_path.parent == RESULTS and filehash(prepare_path) == report["prepare_sha256"], "prepare SHA")
    prepare = json.loads(prepare_path.read_bytes())
    require(report["protocol"] == prepare["protocol"], "protocol binding")
    rows = chain(path / "full_cem_events.jsonl", "cem")
    events = [r["payload"]["event"] for r in rows]
    nodes = report["nodes"][f"s{root}_full_cem"]
    return report, prepare, events, nodes


def verify_cem_node(events, node, boundary):
    i = BOUNDARIES.index(boundary)
    sample_count, horizon = SAMPLES[i], HORIZONS[i]
    selected_events = [e for e in events if e.get("boundary") == boundary]
    banks = [e for e in selected_events if e["type"] == "bank"]
    scores = [e for e in selected_events if e["type"] == "score"]
    updates = [e for e in selected_events if e["type"] == "update"]
    winners = [e for e in selected_events if e["type"] == "selection"]
    replays = [e for e in selected_events if e["type"] == "winner_replay"]
    receipts = [e for e in selected_events if e["type"] == "issue_receipt"]
    require(len(banks) == len(updates) == 4 and len(scores) == 4 * sample_count
            and len(winners) == len(replays) == len(receipts) == 1, "CEM events")
    candidates = []
    center = np.zeros((horizon, 5, 2), np.float32)
    scale = np.full_like(center, .5)
    rng = np.random.default_rng(20261101 + i)
    for iteration, (bank, update) in enumerate(zip(banks, updates)):
        require(bank["iteration"] == update["iteration"] == iteration, "iteration")
        raw = np.asarray(bank["raw"], dtype=np.float32)
        require(raw.shape == (sample_count, horizon, 5, 2), "bank dimensions")
        require(np.array_equal(center, np.asarray(bank["mean"], np.float32))
                and np.array_equal(scale, np.asarray(bank["std"], np.float32)), "bank parameters")
        reconstructed = np.clip(rng.normal(center, scale, raw.shape), -1, 1).astype(np.float32)
        reconstructed[0] = center
        require(np.array_equal(reconstructed, raw), "seeded sampling")
        values = scores[iteration * sample_count:(iteration + 1) * sample_count]
        require([v["index"] for v in values] == list(range(iteration * sample_count, (iteration + 1) * sample_count))
                and all(math.isfinite(v["value"]) for v in values), "score sequence")
        elite = sorted(range(sample_count), key=lambda j: (values[j]["value"], j))[:4]
        require(elite == update["elite_indices"], "elite indices")
        center = raw[elite].mean(axis=0, dtype=np.float32)
        scale = np.clip(raw[elite].std(axis=0, dtype=np.float32), .05, 1).astype(np.float32)
        require(np.array_equal(center, np.asarray(update["mean"], np.float32))
                and np.array_equal(scale, np.asarray(update["std"], np.float32)), "elite update")
        candidates.extend(raw.tolist())
    best = min(range(len(scores)), key=lambda x: (scores[x]["value"], x))
    winner = winners[0]
    require(winner["index"] == best and winner["raw"] == candidates[best]
            and winner["score"] == scores[best]["value"], "winner")
    require(math.isclose(replays[0]["score"], winner["score"], rel_tol=1e-6, abs_tol=1e-8), "winner replay")
    require(winner["raw"][0] == node["receipt"]["commands"], "node packet")
    return receipts[0]["receipt"]


def audit(root=0):
    report, prepare, events, nodes = load_sources(root)
    path = RESULTS / f"codex_p1_variable_cem_host_20261101_exclusive_root{root}_RUN_MAIN01"
    require(not report["errors"] and not report["secondary_errors"] and not report["pending"], "run errors")
    require(report["model_before"] == report["model_after"], "model changed")
    binding = report["cp10_goal_binding"]
    require(prepare["cem_source_identity"]["rebound_goal_binding"] == binding, "root goal binding")
    require(events[0]["identity"] == prepare["cem_source_identity"]["identity"], "controller identity")
    require(events[1]["type"] == "variable_bank_contract"
            and events[1]["contract"] == report["protocol"]["variable_cem"], "variable contract")
    physical = chain(path / "journal.jsonl", "physical")
    require(filehash(path / "journal.jsonl") == report["journal_sha256"]
            and physical[-1]["event_sha256"] == report["journal_chain_tail"], "physical journal SHA/tail")
    root_events = [x for x in physical if x["kind"] == "full_cem_root_verified"]
    require(len(root_events) == 1 and root_events[0]["goal_binding"] == binding, "physical root binding")
    require([n["boundary"] for n in nodes] == list(BOUNDARIES), "roster")
    sealed = [n for n in nodes if n["status"] == "sealed"]
    unstarted = [n for n in nodes if n["status"] == "not_started"]
    require(len(sealed) + len(unstarted) == len(nodes)
            and nodes == sealed + unstarted, "node statuses")
    receipts = []
    for b, count, node in zip(BOUNDARIES, COUNTS, sealed):
        account = node["receipt"]["account"]
        ledger_path = Path(node["receipt"]["ledger"]["journal"])
        require(ledger_path.is_relative_to(path) and filehash(ledger_path)
                == account["journal_sha256"] == node["receipt"]["ledger"]["sha256"], "ledger SHA")
        ledger = chain(ledger_path, "ledger")
        require(ledger[0]["payload"]["contract"] == node["ledger_contract"]
                and digest(node["ledger_contract"]) == node["contract_sha256"], "ledger contract")
        identity = node["ledger_contract"]["identity"]
        require(identity["root"] == binding["root_sha256"] and identity["goal"] == binding["goal_latent_sha256"]
                and identity["model"] == report["model_before"]["original_state_dict"], "node identity")
        caps = node["ledger_contract"]["levels"][0]["caps"]
        require(caps == report["protocol"]["node_costs"][str(b)], "node cap")
        intents = [x for x in ledger if x["kind"] == "intent"]
        require([x["kind"] for x in ledger[1:-1]] == ["intent", "return"] * len(intents)
                and len(intents) == account["attempted_events"] == account["completed_events"], "ledger sequence")
        total = collections.Counter()
        for entry in intents:
            total.update(entry["payload"]["reservation"])
        require(dict(total) == account["charged_upper_bounds"] == caps
                and total["predict_calls"] == total["predict_samples"] == total["action_encode_calls"] == count,
                "ledger charges")
        require(account["status"] == "closed_unverified" and not account["actual_usage_verified"]
                and not account["coverage_verified"], "account verification status")
        require(json.loads((path / f"s{root}_full_cem_b{b:02}_seal.json").read_bytes()) == node["receipt"], "seal")
        seals = [e for e in physical if e["kind"] == "boundary_sealed" and e["boundary"] == b]
        require(len(seals) == 1 and seals[0]["plan_sha256"] == digest(node["receipt"]["commands"]), "physical seal")
        receipts.append(verify_cem_node(events, node, b))
    observations = [x["receipt"] for x in physical if x["kind"] == "fresh_observation"
                    and x["receipt"]["phase"] == "plan"]
    dispatch = [x for x in physical if x["kind"] == "authorized_dispatch"]
    returns = [x for x in physical if x["kind"] == "control_return"]
    row = report["rows"][0]
    n = len(dispatch)
    require([o["step"] for o in observations] == list(range(n + 1)), "observations")
    require([x["boundary"] for x in sealed] == list(range(0, n, 5)), "sealed boundaries")
    require(len(returns) == len(row["executed_raw"]) == row["control_returned"] == n,
            "physical step count")
    goal = observations[0]["goal"]
    metrics = []
    for step, obs in enumerate(observations):
        require(obs["goal"] == goal and obs["state"] == obs["snapshot"]["state"]
                and Path(obs["path"]).is_relative_to(path) and filehash(obs["path"]) == obs["file_sha256"],
                "observation")
        metric = native(obs["state"], goal)
        metrics.append(metric)
        if not step:
            continue
        d, r = dispatch[step-1], returns[step-1]
        require(d["step"] == r["step"] == step and d["index"] < obs["observed_event_index"] < r["index"],
                "physical event ordering")
        require(d["command"] == r["command"] == row["executed_raw"][step-1]
                == nodes[(step-1)//5]["receipt"]["commands"][(step-1)%5], "physical command")
        require(math.isclose(r["reward"], -metric[2], rel_tol=1e-12, abs_tol=1e-10)
                and r["terminated"] == metric[3] and not r["truncated"], "physical return")
    require(not any(x[3] for x in metrics[:-1]) and metrics[-1][3], "first native success")
    require(row["native_success"] and row["terminated"] and not row["truncated"]
            and row["final_state"] == observations[-1]["state"]
            and math.isclose(row["final_metrics"]["native_distance"], metrics[-1][2], rel_tol=1e-12)
            and math.isclose(row["final_metrics"]["native_angle_term"], metrics[-1][1], rel_tol=1e-12),
            "final state")
    steps = [r["executed_steps"] for r in receipts]
    require(steps == [min(5, n - b) for b in range(0, n, 5)]
            and [r["stopped"] for r in receipts] == [False] * (len(steps) - 1) + [True],
            "packet receipt")
    counts = COUNTS[:len(sealed)]
    return {"report_sha256": REPORT_SHAS[root], "prepare_sha256": report["prepare_sha256"],
            "root_sha256": binding["root_sha256"], "goal_latent_sha256": binding["goal_latent_sha256"],
            "model_sha256": report["model_before"]["original_state_dict"], "status": report["status"],
            "node_predict": list(counts), "total_predict": sum(counts), "steps": n, "issued_steps": steps,
            "unstarted_boundaries": [x["boundary"] for x in unstarted],
            "native_success": metrics[-1][3], "final_xy": metrics[-1][0],
            "final_angle": metrics[-1][1], "final_distance": metrics[-1][2],
            "physical_events": len(physical), "cem_events": len(events),
            "actual_usage_verified": False, "coverage_verified": False}


if __name__ == "__main__":
    print(json.dumps([audit(0), audit(1)], sort_keys=True, indent=2, allow_nan=False))
