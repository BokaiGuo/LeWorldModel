"""Read-only audit of two saved CEM runs; does not execute models or physics."""
import collections
import hashlib
import json
import math
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parents[2] / "results"
EXPECTED = (
    "a973757107da8b98aed242cfb01d5a9079115150d0871c26fe255cc7b960e356",
    "0149183cf66149344f28eb3689f80d08149538bc8ad4143be78346c30b8efeb2",
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def filehash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def chain(path, mode):
    raw = Path(path).read_bytes()
    require(raw and raw.endswith(b"\n"), f"incomplete {path}")
    rows = [json.loads(x) for x in raw.splitlines()]
    prev = "0" * 64 if mode == "cem" else None
    for i, row in enumerate(rows):
        previous = "previous_sha256" if mode == "physical" else "previous"
        sha = "event_sha256" if mode == "physical" else "sha256"
        require(row["index"] == i and row[previous] == prev, f"chain order {path}:{i}")
        body = {k: v for k, v in row.items() if k != sha}
        # The physical writer uses default JSON separators and ASCII escaping;
        # the independently implemented CEM/ledger writers use compact JSON.
        if mode == "physical":
            actual = hashlib.sha256(json.dumps(body, sort_keys=True, allow_nan=False).encode()).hexdigest()
        else:
            actual = digest(body)
        require(actual == row[sha], f"chain hash {path}:{i}")
        prev = row[sha]
    return rows


def native(state, goal):
    delta = np.asarray(state, dtype=np.float64) - np.asarray(goal, dtype=np.float64)
    require(delta.shape == (7,) and np.isfinite(delta).all(), "invalid state")
    xy = float(np.linalg.norm(delta[:4]))
    angle = abs(float(delta[4]))
    angle = min(angle, 2 * math.pi - angle)
    return xy, angle, float(np.linalg.norm(delta)), bool(xy < 20 and angle < math.pi / 9)


def audit(root):
    path = BASE / f"codex_p1_full_cem_host_20261101_exclusive_root{root}_RUN_MAIN01"
    report_path = path / "report.json"
    require(filehash(report_path) == EXPECTED[root], "report hash")
    report = json.loads(report_path.read_text())
    require(filehash(report["prepared"]) == report["prepare_sha256"], "prepare hash")
    prepared = json.loads(Path(report["prepared"]).read_text())
    require(prepared["protocol"] == report["protocol"], "prepared protocol mismatch")
    require(report["model_before"] == report["model_after"], "model drift")
    require(not report["errors"] and not report["secondary_errors"] and not report["pending"], "errors")
    physical = chain(path / "journal.jsonl", "physical")
    require(filehash(path / "journal.jsonl") == report["journal_sha256"], "physical journal file")
    require(physical[-1]["event_sha256"] == report["journal_chain_tail"], "physical tail")
    cem = chain(path / "full_cem_events.jsonl", "cem")
    events = [r["payload"]["event"] for r in cem]
    row = report["rows"][0]
    require(row["source_slot"] == root, "root slot")
    binding = report["cp10_goal_binding"]
    root_event = [x for x in physical if x["kind"] == "full_cem_root_verified"]
    require(len(root_event) == 1 and root_event[0]["goal_binding"] == binding, "root binding")
    contract = events[0]
    require(contract["identity"] == prepared["cem_source_identity"]["identity"], "prepared controller identity")
    require(prepared["cem_source_identity"]["rebound_goal_binding"] == binding, "prepared rebound binding")
    require(contract["identity"]["root_sha256"] == binding["root_sha256"], "controller root")
    require(contract["identity"]["model_sha256"] == report["model_before"]["original_state_dict"], "model identity")
    nodes = report["nodes"][row["episode_id"]]
    selections = {}
    costs = []
    for node in nodes:
        if node["status"] == "not_started":
            continue
        require(node["status"] == "sealed", "node status")
        b = node["boundary"]
        account = node["receipt"]["account"]
        ledger_path = Path(node["receipt"]["ledger"]["journal"])
        require(filehash(ledger_path) == account["journal_sha256"] == node["receipt"]["ledger"]["sha256"], "ledger hash")
        ledger = chain(ledger_path, "ledger")
        require(ledger[0]["payload"]["contract"] == node["ledger_contract"], "ledger contract")
        require(digest(node["ledger_contract"]) == node["contract_sha256"], "contract hash")
        identity = node["ledger_contract"]["identity"]
        require(identity["root"] == binding["root_sha256"] and identity["goal"] == binding["goal_latent_sha256"]
                and identity["model"] == report["model_before"]["original_state_dict"], "node identity")
        caps = node["ledger_contract"]["levels"][0]["caps"]
        require(caps == report["protocol"]["node_costs"][str(b)], "effective CEM node cap")
        intents = [x for x in ledger if x["kind"] == "intent"]
        returns = [x for x in ledger if x["kind"] == "return"]
        require(len(intents) == len(returns) == account["attempted_events"] == account["completed_events"], "ledger counts")
        require([x["kind"] for x in ledger[1:-1]] == ["intent", "return"] * len(intents), "ledger sequence")
        total = collections.Counter()
        for event in intents:
            total.update(event["payload"]["reservation"])
        require(dict(total) == account["charged_upper_bounds"], "ledger charges")
        require(not account["actual_usage_verified"] and not account["coverage_verified"], "coverage claim")
        costs.append({k: total[k] for k in ("action_encode_calls", "predict_calls", "predict_samples")})
        e = [x for x in events if x.get("boundary") == b]
        banks = [x for x in e if x["type"] == "bank"]
        scores = [x for x in e if x["type"] == "score"]
        updates = [x for x in e if x["type"] == "update"]
        require(len(banks) == 4 and len(scores) == 64 and len(updates) == 4, "CEM shape")
        all_plans = []
        for i, (bank, update) in enumerate(zip(banks, updates)):
            require(bank["iteration"] == update["iteration"] == i, "iteration")
            samples = np.asarray(bank["raw"], dtype=np.float32)
            values = scores[i * 16:(i + 1) * 16]
            require([v["index"] for v in values] == list(range(i * 16, (i + 1) * 16)), "scores")
            require(samples.shape == (16, 3 - b // 5, 5, 2), "bank shape")
            elites = sorted(range(16), key=lambda j: (values[j]["value"], j))[:4]
            require(elites == update["elite_indices"], "elite selection")
            require(np.array_equal(samples[elites].mean(axis=0, dtype=np.float32), np.asarray(update["mean"], dtype=np.float32)), "elite mean")
            require(np.array_equal(np.clip(samples[elites].std(axis=0, dtype=np.float32), .05, 1), np.asarray(update["std"], dtype=np.float32)), "elite std")
            if i:
                require(bank["mean"] == updates[i-1]["mean"] and bank["std"] == updates[i-1]["std"], "update continuity")
            all_plans.extend(samples.tolist())
        selected = [x for x in e if x["type"] == "selection"]
        replay = [x for x in e if x["type"] == "winner_replay"]
        require(len(selected) == len(replay) == 1, "winner count")
        winner = min(range(64), key=lambda i: (scores[i]["value"], i))
        require(selected[0]["index"] == winner and selected[0]["raw"] == all_plans[winner], "winner")
        require(math.isclose(replay[0]["score"], scores[winner]["value"], rel_tol=1e-6, abs_tol=1e-8), "winner replay")
        require(selected[0]["raw"][0] == node["receipt"]["commands"], "sealed packet")
        seal_file = path / f"{row['episode_id']}_b{b:02}_seal.json"
        require(json.loads(seal_file.read_text()) == node["receipt"], "saved seal")
        sealed_events = [x for x in physical if x["kind"] == "boundary_sealed" and x["boundary"] == b]
        require(len(sealed_events) == 1 and sealed_events[0]["plan_sha256"] == digest(node["receipt"]["commands"]), "physical packet seal")
        selections[b] = node["receipt"]["commands"]
    observations = {e["receipt"]["step"]: e["receipt"] for e in physical
                    if e["kind"] == "fresh_observation" and e["receipt"]["phase"] == "plan"}
    dispatch = [x for x in physical if x["kind"] == "authorized_dispatch"]
    returns = [x for x in physical if x["kind"] == "control_return"]
    n = row["control_returned"]
    require(len(dispatch) == len(returns) == n == len(row["executed_raw"]), "step count")
    require(set(observations) == set(range(n + 1)), "observation count")
    initial_goal = observations[0]["goal"]
    metrics = []
    for step in range(n + 1):
        obs = observations[step]
        require(obs["goal"] == initial_goal and filehash(obs["path"]) == obs["file_sha256"], "observation provenance")
        require(obs["state"] == obs["snapshot"]["state"], "snapshot state")
        xy, angle, distance, success = native(obs["state"], initial_goal)
        metrics.append((xy, angle, distance, success))
        if step:
            d, r = dispatch[step - 1], returns[step - 1]
            require(d["step"] == r["step"] == step and d["index"] < obs["observed_event_index"] < r["index"], "step order")
            require(d["command"] == r["command"] == row["executed_raw"][step - 1]
                    == selections[(step - 1) // 5 * 5][(step - 1) % 5], "command mismatch")
            require(math.isclose(r["reward"], -distance, rel_tol=1e-12, abs_tol=1e-10), "reward")
            require(r["terminated"] == success and not r["truncated"], "termination")
    require(not any(m[3] for m in metrics[:-1]) and metrics[-1][3], "first native success")
    require(row["final_state"] == observations[n]["state"] and row["native_success"] and row["terminated"], "final")
    require(math.isclose(row["final_metrics"]["native_distance"], metrics[-1][2], rel_tol=1e-12)
            and math.isclose(row["final_metrics"]["native_angle_term"], metrics[-1][1], rel_tol=1e-12), "final metrics")
    return {"root": root, "report_sha256": EXPECTED[root], "status": report["status"],
            "root_sha256": binding["root_sha256"], "goal_latent_sha256": binding["goal_latent_sha256"],
            "model_sha256": report["model_before"]["original_state_dict"], "steps": n,
            "sealed_boundaries": sorted(selections), "node_charges": costs,
            "final_xy": metrics[-1][0], "final_angle": metrics[-1][1], "final_distance": metrics[-1][2],
            "physical_events": len(physical), "cem_events": len(cem)}


if __name__ == "__main__":
    print(json.dumps([audit(0), audit(1)], indent=2, allow_nan=False))
