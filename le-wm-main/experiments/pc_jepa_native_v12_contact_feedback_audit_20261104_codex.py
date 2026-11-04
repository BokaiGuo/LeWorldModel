"""Saved-data audit of constructed-feasible native JEPA v12 feedback."""
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/pc_jepa_native_v12_contact_feedback_dev_20261104_codex"
WITNESS = ROOT / "results/pc_jepa_native_contact_witness_dev_20261104_codex"
REFERENCE = ROOT / "results/p5_jepa_factual_nonzero_20261027_v2/factual_main_v1/a"


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1048576), b""):
            h.update(b)
    return h.hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def journal_events(path):
    previous, events = None, []
    for index, line in enumerate(path.read_bytes().splitlines()):
        row = json.loads(line)
        digest = row.pop("entry_sha256")
        actual = hashlib.sha256(json.dumps(row, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
        require(row["sequence"] == index and row["previous_sha256"] == previous and digest == actual, "journal chain")
        previous = digest
        events.append(row["event"])
    return events


def main():
    manifest = json.loads((OUT / "manifest.json").read_text())
    summary = json.loads((OUT / "summary.json").read_text())
    require(summary["manifest_sha256"] == sha(OUT / "manifest.json"), "manifest binding")
    require(summary["journal_sha256"] == sha(OUT / "journal.jsonl"), "journal binding")
    require(manifest["v12_sha256"] == sha(ROOT / "experiments/spectral_reachability/pc_recursive_response_v12.py"),
            "compiler source hash")
    require(manifest["worker_sha256"] == sha(ROOT / "experiments/pc_jepa_native_v12_feedback_worker_20261104_codex.py"),
            "worker source hash")
    events = journal_events(OUT / "journal.jsonl")
    goal = np.load(OUT / "goal_state_evaluation_only.npy", allow_pickle=False).astype(float)
    goal_rgb = np.load(OUT / "goal_rgb.npy", allow_pickle=False)
    witness = np.load(OUT / "witness_actions_evaluation_only.npy", allow_pickle=False)
    require(manifest["witness_source_sha256"] == sha(WITNESS / "witness.npz"), "witness hash")
    require(manifest["witness_manifest_sha256"] == sha(WITNESS / "manifest.json")
            and manifest["witness_summary_sha256"] == sha(WITNESS / "summary.json"), "witness records")
    with np.load(WITNESS / "witness.npz", allow_pickle=False) as original:
        require(np.array_equal(witness, original["actions"]), "witness actions")
    witness_arrays = {key: np.load(OUT / f"witness_{key}_evaluation_only.npy", allow_pickle=False)
                      for key in ("rgb", "proprio", "state")}
    require(all(len(value) == 26 for value in witness_arrays.values()), "witness timeline")
    require(np.array_equal(witness_arrays["rgb"][-1], goal_rgb)
            and np.array_equal(witness_arrays["state"][-1], goal), "witness target")
    stats = {k: np.asarray(v, np.float32) for k, v in manifest["action_stats"].items()}
    results = {}
    for arm in ("response", "cem"):
        result = json.loads((OUT / f"{arm}_result.json").read_text())
        require(result["status"] == "completed" and len(result["events"]) == 26, "incomplete physical run")
        physical = OUT / (arm + "_physical")
        for frame in range(26):
            entry = result["events"][frame]
            require(entry["frame"] == frame, "frame order")
            for name, info in entry["record"]["files"].items():
                require(sha(physical / f"frame_{frame:02d}" / name) == info["sha256"], "physical asset hash")
            if frame <= 10:
                for key in ("rgb", "state", "proprio"):
                    observed = np.load(physical / f"frame_{frame:02d}" / f"{key}.npy", allow_pickle=False)
                    require(np.array_equal(observed, witness_arrays[key][frame]), "warmup mismatch")
        rows, previous_chain = [], None
        for step in range(3):
            path = OUT / f"{arm}_{step}"
            decision_path = path / "output/decision.json"
            data = json.loads(decision_path.read_text())
            require(data["input_sha256"] == sha(path / "input/context.npz"), "input binding")
            require(data["config_sha256"] == sha(path / "input/config.json"), "config binding")
            source_metadata = ROOT / "results/pc_jepa_v7_isolated_input_dev_20261104_codex/input/model_metadata.json"
            require(sha(source_metadata) == manifest["metadata_sha256"], "source metadata hash")
            require(data["metadata_sha256"] == sha(path / "input/model_metadata.json"), "worker metadata hash")
            require(json.loads(source_metadata.read_text()) == json.loads((path / "input/model_metadata.json").read_text()),
                    "metadata content")
            require(data["worker_sha256"] == manifest["worker_sha256"], "worker binding")
            require(result["decisions"][step]["decision_sha256"] == sha(decision_path), "decision binding")
            current = 10 + 5 * step
            with np.load(path / "input/context.npz", allow_pickle=False) as pkg:
                images = np.stack([np.load(physical / f"frame_{i:02d}/rgb.npy", allow_pickle=False)
                                   for i in (current - 10, current - 5, current)])
                props = np.stack([np.load(physical / f"frame_{i:02d}/proprio.npy", allow_pickle=False)
                                  for i in (current - 10, current - 5, current)]).astype(np.float32)
                require(np.array_equal(pkg["rgb_history"], images) and np.array_equal(pkg["goal_rgb"][0], goal_rgb), "RGB history")
                require(np.array_equal(pkg["normalized_proprio"], (props - stats["PROPRIO_MEAN"]) / stats["PROPRIO_STD"]), "proprio history")
                warm = np.array(manifest["warmup"], dtype=np.float64)
                past = [*warm]
                for j in range(11, current + 1):
                    past.append(json.loads((physical / f"command_{j:02d}.json").read_text())["action"])
                u = (100 * np.asarray(past[-10:], np.float64)).astype(np.float32) / np.float32(100)
                normalized = ((u - stats["ACTION_MEAN"]) / stats["ACTION_STD"]).reshape(2, 10)
                require(np.array_equal(pkg["normalized_history_packets"], normalized), "action history")
                require(np.array_equal(pkg["carry"], np.empty((0, 2), np.float32) if step == 0 else previous_chain[5:]), "carry history")
            horizon = 15 - 5 * step
            chain = np.asarray(data["decision"]["selected"]["chain"], np.float32)
            require(chain.shape == (horizon, 2), "horizon")
            token = samples = full = 0
            matches = []
            full_scores, incumbent_matches = [], []
            for batch_index, batch in enumerate(data["evaluations"]):
                actions = np.asarray(batch["actions"], np.float32)
                scores = np.asarray(batch["scores"], float)
                require(actions.ndim == 3 and actions.shape[2] == 2 and len(actions) == len(scores), "batch shape")
                token += len(actions) * (actions.shape[1] // 5)
                samples += len(actions)
                if actions.shape[1] == horizon:
                    full += len(actions)
                    if batch_index < len(data["evaluations"]) - 1:
                        full_scores.extend(scores.tolist())
                        if arm == "response":
                            incumbent = np.asarray(data["decision"]["compiler"]["incumbent"], np.float32)
                            incumbent_matches.extend(float(s) for a, s in zip(actions, scores)
                                                     if np.array_equal(a, incumbent))
                    matches.extend(float(s) for a, s in zip(actions, scores) if np.array_equal(a, chain))
            require(token == data["meter"]["prediction_token_samples"] and token <= 1920, "token counts")
            require(samples == data["meter"]["trajectory_samples"], "sample counts")
            selected_score = data["decision"]["selected"]["terminal_score"]
            require(matches and any(s == selected_score for s in matches), "selected not evaluated")
            if arm == "response":
                compiler = data["decision"]["compiler"]
                require(compiler["method"] == "recursive_response_v12_development", "wrong compiler")
                require(compiler["resource"]["prediction_tokens"] == token, "compiler ledger")
                require(np.array_equal(np.asarray(compiler["selected"], np.float32), chain), "compiler selection")
                require(incumbent_matches and min(incumbent_matches) == min(full_scores),
                        "discarded pre-comparison full-chain incumbent")
                require(all(1 <= generation["batch_size"] <= 12 for generation in compiler["refinement"]),
                        "invalid refinement batch size")
                if step:
                    initial = np.asarray(data["evaluations"][0]["actions"], np.float32)
                    require(len(initial) == 2 and np.array_equal(initial[1], previous_chain[5:]),
                            "carry not re-evaluated")
                pair = data["evaluations"][-1]
                pair_actions = np.asarray(pair["actions"], np.float32)
                require(pair_actions.shape == (2, horizon, 2)
                        and np.array_equal(pair_actions[int(compiler["abstained"])], chain),
                        "paired selection")
                require(pair["scores"] == compiler["paired_scores"], "paired scores")
            for i, action in enumerate(chain[:5], start=current + 1):
                command = json.loads((physical / f"command_{i:02d}.json").read_text())
                require(command["decision_sha256"] == sha(decision_path), "command decision hash")
                require(np.array_equal(np.asarray(command["action"], np.float32), action), "dispatched action")
            sealed = [e for e in events if e.get("phase") == "decision_sealed" and e.get("arm") == arm and e.get("step") == step]
            require(len(sealed) == 1 and sealed[0]["decision_sha256"] == sha(decision_path), "decision seal")
            if arm == "cem" and step:
                candidates = data["decision"]["candidates"]
                require(len(candidates) == 2 and abs(candidates[0]["samples"] - candidates[1]["samples"]) <= 1, "CEM multistart")
            record = {"step": step, "horizon": horizon, "token": token, "samples": samples,
                      "full_horizon_samples": full, "selected_score": selected_score}
            if arm == "response":
                record["incumbent_min_pre_pair"] = min(incumbent_matches)
                record["incumbent_pair_score"] = compiler["paired_scores"][0]
                record["incumbent_rescore_drift"] = compiler["paired_scores"][0] - min(incumbent_matches)
            rows.append(record)
            previous_chain = chain
        metrics = []
        for frame in (10, 15, 20, 25):
            state = np.load(physical / f"frame_{frame:02d}/state.npy", allow_pickle=False).astype(float)
            delta = abs((state[4] - goal[4] + np.pi) % (2 * np.pi) - np.pi)
            joint = float(np.linalg.norm(state[:4] - goal[:4]))
            recomputed = {"agent_error": float(np.linalg.norm(state[:2] - goal[:2])),
                          "object_error": float(np.linalg.norm(state[2:4] - goal[2:4])),
                          "angle_error_radians": float(delta), "joint_xy_error": joint,
                          "success": bool(joint < 20 and delta < np.pi / 9)}
            require(recomputed == result["events"][frame]["metrics"], "physical metric mismatch")
            metrics.append(recomputed)
        results[arm] = {"decisions": rows, "boundary_metrics": metrics}
    audit = {"status": "saved_data_audit_passed", "summary_sha256": sha(OUT / "summary.json"),
             "journal_events": len(events), "arms": results,
             "scope": "saved-data arithmetic and bindings only; no independent physics or neural replay"}
    with (OUT / "audit.json").open("x") as f:
        json.dump(audit, f, indent=2, allow_nan=False)
        f.write("\n")
    print(json.dumps({"status": audit["status"], "arms": results}, indent=2))


if __name__ == "__main__":
    main()
