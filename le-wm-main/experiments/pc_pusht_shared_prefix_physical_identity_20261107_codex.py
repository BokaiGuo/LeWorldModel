"""Audit action and physical-state identity on the recorded shared-prefix pool."""
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

from experiments.pc_input_permissions import file_hash
from experiments.pc_pusht_all_shared_prefix_rescore_20261107_codex import BASE, OUT
from experiments.pc_scored_incumbent_real_policy_parity_20261107_codex import DATA, ROOT, records

TARGET = ROOT / "results/pc_pusht_shared_prefix_physical_identity_20261107_codex.json"
PROTOCOL = ROOT.parent / "docs/PC_PUSHT_SHARED_PREFIX_PHYSICAL_IDENTITY_PROTOCOL_20261107_CODEX.md"


def physical_differences(candidates, selected_index, online_steps):
    states = np.asarray([record["packet_states"][:2] for record in candidates], dtype=np.float64)
    if states.ndim != 3 or states.shape[1:] != (2, 7) or not np.isfinite(states).all():
        raise ValueError("invalid recorded packet states")
    reference = states[selected_index]
    online = np.asarray([online_steps[4]["observed_state"], online_steps[9]["observed_state"]])
    return {"max_candidate_state_abs_error_by_packet": np.abs(states - reference).max(axis=(0, 2)).tolist(),
            "candidate_count_exceeding_1e_4_by_packet":
                (np.abs(states - reference).max(axis=2) > 1e-4).sum(axis=0).tolist(),
            "selected_online_state_abs_error_by_packet": np.abs(reference - online).max(axis=1).tolist()}


def main():
    if TARGET.exists():
        raise FileExistsError(TARGET)
    sealed = json.loads(OUT.with_suffix(".json").read_text())
    evaluation_path = OUT.parent / (OUT.name + ".evaluation.json")
    evaluation = json.loads(evaluation_path.read_text())
    if file_hash(OUT.with_suffix(".json")) != evaluation["sealed_json_sha256"]:
        raise ValueError("sealed selection changed")
    trace = records(BASE / "trace.jsonl")
    with h5py.File(DATA, "r") as data:
        actions = data["action"][:]
    scaler = StandardScaler().fit(actions[~np.isnan(actions).any(axis=1)])
    rows = []
    for plan in sealed["plans"]:
        task = plan["task"]
        path = BASE / f"physical_{task}.json"
        if file_hash(path) != evaluation["physical_sha256"].get(str(path)):
            raise ValueError("physical replay hash mismatch")
        physical = json.loads(path.read_text())
        lookup = {x["action_sha256"]: x for x in physical["records"]}
        batches = [torch.load(BASE / "capture" / f"call_{task * 5 + i:03}.pt",
                              map_location="cpu", weights_only=True) for i in range(5)]
        candidate_actions = []
        for item in plan["candidates"]:
            action = batches[item["call"]]["actions"][0, item["slot"]].numpy()
            if hashlib.sha256(action.tobytes()).hexdigest() != item["sha256"]:
                raise ValueError("candidate action fingerprint mismatch")
            candidate_actions.append(action)
        selected_index = plan["original_selected_position"]
        selected_action = candidate_actions[selected_index]
        if any(not np.array_equal(x[:2], selected_action[:2]) for x in candidate_actions):
            raise ValueError("nonidentical first two packets")
        physical_candidates = [lookup[item["sha256"]] for item in plan["candidates"]]
        if physical_candidates[selected_index]["index"] != physical["selected_index"]:
            raise ValueError("selected replay identity mismatch")
        online = sorted((x for x in trace if x["reset_seed"] == 871001 + task), key=lambda x: x["step"])
        if [x["step"] for x in online[:10]] != list(range(1, 11)):
            raise ValueError("online trace lacks first ten steps")
        reconstructed = scaler.inverse_transform(selected_action.reshape(25, 2))[:10]
        observed = np.asarray([entry["action"] for entry in online[:10]])
        delta = np.abs(reconstructed - observed)
        row = {"task": task, "candidate_count": len(candidate_actions),
               "max_action_abs_error": float(delta.max()),
               "action_steps_exceeding_1e_6": int((delta.max(axis=1) > 1e-6).sum()),
               **physical_differences(physical_candidates, selected_index, online)}
        rows.append(row)
        print(json.dumps(row), flush=True)
    with TARGET.open("x") as stream:
        json.dump({"protocol_sha256": file_hash(PROTOCOL), "runner_sha256": file_hash(Path(__file__)),
                   "sealed_sha256": file_hash(OUT.with_suffix(".json")),
                   "evaluation_sha256": file_hash(evaluation_path),
                   "trace_sha256": file_hash(BASE / "trace.jsonl"), "rows": rows},
                  stream, indent=2, allow_nan=False)


if __name__ == "__main__":
    main()
