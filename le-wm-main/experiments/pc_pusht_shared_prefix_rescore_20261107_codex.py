"""Rescore pre-existing tails after a shared, observed 10-step prefix."""
import json
import time
from pathlib import Path

import numpy as np
import torch

from experiments.checkpoint_compat import load_compatible_pretrained
from experiments.pc_input_permissions import file_hash
from experiments.pc_prefix_novel_actual_encoding_20261105_codex import model_fingerprint
from experiments import pc_pusht_first_plan_alignment_20261107_codex as prior
from experiments import pc_pusht_first_plan_initial_teacher_20261107_codex as teacher

OUT = prior.RESULTS / "pc_pusht_shared_prefix_rescore_20261107_codex"
PROTOCOL = prior.ROOT.parent / "docs/PC_PUSHT_SHARED_PREFIX_SUFFIX_RESCORING_PROTOCOL_20261107_CODEX.md"


def conditional_rollout(model, observations, actions, prefix_packets):
    """Use observations through prefix; actions include past and future packets."""
    if observations.ndim != 2 or actions.ndim != 3 or prefix_packets < 0:
        raise ValueError("invalid dimensions or prefix")
    if len(observations) != prefix_packets + 1 or actions.shape[1] <= prefix_packets:
        raise ValueError("misaligned clock")
    batch = actions.shape[0]
    emb = observations.unsqueeze(0).expand(batch, -1, -1).clone()
    predictions = []
    for t in range(prefix_packets, actions.shape[1]):
        start = max(0, t - 2)
        act = model.action_encoder(actions[:, start:t + 1])
        state = emb[:, start:t + 1]
        if state.shape[1] != act.shape[1]:
            raise ValueError("action/observation history mismatch")
        next_state = model.predict(state, act)[:, -1:]
        predictions.append(next_state)
        emb = torch.cat((emb, next_state), dim=1)
    return torch.cat(predictions, dim=1)


def checked_source(path):
    record = json.loads(path.read_text())
    for name, digest in record["source_sha256"].items():
        if file_hash(name) != digest:
            raise ValueError(f"source changed: {name}")
    return record


def main():
    if OUT.with_suffix(".json").exists():
        raise FileExistsError(OUT)
    previous_json = prior.OUT.with_suffix(".json")
    teacher_json = teacher.OUT.with_suffix(".json")
    previous = checked_source(previous_json)
    teacher_record = checked_source(teacher_json)
    if file_hash(prior.OUT.with_suffix(".npz")) != previous["arrays_sha256"]:
        raise ValueError("previous arrays differ")
    if file_hash(teacher.OUT.with_suffix(".npz")) != teacher_record["arrays_sha256"]:
        raise ValueError("teacher arrays differ")
    with np.load(prior.OUT.with_suffix(".npz")) as arrays:
        real, goals = arrays["actual"].copy(), arrays["goal"].copy()
    with np.load(teacher.OUT.with_suffix(".npz")) as arrays:
        initial = arrays["initial_rendered"].copy()
    audits = [prior.load_first(stem) for stem in (prior.OLD, prior.NEW)]
    model = load_compatible_pretrained("pusht/lewm", device="cuda").eval().requires_grad_(False)
    fingerprint = model_fingerprint(model)
    if fingerprint != previous["model_sha256"] or fingerprint != teacher_record["model_sha256"]:
        raise ValueError("wrong model")
    rows = []
    print("Rescore existing tails at the common 10-step observation; no future labels enter scores.", flush=True)
    with torch.inference_mode():
        for i, source in enumerate(previous["rows"]):
            actions = np.stack([np.asarray(a[i]["selected_actions"], np.float32) for a in audits])
            if actions.shape != (2, 5, 10) or source["index"] != i:
                raise ValueError("candidate mismatch")
            common = bool(np.array_equal(actions[0, :2], actions[1, :2]))
            record = {"index": i, "common_prefix": common, "same_chain": source["same_chain"]}
            if common:
                np.testing.assert_allclose(initial[i, 0], initial[i, 1], atol=1e-6, rtol=0)
                np.testing.assert_allclose(real[i, 0, :2], real[i, 1, :2], atol=1e-5, rtol=0)
                history = torch.from_numpy(np.concatenate((initial[i, 0:1], real[i, 0, :2]))).to("cuda")
                chain = torch.from_numpy(actions).to("cuda")
                goal = torch.from_numpy(goals[i]).to("cuda")
                record.update({"success_before_branch": source["arms"][0]["success_by_packet"][1],
                               "first_success": [a["first_success"] for a in source["arms"]],
                               "actual_final_cost": [a["actual_encoded_costs"][-1] for a in source["arms"]],
                               "original_score": [a["score_recomputed"] for a in source["arms"]]})
                for mode in ("one_frame", "three_frames"):
                    obs = history[-1:] if mode == "one_frame" else history
                    act = chain[:, 2:] if mode == "one_frame" else chain
                    prefix = 0 if mode == "one_frame" else 2
                    torch.cuda.synchronize()
                    start = time.perf_counter()
                    pred = conditional_rollout(model, obs, act, prefix)
                    torch.cuda.synchronize()
                    elapsed = time.perf_counter() - start
                    if pred.shape[1] != 3:
                        raise ValueError("wrong suffix horizon")
                    record[mode] = {"scores": (pred[:, -1] - goal).square().sum(-1).cpu().tolist(),
                                    "elapsed_seconds": elapsed, "candidate_packets": 6,
                                    "predict_calls": 3}
            rows.append(record)
    if model_fingerprint(model) != fingerprint:
        raise ValueError("model changed")
    with OUT.with_suffix(".json").open("x") as stream:
        json.dump({"rows": rows, "model_sha256": fingerprint, "source_sha256": {
            str(p): file_hash(p) for p in (Path(__file__), PROTOCOL, previous_json, teacher_json,
                                           prior.OUT.with_suffix(".npz"), teacher.OUT.with_suffix(".npz"))},
                   "scope": "fixed pre-existing tails; legal shared observation; offline physical evaluation"},
                  stream, indent=2, allow_nan=False)
        stream.write("\n")


if __name__ == "__main__":
    main()
