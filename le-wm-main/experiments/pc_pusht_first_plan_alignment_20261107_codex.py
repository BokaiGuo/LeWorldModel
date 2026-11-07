"""Replay paired first plans and align model and physical successor encodings."""
import json
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

from eval import img_transform
from experiments.checkpoint_compat import load_compatible_pretrained
from experiments.pc_input_permissions import file_hash
from experiments.pc_prefix_novel_actual_encoding_20261105_codex import model_fingerprint, native_world_pixels
from experiments.pc_pusht_reset_hidden_audit_20261105_codex import snapshot
from experiments.pc_split_seed_action_audit_analysis_20261105_codex import load
from experiments.pc_standard_pusht_reference_replay_20261105_codex import panel_rows

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OLD = RESULTS / "pc_episode_expansion_prefix_task85_solver42_dev1106"
NEW = RESULTS / "pc_pusht_explicit_prefix_scored_incumbent_20261107_codex"
OUT = RESULTS / "pc_pusht_first_plan_alignment_20261107_codex"
PROTOCOL = ROOT.parent / "docs/PC_PUSHT_FIRST_PLAN_ALIGNMENT_PROTOCOL_20261107_CODEX.md"
DATA = Path("/home/gbk/.stable_worldmodel/datasets/pusht_expert_train.h5")


def load_first(stem):
    with stem.with_suffix(".audit.jsonl").open() as source:
        record = json.loads(next(source))
    if record["solve"] != 0 or len(record["environments"]) != 50:
        raise ValueError("invalid first-plan audit")
    return record["environments"]


def main():
    import gymnasium as gym
    import stable_worldmodel  # noqa: F401

    if OUT.with_suffix(".json").exists() or OUT.with_suffix(".npz").exists():
        raise FileExistsError(OUT)
    audits = [load_first(OLD), load_first(NEW)]
    traces = [load(stem.with_suffix(".trace.jsonl")) for stem in (OLD, NEW)]
    model = load_compatible_pretrained("pusht/lewm", device="cuda").eval().requires_grad_(False)
    model.interpolate_pos_encoding = True
    fingerprint = model_fingerprint(model)
    transform = img_transform(SimpleNamespace(eval=SimpleNamespace(img_size=224)))
    rows, predictions, actuals, goals = [], [], [], []
    print("Replay 50 paired first plans; rescore and compare prediction with real RGB.", flush=True)
    with h5py.File(DATA, "r") as data:
        panel = panel_rows(data, seed=85, count=50)
        source_actions = data["action"][:]
        scaler = StandardScaler().fit(source_actions[~np.isnan(source_actions).any(axis=1)])
        env = gym.make("swm/PushT-v1", render_mode="rgb_array", max_episode_steps=100)
        try:
            for index, data_row in enumerate(panel):
                items = [a[index] for a in audits]
                if any(item["index"] != index for item in items):
                    raise ValueError("audit order mismatch")
                chains = np.stack([np.asarray(x["selected_actions"], np.float32) for x in items])
                if chains.shape != (2, 5, 10) or not np.isfinite(chains).all():
                    raise ValueError("invalid action chain")
                initial = transform(np.asarray(data["pixels"][data_row])).to("cuda")
                goal = transform(np.asarray(data["pixels"][data_row + 25])).to("cuda")
                context = {"pixels": initial[None, None, None].expand(1, 2, -1, -1, -1, -1).clone(),
                           "goal": goal[None, None, None].expand(1, 2, -1, -1, -1, -1).clone(),
                           "action": torch.zeros((1, 2, 1, 10), device="cuda")}
                with torch.inference_mode():
                    scores = model.get_cost(context, torch.from_numpy(chains[None]).to("cuda"))[0]
                expected = np.array([items[0]["branch_final_costs"][items[0]["winner"]],
                                     items[1]["selected_cost"]])
                np.testing.assert_allclose(scores.cpu().numpy(), expected, atol=0.002, rtol=0.0002,
                                           err_msg=f"score mismatch task {index}")
                predicted = context["predicted_emb"][0, :, 1:].cpu().numpy()
                goal_emb = context["goal_emb"][0, -1].cpu().numpy()
                if predicted.shape[1] != 5:
                    raise ValueError("unexpected packet timing")
                arms, frames = [], []
                for arm in range(2):
                    physical = scaler.inverse_transform(chains[arm].reshape(25, 2)).astype(np.float32)
                    if np.abs(physical).max() > 1.000001:
                        raise ValueError("illegal action")
                    seed = 871001 + index
                    env.reset(seed=seed)
                    actor = env.unwrapped
                    actor._set_state(np.asarray(data["state"][data_row], dtype=np.float64))
                    actor._set_goal_state(np.asarray(data["state"][data_row + 25], dtype=np.float64))
                    first, termination, matched = None, [], 0
                    trace = traces[arm][seed]
                    for step, action in enumerate(physical, 1):
                        # Existing trace includes later replans; verify only the initial dispatched plan.
                        if step <= min(25, len(trace)):
                            prior = trace[step - 1]
                            np.testing.assert_allclose(action, prior["action"], atol=1e-6, rtol=0)
                            np.testing.assert_allclose(snapshot(actor), prior["before"], atol=1e-4, rtol=0)
                        _, _, term, trunc, _ = actor.step(action.copy())
                        if step <= len(trace) and step == matched + 1:
                            np.testing.assert_allclose(snapshot(actor), prior["after"], atol=1e-4, rtol=0)
                            if bool(term) != prior["terminated"] or bool(trunc) != prior["truncated"]:
                                raise ValueError("trace outcome mismatch")
                            matched += 1
                        termination.append(bool(term))
                        if term and first is None:
                            first = step
                        if step % 5 == 0:
                            frames.append(transform(native_world_pixels(actor.render())))
                    if len(termination) != 25:
                        raise ValueError(f"trace action divergence task {index} arm {arm} step {step}")
                    arms.append({"first_success": first, "final_success": termination[-1],
                                 "success_by_packet": [any(termination[:i]) for i in (5, 10, 15, 20, 25)],
                                 "matched_trace_steps": matched, "score_audit": float(expected[arm]),
                                 "score_recomputed": float(scores[arm])})
                with torch.inference_mode():
                    pixels = torch.stack(frames).to("cuda")
                    encoded = model.encode({"pixels": pixels[:, None]})["emb"][:, 0].cpu().numpy().reshape(2, 5, -1)
                predictions.append(predicted)
                actuals.append(encoded)
                goals.append(goal_emb)
                rows.append({"index": index, "data_row": int(data_row),
                             "same_chain": bool(np.array_equal(chains[0], chains[1])), "arms": arms})
                print(f"aligned {index + 1}/50", flush=True)
        finally:
            env.close()
    if model_fingerprint(model) != fingerprint:
        raise ValueError("model changed")
    pred, real, goal = np.stack(predictions), np.stack(actuals), np.stack(goals)
    pred_cost = ((pred - goal[:, None, None]) ** 2).sum(-1)
    real_cost = ((real - goal[:, None, None]) ** 2).sum(-1)
    for i, row in enumerate(rows):
        for arm in range(2):
            row["arms"][arm]["predicted_costs"] = pred_cost[i, arm].tolist()
            row["arms"][arm]["actual_encoded_costs"] = real_cost[i, arm].tolist()
            row["arms"][arm]["representation_error_l2"] = np.linalg.norm(pred[i, arm] - real[i, arm], axis=-1).tolist()
    sources = [PROTOCOL, Path(__file__)] + [stem.with_suffix(suffix) for stem in (OLD, NEW)
                                                for suffix in (".audit.jsonl", ".trace.jsonl")]
    with OUT.with_suffix(".npz").open("xb") as stream:
        np.savez_compressed(stream, predicted=pred, actual=real, goal=goal)
    report = {"protocol": str(PROTOCOL), "model_sha256": fingerprint,
              "data": str(DATA), "data_size": DATA.stat().st_size, "data_mtime_ns": DATA.stat().st_mtime_ns,
              "source_sha256": {str(path): file_hash(path) for path in sources},
              "arrays_sha256": file_hash(OUT.with_suffix(".npz")), "rows": rows}
    with OUT.with_suffix(".json").open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")


if __name__ == "__main__":
    main()
