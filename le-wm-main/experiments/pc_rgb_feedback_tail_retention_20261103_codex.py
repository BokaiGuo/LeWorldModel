"""Execute sealed PointMaze plans with budgeted cp10 old-tail retention."""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from experiments import pc_budget_allocation_compare_20261103_codex as previous
from experiments.pc_feedback_tail_retention_20261103_codex import select_feedback_tail
from experiments.pc_pointmaze_shadow_rgb_bridge_20261103_codex import ShadowRGBBridge
from experiments.pointmaze_action_history_unroll_20261103 import unroll_with_action_history

old, h, base, ROOT = previous.old, previous.h, previous.base, previous.ROOT
PROTOCOL = ROOT.parent / "docs/PC_RGB_FEEDBACK_TAIL_RETENTION_PROTOCOL_20261103_CODEX.md"


def select_at_feedback(chain, root, score):
    """Bind the executed prefix and saved tail to the same feedback scorer."""
    chain = np.asarray(chain, dtype=float)
    if chain.shape != (15, 2) or not np.isfinite(chain).all() or np.any(np.abs(chain) > 1):
        raise ValueError("invalid sealed root chain")
    result = select_feedback_tail(
        chain[10:], total_samples=96,
        rng=np.random.default_rng(2026110640 + root), predict=score)
    return chain[:10].tolist(), result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("smoke", "full"), required=True)
    args = parser.parse_args()
    prefix = ROOT / f"results/pc_rgb_feedback_tail_retention_20261103_codex_{args.mode}"
    plan_path = prefix.with_name(prefix.name + "_cp10.json")
    execution_path = prefix.with_name(prefix.name + "_execution.json")
    image_path = prefix.with_name(prefix.name + "_images.npz")
    if any(path.exists() for path in (plan_path, execution_path, image_path)):
        raise FileExistsError("refusing to overwrite integration output")

    sha = old.prior.sha
    source = json.loads(old.DECISIONS.read_text())
    root_record = json.loads(previous.ROOT_PLAN.read_text())
    old_cp = json.loads(previous.CP_PLAN.read_text())
    if (sha(old.HISTORY) != source["history_sha256"]
            or root_record["source_sha256"] != sha(old.DECISIONS)
            or root_record["history_sha256"] != sha(old.HISTORY)
            or root_record["runner_sha256"] != sha(Path(previous.__file__))
            or root_record["protocol_sha256"] != sha(previous.PROTOCOL)
            or root_record["generator_sha256"] != sha(Path(previous.compiler.__file__))
            or old_cp["root_plan_sha256"] != sha(previous.ROOT_PLAN)
            or old_cp["context_sha256"] != sha(previous.CONTEXT)):
        raise ValueError("sealed input dependency mismatch")
    saved_cp = {(r["root"], r["budget"], r["method"]): r for r in old_cp["rows"]}
    rows = root_record["rows"][:1] if args.mode == "smoke" else root_record["rows"]
    if len(saved_cp) != len(root_record["rows"]) or len(rows) != (1 if args.mode == "smoke" else 32):
        raise ValueError("unexpected sealed panel")
    camera, calibration = (json.loads(path.read_text()) for path in (h.CAMERA, h.OLD))
    metadata, offset = camera["metadata"], calibration["body_offset_from_static_scene"]
    planned, executions, images = [], [], {}
    print(f"{args.mode}: replay sealed RGB plans, rescore old tail and execute selection", flush=True)
    with np.load(old.HISTORY, allow_pickle=False) as histories, \
            np.load(h.OBSTACLE.with_suffix(".npz"), allow_pickle=False) as obstacle, \
            h.lock.gpu_lock(ROOT / "results/e3_p2_host_cuda_cooperative.lock"):
        device = h.require_cuda()
        options = argparse.Namespace(dataset_root=h.DEFAULT_DATASET, checkpoint=h.DEFAULT_CHECKPOINT,
                                     config=h.DEFAULT_CONFIG, dinov2_source=ROOT / "external/local_dinov2")
        model, dataset = h.load_model_and_dataset(options, device)
        model.eval()
        model_hash = h.parameter_hash(model)
        if (model_hash != source["model_hash"] or model_hash != root_record["model_hash"]
                or sha(h.DEFAULT_CHECKPOINT) != source["checkpoint_sha256"]):
            raise ValueError("checkpoint mismatch")
        mean, std = dataset.action_mean.to(device), dataset.action_std.to(device)
        if (not np.array_equal(mean.cpu().numpy(), source["action_mean"])
                or not np.array_equal(std.cpu().numpy(), source["action_std"])):
            raise ValueError("action normalization mismatch")
        weights, _, _ = h.get_encoder()
        pinv = torch.as_tensor(np.linalg.pinv(weights), dtype=torch.float32, device=device)
        pos_std = torch.as_tensor(h.STD[:2], dtype=torch.float32, device=device)
        goal = obstacle["goal_rgb"]
        if old.digest_image(goal) != source["goal_sha256"]:
            raise ValueError("goal mismatch")
        with torch.no_grad():
            goal_state, _ = h.estimate_state(goal, metadata, offset)
            target = h.encode_with_state(model, goal, goal_state, device)["proprio"].reshape(256, 16)
            torch.cuda.synchronize()

        for row in rows:
            root, budget, method = row["root"], row["budget"], row["method"]
            key = (root, budget, method)
            chain = np.asarray(row["chosen"]["chain"], dtype=float)
            if chain.shape != (15, 2) or chain[:10].tolist() != saved_cp[key]["first10"]:
                raise ValueError("root action binding mismatch")
            with ShadowRGBBridge(np.asarray([*old.STARTS[root], 0., 0.])) as env:
                frames = [env.initial_rgb.copy()]
                warm, _ = old.do_actions(env, old.WARMUP)
                frames += warm
                if ([old.digest_image(x) for x in frames] != source["rows"][root]["rgb_sha256"]
                        or not np.array_equal(np.stack(frames), histories[f"root{root}_history_rgb"])):
                    raise ValueError("warmup RGB mismatch")
                observed, times = old.do_actions(env, chain[:10])
                frames += observed
            if old.digest_image(frames[25]) != saved_cp[key]["cp10_rgb_sha256"]:
                raise ValueError("feedback RGB mismatch")
            start = time.monotonic()
            with torch.no_grad():
                entries = [h.encode_with_state(model, frames[t],
                           h.causal_state(frames[t-1], frames[t], metadata, offset, .05), device)
                           for t in (15, 20, 25)]
                torch.cuda.synchronize()
            context = {k: torch.cat([x[k] for x in entries], dim=1) for k in ("visual", "proprio")}
            history = ((torch.as_tensor(chain[:10], dtype=torch.float32, device=device) - mean) / std).reshape(2, 1, 10)
            encoding_seconds = time.monotonic() - start
            meter = base.Meter(96)

            def score(actions):
                def raw(batch):
                    n, steps, _ = batch.shape
                    packets = ((torch.as_tensor(batch, dtype=torch.float32, device=device) - mean) / std).reshape(n, steps // 5, 10).transpose(0, 1)
                    with torch.no_grad():
                        pred = unroll_with_action_history(model, context, history.expand(2, n, 10), packets, 3)["proprio"][-1]
                        features = pred.reshape(n, 256, 16)
                        scores = h.projected_cost(features, target, pinv, pos_std)
                        torch.cuda.synchronize()
                    return scores.cpu().numpy()
                return meter.predict(np.asarray(actions, dtype=np.float32), raw)

            start = time.monotonic()
            first10, decision = select_at_feedback(chain, root, score)
            if meter.token_samples != 96 or meter.samples != 96:
                raise ValueError("feedback prediction budget mismatch")
            planned.append({"root": root, "budget": budget, "method": method, "first10": first10,
                            "old_tail": chain[10:].tolist(), "decision": decision,
                            "cp10_rgb_sha256": old.digest_image(frames[25]),
                            "first10_times_ns": times, "encoding_calls": 3,
                            "encoding_seconds": encoding_seconds, "counts": meter.as_dict(),
                            "planning_seconds": time.monotonic() - start})
            print(f"{key}: selected {decision['source']}", flush=True)
        if h.parameter_hash(model) != model_hash:
            raise ValueError("model changed during planning")
        old.write_json(plan_path, {"rows": planned, "protocol_sha256": sha(PROTOCOL),
                                  "runner_sha256": sha(Path(__file__)), "root_plan_sha256": sha(previous.ROOT_PLAN),
                                  "prior_cp10_sha256": sha(previous.CP_PLAN), "model_hash": model_hash,
                                  "full_cost_certified": False})
        print("all feedback selections sealed before tail execution", flush=True)
        for row in planned:
            root = row["root"]
            with ShadowRGBBridge(np.asarray([*old.STARTS[root], 0., 0.])) as env:
                old.do_actions(env, old.WARMUP)
                first, _ = old.do_actions(env, row["first10"])
                if old.digest_image(first[-1]) != row["cp10_rgb_sha256"]:
                    raise ValueError("execution feedback replay mismatch")
                frames, times = old.do_actions(env, row["decision"]["tail"])
            key = f"root{root}_b{row['budget']}_{row['method']}"
            images[key] = np.stack(frames)
            executions.append({"root": root, "budget": row["budget"], "method": row["method"],
                               "first10": row["first10"], "suffix": row["decision"]["tail"],
                               "rgb_sha256": [old.digest_image(x) for x in frames], "times_ns": times})
        with image_path.open("xb") as file:
            np.savez_compressed(file, **images)
        old.write_json(execution_path, {"rows": executions, "cp10_sha256": sha(plan_path),
                                        "images_sha256": sha(image_path)})
        print(f"executed {len(executions)} selected tails", flush=True)


if __name__ == "__main__":
    main()
