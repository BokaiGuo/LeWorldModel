"""Audit sealed v7 Reacher feedback and its paired final comparisons."""

import json
import numpy as np

from experiments import pc_v7_reacher_rgb_live_20261104_codex as configured


def main():
    configured.configure()
    run = configured.live
    out = configured.PREFIX.with_name(configured.PREFIX.name + "_audit.json")
    if out.exists():
        raise FileExistsError(out)
    sha = run.host.bridge.sha_file
    source = json.loads(run.run.SOURCE.read_text())
    manifest = json.loads(run.MANIFEST.read_text())
    if manifest["dependencies"] != run.dependencies(source):
        raise ValueError("dependency mismatch")
    execution_path = run.path("response", "execution.json")
    receipt = json.loads(execution_path.read_text())
    if receipt["manifest_sha256"] != sha(run.MANIFEST) or len(receipt["rows"]) != 3:
        raise ValueError("execution receipt mismatch")
    carry, steps = None, []
    for step, execution in enumerate(receipt["rows"]):
        current = run.path("response", f"context{step}.npz")
        successor = run.path("response", f"context{step + 1}.npz")
        decision = run.path("response", f"decision{step}.json")
        data = json.loads(decision.read_text())
        if (data["manifest_sha256"] != sha(run.MANIFEST) or data["context_sha256"] != sha(current)
                or execution["decision_sha256"] != sha(decision) or execution["context_sha256"] != sha(current)
                or data["carry_in"] != carry or execution["step"] != step
                or execution["execution_return_ns"] < execution["sealed_ns"]):
            raise ValueError("clock or decision binding mismatch")
        with np.load(current, allow_pickle=False) as ctx, np.load(successor, allow_pickle=False) as nxt:
            if (run.context_lib.rgb_digest(ctx["root"]) != data["root_rgb_sha256"]
                    or run.context_lib.rgb_digest(ctx["goal"]) != data["goal_rgb_sha256"]
                    or [run.context_lib.rgb_digest(frame) for frame in ctx["frames"]] != data["history_rgb_sha256"]
                    or not np.array_equal(nxt["frames"], np.concatenate((ctx["frames"][1:], ctx["root"][None])))
                    or not np.array_equal(nxt["history"][:-1], ctx["history"][1:])
                    or not np.array_equal(nxt["history"][-1], np.asarray(execution["dispatch"], np.float32).reshape(-1))
                    or not np.array_equal(nxt["goal"], ctx["goal"])
                    or run.context_lib.rgb_digest(nxt["root"]) != execution["successor_root_sha256"]):
                raise ValueError("causal observation/action mismatch")
        result, trace = data["result"], data["trace"]
        selected = np.asarray(result["selected"]["chain"])
        if (selected.shape != (15-5*step, 2)
                or selected[:5].astype(np.float32).tolist() != execution["dispatch"]
                or data["dispatch_float32"] != execution["dispatch"]):
            raise ValueError("dispatch mismatch")
        tokens, samples = 0, 0
        for batch in trace:
            requested, raw = np.asarray(batch["requested"]), np.asarray(batch["raw"], dtype=np.float32)
            if (not np.array_equal(requested.astype(np.float32), raw) or np.any(np.abs(raw)>1)
                    or not np.isfinite(batch["positions"]).all() or not np.isfinite(batch["scores"]).all()
                    or len(raw) != len(batch["scores"]) or len(raw) != len(batch["positions"])
                    or batch["tokens"] != len(raw)*raw.shape[1]//5):
                raise ValueError("trace request mismatch")
            tokens, samples = tokens + batch["tokens"], samples + len(raw)
        paired = trace[-1]
        request, raw, scores = np.asarray(paired["requested"]), np.asarray(paired["raw"], np.float32), paired["scores"]
        expected = np.stack((np.asarray(result["precomparison"]["chain"]), np.asarray(result["comparison_chain"])))
        same = np.array_equal(raw[0], raw[1])
        abstained = bool(same or scores[1] <= scores[0])
        index = int(abstained)
        record = result["paired_comparison"]
        if (not np.array_equal(request, expected) or result["abstained"] != abstained
                or record["same_actual_float32_actions"] != same
                or record["incumbent_score"] != scores[0] or record["zero_prefix_score"] != scores[1]
                or record["reserved_token_samples"] != 2*(3-step)
                or not np.array_equal(selected, request[index])
                or result["selected"]["terminal_score"] != scores[index]
                or data["counts"]["prediction_token_samples"] != tokens or tokens > 1920
                or data["counts"]["trajectory_samples"] != samples
                or data["counts"]["prediction_calls"] != len(trace)):
            raise ValueError("paired comparison or cost mismatch")
        if step and (trace[0]["requested"] != [carry] or result["carry_score"] != trace[0]["scores"][0]):
            raise ValueError("carry mismatch")
        generations = result["refinement"]
        boundary = len(trace)-len(generations)-2
        earlier = [(action, score) for batch in trace[:boundary] if len(batch["requested"][0]) == len(selected)
                   for action, score in zip(batch["requested"], batch["scores"], strict=True)]
        start = result["refinement_start"]
        if (boundary < 1 or len(earlier) != start["full_samples_seen"]
                or min(score for _, score in earlier) != start["terminal_score"]
                or (start["chain"], start["terminal_score"]) not in earlier
                or any(gen["size"] != len(batch["scores"])
                       for gen, batch in zip(generations, trace[boundary:-2], strict=True))):
            raise ValueError("refinement accounting mismatch")
        carry = selected[5:].astype(np.float32).tolist() if step < 2 else None
        steps.append({"step": step, "tokens": tokens, "prediction_calls": len(trace), "samples": samples,
                      "selected_score": scores[index], "abstained": abstained, "same_actions": same})
    if receipt["final_context_sha256"] != sha(run.path("response", "context3.npz")):
        raise ValueError("final context mismatch")
    distances = []
    for outcome in receipt["initial_and_block_outcomes"]:
        distance = float(run.p0._qpos_distance(np.asarray(outcome["qpos"]), np.asarray(outcome["target_qpos"])))
        if abs(distance-outcome["distance"]) > 1e-12:
            raise ValueError("physical metric mismatch")
        distances.append(distance)
    if len(distances) != 4:
        raise ValueError("incorrect physical boundary count")
    run.host.write_new(out, {"manifest_sha256": sha(run.MANIFEST), "execution_sha256": sha(execution_path),
        "steps": steps, "distances": distances,
        "scope": "record and causal history audit; prediction and physical replay separate"})
    print("Audited v7 Reacher decisions, paired comparison and causal history.", flush=True)


if __name__ == "__main__":
    main()
