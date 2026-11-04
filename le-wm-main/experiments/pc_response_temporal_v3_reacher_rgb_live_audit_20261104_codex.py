"""Audit saved v3 Reacher RGB contexts, selection, accounting and dispatch."""
import json
from pathlib import Path

import numpy as np

from experiments import pc_response_temporal_v3_reacher_rgb_live_20261104_codex as run

OUT = run.PREFIX.with_name(run.PREFIX.name + "_audit.json")


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    sha = run.host.bridge.sha_file
    source = json.loads(run.run.SOURCE.read_text())
    manifest = json.loads(run.MANIFEST.read_text())
    if manifest["dependencies"] != run.dependencies(source):
        raise ValueError("source dependency mismatch")
    findings = []
    for arm in run.ARMS:
        receipt_path = run.path(arm, "execution.json")
        receipt = json.loads(receipt_path.read_text())
        if receipt["manifest_sha256"] != sha(run.MANIFEST) or receipt["arm"] != arm or len(receipt["rows"]) != 3:
            raise ValueError("receipt mismatch")
        carry, steps = None, []
        for step, execution in enumerate(receipt["rows"]):
            context_file = run.path(arm, f"context{step}.npz")
            next_file = run.path(arm, f"context{step + 1}.npz")
            decision_file = run.path(arm, f"decision{step}.json")
            data = json.loads(decision_file.read_text())
            if (data["manifest_sha256"] != sha(run.MANIFEST) or data["context_sha256"] != sha(context_file)
                    or execution["context_sha256"] != sha(context_file) or execution["decision_sha256"] != sha(decision_file)
                    or data["carry_in"] != carry or execution["step"] != step
                    or execution["execution_return_ns"] < execution["sealed_ns"]):
                raise ValueError("decision binding or clock mismatch")
            with np.load(context_file, allow_pickle=False) as context, np.load(next_file, allow_pickle=False) as next_context:
                if (run.context_lib.rgb_digest(context["root"]) != data["root_rgb_sha256"]
                        or run.context_lib.rgb_digest(context["goal"]) != data["goal_rgb_sha256"]
                        or [run.context_lib.rgb_digest(frame) for frame in context["frames"]] != data["history_rgb_sha256"]
                        or not np.array_equal(next_context["frames"], np.concatenate((context["frames"][1:], context["root"][None])))
                        or not np.array_equal(next_context["history"][:-1], context["history"][1:])
                        or not np.array_equal(next_context["history"][-1], np.asarray(execution["dispatch"], np.float32).reshape(-1))
                        or not np.array_equal(next_context["goal"], context["goal"])
                        or run.context_lib.rgb_digest(next_context["root"]) != execution["successor_root_sha256"]):
                    raise ValueError("actual RGB/action history mismatch")
            result, trace = data["result"], data["trace"]
            selected, selected_score = np.asarray(result["selected"]["chain"]), result["selected"]["terminal_score"]
            if (selected.shape != (15 - 5 * step, 2)
                    or selected[:5].astype(np.float32).tolist() != execution["dispatch"]
                    or execution["dispatch"] != data["dispatch_float32"]):
                raise ValueError("physical dispatch mismatch")
            values, found, tokens, samples = [], False, 0, 0
            for batch in trace:
                requested, raw = np.asarray(batch["requested"]), np.asarray(batch["raw"], dtype=np.float32)
                if (not np.array_equal(requested.astype(np.float32), raw)
                        or len(batch["positions"]) != len(raw) or len(batch["scores"]) != len(raw)
                        or not np.isfinite(batch["positions"]).all() or not np.isfinite(batch["scores"]).all()
                        or batch["tokens"] != len(raw) * raw.shape[1] // 5):
                    raise ValueError("prediction request mismatch")
                tokens += batch["tokens"]
                samples += len(raw)
                if raw.shape[1] == len(selected):
                    values.extend(batch["scores"])
                    found |= any(batch["scores"][i] == selected_score
                                 for i in np.flatnonzero(np.all(requested == selected, axis=(1, 2))))
            counts = data["counts"]
            if (not found or min(values) != selected_score or tokens > 1920
                    or counts["prediction_token_samples"] != tokens or counts["trajectory_samples"] != samples
                    or counts["prediction_calls"] != len(trace)):
                raise ValueError("selection or cost mismatch")
            if step and (trace[0]["requested"] != [carry] or result["carry_score"] != trace[0]["scores"][0]):
                raise ValueError("carry rescore mismatch")
            if arm == "response":
                generations = result["refinement"]
                boundary = len(trace) - len(generations) - 1
                pre = [(action, score) for batch in trace[:boundary] if len(batch["requested"][0]) == len(selected)
                       for action, score in zip(batch["requested"], batch["scores"], strict=True)]
                start = result["refinement_start"]
                if (boundary < 1 or len(pre) != start["full_samples_seen"]
                        or min(score for _, score in pre) != start["terminal_score"]
                        or (start["chain"], start["terminal_score"]) not in pre
                        or any(gen["size"] != len(batch["scores"])
                               for gen, batch in zip(generations, trace[boundary:-1], strict=True))):
                    raise ValueError("refinement incumbent mismatch")
            carry = selected[5:].astype(np.float32).tolist() if step < 2 else None
            steps.append({"step": step, "tokens": tokens, "prediction_calls": len(trace),
                          "selected_score": selected_score, "abstained": result["abstained"]})
        if receipt["final_context_sha256"] != sha(run.path(arm, "context3.npz")):
            raise ValueError("final observation mismatch")
        distances = []
        for outcome in receipt["initial_and_block_outcomes"]:
            distance = float(run.p0._qpos_distance(np.asarray(outcome["qpos"]), np.asarray(outcome["target_qpos"])))
            if abs(distance - outcome["distance"]) > 1e-12:
                raise ValueError("physical outcome mismatch")
            distances.append(distance)
        if len(distances) != 4:
            raise ValueError("invalid physical outcome clock")
        findings.append({"arm": arm, "steps": steps, "distances": distances, "execution_sha256": sha(receipt_path)})
    run.host.write_new(OUT, {"manifest_sha256": sha(run.MANIFEST), "rows": findings,
                             "limitations": ["recorded model predictions not independently recomputed",
                                             "physical snapshots not independently replayed",
                                             "one exposed root"]})
    print("Audited six RGB decisions, history, selection, resources and recorded dispatch.", flush=True)


if __name__ == "__main__":
    main()
