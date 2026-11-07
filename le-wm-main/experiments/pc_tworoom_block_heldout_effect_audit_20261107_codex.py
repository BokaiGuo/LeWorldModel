"""Audit frozen held-out effects with reconstructed actions and GPU score replay."""
import json
from pathlib import Path

import numpy as np
import torch

from experiments import pc_tworoom_block_heldout_effect_20261107_codex as exp
from experiments.pc_input_permissions import file_hash
from experiments.pc_prefix_novel_actual_encoding_20261105_codex import model_fingerprint

OUTPUT = Path(str(exp.PREFIX) + "_audit.json")
REPLAY_ROOTS = (0, 4, 9)


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    plan = json.loads(exp.PLAN.read_text())
    physics = json.loads(exp.PHYSICAL.read_text())
    result = json.loads(exp.RESULT.read_text())
    for actual, expected in ((plan["pool_sha256"], file_hash(exp.POOL)),
                             (plan["source_pool_sha256"], file_hash(exp.source.POOL)),
                             (plan["runner_sha256"], file_hash(exp.__file__)),
                             (plan["protocol_sha256"], file_hash(exp.PROTOCOL)),
                             (physics["plan_sha256"], file_hash(exp.PLAN)),
                             (result["physical_sha256"], file_hash(exp.PHYSICAL))):
        if actual != expected:
            raise ValueError("artifact binding mismatch")
    model, scaler, transform, normalizer = exp.source.block.prior.common.model_setup()
    if (model_fingerprint(model) != plan["model_sha256"] or
            normalizer != plan["normalizer_sha256"]):
        raise ValueError("model or normalizer mismatch")
    errors = []
    with (np.load(exp.POOL, allow_pickle=False) as pool,
          np.load(exp.source.POOL, allow_pickle=False) as old,
          np.load(exp.source.block.prior.inputs.INPUT, allow_pickle=False) as inputs,
          torch.inference_mode()):
        for root in range(10):
            p, q = plan["rows"][root], physics["rows"][root]
            if p["root"] != root or q["root"] != root or p["row"] != q["row"]:
                raise ValueError("task binding mismatch")
            prefixes = exp.select_prefixes(old[f"r{root}_block_cem_actions"],
                                           old[f"r{root}_block_cem_scores"],
                                           old[f"r{root}_block_random_actions"])
            for label, seed in (("a", 29111000 + root), ("b", 29112000 + root)):
                tail = np.random.default_rng(seed).uniform(-1, 1, (8, 3, 2)).astype(np.float32)
                np.testing.assert_array_equal(tail, pool[f"r{root}_{label}"])
                chain = pool[f"r{root}_{label}_actions"]
                np.testing.assert_array_equal(chain[:, :, :10],
                                              np.broadcast_to(prefixes[:, None, :10], (8, 8, 10, 2)))
                np.testing.assert_array_equal(chain[:, :, 10:],
                                              np.broadcast_to(np.repeat(tail, 5, axis=1)[None], (8, 8, 15, 2)))
                if root in REPLAY_ROOTS:
                    norm = scaler.transform(chain.reshape(-1, 2)).reshape(64, 25, 2)
                    context = {"pixels": transform(inputs["current"][root])[None, None, None].cuda(),
                               "goal": transform(inputs["goal"][root])[None, None, None].cuda(),
                               "action": torch.empty((1, 1, 0, 10), device="cuda")}
                    replay = []
                    for offset in (0, 32):
                        chunk = torch.tensor(norm[offset:offset+32].reshape(1, -1, 5, 10),
                                             device="cuda", dtype=torch.float32)
                        replay.extend(model.get_cost(dict(context), chunk)[0].cpu().tolist())
                    error = float(np.max(np.abs(np.asarray(replay).reshape(8, 8) -
                                                pool[f"r{root}_{label}_cost"])))
                    if error > 0.001:
                        raise ValueError(f"score replay mismatch root={root} {label} error={error}")
                    errors.append({"root": root, "bank": label, "max_error": error})
            pred = exp.predictions(pool[f"r{root}_a"], pool[f"r{root}_b"],
                                   pool[f"r{root}_a_cost"], pool[f"r{root}_parent"], 29113000 + root)
            for method in exp.METHODS:
                np.testing.assert_array_equal(pred[method], pool[f"r{root}_pred_{method}"])
                if int(np.argmin(pred[method])) != p["choices"][method]:
                    raise ValueError("prediction choice mismatch")
            if len(q["outcomes"]) != 64:
                raise ValueError("missing physical evaluations")
            for outcome in q["outcomes"]:
                trace = outcome["trace"]
                if (len(trace) != 25 or outcome["ever"] != any(x["terminated"] for x in trace)
                        or outcome["prefix_success"] != any(x["terminated"] for x in trace[:10])
                        or outcome["final_distance"] != trace[-1]["distance"]):
                    raise ValueError("physical trace mismatch")
            print(f"root={root} verified", flush=True)
    with OUTPUT.open("x") as f:
        json.dump({"roots": 10, "physical_traces": 640, "gpu_replays": errors,
                   "plan_sha256": file_hash(exp.PLAN), "physical_sha256": file_hash(exp.PHYSICAL),
                   "result_sha256": file_hash(exp.RESULT),
                   "scope": "same implementation replay, not independent environment certification"},
                  f, indent=2, allow_nan=False)


if __name__ == "__main__":
    main()
