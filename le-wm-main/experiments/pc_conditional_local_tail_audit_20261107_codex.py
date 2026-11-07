"""Reconstruct all proposals and selections, plus GPU-score a fixed three-root subset."""
import json
from pathlib import Path

import numpy as np
import torch

from experiments import pc_tworoom_conditional_local_tail_20261107_codex as run
from experiments.pc_input_permissions import RGB, file_hash, freeze_audit
from experiments.pc_prefix_novel_actual_encoding_20261105_codex import model_fingerprint
from experiments.pc_rgb_observation_cache_20261107_codex import RGBObservationScoringSession

OUT = run.ROOT / "results/pc_conditional_local_tail_audit_20261107_codex.json"
CHECK_ROOTS = (18, 38, 49)


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    plan, physical = json.loads(run.PLAN.read_text()), json.loads(run.PHYSICAL.read_text())
    for key, path in (("pool_sha256", run.POOL), ("source_plan_sha256", run.source.PLAN),
                      ("source_pool_sha256", run.source.POOL), ("protocol_sha256", run.PROTOCOL),
                      ("runner_sha256", run.__file__)):
        if plan[key] != file_hash(path):
            raise ValueError("plan binding mismatch")
    if physical["plan_sha256"] != file_hash(run.PLAN) or physical["pool_sha256"] != file_hash(run.POOL):
        raise ValueError("physical binding mismatch")
    if [x["root"] for x in plan["rows"]] != list(range(10, 50)) or \
       [x["root"] for x in physical["rows"]] != list(range(10, 50)):
        raise ValueError("incomplete roots")
    common = run.source.old.base.source.block.prior.common
    inp = run.source.old.base.source.block.prior.inputs.INPUT
    if file_hash(inp) != plan["input_sha256"]:
        raise ValueError("input drift")
    model, scaler, transform, normalizer = common.model_setup()
    if model_fingerprint(model) != plan["model_sha256"] or normalizer != plan["normalizer_sha256"]:
        raise ValueError("model drift")
    audit = freeze_audit(RGB, "lewm", [__file__, run.PLAN, run.POOL, inp, common.CHECKPOINT / "weights.pt"],
                         preprocessing="ImageNet 224, TwoRoom action scaler",
                         clock="five packets of five environment steps")
    max_diff = 0.0
    with (np.load(run.POOL, allow_pickle=False) as saved,
          np.load(run.source.POOL, allow_pickle=False) as prev,
          np.load(run.source.old.POOL, allow_pickle=False) as old,
          np.load(inp, allow_pickle=False) as inputs, torch.inference_mode()):
        for row, real in zip(plan["rows"], physical["rows"], strict=True):
            root = row["root"]
            actions, winners = run.proposals(old[f"r{root}_a"], prev[f"r{root}_a_cost"], root)
            np.testing.assert_array_equal(actions, saved[f"r{root}_local"])
            np.testing.assert_array_equal(winners, row["winners"])
            ids, scores = saved[f"r{root}_ids"], saved[f"r{root}_scores"]
            expected = sorted({p * 8 + j for m in run.METHODS
                               for p in row["decisions"][m]["retained"] for j in range(8)})
            np.testing.assert_array_equal(ids, expected)
            for i, m in enumerate(run.METHODS):
                bank, index, cost = run.pick(row["common_cost"], scores, ids, row["decisions"][m]["retained"])
                if (bank, index, cost) != tuple(row["decisions"][m][k] for k in ("bank", "index", "cost")):
                    raise ValueError("decision drift")
                action = saved[f"r{root}_actions"][i]
                expected_action = (prev[f"r{root}_actions"][0] if bank == "common"
                                   else actions.reshape(64, 25, 2)[index])
                np.testing.assert_array_equal(action, expected_action)
                if run.source.digest(action) != row["decisions"][m]["action_sha256"]:
                    raise ValueError("action mismatch")
                trace = real["outcomes"][m]["trace"]
                if len(trace) != 25 or real["outcomes"][m]["ever"] != any(x["terminated"] for x in trace):
                    raise ValueError("physical trace mismatch")
            if root in CHECK_ROOTS:
                raw = np.concatenate((prev[f"r{root}_actions"][:1], actions.reshape(64, 25, 2)[ids]))
                norm = scaler.transform(raw.reshape(-1, 2)).reshape(-1, 5, 10)
                ctx = {"pixels": transform(inputs["current"][root])[None, None, None].cuda(),
                       "goal": transform(inputs["goal"][root])[None, None, None].cuda(),
                       "action": torch.empty((1, 1, 0, 10), device="cuda")}
                session = RGBObservationScoringSession(model, audit)
                vals = []
                for offset in range(0, len(raw), 32):
                    result, _ = session.score(ctx, torch.as_tensor(norm[offset:offset+32][None],
                                                                   dtype=torch.float32, device="cuda"))
                    vals.extend(result[0].cpu().tolist())
                err = float(np.max(np.abs(np.asarray(vals) - np.r_[row["common_cost"], scores])))
                max_diff = max(max_diff, err)
                if err > 0.005:
                    raise ValueError("GPU replay score drift")
                print(f"root={root} score max error={err}", flush=True)
    if model_fingerprint(model) != plan["model_sha256"]:
        raise ValueError("model changed")
    with OUT.open("x") as f:
        json.dump({"all_40_reconstructed": True, "gpu_roots": CHECK_ROOTS,
                   "maximum_score_difference": max_diff, "physical_replayed": False,
                   "plan_sha256": file_hash(run.PLAN), "pool_sha256": file_hash(run.POOL),
                   "physical_sha256": file_hash(run.PHYSICAL), "runner_sha256": file_hash(__file__)},
                  f, indent=2, allow_nan=False)


if __name__ == "__main__":
    main()
