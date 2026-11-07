"""Rebuild block candidates, rescore on GPU, and verify physical bindings."""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from experiments.pc_input_permissions import file_hash
from experiments.pc_prefix_novel_actual_encoding_20261105_codex import model_fingerprint
from experiments import pc_tworoom_block_action_coverage_20261107_codex as exp
from experiments import pc_tworoom_failed_root_full_pool_20261107_codex as old_pool

OUTPUT = exp.PREFIX.with_name(exp.PREFIX.name + "_audit.json")


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    plan = json.loads(exp.PLAN.read_text())
    physical = json.loads(exp.PHYSICAL.read_text())
    result = json.loads(exp.RESULT.read_text())
    old = json.loads(exp.prior.PLAN.read_text())
    old_physical = json.loads(old_pool.OUTPUT.read_text())
    if (physical["plan_sha256"] != file_hash(exp.PLAN) or result["plan_sha256"] != file_hash(exp.PLAN)
            or result["physical_sha256"] != file_hash(exp.PHYSICAL)
            or plan["protocol_sha256"] != file_hash(exp.PROTOCOL)
            or plan["runner_sha256"] != file_hash(exp.__file__)
            or plan["source_sha256"] != file_hash(exp.prior.PLAN)
            or old_physical["plan_sha256"] != file_hash(exp.prior.PLAN)):
        raise ValueError("artifact binding mismatch")
    model, scaler, transform, action_hash = exp.prior.common.model_setup()
    fingerprint = model_fingerprint(model)
    if fingerprint != plan["model_sha256"] or action_hash != plan["action_scaler_sha256"]:
        raise ValueError("model mismatch")
    rows = []
    with np.load(exp.prior.inputs.INPUT, allow_pickle=False) as inputs, torch.inference_mode():
        for root, (p, q, previous) in enumerate(zip(plan["rows"], physical["rows"], old["rows"], strict=True)):
            if p["row"] != q["row"] or p["row"] != previous["row"] or p["root"] != root:
                raise ValueError("root mismatch")
            raw = exp.block_actions(p["seed"])
            np.testing.assert_array_equal(raw, np.asarray(p["actions"], dtype=np.float32))
            index = int(np.argmin(previous["generation_costs"]))
            baseline = exp.prior.to_physical(np.asarray(previous["generation_actions"][index]), scaler)
            np.testing.assert_array_equal(baseline, np.asarray(p["baseline_actions"]))
            if index != p["baseline_index"] or int(np.argmin(p["scores"])) != p["choice"]:
                raise ValueError("candidate choice mismatch")
            norm = scaler.transform(raw.reshape(-1, 2)).reshape(300, 25, 2)
            context = {"pixels": transform(inputs["current"][root])[None, None, None].cuda(),
                       "goal": transform(inputs["goal"][root])[None, None, None].cuda(),
                       "action": torch.empty((1, 1, 0, 10), device="cuda")}
            costs = []
            for offset in range(0, 300, 32):
                chunk = torch.tensor(norm[offset:offset + 32].reshape(1, -1, 5, 10),
                                     device="cuda", dtype=torch.float32)
                values = model.get_cost(dict(context), chunk)
                costs.extend(values[0].cpu().tolist())
            error = float(np.max(np.abs(np.asarray(costs) - p["scores"])))
            if error > 1e-3 or int(np.argmin(costs)) != p["choice"]:
                raise ValueError("GPU score replay mismatch")
            for key, actions in (("selected", raw[p["choice"]]), ("baseline", baseline)):
                digest = hashlib.sha256(np.asarray(actions, dtype=np.float32).tobytes()).hexdigest()
                if digest != q[key]["action_sha256"] or len(q[key]["trace"]) != 25:
                    raise ValueError("selected physical witness mismatch")
                if q[key]["ever_success"] != any(t["terminated"] for t in q[key]["trace"]):
                    raise ValueError("success predicate mismatch")
            if p["row"] in exp.ORACLE_ROWS:
                if len(q["all_candidates"]) != 300:
                    raise ValueError("oracle pool incomplete")
                for i, candidate in enumerate(q["all_candidates"]):
                    if (candidate["candidate_index"] != i or len(candidate["trace"]) != 25
                            or candidate["action_sha256"] != hashlib.sha256(raw[i].tobytes()).hexdigest()
                            or candidate["ever_success"] != any(t["terminated"] for t in candidate["trace"])):
                        raise ValueError("oracle action binding mismatch")
                old_row = next(row for row in old_physical["rows"] if row["row"] == p["row"])
                old_candidate = next(candidate for candidate in old_row["candidates"] if candidate["id"] == f"cem_{index}")
                if old_candidate["trace"] != q["baseline"]["trace"]:
                    raise ValueError("baseline physical replay differs from prior full pool")
            rows.append({"root": root, "row": p["row"], "max_score_error": error,
                         "baseline_prior_trace_checked": p["row"] in exp.ORACLE_ROWS})
            print(f"verified row={p['row']} score_max_error={error:.6g}", flush=True)
    if model_fingerprint(model) != fingerprint:
        raise ValueError("frozen weights changed")
    output = {"rows": rows, "model_sha256": fingerprint, "plan_sha256": file_hash(exp.PLAN),
              "physical_sha256": file_hash(exp.PHYSICAL), "result_sha256": file_hash(exp.RESULT),
              "old_pool_sha256": file_hash(old_pool.OUTPUT), "runner_sha256": file_hash(Path(__file__)),
              "scope": "same implementation replay, not independent environment certification"}
    with OUTPUT.open("x") as f:
        json.dump(output, f, indent=2, allow_nan=False)


if __name__ == "__main__":
    main()
