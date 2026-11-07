"""GPU parity between historical RGB compiler and shared search core."""
import json

import numpy as np
import torch

from experiments import pc_tworoom_block_cem_50dev_20261107_codex as source
from experiments import pc_fixed_prefix_compiler_20261107_codex as previous
from experiments import pc_fixed_prefix_shared_20261107_codex as shared
from experiments import pc_fixed_prefix_rgb_shared_adapter_20261107_codex as adapter
from experiments.pc_input_permissions import RGB, RGBLeWM, file_hash, freeze_audit
from experiments.pc_prefix_novel_actual_encoding_20261105_codex import model_fingerprint

ROOT = source.ROOT
PROTOCOL = ROOT.parent / "docs/PC_FIXED_PREFIX_SHARED_RGB_PARITY_PROTOCOL_20261107_CODEX.md"
OUT = ROOT / "results/pc_fixed_prefix_shared_rgb_parity_20261107_codex.json"


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    plan = json.loads(source.PLAN.read_text())
    if plan["pool_sha256"] != file_hash(source.POOL):
        raise ValueError("candidate source changed")
    model, scaler, transform, normalizer = source.block.prior.common.model_setup()
    fingerprint = model_fingerprint(model)
    if fingerprint != plan["model_sha256"] or normalizer != plan["normalizer_sha256"]:
        raise ValueError("model or normalizer mismatch")
    dependencies = [__file__, PROTOCOL, previous.__file__, shared.__file__, adapter.__file__, source.PLAN,
                    source.POOL, source.block.prior.inputs.INPUT,
                    source.block.prior.common.CHECKPOINT / "weights.pt"]
    audit = freeze_audit(RGB, "lewm", dependencies, preprocessing="ImageNet 224 and TwoRoom action scaler",
                         clock="five packets of five environment steps")
    rows = []
    with (np.load(source.POOL, allow_pickle=False) as pool,
          np.load(source.block.prior.inputs.INPUT, allow_pickle=False) as inputs, torch.inference_mode()):
        for root in (10, 11, 12):
            parents = pool[f"r{root}_block_cem_actions"][:3]
            rng = np.random.default_rng(202611070 + root)
            probes = np.repeat(rng.uniform(-1, 1, (2, 3, 2)).astype(np.float32), 5, axis=1)
            tails = np.repeat(rng.uniform(-1, 1, (2, 3, 2)).astype(np.float32), 5, axis=1)
            ctx = {"pixels": transform(inputs["current"][root])[None, None, None].cuda(),
                   "goal": transform(inputs["goal"][root])[None, None, None].cuda(),
                   "action": torch.empty((1, 1, 0, 10), device="cuda")}
            for strategy in ("parent", "lookahead", "vector", "shuffled"):
                before, after = RGBLeWM(model, audit), RGBLeWM(model, audit)
                params = dict(scaler=scaler, prefix_steps=10, retain_count=1, strategy=strategy, seed=42)
                old = previous.compile_fixed_prefix(before, ctx, parents, probes, tails, max_queries=11, **params)
                new = adapter.compile_rgb(after, ctx, parents, probes, tails, budget=11, **params)
                max_error = 0.
                for key in ("parent_cost", "probe_cost", "tail_cost"):
                    np.testing.assert_allclose(old[key], new[key], atol=1e-5, rtol=1e-6)
                    max_error = max(max_error, float(np.max(np.abs(old[key] - new[key]))))
                if (old["retained_prefixes"] != new["retained_prefixes"]
                        or old["choice"] != new["choice"] or old["model_queries"] != new["model_queries"]):
                    raise ValueError("compiler selection or budget mismatch")
                np.testing.assert_array_equal(old["actions"], new["actions"])
                if any(r["track_id"] != RGB for r in before.receipts + after.receipts):
                    raise ValueError("invalid track receipt")
                rows.append({"root": root, "strategy": strategy, "max_cost_error": max_error,
                             "choice": {"source": new["choice"].source, "index": new["choice"].index},
                             "retained": new["retained_prefixes"], "per_implementation_queries": 11,
                             "old_receipt_count": len(before.receipts), "new_receipt_count": len(after.receipts)})
    if fingerprint != model_fingerprint(model):
        raise ValueError("model changed")
    record = {"track": RGB, "scope": "engineering parity on exposed roots; not scientific performance",
              "protocol_sha256": file_hash(PROTOCOL), "source_plan_sha256": file_hash(source.PLAN),
              "old_sha256": file_hash(previous.__file__), "shared_sha256": file_hash(shared.__file__),
              "adapter_sha256": file_hash(adapter.__file__), "runner_sha256": file_hash(__file__),
              "model_sha256": fingerprint, "rows": rows}
    with OUT.open("x") as stream:
        json.dump(record, stream, indent=2, allow_nan=False)
    print(json.dumps({"conditions": len(rows), "max_score_error": max(r["max_cost_error"] for r in rows),
                      "logical_queries_total": len(rows) * 22}, indent=2))


if __name__ == "__main__":
    main()
