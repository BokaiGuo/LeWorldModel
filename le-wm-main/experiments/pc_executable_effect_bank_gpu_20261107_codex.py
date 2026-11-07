"""GPU numerical integration check on an existing, exposed RGB panel."""
import json
from pathlib import Path

import numpy as np
import torch

from experiments import pc_tworoom_vector_retention_40dev_20261107_codex as prior
from experiments.pc_executable_effect_bank_20261107_codex import effect_bank
from experiments.pc_input_permissions import RGB, file_hash, freeze_audit
from experiments.pc_prefix_novel_actual_encoding_20261105_codex import model_fingerprint
from experiments.pc_rgb_observation_cache_20261107_codex import RGBObservationScoringSession

OUT = prior.ROOT / "results/pc_executable_effect_bank_gpu_20261107_codex.json"
PROTOCOL = prior.ROOT.parent / "docs/PC_EXECUTABLE_EFFECT_BANK_GPU_PROTOCOL_20261107_CODEX.md"


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    original = json.loads(prior.PLAN.read_text())
    if original["pool_sha256"] != file_hash(prior.POOL):
        raise ValueError("prior pool mismatch")
    common = prior.base.source.block.prior.common
    inputs_path = prior.base.source.block.prior.inputs.INPUT
    if file_hash(inputs_path) != original["input_sha256"]:
        raise ValueError("input mismatch")
    model, scaler, transform, normalizer = common.model_setup()
    if model_fingerprint(model) != original["model_sha256"] or normalizer != original["normalizer_sha256"]:
        raise ValueError("checkpoint or normalizer mismatch")
    audit = freeze_audit(RGB, "lewm",
                         [__file__, PROTOCOL, prior.PLAN, prior.POOL, inputs_path,
                          common.CHECKPOINT / "weights.pt"],
                         preprocessing="ImageNet 224, TwoRoom action scaler",
                         clock="five packets of five steps")
    rows = []
    print("Verify executable effect scores against saved A-bank scores; no physical labels.", flush=True)
    with (np.load(prior.POOL, allow_pickle=False) as pool,
          np.load(inputs_path, allow_pickle=False) as inputs, torch.inference_mode()):
        for root in (10, 20, 30, 40):
            original_actions = pool[f"r{root}_a"]
            prefixes, probes = original_actions[:, 0, :10], original_actions[0, :, 10:]
            reconstructed = np.concatenate(
                (np.broadcast_to(prefixes[:, None], (8, 8, 10, 2)),
                 np.broadcast_to(probes[None], (8, 8, 15, 2))), axis=2)
            if not np.array_equal(reconstructed, original_actions):
                raise ValueError("probe identity differs across prefixes")
            ctx = {"pixels": transform(inputs["current"][root])[None, None, None].cuda(),
                   "goal": transform(inputs["goal"][root])[None, None, None].cuda(),
                   "action": torch.empty((1, 1, 0, 10), device="cuda")}
            session = RGBObservationScoringSession(model, audit)

            def score(chains):
                norm = scaler.transform(chains.reshape(-1, 2)).reshape(-1, 5, 10)
                values = []
                for offset in range(0, len(norm), 32):
                    cost, _ = session.score(ctx, torch.as_tensor(norm[offset:offset + 32][None],
                                                                  dtype=torch.float32, device="cuda"))
                    values.extend(cost[0].cpu().tolist())
                return np.asarray(values)

            d = effect_bank(prefixes, probes, np.zeros((15, 2), dtype=np.float32), score)
            difference = float(np.max(np.abs(d["probe_costs"] - pool[f"r{root}_a_cost"])))
            if difference > 0.005:
                raise ValueError(f"probe scores drift {difference}")
            algebra = float(np.max(np.abs(d["opportunity"]
                                           - (d["reference_cost"] - d["effect"].max(axis=1)))))
            if algebra > 1e-9:
                raise ValueError("lookahead identity broken")
            rows.append({"root": root, "score_max_abs_error": difference,
                         "lookahead_identity_max_abs_error": algebra,
                         "full_chain_scores": d["num_full_chain_scores"],
                         "chains_sha256": d["chains_sha256"],
                         "reference_cost": d["reference_cost"].tolist(),
                         "effect": d["effect"].tolist()})
            print(f"root={root} A max error={difference:.8g}", flush=True)
    if model_fingerprint(model) != original["model_sha256"]:
        raise ValueError("model changed")
    with OUT.open("x") as f:
        json.dump({"track": RGB, "reference": "zero_control_not_certified_hold",
                   "protocol_sha256": file_hash(PROTOCOL), "runner_sha256": file_hash(__file__),
                   "component_sha256": file_hash(Path(effect_bank.__code__.co_filename)),
                   "prior_pool_sha256": file_hash(prior.POOL), "model_sha256": original["model_sha256"],
                   "normalizer_sha256": normalizer, "rows": rows}, f, indent=2, allow_nan=False)


if __name__ == "__main__":
    main()
