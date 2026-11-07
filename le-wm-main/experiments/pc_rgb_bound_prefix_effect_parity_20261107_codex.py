"""GPU parity with the sealed RGB fixed-prefix physical-diagnostic matrix."""
import json
from pathlib import Path

import numpy as np
import torch

from experiments import pc_tworoom_rgb_action_effect_alignment_20261107_codex as source
from experiments.pc_input_permissions import RGB, RGBLeWM, file_hash, freeze_audit
from experiments.pc_prefix_novel_actual_encoding_20261105_codex import model_fingerprint
from experiments.pc_rgb_bound_prefix_effect_20261107_codex import score_prefix_effects

OUTPUT = source.ROOT / "results/pc_rgb_bound_prefix_effect_parity_20261107_codex.json"
PROTOCOL = source.ROOT.parent / "docs/PC_RGB_BOUND_PREFIX_EFFECT_PARITY_PROTOCOL_20261107_CODEX.md"


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    previous = json.loads(source.PRED.read_text())
    if previous["input_sha256"] != file_hash(source.INPUT):
        raise ValueError("sealed RGB input mismatch")
    model, scaler, transform, action_hash = source.model_setup()
    fingerprint = model_fingerprint(model)
    if fingerprint != previous["model_sha256"] or action_hash != previous["normalizer_action_sha256"]:
        raise ValueError("model or normalizer mismatch")
    audit = freeze_audit(RGB, "lewm", [Path(__file__), PROTOCOL, source.INPUT,
                                      source.CHECKPOINT / "weights.pt", source.CHECKPOINT / "config.json"],
                         preprocessing="ImageNet 224; TwoRoom dataset action scaler",
                         clock="two five-environment-step packets")
    prefixes = np.stack([np.tile(a, (5, 1)) for a in source.ACTIONS])
    tails = np.stack([np.tile(a, (5, 1)) for a in source.ACTIONS])
    rows = []
    print("Compare four sealed RGB cost matrices and hold-relative effects on GPU.", flush=True)
    with np.load(source.INPUT, allow_pickle=False) as images, torch.inference_mode():
        for old in previous["rows"]:
            seed = old["seed"]
            current = {"pixels": transform(images[f"s{seed}_current"])[None, None, None].cuda(),
                       "goal": transform(images[f"s{seed}_goal"])[None, None, None].cuda(),
                       "action": torch.empty((1, 1, 0, 10), device="cuda")}
            wrapper = RGBLeWM(model, audit)

            def score(chains):
                normal = scaler.transform(chains.reshape(-1, 2)).reshape(1, len(chains), 2, 10)
                values, _ = wrapper.score(current, torch.tensor(normal, dtype=torch.float32, device="cuda"))
                return values[0].cpu().numpy()

            desc = score_prefix_effects(score, prefixes, tails[0], tails[1:], remaining_steps=10)
            values = np.concatenate((desc["hold_cost"][:, None], desc["probe_cost"]), axis=1)
            prior = np.asarray(old["cost"])
            old_effect = prior[:, :1] - prior[:, 1:]
            max_cost = float(np.abs(values - prior).max())
            max_effect = float(np.abs(desc["effects"] - old_effect).max())
            if max_cost > 1e-3 or max_effect > 2e-3 or desc["queries"] != 25:
                raise ValueError("sealed model parity failed")
            rows.append({"seed": seed, "cost": values.tolist(), "effects": desc["effects"].tolist(),
                         "max_cost_error": max_cost, "max_effect_error": max_effect,
                         "queries": desc["queries"], "model_packet_steps": desc["model_packet_steps"],
                         "receipts": wrapper.receipts})
    if model_fingerprint(model) != fingerprint:
        raise ValueError("model weights changed")
    output = {"track": RGB, "scope": "interface parity, exposed roots, no gate claim",
              "model_sha256": fingerprint, "normalizer_sha256": action_hash,
              "source_prediction_sha256": file_hash(source.PRED), "source_input_sha256": file_hash(source.INPUT),
              "runner_sha256": file_hash(Path(__file__)), "adapter_sha256": file_hash(
                  source.ROOT / "experiments/pc_rgb_bound_prefix_effect_20261107_codex.py"),
              "protocol_sha256": file_hash(PROTOCOL), "rows": rows}
    with OUTPUT.open("x") as stream:
        json.dump(output, stream, indent=2, allow_nan=False)
    print(json.dumps([{"seed": x["seed"], "max_cost_error": x["max_cost_error"],
                       "max_effect_error": x["max_effect_error"]} for x in rows], indent=2))


if __name__ == "__main__":
    main()
