"""Check the reusable loader against the sealed repaired predictions, no outcomes."""
from pathlib import Path

import numpy as np
import torch

from experiments.pc1_layer_audit import camera, dino_source_loader, dino_repair_v2_run as run
from experiments.pc1_layer_audit import dino_repair_run as base
from experiments.pc_input_permissions import STATE, file_hash
from experiments.pc_input_permissions_v2_20261107_codex import StateAssistedPointMazev2
from experiments.pc_prefix_novel_actual_encoding_20261105_codex import model_fingerprint


def main():
    output = run.OUT/"loader_smoke.json"
    if output.exists():
        raise FileExistsError(output)
    model, _, audit = dino_source_loader.load_model(track=STATE)
    boundary = StateAssistedPointMazev2(model, audit)
    fingerprint = model_fingerprint(model)
    _, packet = camera.packet_actions(model.preprocessor)
    checks = {}
    with (np.load(base.BASE/"prepare.npz", allow_pickle=False) as inputs,
          np.load(run.OUT/"predict_r1832.npz", allow_pickle=False) as sealed,
          torch.inference_mode()):
        obs = dict(visual=camera.raw_rgb(np.repeat(inputs["frames"][0, 0, 0][None], 3, axis=0)),
                   proprio=torch.tensor(inputs["states"][0, 0], dtype=torch.float32, device="cuda")[None, None].expand(3, 1, 4).contiguous())
        prediction = boundary.predict(obs, packet)
        for key in ("visual", "proprio"):
            actual = prediction[key].cpu().numpy()
            np.testing.assert_array_equal(actual, sealed[f"v0_pred_{key}"])
            checks[key+"_maximum_error"] = float(np.max(np.abs(actual-sealed[f"v0_pred_{key}"])))
    if fingerprint != model_fingerprint(model):
        raise ValueError("weights changed")
    base.write_json(output, dict(track=STATE, passed=True, checks=checks,
        input_audit_hash=audit.fingerprint, model_fingerprint=fingerprint,
        sources={str(p): file_hash(p) for p in (
            Path(__file__), Path(dino_source_loader.__file__), run.OUT/"manifest.json",
            run.OUT/"predict_r1832.json", run.OUT/"predict_r1832.npz")},
        permission="explicit state-assisted; requesting primary_rgb raises before model load"))
    print(output, checks, flush=True)


if __name__ == "__main__":
    main()
