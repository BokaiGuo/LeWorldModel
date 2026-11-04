"""Single exposed-root, native JEPA first-step history-packet diagnostic."""
import hashlib
import json
import os
from pathlib import Path

import torch

from experiments.spectral_reachability import p5_jepa_native_action_response_20261027_v8 as native
from experiments.spectral_reachability.e3_p2_cooperative_gpu_lock_codex_20261030_v1 import gpu_lock

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT.parent / "docs/PC_JEPA_PUSHT_HISTORY_PACKET_DIAG_PROTOCOL_20261104_CODEX.md"
MANIFEST = ROOT / "results/p5_jepa_native_action_response_20261027_v8/manifest_frozen.json"
OUTPUT = ROOT / "results/pc_jepa_pusht_history_packet_diag_20261104_codex.json"
LOCK = ROOT / "results/e3_p2_host_cuda_cooperative.lock"
UUID = "4e8973ff-0278-53f3-2bae-2615d461e441"


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def run():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    manifest = json.loads(MANIFEST.read_text())
    native.verify_manifest(ROOT, manifest)
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "0" or not torch.cuda.is_available():
        raise RuntimeError("CUDA device 0 required")
    actual_uuid = str(torch.cuda.get_device_properties(0).uuid).lower().removeprefix("gpu-")
    if actual_uuid != UUID:
        raise RuntimeError("unexpected physical GPU")
    with gpu_lock(LOCK):
        models, strict, sources, before = native.load_cuda(ROOT, manifest)
        v7 = native.helper(ROOT)
        _, v5, _ = v7.dependencies(ROOT)
        v4 = v5.import_v4(ROOT)
        arrays = v5.read_arrays(v4, ROOT)
        stats = v4.source_stats(v4.native_root(ROOT) / "app/plan_common/datasets/pusht_dset.py")
        values, rgb, length = v5.row_inputs(v4, ROOT, arrays, stats, native.EPISODE)
        saved = manifest["row"]
        if v4.array_sha(rgb) != saved["rgb_sha256"] or any(
            v4.array_sha(value) != saved["inputs"][key]["sha256"] for key, value in values.items()
        ):
            raise ValueError("fixed input binding differs")
        from app.plan_common.datasets.transforms import make_transforms
        from torch.nn.attention import SDPBackend, sdpa_kernel

        transform = make_transforms(img_size=224, random_horizontal_flip=False,
            random_resize_aspect_ratio=(1., 1.), random_resize_scale=(1., 1.),
            reprob=0., auto_augment=False, motion_shift=False)
        with torch.inference_mode(), sdpa_kernel(SDPBackend.MATH):
            pixels = torch.from_numpy(rgb.copy()).permute(0, 3, 1, 2).to("cuda", torch.float32) / 255.
            patches = models["official_encoder"](transform(pixels), is_training=True)["x_norm_patchtokens"]
            if v4.array_sha(patches.cpu().numpy()) != manifest["expected_v7_features"]["official_features_sha256"]:
                raise ValueError("encoded features differ from sealed receipt")
            visual = patches[:3].reshape(1, 3, 1, 16, 16, 384)
            target = patches[3:4].double()
            prop = torch.from_numpy(values["normalized_proprio"][:3].copy()).unsqueeze(0).to("cuda")
            features = models["proprio_encoder"](prop).expand(1, 3, 256, 16)
            packets = torch.from_numpy(values["packets"].copy()).to("cuda")
            predictions, scores = {}, {}
            for hist in ("actual", "normalized_zero"):
                for suffix in ("actual", "normalized_zero"):
                    actions = packets.clone()
                    if hist == "normalized_zero":
                        actions[:2] = 0
                    if suffix == "normalized_zero":
                        actions[2] = 0
                    out = models["predictor"](visual, actions.unsqueeze(0), features)
                    prediction = out[0][:, -1].double()
                    if prediction.shape != target.shape or not torch.isfinite(prediction).all():
                        raise ValueError("bad predictor output")
                    key = f"{hist}_history__{suffix}_suffix"
                    predictions[key] = prediction
                    scores[key] = (prediction - target).square().mean().item()
            torch.cuda.synchronize()
            def diff(a, b):
                delta = predictions[a] - predictions[b]
                return {"rmse": delta.square().mean().sqrt().item(),
                        "max_abs": delta.abs().max().item()}
            after = {key: v4.model_receipt(model) for key, model in models.items()}
            if before != after:
                raise ValueError("model state changed")
            summary = {
                "schema": "pc_jepa_pusht_history_packet_diag_20261104_codex",
                "scope": "one exposed episode; agent position and velocity included; no physical execution",
                "episode": native.EPISODE, "frames": [0, 5, 10, 15], "video_length": length,
                "protocol_sha256": digest(PROTOCOL), "script_sha256": digest(Path(__file__)),
                "native_manifest_sha256": digest(MANIFEST),
                "checkpoint_sha256": digest(ROOT / "data/jepa_wm_checkpoints/jepa_wm_pusht.pth.tar"),
                "rgb_sha256": saved["rgb_sha256"], "input_receipts": saved["inputs"],
                "gpu_uuid": actual_uuid, "strict_load": strict, "imported_sources": sources,
                "encoder_calls": 1, "proprio_encoder_calls": 1, "predictor_calls": 4,
                "target_patch_mse": scores,
                "same_suffix_history_differences": {
                    suffix: diff(f"actual_history__{suffix}_suffix", f"normalized_zero_history__{suffix}_suffix")
                    for suffix in ("actual", "normalized_zero")},
                "zero_minus_actual_suffix_mse": {
                    hist: scores[f"{hist}_history__normalized_zero_suffix"] - scores[f"{hist}_history__actual_suffix"]
                    for hist in ("actual", "normalized_zero")},
                "target_access": "evaluation only", "physical_execution": False,
                "training": False, "pc5_pass": False,
            }
    with OUTPUT.open("x") as stream:
        json.dump(summary, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"output": str(OUTPUT), "sha256": digest(OUTPUT),
                      "target_patch_mse": scores,
                      "same_suffix_history_differences": summary["same_suffix_history_differences"]}))


if __name__ == "__main__":
    run()
