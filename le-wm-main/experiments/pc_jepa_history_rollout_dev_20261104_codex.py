"""Compare original JEPA wrapper and explicit-history adapter on one exposed episode."""
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch

from experiments.spectral_reachability import p5_jepa_native_action_response_20261027_v8 as native
from experiments.spectral_reachability.e3_p2_cooperative_gpu_lock_codex_20261030_v1 import gpu_lock
from experiments.spectral_reachability.pc_jepa_history_rollout import unroll_history

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT.parent / "docs/PC_JEPA_PUSHT_HISTORY_AWARE_ADAPTER_PROTOCOL_20261104_CODEX.md"
MANIFEST = ROOT / "results/p5_jepa_native_action_response_20261027_v8/manifest_frozen.json"
PRIOR = ROOT / "results/pc_jepa_pusht_history_packet_diag_20261104_codex.json"
OUTPUT = ROOT / "results/pc_jepa_history_rollout_dev_20261104_codex.json"
LOCK = ROOT / "results/e3_p2_host_cuda_cooperative.lock"
UUID = "4e8973ff-0278-53f3-2bae-2615d461e441"


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1048576), b""):
            h.update(chunk)
    return h.hexdigest()


class NativeCore(torch.nn.Module):
    def __init__(self, predictor):
        super().__init__()
        self.predictor = predictor
        self.calls = 0

    def encode_act(self, actions):
        return actions

    def forward_pred(self, video, actions, proprio, debug=False):
        del debug
        self.calls += 1
        predicted, action, prop = self.predictor(video, actions, proprio)
        batch, time, views, height, width, dim = video.shape
        if predicted.shape != (batch, time, views * height * width, dim):
            raise ValueError("native predictor patch shape differs")
        return predicted.reshape(batch, time, views, height, width, dim), action, prop


def run():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    manifest = json.loads(MANIFEST.read_text())
    previous = json.loads(PRIOR.read_text())
    native.verify_manifest(ROOT, manifest)
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "0" or not torch.cuda.is_available():
        raise RuntimeError("CUDA device 0 required")
    uuid = str(torch.cuda.get_device_properties(0).uuid).lower().removeprefix("gpu-")
    if uuid != UUID:
        raise RuntimeError("unexpected GPU")
    with gpu_lock(LOCK):
        models, strict, sources, before = native.load_cuda(ROOT, manifest)
        v7 = native.helper(ROOT)
        _, v5, _ = v7.dependencies(ROOT)
        v4 = v5.import_v4(ROOT)
        arrays = v5.read_arrays(v4, ROOT)
        stats = v4.source_stats(v4.native_root(ROOT) / "app/plan_common/datasets/pusht_dset.py")
        values, rgb, length = v5.row_inputs(v4, ROOT, arrays, stats, native.EPISODE)
        if v4.array_sha(rgb) != manifest["row"]["rgb_sha256"] or any(
            v4.array_sha(value) != manifest["row"]["inputs"][key]["sha256"]
            for key, value in values.items()
        ):
            raise ValueError("fixed context mismatch")
        raw_actions = arrays[2][native.EPISODE, :25]
        packets = (raw_actions / 100. - np.asarray(stats["ACTION_MEAN"], np.float32)) / np.asarray(
            stats["ACTION_STD"], np.float32)
        packets = torch.from_numpy(packets.reshape(5, 10).copy()).to("cuda")
        if not np.array_equal(packets[:3].cpu().numpy(), values["packets"]):
            raise ValueError("action preprocessing mismatch")
        from app.plan_common.datasets.transforms import make_transforms
        from app.vjepa_wm.modelcustom.simu_env_planning.vit_enc_preds import EncPredWM
        from tensordict import TensorDict
        from torch.nn.attention import SDPBackend, sdpa_kernel

        transform = make_transforms(img_size=224, random_horizontal_flip=False,
            random_resize_aspect_ratio=(1., 1.), random_resize_scale=(1., 1.),
            reprob=0., auto_augment=False, motion_shift=False)
        with torch.inference_mode(), sdpa_kernel(SDPBackend.MATH):
            pixels = torch.from_numpy(rgb.copy()).permute(0, 3, 1, 2).to("cuda", torch.float32) / 255.
            patches = models["official_encoder"](transform(pixels), is_training=True)["x_norm_patchtokens"]
            if v4.array_sha(patches.cpu().numpy()) != manifest["expected_v7_features"]["official_features_sha256"]:
                raise ValueError("prior encoded feature mismatch")
            visual = patches[:3].reshape(1, 3, 1, 16, 16, 384)
            prop = torch.from_numpy(values["normalized_proprio"][:3].copy()).unsqueeze(0).to("cuda")
            features = models["proprio_encoder"](prop).expand(1, 3, 256, 16)
            suffix = packets[2:5].unsqueeze(1)
            core = NativeCore(models["predictor"])
            # Bypass only the constructor; execute the original unroll method,
            # with its actual encode_act/forward_pred dependencies supplied.
            wrapper = EncPredWM.__new__(EncPredWM)
            torch.nn.Module.__init__(wrapper)
            wrapper.model = core
            wrapper.ctxt_window = 3
            wrapper.proprio_mode = "predict_proprio"
            ctxt = TensorDict({"visual": visual, "proprio": features}, batch_size=[1, 3])
            original = wrapper.unroll(ctxt, suffix)
            corrected = unroll_history(core, visual, features, packets[:2].unsqueeze(0), suffix, 3)
            if core.calls != 6:
                raise ValueError("unexpected predictor count")
            # Read future images only after both predictions are fully determined.
            import decord
            video = v4.data_root(ROOT) / "obses/episode_001.mp4"
            reader = decord.VideoReader(str(video), num_threads=1, ctx=decord.cpu(0))
            future = reader.get_batch([15, 20, 25]).asnumpy()
            x = torch.from_numpy(future.copy()).permute(0, 3, 1, 2).to("cuda", torch.float32) / 255.
            targets = models["official_encoder"](transform(x), is_training=True)["x_norm_patchtokens"].double()
            rows = []
            for step in range(3):
                a = original["visual"][3 + step].reshape(256, 384).double()
                b = corrected["visual"][3 + step].reshape(256, 384).double()
                target = targets[step]
                rows.append({"horizon_packets": step + 1,
                    "original_target_mse": (a-target).square().mean().item(),
                    "history_aware_target_mse": (b-target).square().mean().item(),
                    "prediction_rmse": (a-b).square().mean().sqrt().item(),
                    "prediction_max_abs": (a-b).abs().max().item()})
            prior_scores = previous["target_patch_mse"]
            if abs(rows[0]["original_target_mse"] - prior_scores["normalized_zero_history__actual_suffix"]) > 1e-6:
                raise ValueError("original first-step replay mismatch")
            if abs(rows[0]["history_aware_target_mse"] - prior_scores["actual_history__actual_suffix"]) > 1e-6:
                raise ValueError("history-aware first-step replay mismatch")
            torch.cuda.synchronize()
            after = {key: v4.model_receipt(model) for key, model in models.items()}
            if before != after:
                raise ValueError("model state changed")
            result = {"scope": "one exposed JEPA PushT episode; RGB plus position and velocity",
                "protocol_sha256": sha(PROTOCOL), "script_sha256": sha(Path(__file__)),
                "adapter_sha256": sha(ROOT / "experiments/spectral_reachability/pc_jepa_history_rollout.py"),
                "manifest_sha256": sha(MANIFEST), "prior_sha256": sha(PRIOR), "video_sha256": sha(video),
                "checkpoint_sha256": sha(ROOT / "data/jepa_wm_checkpoints/jepa_wm_pusht.pth.tar"),
                "gpu_uuid": uuid, "strict_load": strict, "imported_sources": sources,
                "predictor_calls": core.calls, "encoder_calls": 2, "proprio_encoder_calls": 1,
                "rows": rows, "target_access": "evaluation after predictions",
                "physical_execution": False, "training": False, "pc5_pass": False}
    with OUTPUT.open("x") as f:
        json.dump(result, f, indent=2, allow_nan=False)
        f.write("\n")
    print(json.dumps({"output": str(OUTPUT), "sha256": sha(OUTPUT), "rows": rows}))


if __name__ == "__main__":
    run()
