"""Development comparison of zero-padded and observed action histories."""
import hashlib
import json
import os
from pathlib import Path

import decord
import numpy as np
import torch
from tensordict import TensorDict
from torch.nn.attention import SDPBackend, sdpa_kernel

from experiments.pc_jepa_history_rollout_dev_20261104_codex import NativeCore
from experiments.spectral_reachability import p5_jepa_native_action_response_20261027_v8 as native
from experiments.spectral_reachability.e3_p2_cooperative_gpu_lock_codex_20261030_v1 import gpu_lock
from experiments.spectral_reachability.pc_jepa_history_rollout import unroll_history

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/pc_jepa_multiepisode_history_20261106_codex.json"
MANIFEST = ROOT / "results/p5_jepa_native_action_response_20261027_v8/manifest_frozen.json"
EPISODES = (2, 3, 4)


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    manifest = json.loads(MANIFEST.read_text())
    native.verify_manifest(ROOT, manifest)
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "0" or not torch.cuda.is_available():
        raise RuntimeError("CUDA device 0 required")
    with gpu_lock(ROOT / "results/e3_p2_host_cuda_cooperative.lock"):
        models, strict, sources, before = native.load_cuda(ROOT, manifest)
        v7 = native.helper(ROOT)
        _, v5, _ = v7.dependencies(ROOT)
        v4 = v5.import_v4(ROOT)
        arrays = v5.read_arrays(v4, ROOT)
        stats = v4.source_stats(v4.native_root(ROOT) / "app/plan_common/datasets/pusht_dset.py")
        from app.plan_common.datasets.transforms import make_transforms
        from app.vjepa_wm.modelcustom.simu_env_planning.vit_enc_preds import EncPredWM
        transform = make_transforms(img_size=224, random_horizontal_flip=False,
            random_resize_aspect_ratio=(1., 1.), random_resize_scale=(1., 1.),
            reprob=0., auto_augment=False, motion_shift=False)
        rows = []
        with torch.inference_mode(), sdpa_kernel(SDPBackend.MATH):
            for episode in EPISODES:
                values, rgb, _ = v5.row_inputs(v4, ROOT, arrays, stats, episode)
                actions = arrays[2][episode, :25]
                actions = (actions / 100. - np.asarray(stats["ACTION_MEAN"], np.float32)) / np.asarray(stats["ACTION_STD"], np.float32)
                packets = torch.from_numpy(actions.reshape(5, 10).copy()).to("cuda")
                pixels = torch.from_numpy(rgb.copy()).permute(0, 3, 1, 2).to("cuda", torch.float32) / 255.
                patches = models["official_encoder"](transform(pixels), is_training=True)["x_norm_patchtokens"]
                visual = patches[:3].reshape(1, 3, 1, 16, 16, 384)
                prop = torch.from_numpy(values["normalized_proprio"][:3].copy()).unsqueeze(0).to("cuda")
                features = models["proprio_encoder"](prop).expand(1, 3, 256, 16)
                core = NativeCore(models["predictor"])
                wrapper = EncPredWM.__new__(EncPredWM)
                torch.nn.Module.__init__(wrapper)
                wrapper.model, wrapper.ctxt_window, wrapper.proprio_mode = core, 3, "predict_proprio"
                suffix = packets[2:5].unsqueeze(1)
                context = TensorDict({"visual": visual, "proprio": features}, batch_size=[1, 3])
                zero = wrapper.unroll(context, suffix)
                history = unroll_history(core, visual, features, packets[:2].unsqueeze(0), suffix, 3)
                video = v4.data_root(ROOT) / f"obses/episode_{episode:03d}.mp4"
                reader = decord.VideoReader(str(video), num_threads=1, ctx=decord.cpu(0))
                future = reader.get_batch([15, 20, 25]).asnumpy()
                x = torch.from_numpy(future.copy()).permute(0, 3, 1, 2).to("cuda", torch.float32) / 255.
                targets = models["official_encoder"](transform(x), is_training=True)["x_norm_patchtokens"].double()
                for step in range(3):
                    a = zero["visual"][3 + step].reshape(256, 384).double()
                    b = history["visual"][3 + step].reshape(256, 384).double()
                    rows.append({"episode": episode, "horizon_packets": step + 1,
                                 "zero_mse": (a-targets[step]).square().mean().item(),
                                 "history_mse": (b-targets[step]).square().mean().item(),
                                 "prediction_rmse": (a-b).square().mean().sqrt().item()})
        torch.cuda.synchronize()
        if before != {key: v4.model_receipt(model) for key, model in models.items()}:
            raise ValueError("model state changed")
        result = {"scope": "development data episodes 2,3,4; no independent confirmation",
                  "episodes": EPISODES, "strict_load": strict, "sources": sources,
                  "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  "future_use": "targets read only after both predictions", "rows": rows}
    with OUT.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
