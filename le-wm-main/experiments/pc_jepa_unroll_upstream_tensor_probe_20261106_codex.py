"""Fixed-input comparison of local and pinned-git JEPA unroll implementations."""
import ast
import hashlib
import json
import os
import subprocess
from pathlib import Path

import numpy as np
import torch

from experiments.pc_jepa_history_rollout_dev_20261104_codex import NativeCore
from experiments.spectral_reachability import p5_jepa_native_action_response_20261027_v8 as native
from experiments.spectral_reachability.e3_p2_cooperative_gpu_lock_codex_20261030_v1 import gpu_lock


ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT / "external/jepa-wms"
SOURCE = "app/vjepa_wm/modelcustom/simu_env_planning/vit_enc_preds.py"
COMMIT = "13cf1d9c7e476f53c17714d2e0f1dc239a883ce0"
OUTPUT = ROOT / "results/pc_jepa_unroll_upstream_tensor_probe_20261106_codex.json"


class RecordingCore(NativeCore):
    def __init__(self, predictor):
        super().__init__(predictor)
        self.inputs = []

    def forward_pred(self, video, actions, proprio, debug=False):
        self.inputs.append({"visual_frames": video.shape[1], "action_frames": actions.shape[1],
                            "action_values": actions.detach().cpu().tolist()})
        return super().forward_pred(video, actions, proprio, debug)


def upstream_unroll(module):
    source = subprocess.check_output(["git", "show", f"{COMMIT}:{SOURCE}"], cwd=REPO)
    tree = ast.parse(source)
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "EncPredWM")
    method = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "unroll")
    compiled = compile(ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[])),
                       f"{COMMIT}:{SOURCE}", "exec")
    scope = dict(vars(module))
    exec(compiled, scope)
    return scope["unroll"], hashlib.sha256(source).hexdigest()


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "0" or not torch.cuda.is_available():
        raise RuntimeError("CUDA device 0 required")
    manifest_path = ROOT / "results/p5_jepa_native_action_response_20261027_v8/manifest_frozen.json"
    manifest = json.loads(manifest_path.read_text())
    native.verify_manifest(ROOT, manifest)
    with gpu_lock(ROOT / "results/e3_p2_host_cuda_cooperative.lock"):
        models, strict, sources, before = native.load_cuda(ROOT, manifest)
        v7 = native.helper(ROOT)
        _, v5, _ = v7.dependencies(ROOT)
        v4 = v5.import_v4(ROOT)
        arrays = v5.read_arrays(v4, ROOT)
        stats = v4.source_stats(v4.native_root(ROOT) / "app/plan_common/datasets/pusht_dset.py")
        values, rgb, _ = v5.row_inputs(v4, ROOT, arrays, stats, native.EPISODE)
        if v4.array_sha(rgb) != manifest["row"]["rgb_sha256"]:
            raise ValueError("context fingerprint mismatch")
        packets = (arrays[2][native.EPISODE, :25] / 100. -
                   np.asarray(stats["ACTION_MEAN"], np.float32)) / np.asarray(stats["ACTION_STD"], np.float32)
        packets = torch.from_numpy(packets.reshape(5, 10).copy()).to("cuda")
        from app.plan_common.datasets.transforms import make_transforms
        from app.vjepa_wm.modelcustom.simu_env_planning import vit_enc_preds
        from tensordict import TensorDict
        from torch.nn.attention import SDPBackend, sdpa_kernel

        upstream, source_sha = upstream_unroll(vit_enc_preds)
        transform = make_transforms(img_size=224, random_horizontal_flip=False,
                                    random_resize_aspect_ratio=(1., 1.),
                                    random_resize_scale=(1., 1.), reprob=0.,
                                    auto_augment=False, motion_shift=False)
        records = []
        with torch.inference_mode(), sdpa_kernel(SDPBackend.MATH):
            pixels = torch.from_numpy(rgb.copy()).permute(0, 3, 1, 2).to("cuda", torch.float32) / 255.
            patches = models["official_encoder"](transform(pixels), is_training=True)["x_norm_patchtokens"]
            visual = patches[:3].reshape(1, 3, 1, 16, 16, 384)
            prop = torch.from_numpy(values["normalized_proprio"][:3].copy()).unsqueeze(0).to("cuda")
            prop = models["proprio_encoder"](prop).expand(1, 3, 256, 16)
            for context_len in (1, 2, 3):
                context = TensorDict({"visual": visual[:, -context_len:],
                                     "proprio": prop[:, -context_len:]}, batch_size=[1, context_len])
                suffix = packets[2:5].unsqueeze(1)
                results = {}
                tensors = {}
                for name, method in (("local", vit_enc_preds.EncPredWM.unroll), ("pinned", upstream)):
                    core = RecordingCore(models["predictor"])
                    wrapper = vit_enc_preds.EncPredWM.__new__(vit_enc_preds.EncPredWM)
                    torch.nn.Module.__init__(wrapper)
                    wrapper.model, wrapper.ctxt_window = core, context_len
                    wrapper.proprio_mode = "predict_proprio"
                    try:
                        predicted = method(wrapper, context, suffix)
                        tensors[name] = predicted["visual"].detach()
                        results[name] = {"ok": True, "calls": core.inputs}
                    except Exception as exc:
                        results[name] = {"ok": False, "error_type": type(exc).__name__,
                                         "error": str(exc), "calls": core.inputs}
                if len(tensors) == 2:
                    diff = (tensors["local"] - tensors["pinned"]).abs()
                    results["max_abs_difference"] = diff.max().item()
                    results["rms_difference"] = diff.square().mean().sqrt().item()
                records.append({"context_frames": context_len, "suffix_packets": 3, "results": results})
            torch.cuda.synchronize()
            if before != {key: v4.model_receipt(model) for key, model in models.items()}:
                raise ValueError("model mutated")
        with OUTPUT.open("x", encoding="utf-8") as stream:
            json.dump({"fixed_commit": COMMIT, "fixed_source_sha256": source_sha,
                       "scope": "one previously exposed PushT context, fixed suffix, local DINO",
                       "strict_load": strict, "sources": sources, "records": records},
                      stream, indent=2, allow_nan=False)


if __name__ == "__main__":
    main()
