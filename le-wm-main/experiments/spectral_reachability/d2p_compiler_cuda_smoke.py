"""Development-only frozen PushT compiler smoke; never scientific admission."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from .d2p_local_compiler import CompilerConfig, LocalCompiler
from .d2p_pusht_cuda_backend import PushTCudaBackend
from .e3_p2_cooperative_gpu_lock_codex_20261030_v1 import gpu_lock


def run(inputs, policy, output):
    import torch
    from experiments.checkpoint_compat import load_compatible_pretrained
    from experiments.run_pusht_counterfactual import image_transform
    from . import preflight_pusht_d2p_p0 as p0
    from .run_pusht_d2p_budget_guard_cuda import tensor_state_hash
    from .pusht_composition_cuda_replay_v2 import CudaReplay

    if output.exists():
        raise FileExistsError(output)
    root = Path(__file__).resolve().parents[2]
    with gpu_lock(root / "results/e3_p2_host_cuda_cooperative.lock"):
        if not torch.cuda.is_available():
            raise RuntimeError("host CUDA unavailable; no neural CPU fallback")
        device = torch.device("cuda:0")
        model = load_compatible_pretrained(str(policy), device=device).eval().requires_grad_(False)
        before = tensor_state_hash(model)
        with np.load(inputs, allow_pickle=False) as data:
            frames = [x.copy() for x in data["rgb_history"]]
            goal_rgb = data["goal_rgb"].copy()
            history = data["raw_action_history"].copy()
        transform = image_transform()
        with torch.no_grad():
            emb = model.encode({"pixels": p0._frames_tensor(frames, transform=transform, device=device)})["emb"][0]
            goal = p0._goal_latent(model, goal_rgb, transform=transform, device=device)
            tokens = p0.packets.normalize_live_packet(
                torch.tensor(history.reshape(2, 5, 2), dtype=torch.float32, device=device),
                mean=p0.PUBLIC_ACTION_MEAN, std=p0.PUBLIC_ACTION_STD)
        context = dict(layout="states_with_between_state_actions_v1", context_length=3,
                       latent_dim=192, action_dim=10, latent_history=emb.cpu().tolist(),
                       action_history=tokens.cpu().tolist())
        backend = PushTCudaBackend(model, device)
        config = CompilerConfig(packet_steps=5, suffix_packets=1, spectral_modes=1,
                                line_scales=(0., .5, 1.))
        compiler = LocalCompiler(
            backend, goal=goal.cpu().numpy(), sqrt_q=np.eye(192),
            action_scale=np.ones(2), low=-np.ones(2), high=np.ones(2), config=config)
        report = compiler.compile(context)
        replay = CudaReplay(model, device)
        exact = all(replay(context, p["actions"]) == p["output_context"] for p in report["programs"])
        after = tensor_state_hash(model)
        report.update(input_sha256=hashlib.sha256(inputs.read_bytes()).hexdigest(),
                      model_before=before, model_after=after,
                      independent_replay_exact=exact, model_unchanged=before == after,
                      scope="known development input; compiler engineering smoke only",
                      physical_steps=0, independent_confirmation=False)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("x") as stream:
            json.dump(report, stream, sort_keys=True, allow_nan=False)
        if not exact or before != after:
            raise RuntimeError("compiler replay/model integrity failed")
        print(json.dumps({k: report[k] for k in ("selected_index", "operation_counts",
                                                "independent_replay_exact", "model_unchanged")}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("inputs", "policy", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    run(**vars(parser.parse_args()))
