"""Reduced JEPA runner parity: one changed unroll method, with full step traces."""
import argparse
import hashlib
import json
import os
import runpy
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT / "external/jepa-wms"
sys.path.insert(0, str(REPO))

from experiments.pc_jepa_unroll_upstream_tensor_probe_20261106_codex import upstream_unroll


def digest(tensor):
    array = tensor.detach().cpu().contiguous().numpy()
    return {"shape": list(array.shape), "dtype": str(array.dtype),
            "sha256": hashlib.sha256(array.tobytes()).hexdigest()}


def decision_parity(reference, split, elites):
    reference = reference.detach().reshape(-1).cpu()
    split = split.detach().reshape(-1).cpu()
    if reference.shape != split.shape or not torch.isfinite(reference).all() or not torch.isfinite(split).all():
        raise ValueError("invalid parity costs")
    order_ref = torch.argsort(reference, stable=True)
    order_split = torch.argsort(split, stable=True)
    elite_ref = torch.topk(-reference, elites).indices
    elite_split = torch.topk(-split, elites).indices
    ordered = reference[order_ref]
    gaps = ordered[1:] - ordered[:-1]
    return {
        "max_abs_cost_difference": (reference - split).abs().max().item(),
        "within_tolerance": bool(torch.all((reference - split).abs() <= 1e-4 + 1e-5 * reference.abs())),
        "full_stable_order_equal": torch.equal(order_ref, order_split),
        "elite_set_equal": set(elite_ref.tolist()) == set(elite_split.tolist()),
        "reference_order": order_ref.tolist(),
        "split_order": order_split.tolist(),
        "reference_elites": elite_ref.tolist(),
        "split_elites": elite_split.tolist(),
        "reference_min_gap": gaps.min().item() if len(gaps) else None,
        "reference_elite_boundary_gap": (ordered[elites] - ordered[elites - 1]).item() if elites < len(ordered) else None,
        "reference_costs": reference.tolist(),
        "split_costs": split.tolist(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=("local", "pinned"), required=True)
    parser.add_argument("--seed", type=int, choices=(1, 2), required=True)
    parser.add_argument("--run-id", default="r3")
    parser.add_argument("--execution-packets", type=int, choices=(3, 6), default=6)
    parser.add_argument("--dual-check", action="store_true")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--iterations", type=int, default=2)
    parser.add_argument("--elites", type=int, default=2)
    parser.add_argument("--chunk-size", type=int, default=0,
                        help="split candidate batch at unroll; 0 preserves original execution")
    parser.add_argument("--objective-parity", action="store_true")
    args = parser.parse_args()
    if min(args.episodes, args.samples, args.iterations, args.elites) < 1 or args.elites > args.samples or args.chunk_size < 0:
        raise ValueError("invalid episode or CEM budget")
    if not args.run_id.isalnum():
        raise ValueError("run id must be alphanumeric")
    if args.objective_parity and (args.arm != "local" or args.chunk_size != 4 or args.samples != 8 or args.elites != 2):
        raise ValueError("objective parity requires local arm, 8 samples, 4 chunk size and 2 elites")
    suffix = "" if args.execution_packets == 6 else "_step3"
    stem = f"pc_jepa_multi_unroll_{args.arm}_seed{args.seed}_{args.run_id}{suffix}_20261106_codex"
    config_path = ROOT / f"results/{stem}.yaml"
    trace_path = ROOT / f"results/{stem}.trace.jsonl"
    output_path = ROOT / f"results/{stem}.run"
    if any(p.exists() for p in (config_path, trace_path, output_path)):
        raise FileExistsError(stem)
    with (ROOT / "results/pc_jepa_official_runner_smoke_config_20261105_codex.yaml").open() as stream:
        config = yaml.safe_load(stream)
    config["folder"] = str(output_path)
    config["meta"]["seed"] = args.seed
    config["meta"]["eval_episodes"] = args.episodes
    config["planner"]["num_act_stepped"] = args.execution_packets
    config["planner"]["num_samples"] = args.samples
    config["planner"]["iterations"] = args.iterations
    config["planner"]["num_elites"] = args.elites
    with config_path.open("x") as stream:
        yaml.safe_dump(config, stream, sort_keys=False)
    trace_path.open("x").close()
    os.environ["JEPAWM_DSET"] = str(ROOT / "data/jepa_wm_osf/extracted")
    os.environ["JEPAWM_DINOV2_SOURCE"] = str(ROOT / "external/local_dinov2")
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
    from app.vjepa_wm.modelcustom.simu_env_planning import vit_enc_preds
    from evals.simu_env_planning.envs.pusht_gym_wrap import PushTWrapper

    def record(row):
        with trace_path.open("a") as stream:
            stream.write(json.dumps(row, allow_nan=False) + "\n")

    method = (vit_enc_preds.EncPredWM.unroll if args.arm == "local"
              else upstream_unroll(vit_enc_preds)[0])
    alternate = upstream_unroll(vit_enc_preds)[0] if args.dual_check else None
    if args.dual_check and args.arm != "local":
        raise ValueError("dual check uses local arm as the executed method")

    checked = False
    force_unchunked = False

    def select_context(context, start, end):
        if hasattr(context, "keys"):
            visual = context["visual"]
            if visual.shape[0] == 1:
                return context
            return context.apply(lambda value: value[start:end])
        return context if context.shape[0] == 1 else context[start:end]

    def chunked(self, context, actions, debug, chunk_size):
        parts = [method(self, select_context(context, start, min(start + chunk_size, actions.shape[1])),
                        actions[:, start:start + chunk_size], debug)
                 for start in range(0, actions.shape[1], chunk_size)]
        if hasattr(parts[0], "keys"):
            from tensordict import TensorDict
            return TensorDict({key: torch.cat([part[key] for part in parts], dim=1)
                               for key in parts[0].keys()}, batch_size=[])
        return torch.cat(parts, dim=1)

    def traced_unroll(self, context, act_suffix=None, debug=False):
        nonlocal checked
        if args.chunk_size and not checked:
            n = min(8, act_suffix.shape[1])
            probe_context = select_context(context, 0, n)
            probe_actions = act_suffix[:, :n]
            cpu_rng, cuda_rng = torch.random.get_rng_state(), torch.cuda.get_rng_state()
            full = method(self, probe_context, probe_actions, debug)
            torch.random.set_rng_state(cpu_rng)
            torch.cuda.set_rng_state(cuda_rng)
            split = chunked(self, probe_context, probe_actions, debug, max(1, n // 2))
            torch.random.set_rng_state(cpu_rng)
            torch.cuda.set_rng_state(cuda_rng)
            full_visual = full["visual"] if hasattr(full, "keys") else full
            split_visual = split["visual"] if hasattr(split, "keys") else split
            delta = (full_visual - split_visual).abs()
            record({"kind": "chunk_probe", "n": n, "max_abs": delta.max().item(),
                    "rms": delta.square().mean().sqrt().item(),
                    "exact": torch.equal(full_visual, split_visual)})
            checked = True
        result = (chunked(self, context, act_suffix, debug, args.chunk_size)
                  if args.chunk_size and not force_unchunked else method(self, context, act_suffix, debug))
        visual = context["visual"] if isinstance(context, dict) or hasattr(context, "keys") else context
        result_visual = result["visual"] if isinstance(result, dict) or hasattr(result, "keys") else result
        entry = {"kind": "unroll_reference" if force_unchunked else "unroll", "context": digest(visual), "actions": digest(act_suffix),
                 "prediction": digest(result_visual)}
        if alternate is not None:
            cpu_rng, cuda_rng = torch.random.get_rng_state(), torch.cuda.get_rng_state()
            try:
                other = alternate(self, context, act_suffix, debug)
            finally:
                torch.random.set_rng_state(cpu_rng)
                torch.cuda.set_rng_state(cuda_rng)
            other_visual = other["visual"] if isinstance(other, dict) or hasattr(other, "keys") else other
            delta = (result_visual - other_visual).abs()
            entry["dual_same_input"] = {
                "max_abs": delta.max().item(), "rms": delta.square().mean().sqrt().item(),
                "exact": torch.equal(result_visual, other_visual),
                "input_unchanged": entry["context"] == digest(visual) and entry["actions"] == digest(act_suffix),
            }
        record(entry)
        return result

    original_prepare, original_step = PushTWrapper.prepare, PushTWrapper.step

    def prepare(self, seed, init_state, env_info=None):
        result = original_prepare(self, seed, init_state, env_info)
        record({"kind": "prepare", "seed": int(seed),
                "init_state": np.asarray(init_state).tolist()})
        return result

    def step(self, action):
        result = original_step(self, action)
        record({"kind": "step", "action": np.asarray(action).tolist(),
                "state": np.asarray(result[0]).tolist()})
        return result

    vit_enc_preds.EncPredWM.unroll = traced_unroll
    if args.objective_parity:
        from evals.simu_env_planning.planning.planning.planner import Planner
        original_cost = Planner.cost_function

        def parity_cost(self, actions, z_init):
            nonlocal force_unchunked
            before_cpu, before_cuda = torch.random.get_rng_state(), torch.cuda.get_rng_state()
            split = original_cost(self, actions, z_init)
            after_cpu, after_cuda = torch.random.get_rng_state(), torch.cuda.get_rng_state()
            try:
                torch.random.set_rng_state(before_cpu)
                torch.cuda.set_rng_state(before_cuda)
                force_unchunked = True
                reference = original_cost(self, actions, z_init)
                row = decision_parity(reference, split, args.elites)
                row.update({"kind": "objective_parity", "actions": digest(actions)})
                record(row)
            finally:
                force_unchunked = False
                torch.random.set_rng_state(after_cpu)
                torch.cuda.set_rng_state(after_cuda)
            return split

        Planner.cost_function = parity_cost
    PushTWrapper.prepare, PushTWrapper.step = prepare, step
    print(f"JEPA parity: arm={args.arm} seed={args.seed} episodes={args.episodes}, "
          f"CEM {args.samples}x{args.iterations}, elites={args.elites}, "
          f"execute {args.execution_packets} packets per plan.", flush=True)
    sys.argv = ["evals.main", "--fname", str(config_path), "--debug"]
    runpy.run_module("evals.main", run_name="__main__")


if __name__ == "__main__":
    main()
