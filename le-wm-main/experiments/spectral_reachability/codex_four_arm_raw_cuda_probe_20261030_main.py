"""Compare real CUDA raw-action optimizer against a direct reference.

Known root and synthetic zero initialization; not physical or performance data.
"""
import argparse
import json
import os
from pathlib import Path
import time

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

from experiments.checkpoint_compat import load_compatible_pretrained
from experiments.gpu_preflight import require_cuda_allocation
from experiments.spectral_reachability import agent_p1_joint_vreach_host_cuda_20261029_v1 as host
from experiments.spectral_reachability import codex_four_arm_consumer_20261030 as consumer
from experiments.spectral_reachability import codex_four_arm_cuda_abi_probe_20261030 as previous
from experiments.spectral_reachability import codex_four_arm_durable_20261030 as durable
from experiments.spectral_reachability import codex_four_arm_full_cost_20261030 as spec
from experiments.spectral_reachability import codex_four_arm_leaf_suffix_20261030 as leaf
from experiments.spectral_reachability import codex_four_arm_raw_suffix_20261030_main as method

NAME = "codex_four_arm_raw_cuda_probe_20261030_main"


def run(output):
    output = Path(output)
    spec.require(output.is_absolute() and output.parent == previous.ROOT / "results"
                 and output.name.startswith(NAME + "_") and not output.exists()
                 and not output.is_symlink(), "exclusive output")
    prep = host.verify(str(previous.PREP), previous.PREP_SHA)
    task = prep["requests"][0]["task"]
    config = {"steps": 2, "learning_rate": .05, "history_size": 3}
    ids = {k: spec.digest({"probe": NAME, "field": k}) for k in spec.IDENTITY}
    ids.update(model=prep["expected_model_hash"], task=spec.digest(task),
               history=spec.digest(task["root_context"]), goal=spec.digest(task["goal"]),
               optimizer=previous.sha(method.__file__), optimizer_config=spec.digest(config),
               meter=previous.sha(durable.__file__))
    arms = {a: dict(initializer_sha256=spec.digest({"unused_arm": a}),
                    optimizer_sha256=ids["optimizer"], optimizer_config_sha256=ids["optimizer_config"])
            for a in spec.ARMS}
    caps = dict.fromkeys(spec.AXES, 100)
    contract = dict(schema=spec.NAME + ".contract.v1", identity=ids, arms=arms,
                    scope="standalone_per_root_per_arm_including_attributable_setup",
                    phase_policy_sha256=spec.digest({"diagnostic_only": True}),
                    levels=[dict(id="probe", caps=caps),
                            dict(id="unused", caps={k: v + 1 for k, v in caps.items()})])
    contract_sha = spec.digest(contract)
    device_info = require_cuda_allocation("cuda:0")
    os.environ["HF_HUB_OFFLINE"] = os.environ["TRANSFORMERS_OFFLINE"] = "1"
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    start = time.monotonic()
    with torch.autocast(device_type="cuda", enabled=False), sdpa_kernel(SDPBackend.MATH):
        model = load_compatible_pretrained(prep["policy"], device=torch.device("cuda:0")).eval().requires_grad_(False)
        before = host.signal.model_hash(model)
        spec.require(before["original_state_dict"] == prep["expected_model_hash"], "checkpoint identity")
        tensor = lambda value: torch.tensor(value, dtype=torch.float32, device="cuda")
        states = tensor(task["root_context"]["latent_history"])[None]
        history = tensor(task["root_context"]["action_history"])[None]
        goal = tensor(task["goal"])[None]
        mean, std = tensor(task["action_mean"]), tensor(task["action_std"])
        initial = torch.zeros((1, 2, 5, 2), dtype=torch.float32, device="cuda")
        output.mkdir()
        ledger = durable.Driver(output / "ledger", contract, contract_sha, "structured", "probe", "raw_abi_probe")
        try:
            adapter = leaf.LeafAdapter(model, ledger, contract, contract_sha)
            actual = method.optimize(adapter, states, history, initial, goal, mean, std, config)
        finally:
            receipt = ledger.finish()
        account = consumer.consume(receipt["journal"], receipt["sha256"], contract_sha,
                                   "structured", "probe", "raw_abi_probe")
        ref = initial.clone()
        for _ in range(config["steps"]):
            ref = ref.detach().requires_grad_(True)
            z, past = states.detach(), history.detach()
            for i in range(2):
                token = ((ref[:, i] - mean) / std).reshape(1, 1, 10)
                context = torch.cat((past, token), dim=1)
                z_next = model.predict(z, model.action_encoder(context))
                z = torch.cat((z, z_next[:, -1:]), dim=1)[:, -3:]
                past = context[:, -2:]
            gradient = torch.autograd.grad((z[:, -1] - goal).square().mean(), ref)[0]
            ref = (ref - config["learning_rate"] * gradient).clamp(-1, 1).detach()
        torch.cuda.synchronize()
        difference = float((actual - ref).abs().max().item())
        after = host.signal.model_hash(model)
        spec.require(before == after and difference <= 1e-6 and bool((actual.abs() <= 1).all().item()),
                     "reference mismatch, raw bounds or model drift")
        report = dict(schema=NAME + ".v1", status="raw_cuda_probe_passed", source_prepare_sha256=previous.PREP_SHA,
                      source_code_sha256=previous.sha(__file__), optimizer_sha256=ids["optimizer"],
                      normalizer_mean=task["action_mean"], normalizer_std=task["action_std"],
                      raw_output=actual.cpu().tolist(), max_abs_action_difference=difference,
                      model_before=before, model_after=after, device=device_info, ledger=account,
                      wall_seconds=time.monotonic()-start, neural_device="cuda:0",
                      physical_executed=False, fairness_proven=False, P1_admission=False,
                      limitation="known root, unmetered reference, no prefix production or physical execution")
        raw = json.dumps(report, sort_keys=True, allow_nan=False, default=str).encode() + b"\n"
        with (output / "report.json").open("xb") as f:
            f.write(raw)
            f.flush()
            os.fsync(f.fileno())
        print(json.dumps({"status": report["status"], "report_sha256": previous.sha(output / "report.json"),
                          "max_abs_action_difference": difference}))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--output", required=True)
    run(p.parse_args().output)
