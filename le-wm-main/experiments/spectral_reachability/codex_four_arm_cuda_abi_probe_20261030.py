"""Known-root CUDA ABI probe, not a four-arm performance experiment."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

from experiments.checkpoint_compat import load_compatible_pretrained
from experiments.gpu_preflight import require_cuda_allocation
from experiments.spectral_reachability import agent_p1_joint_vreach_host_cuda_20261029_v1 as host
from experiments.spectral_reachability import codex_four_arm_durable_20261030 as durable
from experiments.spectral_reachability import codex_four_arm_full_cost_20261030 as spec
from experiments.spectral_reachability import codex_four_arm_leaf_suffix_20261030 as suffix
from experiments.spectral_reachability import codex_four_arm_consumer_20261030 as consumer

NAME = "codex_four_arm_cuda_abi_probe_20261030"
ROOT = Path(__file__).resolve().parents[2]
PREP = ROOT / "results/agent_p1_joint_vreach_host_cuda_20261029_v1_prepare_MAIN01.json"
PREP_SHA = "97a312976831ca4104cae2f69283679e9b4a432a7339011bb122eeb8f46f82dd"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(output):
    path = Path(output)
    spec.require(path.is_absolute() and path.parent == ROOT / "results" and
                 path.name.startswith(NAME + "_") and not path.exists() and not path.is_symlink(),
                 "exclusive output")
    prepared = host.verify(str(PREP), PREP_SHA)
    task = prepared["requests"][0]["task"]
    config = dict(steps=2, learning_rate=0.05, history_size=3, lower=-1.0, upper=1.0)
    source_hash = sha(__file__)
    identities = {key: spec.digest({"diagnostic": NAME, "field": key}) for key in spec.IDENTITY}
    identities.update(model=prepared["expected_model_hash"], task=spec.digest(task),
                      history=spec.digest(task["root_context"]), goal=spec.digest(task["goal"]),
                      optimizer=sha(suffix.__file__), optimizer_config=spec.digest(config),
                      meter=sha(durable.__file__))
    arms = {a: dict(initializer_sha256=spec.digest({"unused_arm": a}),
                    optimizer_sha256=identities["optimizer"],
                    optimizer_config_sha256=identities["optimizer_config"]) for a in spec.ARMS}
    caps = dict.fromkeys(spec.AXES, 100)
    contract = dict(schema=spec.NAME + ".contract.v1", identity=identities, arms=arms,
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
        model = load_compatible_pretrained(prepared["policy"], device=torch.device("cuda:0")).eval().requires_grad_(False)
        before = host.signal.model_hash(model)
        spec.require(before["original_state_dict"] == prepared["expected_model_hash"], "checkpoint identity")
        context = task["root_context"]
        z = torch.tensor(context["latent_history"], dtype=torch.float32, device="cuda")[None]
        h = torch.tensor(context["action_history"], dtype=torch.float32, device="cuda")[None]
        goal = torch.tensor(task["goal"], dtype=torch.float32, device="cuda")[None]
        initial = torch.zeros((1, 2, 10), dtype=torch.float32, device="cuda")
        path.mkdir()
        ledger = durable.Driver(path / "ledger", contract, contract_sha, "structured", "probe", "abi_probe")
        try:
            leaf = suffix.LeafAdapter(model, ledger, contract, contract_sha)
            observed = suffix.optimize(leaf, z, h, initial, goal, config)
        finally:
            receipt = ledger.finish()
        accounted = consumer.consume(receipt["journal"], receipt["sha256"], contract_sha,
                                     "structured", "probe", "abi_probe")
        # An unmetered reference repeats the same computation solely for ABI
        # validation; these calls are not a budgeted planning comparison.
        ref = initial.detach().clone()
        for _ in range(config["steps"]):
            ref = ref.detach().requires_grad_(True)
            states, past = z.detach(), h.detach()
            for i in range(ref.shape[1]):
                actions = torch.cat((past, ref[:, i:i + 1]), dim=1)
                predicted = model.predict(states, model.action_encoder(actions))
                states = torch.cat((states, predicted[:, -1:]), dim=1)[:, -3:]
                past = actions[:, -2:]
            loss = (states[:, -1] - goal).square().mean()
            gradient = torch.autograd.grad(loss, ref)[0]
            ref = (ref - config["learning_rate"] * gradient).clamp(-1, 1).detach()
        torch.cuda.synchronize()
        diff = float((observed - ref).abs().max().item())
        after = host.signal.model_hash(model)
        spec.require(before == after and np.isfinite(diff) and diff <= 1e-6, "ABI mismatch or model drift")
        result = dict(schema=NAME + ".v1", status="cuda_abi_probe_passed",
                      source_prepare_sha256=PREP_SHA, source_code_sha256=source_hash,
                      optimizer_sha256=identities["optimizer"], contract_sha256=contract_sha,
                      ledger_sha256=receipt["sha256"], ledger=accounted, device=device_info,
                      model_before=before, model_after=after, max_abs_action_difference=diff,
                      wall_seconds=time.monotonic()-start, neural_device="cuda:0",
                      true_physical_actions=False, independent_confirmation=False,
                      fairness_proven=False, P1_admission=False,
                      limitations="known-root ABI probe; raw physical action normalization and unmetered reference are not a four-arm comparison")
        raw = json.dumps(result, sort_keys=True, allow_nan=False, default=str).encode() + b"\n"
        with (path / "report.json").open("xb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        print(json.dumps({"status": result["status"], "report_sha256": sha(path / "report.json"),
                          "max_abs_action_difference": diff}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    run(parser.parse_args().output)
