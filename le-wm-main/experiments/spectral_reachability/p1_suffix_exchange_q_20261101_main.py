"""Prospective complete prefix/tail exchange using only frozen-model scores."""
import argparse
import json
from pathlib import Path

import torch

from . import p1_horizon_q_20261101_main as parent

NAME = "p1_suffix_exchange_q_20261101_main"
ROOT = parent.parent.ROOT
DOC = ROOT.parent / "docs/D2P_P1_SUFFIX_EXCHANGE_PROTOCOL_20261101_MAIN.md"
BASE_COST = parent.cost
BASE_PLAN = parent.parent.kernels.plan


def cross_bank(bank):
    if bank.shape != (8, 3, 5, 2) or not bool(torch.isfinite(bank).all()) or not bool((bank.abs() <= 1).all()):
        raise ValueError("bounded eight-plan bank")
    prefixes = bank[:, None, :1].expand(8, 8, 1, 5, 2)
    tails = bank[None, :, 1:].expand(8, 8, 2, 5, 2)
    return torch.cat((prefixes, tails), dim=2).reshape(64, 3, 5, 2).clone()


def best_per_prefix(scores):
    if len(scores) != 64 or not all(parent.base.math.isfinite(x) for x in scores):
        raise ValueError("64 finite scores")
    # Preserve the existing own tail on exact ties.
    return [min(range(8), key=lambda j: (scores[8*i+j], j != i, j)) for i in range(8)]


def cost(arm):
    value = BASE_COST(arm)
    h = parent.horizon()
    if h > 1:
        for key in ("predict_calls", "predict_samples", "action_encode_calls"):
            value[key] += 64 * h
        for key in ("replay_calls", "selection_evaluations"):
            value[key] += 64
    return value


def plan(leaf, arm, task, config, policy, contract, *, device, emit):
    h = parent.horizon()

    def record(kind, payload):
        if h > 1 and kind in ("optimized_candidates", "final_selection"):
            emit("before_exchange_" + kind, payload)
        else:
            emit(kind, payload)

    result = BASE_PLAN(leaf, arm, task, config, policy, contract, device=device, emit=record)
    if h == 1:
        return result
    bank = cross_bank(result["optimized_raw"])
    tensor = lambda x: torch.as_tensor(x, device=device, dtype=torch.float32)
    scores = parent.parent.kernels.evaluate(
        leaf, tensor(task["root_context"]["latent_history"])[None],
        tensor(task["root_context"]["action_history"])[None], bank[:, :h],
        tensor(task["goal"])[None], tensor(task["action_mean"]), tensor(task["action_std"]))
    tails = best_per_prefix(scores)
    selected = torch.stack([bank[8*i+j] for i, j in enumerate(tails)])
    selected_scores = [scores[8*i+j] for i, j in enumerate(tails)]
    winner = min(range(8), key=lambda i: (selected_scores[i], i))
    parent.spec.require(all(selected_scores[i] <= scores[8*i+i] for i in range(8)),
                        "exchange includes original diagonal")
    emit("suffix_exchange", dict(horizon=h, scores=scores, tail_indices=tails,
                                layout="prefix_major_tail_minor", physical_labels_used=False))
    emit("optimized_candidates", dict(raw=selected.detach().cpu().tolist()))
    emit("final_selection", dict(index=winner, scores=selected_scores,
                                id=policy["candidate_ids"][winner]))
    result.update(optimized_raw=selected.detach(), selected_raw=selected[winner].detach(),
                  selected_index=winner, scores=selected_scores)
    return result


def setup(arm):
    parent.cost = cost
    parent.parent.kernels.plan = plan
    experiment = parent.setup(arm)
    runner = experiment.runner
    old_build, old_protocol, old_contract = runner.build, runner.protocol, parent.control.contract_for
    source_sha = runner.sha(Path(__file__).resolve())
    parent.base.CONFIG = dict(parent.base.CONFIG, suffix_exchange="full8x8_H_ge2_v1")

    def contract_for(task, model_sha, hardware_sha):
        policy, contract, config = old_contract(task, model_sha, hardware_sha)
        contract["identity"]["optimizer"] = source_sha
        for entry in contract["arms"].values():
            entry["optimizer_sha256"] = source_sha
        parent.spec.preflight(contract, parent.spec.digest(contract))
        return policy, contract, config

    def build():
        value = old_build()
        for path in (Path(__file__).resolve(), DOC):
            value["sources"][str(path)] = runner.sha(path)
        return value

    def protocol():
        value = old_protocol()
        value.update(suffix_exchange="all_64_before_true_outcome", exchange_boundaries=[0, 5],
                     all_arms_same_exchange=True, extra_predict_calls_full_trajectory=320,
                     no_extra_backward_calls=True, P1_admission=False,
                     note="same optimizer opportunity, not identical total cost across initializers")
        return value

    parent.control.contract_for = contract_for
    runner.build, runner.protocol = build, protocol
    runner.NAME = f"{NAME}_{arm}"
    return experiment


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=parent.parent.parent.ARMS, required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--authorize-host-cuda", action="store_true", required=True)
    args = parser.parse_args()
    if not args.tag.isalnum():
        raise ValueError("simple tag")
    experiment = setup(args.arm)
    prefix = ROOT / "results" / f"{NAME}_{args.arm}"
    prepared, output = Path(str(prefix) + f"_PREP_{args.tag}.json"), Path(str(prefix) + f"_RUN_{args.tag}")
    if prepared.exists() or output.exists():
        raise ValueError("exclusive output")
    receipt = experiment.runner.prepare(str(prepared))
    print(json.dumps(receipt), flush=True)
    with experiment.extra.gpu.locking.gpu_lock(experiment.extra.gpu.LOCK):
        result = experiment.runner.run(str(prepared), receipt["sha256"], str(output),
                                      "MAIN_AUTHORIZES_COMMON_SUFFIX_RGB_CUDA_AND_PHYSICS")
    print(json.dumps(result), flush=True)
    raise SystemExit(0 if result["status"] == "recorded_known_root_not_admitted" else 1)
