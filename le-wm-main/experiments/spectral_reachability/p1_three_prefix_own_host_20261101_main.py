"""Selected/zero/library-greedy prefixes followed by one common RGB planner."""
import argparse
import json
from pathlib import Path
from unittest.mock import patch

import torch

from . import e3_cp10_dls_feedback_mix_20261031 as mix
from . import p1_four_arm_extra32_budget_20261031 as control
from . import p1_rgb_carry_replan_codex_20261101 as selector
from . import p1_carry_binding_20261101 as binding

NAME = "p1_three_prefix_own_host_20261101_main"
DOC = Path(__file__).resolve().parents[3] / "docs/D2P_P1_THREE_PREFIX_OWN_PROTOCOL_20261101_MAIN.md"


def choose_prefix(result, scores, role):
    if role not in ("selected", "zero", "greedy") or len(scores) != 8:
        raise ValueError("prefix policy")
    bank = result["optimized_raw"]
    if role == "selected":
        return result["selected_raw"].detach().clone()
    if role == "zero":
        return torch.zeros_like(result["selected_raw"])
    index = min(range(8), key=lambda i: (scores[i], i))
    return bank[index].detach().clone()


def setup(root, role):
    experiment = mix.setup(root)
    runner = experiment.runner
    old_build, old_protocol = runner.build, runner.protocol
    old_node = runner.bridge.plan_node
    old_cost, old_plan = control.cost, control.plan
    old_roster = runner.rgb.roster
    runner.rgb.roster = lambda: [row for row in old_roster() if row["arm"] == "structured"]

    def node(task, arm, boundary, store, runtime):
        extra = binding.extra_cost(8, 1, has_carry=False) if boundary == 0 else {}

        def cost(which):
            value = old_cost(which)
            for key, amount in extra.items():
                value[key] += amount
            return value

        def plan(leaf, which, local_task, config, policy, contract, *, device, emit):
            result = old_plan(leaf, which, local_task, config, policy, contract,
                              device=device, emit=emit)
            if boundary == 0:
                tensor = lambda x: torch.as_tensor(x, dtype=torch.float32, device=device)
                scored = selector.select(
                    leaf, states=tensor(local_task["root_context"]["latent_history"])[None],
                    history=tensor(local_task["root_context"]["action_history"])[None],
                    goal=tensor(local_task["goal"])[None], mean=tensor(local_task["action_mean"]),
                    std=tensor(local_task["action_std"]),
                    new_candidates=result["optimized_raw"][:, :1])
                selected = choose_prefix(result, scored["scores"], role)
                emit("prefix_intervention", {"role": role, "stage_scores": scored["scores"],
                     "bank": result["optimized_raw"].detach().cpu().tolist(),
                     "original_selected_index": result["selected_index"],
                     "full_selected": selected.cpu().tolist(), "incremental_cost": extra})
                result["selected_raw"] = selected
                result["prefix_intervention"] = True
            emit("selected_full_plan", {"boundary": boundary,
                                       "raw": result["selected_raw"].detach().cpu().tolist()})
            return result

        original_first = control.base.first_packet

        def first(result):
            if not result.get("prefix_intervention"):
                return original_first(result)
            value = result["selected_raw"]
            runner.require(value.shape == (3, 5, 2) and bool(torch.isfinite(value).all())
                           and bool((value.abs() <= 1).all()), "bounded prefix")
            return value[0].detach().clone()

        with patch.object(control, "cost", cost), patch.object(control, "plan", plan), \
                patch.object(control.base, "first_packet", first):
            return old_node(task, arm, boundary, store, runtime)

    def protocol():
        value = old_protocol()
        value.update(arms=["structured"], prefix_role=role,
                     prefix_rule="same_optimized_H3_bank_stage1_greedy_or_zero_or_original",
                     boundary0_extra=binding.extra_cost(8, 1, has_carry=False),
                     common_suffix="same_structured_DLS_mix_real_RGB_at_5_10_15",
                     P1_admission=False, independent_confirmation=False)
        return value

    def build():
        value = old_build()
        for path in (Path(__file__).resolve(), DOC, Path(selector.__file__).resolve(),
                     Path(binding.__file__).resolve()):
            value["sources"][str(path)] = runner.sha(path)
        return value

    runner.NAME = f"{NAME}_root{root}_{role}"
    runner.build, runner.protocol, runner.bridge.plan_node = build, protocol, node
    return experiment


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=int, choices=(0, 1), required=True)
    parser.add_argument("--role", choices=("selected", "zero", "greedy"), required=True)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--output", required=True)
    p = sub.add_parser("run")
    for key in ("prepared", "sha256", "output", "authorization"):
        p.add_argument("--" + key, required=True)
    args = parser.parse_args()
    experiment = setup(args.root, args.role)
    if args.command == "prepare":
        result = experiment.runner.prepare(args.output)
    else:
        with experiment.extra.gpu.locking.gpu_lock(experiment.extra.gpu.LOCK):
            result = experiment.runner.run(args.prepared, args.sha256, args.output, args.authorization)
    print(json.dumps(result, sort_keys=True), flush=True)
    if args.command == "run" and result["status"] != "recorded_known_root_not_admitted":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
