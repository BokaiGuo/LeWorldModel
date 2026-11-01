"""Fixed-root three-prefix, common-budget own-suffix planning primitives.

Preparation and pure wiring only: this module cannot authorize physical execution
or authenticate an RGB encoder response. No neural inference is run by the CLI.
"""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import numpy as np

from . import codex_p1_blind_uphill_bank_20261101_exclusive as blind
from . import codex_p1_bounded_joint_diag32_20261101_exclusive as diag
from . import codex_p1_bounded_joint_diag32_verify_20261101_exclusive as audit

NAME = "codex_p1_diag32_cp5_own_suffix_20261101_exclusive"
ROOT = blind.ROOT
PREVIOUS_PREP = ROOT / "results/codex_p1_bounded_joint_diag32_20261101_exclusive_PREP_CODEXDIAG01.json"
PREVIOUS_DIR = ROOT / "results/codex_p1_bounded_joint_diag32_20261101_exclusive_RUN_CODEXDIAG01"
PREVIOUS_SHA = "08a6ca7dad502f80f72758262557199656ff62ad9b3b1b20c1d614ee1216ff05"
UPDATES = 32
RATE = 0.05


def raw_bytes(value, shape):
    a = np.asarray(value, np.float32)
    if a.shape != shape or not np.isfinite(a).all() or (np.abs(a) > 1).any():
        raise ValueError("bounded raw action shape")
    return a


def prepare():
    host = blind.host
    host.require(host.sha(PREVIOUS_DIR / "report.json") == PREVIOUS_SHA, "fixed model report")
    audit.verify(PREVIOUS_PREP, PREVIOUS_DIR)
    task, bank, source_paths, policy = blind.load(0)
    report = json.loads((PREVIOUS_DIR / "report.json").read_bytes())
    rows = report["rows"][0]["evaluations"]
    initial = [r for r in rows if r["iteration"] == 0]
    host.require(len(initial) == 8 and [r["candidate"] for r in initial] == list(range(8)), "initial roster")
    host.require(all(np.array_equal(np.asarray(row["actions"], np.float32), bank[i])
                     for i, row in enumerate(initial)), "original bank identity")
    selected = report["rows"][0]["selected"]
    host.require(selected is not None and (selected["candidate"], selected["iteration"]) == (4, 13),
                 "selected fixed identity")
    scores = [r["q1"] for r in initial]
    host.require(np.isfinite(scores).all(), "finite greedy score")
    greedy = min(range(8), key=lambda i: (scores[i], i))
    selected_prefix = raw_bytes(selected["actions"], (3, 5, 2))[0]
    zero = np.zeros((5, 2), np.float32)
    prefixes = {"selected": selected_prefix, "zero": zero, "same_bank_greedy": bank[greedy, 0]}
    host.require(not np.array_equal(prefixes["selected"], prefixes["same_bank_greedy"]), "distinct arms")
    raw10 = np.asarray(json.loads((ROOT / "results" / blind.ITEMS[0][2] /
                                   (blind.ITEMS[0][3] + "_b00_seal.json")).read_bytes())["identity"]["raw10"],
                       np.float32)
    raw_bytes(raw10, (10, 2))
    host.require(host.rgb.moving.normalized(task, raw10) == task["root_context"]["action_history"],
                 "root normalized history")
    normalized = {role: host.rgb.moving.normalized(task, np.concatenate((raw10[5:], prefix)))
                  for role, prefix in prefixes.items()}
    bindings = dict(json.loads((ROOT / "results" / blind.ITEMS[0][0]).read_bytes())["sources"])
    for path in (Path(__file__).resolve(), Path(blind.__file__).resolve(), Path(diag.__file__).resolve(),
                 Path(audit.__file__).resolve(), PREVIOUS_PREP, PREVIOUS_DIR / "report.json",
                 PREVIOUS_DIR / "events.jsonl"):
        path = path.resolve()
        h = host.sha(path)
        host.require(str(path) not in bindings or bindings[str(path)] == h, "source collision")
        bindings[str(path)] = h
    bindings.update(source_paths)
    host.require(all(host.sha(path) == h for path, h in bindings.items()), "source drift")
    return {"schema": NAME + ".protocol.v1", "development_only": True,
            "source_report_sha256": PREVIOUS_SHA, "root_task_sha256": host.digest(task),
            "root_sha256": task["binding"]["root_sha256"], "goal_sha256": host.digest(task["goal"]),
            "model_sha256": task["binding"]["model_sha256"], "model_policy": policy,
            "roles": ["selected", "zero", "same_bank_greedy"], "greedy_candidate": greedy,
            "greedy_q1": scores, "greedy_rule": "minimum root model q1 on eight iteration-zero bank candidates; tie index",
            "prefixes": {k: v.tolist() for k, v in prefixes.items()}, "raw_warmup10": raw10.tolist(),
            "cp5_action_histories_expected": normalized,
            "initial_suffix_bank": bank[:, 1:].tolist(),
            "optimizer": {"candidates": 8, "updates": UPDATES, "rate": RATE,
                          "objective": "cp5 RGB latent predicted 2-packet terminal MSE to same goal",
                          "initialization": "identical original bank suffixes for every role",
                          "projection": "coordinate clamp [-1,1]", "selection": "minimum q2, tie candidate then iteration"},
            "budget_per_role": {"predict": 528, "action_encoder": 528, "reverse": 256,
                                "cp5_rgb_encode": 1, "evaluated_iterates": 264},
            "physical_contract": "unimplemented: real cp5 RGB, actual commands, native stop, durable accounting and replay required",
            "full_end_to_end_cost_equal": False, "physical_future_used_for_selection": False,
            "P1_admission": False, "bindings": bindings}


def feedback_task(base_task, expected_prefix, actual_prefix, previous_warmup, cp5_latent, identity):
    """Construct model-only cp5 input; caller MUST authenticate RGB and actual actions."""
    expected = raw_bytes(expected_prefix, (5, 2))
    actual = raw_bytes(actual_prefix, (5, 2))
    warm = raw_bytes(previous_warmup, (5, 2))
    z = np.asarray(cp5_latent, np.float32)
    if not np.array_equal(expected, actual) or z.shape != (192,) or not np.isfinite(z).all():
        raise ValueError("actual packet/latent mismatch")
    context = deepcopy(base_task["root_context"])
    context["latent_history"] = context["latent_history"][1:] + [z.tolist()]
    context["action_history"] = blind.host.rgb.moving.normalized(base_task, np.concatenate((warm, actual)))
    return blind.host.p3.local_task(base_task, context, identity)


def optimize_suffix_bank(bank, evaluate, reverse, record):
    """Fixed equal-work optimizer; evaluate(candidate, iterate, action) returns scalar q2."""
    import torch
    initial = torch.as_tensor(bank, dtype=torch.float32)
    if initial.shape != (8, 2, 5, 2) or not bool(torch.isfinite(initial).all()) or not bool((initial.abs() <= 1).all()):
        raise ValueError("invalid suffix bank")
    rows = []
    for candidate in range(8):
        action = initial[candidate].detach().clone()
        for iteration in range(UPDATES + 1):
            action = action.detach().requires_grad_(iteration < UPDATES)
            q2 = evaluate(candidate, iteration, action)
            if q2.ndim != 0 or not bool(torch.isfinite(q2)):
                raise ValueError("nonfinite model score")
            result = {"candidate": candidate, "iteration": iteration, "q2": float(q2.detach().item()),
                      "actions": action.detach().cpu().tolist()}
            record(result)
            rows.append(result)
            if iteration < UPDATES:
                gradient = reverse(candidate, iteration, q2, action)
                if gradient.shape != action.shape or not bool(torch.isfinite(gradient).all()):
                    raise ValueError("invalid gradient")
                action = (action - RATE * gradient).clamp(-1, 1).detach()
    return min(rows, key=lambda r: (r["q2"], r["candidate"], r["iteration"]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", required=True)
    args = parser.parse_args()
    blind.host.require(args.tag.isalnum(), "tag")
    output = ROOT / "results" / (NAME + "_PREP_" + args.tag + ".json")
    value = prepare()
    digest = blind.write_new(output, value)
    print(json.dumps({"contract": str(output), "sha256": digest,
                      "status": "prepared_only_no_physical_no_neural", "greedy": value["greedy_candidate"]}))


if __name__ == "__main__":
    main()
