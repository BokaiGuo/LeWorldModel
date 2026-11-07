"""Sealed model-selected prefix interventions for actual RGB-dispatched chains."""
import json
import sys

import h5py
import numpy as np
import torch

from experiments import pc_shared_rgb_feedback_ledger_20261107_codex as ledger
from experiments.pc_input_permissions import RGB, RGBLeWM, file_hash, freeze_audit
from experiments.pc_prefix_novel_actual_encoding_20261105_codex import model_fingerprint

ROOT = ledger.prior.ROOT
PROTOCOL = ROOT.parent / "docs/PC_SHARED_RGB_EXECUTED_CHAIN_INTERVENTION_PROTOCOL_20261107_CODEX.md"
BASE = ROOT / "results/pc_shared_rgb_executed_chain_intervention_20261107_codex"
PLAN = BASE.with_name(BASE.name + "_plan.json")
PHYSICAL = BASE.with_name(BASE.name + "_physical.json")
ARMS = ("original", "zero", "random", "greedy", "lookahead")


def candidate_prefixes(original, seed):
    original = np.asarray(original, dtype=np.float32)
    if original.shape != (5, 2) or not np.isfinite(original).all() or np.abs(original).max() > 1:
        raise ValueError("invalid original prefix")
    rng = np.random.default_rng(seed)
    random = np.repeat(rng.uniform(-1, 1, (30, 1, 2)).astype(np.float32), 5, axis=1)
    return np.concatenate([original[None], np.zeros((1, 5, 2), dtype=np.float32), random])


def build_plan():
    import gymnasium as gym
    import stable_worldmodel  # noqa: F401
    if PLAN.exists():
        raise FileExistsError(PLAN)
    prior = json.loads(ledger.OUT.read_text())
    for key, path in (("runner_sha256", ledger.__file__), ("protocol_sha256", ledger.PROTOCOL),
                      ("shared_sha256", ledger.prior.shared.__file__), ("adapter_sha256", ledger.prior.adapter.__file__)):
        if prior[key] != file_hash(path):
            raise ValueError("ledger binding drift")
    model, scaler, transform, normalizer = ledger.prior.source.block.prior.common.model_setup()
    modelhash = model_fingerprint(model)
    original_plan = json.loads(ledger.prior.source.PLAN.read_text())
    if modelhash != prior["model_sha256"] or normalizer != original_plan["normalizer_sha256"]:
        raise ValueError("model or normalizer mismatch")
    deps = [__file__, PROTOCOL, ledger.OUT, ledger.prior.source.block.prior.inputs.INPUT,
            ledger.prior.source.block.prior.common.CHECKPOINT / "weights.pt"]
    audit = freeze_audit(RGB, "lewm", deps, preprocessing="ImageNet 224, TwoRoom action scaler",
                         clock="five packets of five environment steps")
    env = gym.make("swm/TwoRoom-v1", render_mode="rgb_array")
    rows = []
    try:
        with (np.load(ledger.prior.source.block.prior.inputs.INPUT, allow_pickle=False) as inputs,
              h5py.File(ledger.prior.source.block.prior.common.DATA, "r") as data, torch.inference_mode()):
            for row in prior["rows"]:
                root, task = row["root"], row["row"]
                actions = np.asarray([x["action"] for x in row["trace"]], dtype=np.float32)
                if actions.ndim != 2 or not 5 <= len(actions) <= 25 or actions.shape[1] != 2:
                    raise ValueError("invalid dispatched trajectory")
                prefix, tail = actions[:5], actions[5:]
                candidates = candidate_prefixes(prefix, 27123000 + root)
                env.reset(seed=42)
                actor = env.unwrapped
                actor._set_state(data["proprio"][task])
                actor._set_goal_state(data["proprio"][task + 25])
                ctx = {"pixels": transform(actor.render().copy())[None, None, None].cuda(),
                       "goal": transform(inputs["goal"][root])[None, None, None].cuda(),
                       "action": torch.empty((1, 1, 0, 10), device="cuda")}
                wrapper = RGBLeWM(model, audit)
                values = []
                for continuation in (np.zeros((20, 2), dtype=np.float32),
                                     np.pad(tail, ((0, 20 - len(tail)), (0, 0)))):
                    chains = np.concatenate([candidates, np.broadcast_to(continuation, (32, 20, 2))], axis=1)
                    normalized = scaler.transform(chains.reshape(-1, 2)).reshape(1, 32, 5, 10)
                    score, _ = wrapper.score(ctx, torch.as_tensor(normalized, dtype=torch.float32, device="cuda"))
                    values.append(score[0].cpu().numpy())
                if not all(x.shape == (32,) and np.isfinite(x).all() for x in values):
                    raise ValueError("invalid model score")
                choices = {"original": 0, "zero": 1, "random": 2,
                           "greedy": int(np.argmin(values[0])), "lookahead": int(np.argmin(values[1]))}
                rows.append({"root": root, "row": task, "original_length": len(actions),
                             "prefix_candidates": candidates.tolist(), "common_tail": tail.tolist(),
                             "greedy_scores": values[0].tolist(), "lookahead_scores": values[1].tolist(),
                             "choices": choices, "queries": 64, "receipts": wrapper.receipts})
    finally:
        env.close()
    if model_fingerprint(model) != modelhash:
        raise ValueError("model weights changed")
    payload = {"track": RGB, "scope": "exposed execution-chain intervention; not deployable lookahead",
               "protocol_sha256": file_hash(PROTOCOL), "runner_sha256": file_hash(__file__),
               "ledger_sha256": file_hash(ledger.OUT), "model_sha256": modelhash, "rows": rows}
    with PLAN.open("x") as f:
        json.dump(payload, f, indent=2, allow_nan=False)
    print(json.dumps({"tasks": len(rows), "logical_queries": sum(x["queries"] for x in rows)}, indent=2))


def physical():
    import gymnasium as gym
    import stable_worldmodel  # noqa: F401
    if PHYSICAL.exists():
        raise FileExistsError(PHYSICAL)
    plan, prior = json.loads(PLAN.read_text()), json.loads(ledger.OUT.read_text())
    for key, path in (("protocol_sha256", PROTOCOL), ("runner_sha256", __file__),
                      ("ledger_sha256", ledger.OUT)):
        if plan[key] != file_hash(path):
            raise ValueError("sealed plan drift")
    env = gym.make("swm/TwoRoom-v1", render_mode="rgb_array")
    rows = []
    try:
        with h5py.File(ledger.prior.source.block.prior.common.DATA, "r") as data:
            for item in plan["rows"]:
                root, task = item["root"], item["row"]
                prefixes = np.asarray(item["prefix_candidates"], dtype=np.float32)
                tail = np.asarray(item["common_tail"], dtype=np.float32).reshape(-1, 2)
                if prefixes.shape != (32, 5, 2) or item["choices"]["greedy"] != int(np.argmin(item["greedy_scores"])) \
                        or item["choices"]["lookahead"] != int(np.argmin(item["lookahead_scores"])):
                    raise ValueError("sealed choice mismatch")
                outcomes = {}
                for arm in ARMS:
                    actions = np.concatenate([prefixes[item["choices"][arm]], tail])
                    if actions.shape != (item["original_length"], 2) or not np.isfinite(actions).all() or np.abs(actions).max() > 1:
                        raise ValueError("invalid intervention")
                    env.reset(seed=42)
                    actor = env.unwrapped
                    actor._set_state(data["proprio"][task])
                    actor._set_goal_state(data["proprio"][task + 25])
                    trace = []
                    for action in actions:
                        _, _, term, trunc, info = actor.step(action.copy())
                        trace.append({"action": action.tolist(), "distance": float(info["distance_to_target"]),
                                      "terminated": bool(term), "truncated": bool(trunc)})
                        if term or trunc:
                            break
                    outcomes[arm] = {"trace": trace, "ever": any(x["terminated"] for x in trace),
                                     "prefix_success": any(x["terminated"] for x in trace[:5]),
                                     "prefix_end_distance": trace[min(4, len(trace) - 1)]["distance"]}
                old = next(r for r in prior["rows"] if r["root"] == root)
                if outcomes["original"]["trace"] != old["trace"]:
                    raise ValueError("original executed chain failed replay")
                rows.append({"root": root, "row": task, "outcomes": outcomes})
    finally:
        env.close()
    summary = {arm: {"ever": sum(r["outcomes"][arm]["ever"] for r in rows),
                     "prefix_success": sum(r["outcomes"][arm]["prefix_success"] for r in rows)}
               for arm in ARMS}
    with PHYSICAL.open("x") as f:
        json.dump({"plan_sha256": file_hash(PLAN), "rows": rows, "summary": summary,
                   "scope": "posthoc actual-chain common-tail intervention, not PC2 gate"}, f, indent=2, allow_nan=False)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in ("plan", "physical"):
        raise SystemExit("expected plan|physical")
    {"plan": build_plan, "physical": physical}[sys.argv[1]]()
