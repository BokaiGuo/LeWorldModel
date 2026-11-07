"""Retrospective fixed-first-packet RGB feedback versus unchanged failed plan."""
import hashlib
import json
import time
from pathlib import Path

import h5py
import numpy as np
import torch

from experiments import pc_tworoom_failure_case_temporal_20261107_codex as source
from experiments import pc_rgb_block_incumbent_search_20261107_codex as planner
from experiments.pc_input_permissions import RGB, file_hash, freeze_audit
from experiments.pc_prefix_novel_actual_encoding_20261105_codex import model_fingerprint
from experiments.pc_rgb_observation_cache_20261107_codex import RGBObservationScoringSession

PREFIX = source.b.q.ROOT / "results/pc_tworoom_fixed_prefix_feedback_20261107_codex"
OUTPUT = Path(str(PREFIX) + "_result.json")
PROTOCOL = source.b.q.ROOT.parent / "docs/PC_TWOROOM_FIXED_PREFIX_FEEDBACK_PROTOCOL_20261107_CODEX.md"
PRIOR_CEM = source.b.q.ROOT / "results/pc_tworoom_rgb_deadline_cem_20261107_codex_run.json"
ROOTS = (38, 39)


def digest(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def check_same_prefix(chain, trace):
    if len(trace) < 5:
        raise ValueError("first packet incomplete")
    if not np.array_equal(np.asarray([step["action"] for step in trace[:5]], dtype=np.float32), chain[:5]):
        raise ValueError("shared first packet mismatch")


def main():
    import gymnasium as gym
    import stable_worldmodel  # noqa: F401
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    source.verify()
    before = json.loads(source.RESULT.read_text())
    if before["frame_sha256"] != file_hash(source.FRAMES) or before["features_sha256"] != file_hash(source.FEATURES):
        raise ValueError("prior temporal binding drift")
    plan, model, scaler, transform = source.setup()
    source_phys = json.loads(source.b.q.PHYSICAL.read_text())
    existing_cem = json.loads(PRIOR_CEM.read_text())
    common = source.b.q.old.base.source.block.prior.common
    inp_path = source.b.q.old.base.source.block.prior.inputs.INPUT
    audit = freeze_audit(RGB, "lewm", [__file__, PROTOCOL, planner.__file__, source.b.q.PLAN,
                         source.b.q.POOL, inp_path, common.CHECKPOINT / "weights.pt",
                         common.CHECKPOINT / "config.json"], preprocessing="ImageNet 224 and TwoRoom scaler",
                         clock="5 environment steps per packet, remaining 25-step deadline")
    env = gym.make("swm/TwoRoom-v1", render_mode="rgb_array")
    rows = []
    try:
        with np.load(source.b.q.POOL, allow_pickle=False) as pools, \
                np.load(inp_path, allow_pickle=False) as inputs, \
                h5py.File(common.DATA, "r") as data, torch.inference_mode():
            for root in ROOTS:
                record = next(row for row in plan["rows"] if row["root"] == root)
                prior_trace = next(row for row in source_phys["rows"] if row["root"] == root)["outcomes"]["lookahead"]["trace"]
                chain = pools[f"r{root}_actions"][0]
                if digest(chain) != record["decisions"]["lookahead"]["action_sha256"]:
                    raise ValueError("initial chain drift")
                goal = transform(inputs["goal"][root])[None, None, None].cuda()
                arms = {}
                for arm in ("open", "feedback"):
                    start = time.perf_counter()
                    env.reset(seed=42)
                    actor = env.unwrapped
                    actor._set_state(data["proprio"][record["row"]])
                    actor._set_goal_state(data["proprio"][record["row"] + 25])
                    initial_digest = digest(actor.render().copy())
                    trace, decisions = [], []
                    old_tail = chain.copy()
                    for iteration in range(5):
                        remaining = 25 - len(trace)
                        if iteration == 0 or arm == "open":
                            chosen = old_tail
                        else:
                            image = actor.render().copy()
                            context = {"pixels": transform(image)[None, None, None].cuda(),
                                       "goal": goal, "action": torch.empty((1, 1, 0, 10), device="cuda")}
                            boundary = RGBObservationScoringSession(model, audit)
                            forward_calls = 0

                            def score(actions):
                                nonlocal forward_calls
                                normal = scaler.transform(actions.reshape(-1, 2)).reshape(-1, remaining // 5, 10)
                                costs = []
                                for offset in range(0, len(actions), 32):
                                    batch = torch.as_tensor(normal[offset:offset + 32][None], dtype=torch.float32, device="cuda")
                                    value, _ = boundary.score(context, batch)
                                    forward_calls += 1
                                    costs.extend(value[0].cpu().tolist())
                                return np.asarray(costs)

                            torch.cuda.synchronize()
                            begin = time.perf_counter()
                            result = planner.search(score, remaining=remaining, old_tail=old_tail,
                                                    seed=27121000 + root * 10 + iteration, arm="baseline")
                            torch.cuda.synchronize()
                            elapsed = time.perf_counter() - begin
                            chosen = result["actions"]
                            decisions.append({"iteration": iteration, "remaining": remaining,
                                              "seed": 27121000 + root * 10 + iteration,
                                              "image_sha256": digest(image), "old_tail_sha256": digest(old_tail),
                                              "actions_sha256": result["actions_sha256"], "actions": chosen.tolist(),
                                              "cost": result["cost"], "winner": result["winner"],
                                              "all_scores": result["all_scores"].tolist(), "queries": result["model_queries"],
                                              "packet_steps": result["model_packet_steps"], "forward_calls": forward_calls,
                                              "old_tail_included": result["old_tail_included"],
                                              "planning_seconds": elapsed})
                        for action in chosen[:5]:
                            _, _, term, trunc, info = actor.step(action.copy())
                            trace.append({"action": action.tolist(), "distance": float(info["distance_to_target"]),
                                          "terminated": bool(term), "truncated": bool(trunc)})
                            if term or trunc:
                                break
                        dispatched = len(trace) - (25 - remaining)
                        old_tail = chosen[dispatched:].copy()
                        if term or trunc:
                            break
                    check_same_prefix(chain, trace)
                    if arm == "open":
                        if len(trace) != len(prior_trace) or any(
                            trace[i][key] != prior_trace[i][key]
                            for i in range(len(trace)) for key in ("distance", "terminated", "truncated")):
                            raise ValueError("open-loop replay drift")
                    arms[arm] = {"trace": trace, "decisions": decisions, "initial_rgb_sha256": initial_digest,
                                 "ever_success": any(t["terminated"] for t in trace),
                                 "truncated": any(t["truncated"] for t in trace),
                                 "queries": sum(d["queries"] for d in decisions),
                                 "packet_steps": sum(d["packet_steps"] for d in decisions),
                                 "planning_seconds": sum(d["planning_seconds"] for d in decisions),
                                 "wall_seconds": time.perf_counter() - start}
                if arms["open"]["initial_rgb_sha256"] != arms["feedback"]["initial_rgb_sha256"]:
                    raise ValueError("initial observation mismatch")
                cem = next(row for row in existing_cem["rows"] if row["root"] == root)["arms"]
                references = {name: {"ever_success": item["ever_success"], "queries": item["queries"],
                                     "terminal_distance": item["terminal_distance"],
                                     "decisions": len(item["decisions"])} for name, item in cem.items()}
                rows.append({"root": root, "row": record["row"], "initial_action_sha256": digest(chain),
                             "arms": arms, "separate_prior_cem_reference": references})
                print(f"root={root} open={arms['open']['ever_success']} feedback={arms['feedback']['ever_success']}", flush=True)
    finally:
        env.close()
    if model_fingerprint(model) != plan["model_sha256"]:
        raise ValueError("model changed")
    paths = [Path(__file__), PROTOCOL, Path(planner.__file__), Path(source.__file__), source.PRED,
             source.META, source.RESULT, source.b.q.PLAN, source.b.q.POOL, source.b.q.PHYSICAL,
             inp_path, common.DATA, common.CHECKPOINT / "weights.pt", PRIOR_CEM]
    with OUTPUT.open("x") as f:
        json.dump({"track": RGB, "rows": rows, "model_sha256": plan["model_sha256"],
                   "source_sha256": {str(path): file_hash(path) for path in paths},
                   "scope": "retrospectively selected failure cases; additional feedback budget"},
                  f, separators=(",", ":"), allow_nan=False)


if __name__ == "__main__":
    main()
