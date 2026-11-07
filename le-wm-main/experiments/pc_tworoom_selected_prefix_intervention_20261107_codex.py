"""Replay the sealed descriptor selection with controlled common tails."""
import json

import numpy as np

from experiments import pc_tworoom_common_tail_retention_20261107_codex as base
from experiments.pc_input_permissions import file_hash

ROOT, PANEL = base.ROOT, base.PANEL
PROTOCOL = ROOT.parent / "docs/PC_TWOROOM_SELECTED_PREFIX_COMMON_TAIL_PROTOCOL_20261107_CODEX.md"
OUTPUT = ROOT / "results/pc_tworoom_selected_prefix_intervention_20261107_codex.json"


def interventions(prefix, tails):
    prefix, tails = np.asarray(prefix, np.float32), np.asarray(tails, np.float32)
    if prefix.shape != (10, 2) or tails.shape != (4, 15, 2):
        raise ValueError("bad intervention shape")
    result = {name: np.concatenate((np.broadcast_to(p, (4, 10, 2)), tails), axis=1)
              for name, p in (("selected", prefix), ("zero", np.zeros_like(prefix)),
                              ("reversed", prefix[::-1]))}
    if any(not np.isfinite(v).all() or np.abs(v).max() > 1 for v in result.values()):
        raise ValueError("invalid action")
    return result


def main():
    import gymnasium as gym
    import stable_worldmodel  # noqa: F401

    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    plan, replay = json.loads(base.PLAN.read_text()), json.loads(base.RESULT.read_text())
    if replay["plan_sha256"] != file_hash(base.PLAN) or replay["truth_sha256"] != file_hash(PANEL.TRUTH):
        raise ValueError("prior replay drift")
    for key, path in (("public_sha256", PANEL.PUBLIC), ("protocol_sha256", base.PROTOCOL),
                      ("runner_sha256", base.__file__), ("wide_sha256", base.wide.__file__),
                      ("repair_sha256", base.wide.repair.__file__)):
        if plan[key] != file_hash(path):
            raise ValueError(f"plan drift: {key}")
    reference = json.loads(PANEL.REFERENCE.read_text())
    if reference["truth_sha256"] != file_hash(PANEL.TRUTH) or reference["public_sha256"] != file_hash(PANEL.PUBLIC):
        raise ValueError("input drift")
    if len(plan["rows"]) != 40 or len(replay["rows"]) != 40:
        raise ValueError("root count drift")
    rows = []
    env = gym.make("swm/TwoRoom-v1", render_mode="rgb_array")
    print("Replay sealed descriptor-selected prefix against equal-duration controls with identical B tails.", flush=True)
    try:
        with np.load(PANEL.PUBLIC, allow_pickle=False) as public, np.load(PANEL.TRUTH, allow_pickle=False) as truth:
            for i, (p, old) in enumerate(zip(plan["rows"], replay["rows"])):
                if p["seed"] != old["seed"] or p["seed"] != int(public["seeds"][i]):
                    raise ValueError("root mismatch")
                prefix_index, selected_tail = p["selected"]["descriptor"]
                pools = interventions(p["prefixes"][prefix_index], p["tails"][5:])
                traces = {}
                for name, chains in pools.items():
                    traces[name] = []
                    for chain in chains:
                        env.reset(seed=p["seed"], options={"state": truth["initial"][i],
                                                          "target_state": truth["target"][i]})
                        actor = env.unwrapped
                        np.testing.assert_array_equal(actor.render(), public["current"][i])
                        trace = []
                        for action in chain:
                            _, _, term, trunc, info = actor.step(action.copy())
                            trace.append([float(info["distance_to_target"]), bool(term), bool(trunc)])
                        traces[name].append(trace)
                for j, trace in enumerate(traces["selected"]):
                    old_trace = old["physical"][prefix_index][j + 5]
                    np.testing.assert_allclose(np.asarray(trace)[:, 0], np.asarray(old_trace)[:, 0], atol=1e-5, rtol=0)
                    np.testing.assert_array_equal(np.asarray(trace)[:, 1:], np.asarray(old_trace)[:, 1:])
                rows.append({"seed": p["seed"], "selected_prefix": prefix_index, "selected_tail": selected_tail,
                             "traces": traces})
                print(f"root={i} selected_tail={selected_tail}", flush=True)
    finally:
        env.close()
    with OUTPUT.open("x") as f:
        json.dump({"track": "primary_rgb", "scope": "exposed offline intervention, not closed-loop",
                   "plan_sha256": file_hash(base.PLAN), "prior_replay_sha256": file_hash(base.RESULT),
                   "protocol_sha256": file_hash(PROTOCOL), "runner_sha256": file_hash(__file__),
                   "rows": rows}, f, allow_nan=False, separators=(",", ":"))


if __name__ == "__main__":
    main()
