import copy
import json
from pathlib import Path

import numpy as np
import pytest

from experiments.spectral_reachability import p3_raw_h3_physical_saved_20261031 as saved


@pytest.fixture
def completed():
    root = Path(__file__).resolve().parents[1]
    plans = json.loads((root / "results/p3_raw_h3_physical_prepare_20261031_RUN01.json").read_bytes())["plans"][:6]
    state = np.zeros(7)
    goal = np.array([100., 0., 0., 0., 0., 0., 0.])
    image = np.zeros((224, 224, 3), np.uint8)
    snapshot = {"state": state.tolist()}
    reference = {"goal": goal, "goal_rgb": image, "root_sha256": saved.kernel.source.root_hash(state, goal),
                 "history": [image] * 3, "states": [state] * 3, "dynamics": [snapshot] * 3}
    for plan in plans:
        plan["root_sha256"] = reference["root_sha256"]
        plan["selection_digest"] = saved.kernel.source.digest({k: v for k, v in plan.items()
                                                                if k != "selection_digest"})
    p = {"schema": saved.batch.NAME + ".prepare", "contract": saved.batch.contract(),
         "plans": plans, "bindings": {}}
    events = [{"kind": "run_started", "prepare": p}]
    rows = []
    for i, plan in enumerate(plans):
        arm = f"d{plan['decision_index']}_c{plan['candidate']}_{plan['variant']}"
        row = {"branch_index": i, "root_slot": plan["root_slot"], "arm": arm,
               "selected_plan": {**plan, "arm": arm}, "selection_digest": plan["selection_digest"],
               "initial_exact": True, "first_error": None, "secondary_errors": [], "warmup_returned": 10,
               "own_attempted": 20, "own_returned": 20, "goal": goal.tolist(), "root_state": state.tolist(),
               "final_state": state.tolist(), "initial_metrics": saved.kernel.metric(goal, state),
               "final_metrics": saved.kernel.metric(goal, state), "native_success": False,
               "status": "horizon_completed", "terminated": False, "truncated": False,
               "full_horizon": True, "PNG": 32, "checkpoints": []}
        rows.append(row)
        for j, action in enumerate(saved.kernel.moving.warmup_commands(), 1):
            events.append({"kind": "warmup_return", "branch": i, "step": j,
                           "raw_float32_bits": action.tobytes().hex()})
        for j, action in enumerate(saved.batch.branch.raw20(plan["commands"]), 1):
            events.append({"kind": "returned_transition", "branch": i, "phase": "plan", "step": j,
                           "command": plan["commands"][j-1], "command_bits": action.tobytes().hex(),
                           "actual_bits": action.tobytes().hex(), "state": state.tolist(), "reward": -100.,
                           "terminated": False, "truncated": False, "returned_goal_state": goal.tolist(),
                           "returned_goal_rgb_sha256": saved.kernel.moving.image_sha(image)})
        for phase, count in (("warmup", 11), ("plan", 21)):
            for step in range(count):
                receipt = {"phase": phase, "step": step, "png": f"{i}_{phase}_{step}.png",
                           "file_sha256": "fixture", "pixel_sha256": saved.kernel.moving.image_sha(image),
                           "state": state.tolist(), "observation_state": state.tolist(), "snapshot": snapshot}
                events.append({"kind": "fresh_observation", "branch": i, "receipt": receipt})
                events.append({"kind": "guard_receipt", "branch": i, "accepted": True,
                               "render_preserved_dynamics": True, "phase": phase, "step": step})
                if phase == "plan" and step in (0, 5, 10, 15, 20):
                    row["checkpoints"].append(receipt)
        events.append({"kind": "branch_closed", "receipt": row})
    expected = dict(environment_construct=6, environment_close=6, reset=12, set_state=24,
                    goal_render=12, goal_bind=6, public_step=180, physics_step=1824, explicit_render=192)
    counts = {k: {"attempted": v, "completed": v} for k, v in expected.items()}
    for resource, count in expected.items():
        for _ in range(count):
            events.extend([{"kind": "operation_attempted", "resource": resource},
                           {"kind": "operation_completed", "resource": resource}])
    report = {"schema": saved.batch.NAME + ".report", "status": "recorded_not_admitted",
              "errors": [], "fixed_denominator": 6, "scientific_pass": False,
              "new_neural_calls": 0, "rows": rows, "counts": counts}
    events.append({"kind": "run_finished", "status": report["status"], "counts": counts})
    return p, report, events, [reference, reference], image


def test_complete_batch_and_tampering(completed):
    p, report, events, refs, image = completed
    result = saved.validate(p, report, events, refs, lambda *_: image)
    assert len(result["slots"]) == 6 and len(result["frames"]) == 192
    assert result["scientific_pass"] is False
    altered = copy.deepcopy(events)
    next(e for e in altered if e["kind"] == "returned_transition")["actual_bits"] = "00"
    with pytest.raises(ValueError, match="transition action"):
        saved.validate(p, report, altered, refs, lambda *_: image)
    altered = copy.deepcopy(events)
    next(e for e in altered if e["kind"] == "guard_receipt")["accepted"] = False
    with pytest.raises(ValueError, match="frame axis"):
        saved.validate(p, report, altered, refs, lambda *_: image)


def test_truncated_branch_keeps_last_state_without_full_endpoint(completed):
    p, report, events, refs, image = copy.deepcopy(completed)
    row = report["rows"][0]
    row.update(own_attempted=8, own_returned=8, status="native_stopped",
               truncated=True, full_horizon=False, PNG=20)
    row["checkpoints"] = [x for x in row["checkpoints"] if x["step"] <= 8]
    retained = []
    for event in events:
        if event["kind"] in ("operation_attempted", "operation_completed", "run_finished"):
            continue
        if event.get("branch") == 0:
            if event["kind"] == "returned_transition" and event["phase"] == "plan":
                if event["step"] > 8:
                    continue
                if event["step"] == 8:
                    event["truncated"] = True
            if event["kind"] == "fresh_observation":
                if event["receipt"]["phase"] == "plan" and event["receipt"]["step"] > 8:
                    continue
            if event["kind"] == "guard_receipt" and event["phase"] == "plan" and event["step"] > 8:
                continue
            if event["kind"] == "branch_closed":
                event["receipt"] = row
        retained.append(event)
    expected = dict(environment_construct=6, environment_close=6, reset=12, set_state=24,
                    goal_render=12, goal_bind=6, public_step=168, physics_step=1704, explicit_render=180)
    counts = {k: {"attempted": v, "completed": v} for k, v in expected.items()}
    report["counts"] = counts
    for resource, count in expected.items():
        for _ in range(count):
            retained.extend([{"kind": "operation_attempted", "resource": resource},
                             {"kind": "operation_completed", "resource": resource}])
    retained.append({"kind": "run_finished", "status": report["status"], "counts": counts})
    result = saved.validate(p, report, retained, refs, lambda *_: image)
    assert result["slots"][0]["endpoint"] is None
    assert result["slots"][0]["last_metrics"] == row["final_metrics"]
