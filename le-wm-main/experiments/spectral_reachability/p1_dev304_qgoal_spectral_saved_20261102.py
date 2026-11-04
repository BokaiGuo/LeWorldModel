"""Verify saved full-horizon spectra, selected candidates and all physical branches."""
import argparse
import hashlib
from pathlib import Path

import numpy as np

from . import p1_dev304_qgoal_spectral_physical_20261102 as physical
from . import d2p_goal_mode_rank_20261102 as geometry
from . import p1_dev304_spectral_real_rgb_saved_20261102 as rgb_saved
from .p1_dev303_dynamics_saved_analysis_20261102_main import measures

model_run, base, previous = physical.model_run, physical.base, physical.previous
ROOT = physical.ROOT/"results"
RUN = ROOT/f"{physical.NAME}_RUN_DEV01"
PREP_SHA = "e937e573bdfff4cbd68b08cb5e7d5e96916ebbae7cb1b79fa72aa0526955237c"
REPORT_SHA = "f87394ee94d9323fcf83caf66f01c1606c52124ad490e392dddc8b5202d7e642"
need = physical.need


def analyze():
    contract = base.read(ROOT/f"{physical.NAME}_PREP_DEV01.json", PREP_SHA)
    need(contract == physical.protocol(), "physical contract")
    report = base.read(RUN/"report.json", REPORT_SHA)
    model = base.read(ROOT/f"{model_run.NAME}_RUN_DEV01/report.json", physical.MODEL_SHA)
    model_spec = base.read(ROOT/f"{model_run.NAME}_PREP_DEV01.json", physical.CONTRACT_SHA)
    need(report["contract_sha256"] == PREP_SHA and report["status"] == "recorded" and not report["errors"]
         and base.sha(RUN/"journal.jsonl") == report["journal_sha256"], "report or journal")
    need(model["model_before"] == model["model_after"] == model_spec["model_hash"]
         and model["counts"]["vjp_attempted"] == model["counts"]["vjp_returned"] == 384
         and model["counts"]["svd_calls"] == 2, "frozen model or counts")
    original = base.read(previous.saved.ROOT/f"{previous.saved.NAME}_EXEC_DEV01/report.json", previous.saved.EXEC_SHA)
    capture = base.read(previous.saved.ROOT/f"{previous.saved.NAME}_CAPTURE_DEV01/report.json", previous.saved.CAPTURE_SHA)
    goal = capture["goal_state_evaluation_only"]
    expected = [(role, candidate, old, mrow)
                for role, old, mrow in zip(contract["roles"], original["rows"], model["rows"], strict=True)
                for candidate in role["candidates"]]
    need(len(report["rows"]) == len(expected) == 18, "roster")
    rows, frames = [], 0
    for row, (role, candidate, old, mrow) in zip(report["rows"], expected, strict=True):
        expected_name = f"{role['arm']}_{candidate['index']:02d}"
        need(row["arm"] == expected_name and base.read(RUN/(expected_name+".row.json")) == row
             and row["status"] == "fixed_horizon_completed", "row or censoring")
        actions = np.asarray(role["prefix"], dtype="<f4").tolist() + np.asarray(candidate["actions"], dtype="<f4").reshape(10, 2).tolist()
        need(len(row["transitions"]) == len(actions) == 15, "trajectory size")
        for i, (transition, action) in enumerate(zip(row["transitions"], actions, strict=True)):
            need(transition["phase"] == ("prefix" if i < 5 else "tail")
                 and transition["step"] == (i+1 if i < 5 else i-4)
                 and np.array_equal(transition["action"], action)
                 and not transition["terminated"] and not transition["truncated"], "trajectory action")
            if i < 5:
                need(transition["state"] == old["transitions"][i]["state"], "prefix identity")
        obs = row["observations"]
        need([(o["phase"], o["step"]) for o in obs]
             == [("warmup", x) for x in (0, 5, 10)] + [("prefix", 5)]
             + [("tail", x) for x in range(1, 11)], "frame roster")
        for frame in obs:
            previous.saved.verify_frame(RUN, frame)
            frames += 1
        for a, b in zip(obs[:4], [*old["observations"][:3], old["observations"][7]], strict=True):
            need(all(a[k] == b[k] for k in ("pixel_sha256", "state", "snapshot")), "checkpoint")
        need(all(o["state"] == t["state"] for o, t in zip(obs[4:], row["transitions"][5:], strict=True)), "tail")
        item = mrow["candidates"][candidate["index"]]
        action_array = np.asarray(item["actions"], dtype="<f4")
        need(hashlib.sha256(action_array.tobytes()).hexdigest() == candidate["actions_sha256"], "candidate SHA")
        rows.append({"prefix": role["arm"], "candidate": candidate["index"], "kind": candidate["kind"],
                     "model_score": item["score"], "model_selected": candidate["index"] == role["model_selected_index"],
                     "terminal": measures(row["transitions"][-1]["state"], goal),
                     "reward": row["transitions"][-1]["reward"], "native_success": False})
    goal_latent = np.asarray(base.read(rgb_saved.RUN/"report.json", rgb_saved.REPORT_SHA)["goal_latent"], float)
    metric = np.asarray(model["metric"], float)
    need(metric.shape == (192, 192) and np.isfinite(metric).all()
         and np.allclose(goal_latent, model["metric_info"]["goal_latent"], rtol=1e-5, atol=1e-6),
         "goal metric binding")
    for mrow in model["rows"]:
        need(base.read(ROOT/f"{model_run.NAME}_RUN_DEV01/{mrow['arm']}.json") == mrow, "model shard")
        jac = np.asarray(mrow["jacobian"], float)
        direction = np.asarray(mrow["fd_direction"], float)
        predicted, observed = np.asarray(mrow["fd_jacobian"], float), np.asarray(mrow["fd_observed"], float)
        need(jac.shape == (192, 20) and direction.shape == (20,) and np.isfinite(jac).all()
             and np.allclose(jac@direction, predicted, rtol=1e-6, atol=1e-7)
             and np.isclose(np.linalg.norm(observed-predicted)/(np.linalg.norm(predicted)+1e-12),
                            mrow["fd_relative_error"], rtol=1e-6), "FD or Jacobian")
        need(int(np.argmin([c["score"] for c in mrow["candidates"]])) == mrow["selected_index"], "selector")
        for candidate in mrow["candidates"]:
            delta = np.asarray(candidate["terminal"], float) - goal_latent
            need(np.isclose(np.mean((metric @ delta)**2), candidate["score"], rtol=1e-5, atol=1e-7),
                 "weighted nonlinear score")
        ranked = geometry.rank_goal_modes(metric @ jac,
                                          metric @ (goal_latent - np.asarray(mrow["anchor_terminal"], float)), 0.01)
        need(ranked["order"][:2].tolist() == mrow["mode_indices"], "goal mode ranking")
    return {"schema": physical.NAME+".saved.v1", "rows": rows, "frames_verified": frames,
            "fd_relative_error": {r["arm"]: r["fd_relative_error"] for r in model["rows"]},
            "classification": "one_known_root_post_outcome_development",
            "p1_admission": False, "independent_confirmation": False, "full_cost_certified": False}


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    print(base.write_new(p.parse_args().output, analyze()))
