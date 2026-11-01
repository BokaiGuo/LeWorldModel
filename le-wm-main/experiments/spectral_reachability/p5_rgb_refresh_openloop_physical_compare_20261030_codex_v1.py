"""Compare two sealed prediction journals on the explicitly open known physical panel."""
import argparse
import math
import os
from pathlib import Path

import numpy as np

from experiments.spectral_reachability import p5_rgb_known_physical_compare_20261030_codex_v1 as old
from experiments.spectral_reachability import p5_rgb_refresh_supervision_20261030_codex_v1 as new

bridge = old.bridge
ROOT = bridge.ROOT
RESULTS = ROOT / "results"
NAME = "p5_rgb_refresh_openloop_physical_compare_20261030_codex_v1"
OLD_SOURCE_SHA = "3f52f93aca960837412ef5fbe6db6ed1841612e07223a5680d238c0f75ce30c7"
NEW_SOURCE_SHA = "d5ab673a87a15460a3875d80266cad75819d37d092070f9605667ac7e832a289"
NEW_ACCEPT_SHA = "a1be8db350ad913fbabc4d87ebbbe5d47e14ea57121dc9aa4710f42294b0f799"
NEW_ACCEPT = RESULTS / (new.NAME + "_accept_MAIN01.json")


def require(value, message):
    old.require(value, message)


def sealed_predictions():
    require(bridge.digest(Path(old.__file__).read_bytes()) == OLD_SOURCE_SHA and
            bridge.digest(Path(new.__file__).read_bytes()) == NEW_SOURCE_SHA, "comparison source pins")
    accept_raw = old.checked_file(NEW_ACCEPT, NEW_ACCEPT_SHA, 2 * 1024 * 1024)
    accept = bridge.read_json(accept_raw)
    require(bridge.canonical(accept) == accept_raw and accept["schema"] == new.NAME + ".accept"
            and accept["status"] == "self_report_consistency_only_not_scientific" and
            accept["P5_admission"] is False and accept["producer_sha256"] == NEW_SOURCE_SHA and
            accept["refresh_source_sha256"] == new.REFRESH_SHA and accept["prepared_sha256"] == new.PREP_SHA
            and accept["baseline_sha256"] == new.wiring.BASE_SHA, "refresh accept identity")
    old_prepared, old_review = old.seal()
    base, new_prepared = new.preflight()
    supdir, rundir = new.paths("MAIN01")
    require(supdir.is_dir() and not supdir.is_symlink(), "refresh supervisor directory")
    supervisor = old.checked_file(supdir / "supervisor.jsonl", accept["supervisor_sha256"], 16 * 1024 * 1024)
    require(supervisor.endswith(b"\n"), "refresh supervisor truncation")
    events = [bridge.read_json(line) for line in supervisor.splitlines()]
    require(all(bridge.canonical(e).rstrip(b"\n") == line for e, line in zip(events, supervisor.splitlines(), strict=True)),
            "refresh supervisor canonical")
    expected_review = new.validate_supervisor(events, "MAIN01", NEW_SOURCE_SHA)
    reviewed = new.review_run(rundir, base, new_prepared, NEW_SOURCE_SHA, accept["review"]["journal_sha256"])
    require(reviewed == accept["review"] == expected_review, "refresh accepted review")
    return old_prepared, old_review, new_prepared, reviewed


def compare(old_prepared, old_review, new_prepared, new_review, physical, panel):
    require(old_prepared["normalization_from_prior_strict_load"] ==
            new_prepared["normalization_from_prior_strict_load"], "normalization identity")
    require(new_prepared["source_sha256"]["checkpoint"] == bridge.FILES["checkpoint"][1] and
            old_prepared["source_sha256"]["checkpoint"] == bridge.FILES["checkpoint"][1] and
            old_prepared["pairs"] == new_prepared["pairs"] == panel["pairs"], "checkpoint/pair identity")
    old_result = old.compare(old_prepared, old_review, physical, panel)
    require(len(new_prepared["rows"]) == len(new_review["rows"]) == len(old_result["rows"]) == 164,
            "fixed rows")
    mean = old.finite_vector(old_prepared["normalization_from_prior_strict_load"]["mean"], 6)
    std = old.finite_vector(old_prepared["normalization_from_prior_strict_load"]["std"], 6)
    rows = []
    for i, (a, b, pred, physical_row, old_result_row) in enumerate(
            zip(old_prepared["rows"], new_prepared["rows"], new_review["rows"],
                physical["rows"], old_result["rows"], strict=True)):
        keys = ("index", "root_slot", "condition", "task_id", "selection_digest")
        require(all(a[k] == b[k] == physical_row[k] == old_result_row[k] for k in keys)
                and all(pred[k] == a[k] for k in ("index", "root_slot", "condition")) and
                len(b["steps"]) == len(pred["steps"]) == 5, "row identity")
        actions = np.asarray(a["actions"], dtype="<f4")
        require(actions.shape == (5, 2) and np.isfinite(actions).all() and
                bridge.digest(actions.tobytes()) == a["actions_sha256"], "open actions")
        require(b["steps"][0]["previous_rgb"] == a["previous_rgb"] and
                b["steps"][0]["current_rgb"] == a["current_rgb"], "shared h1 frames")
        steps = []
        for h, (step, nout, o) in enumerate(zip(b["steps"], pred["steps"], old_result_row["steps"], strict=True), 1):
            require((step["horizon"], step["previous_plan_step"], step["current_plan_step"], step["target_plan_step"])
                    == (h, h + 3, h + 4, h + 5) and nout["horizon"] == h and
                    o["horizon"] == h and o["plan_step"] == h + 5, "time alignment")
            action = np.asarray(step["action"], dtype="<f4")
            require(action.shape == (2,) and action.tobytes() == actions[h - 1].tobytes()
                    and bridge.digest(action.tobytes()) == step["action_sha256"], "per-step action")
            for key, plan_step in (("previous_rgb", h + 3), ("current_rgb", h + 4)):
                frame = new.refresh.frame_at(physical_row, plan_step)
                record = step[key]
                require(frame["file_sha256"] == record["file_sha256"] and
                        frame["pixel_sha256"] == record["pixel_sha256"] and
                        frame["png"] == Path(record["path"]).name, "physical frame identity")
            values = old.finite_vector(nout["predict_next"]["values"], 6)
            encoded = np.asarray(values, dtype="<f4")
            require(bridge.digest(encoded.tobytes()) == nout["predict_next"]["sha256"], "refresh output SHA")
            prediction = [values[k] * std[k] + mean[k] for k in range(6)]
            require(all(math.isfinite(x) for x in prediction), "finite decoded output")
            truth = o["physical"]
            old_absolute = [abs(x) for x in o["error"]]
            new_absolute = [abs(prediction[k] - truth[k]) for k in range(6)]
            steps.append({"horizon": h, "plan_step": h + 5, "physical": truth,
                          "open_prediction": o["prediction"], "refresh_prediction": prediction,
                          "open_absolute_error": old_absolute, "refresh_absolute_error": new_absolute,
                          "refresh_minus_open_absolute_error": [new_absolute[k] - old_absolute[k] for k in range(6)]})
        rows.append({**{k: a[k] for k in keys}, "steps": steps})
    summary = {}
    for condition in ("off", "on"):
        selected = [r for r in rows if r["condition"] == condition]
        require(len(selected) == 82, "condition denominator")
        summary[condition] = {"denominator": 82, "horizons": [
            {"horizon": h + 1, **{field: [math.fsum(row["steps"][h][field][k] for row in selected) / 82
                                                   for k in range(6)] for field in
                                 ("open_absolute_error", "refresh_absolute_error",
                                  "refresh_minus_open_absolute_error")}}
            for h in range(5)]}
    return {"schema": NAME + ".result", "status": "known_panel_descriptive_only",
            "source_sha256": {"open_accept": old.ACCEPT_SHA, "refresh_accept": NEW_ACCEPT_SHA,
                              "open_journal": old_review["journal_sha256"],
                              "refresh_journal": new_review["journal_sha256"],
                              "open_prepared": old.runner.PREP_SHA, "refresh_prepared": new.PREP_SHA,
                              "physical_receipt": bridge.FILES["physical_receipt"][1],
                              "panel": bridge.FILES["panel"][1], "checkpoint": bridge.FILES["checkpoint"][1]},
            "coordinates": ["agent_x", "agent_y", "block_x", "block_y", "sin_angle", "cos_angle"],
            "denominators": {"branches": 164, "off": 82, "on": 82, "pairs_in_fixed_panel": 80,
                             "baseline_branches": 4, "known_roots": 2, "fresh_roots": 0},
            "summary": summary, "rows": rows, "P5_admission": False,
            "new_neural_calls": 0, "new_cuda_calls": 0, "new_physics_calls": 0, "new_training_calls": 0,
            "boundary": "known nonindependent roots; refresh has extra true observations; not fair closed-loop or scientific confirmation"}


def build():
    own = [Path(__file__), ROOT / "tests" / ("test_" + NAME + ".py"), ROOT / "innerdocs" / (NAME + ".md")]
    hashes = {str(p.relative_to(ROOT)): bridge.digest(p.read_bytes()) for p in own}
    old_prepared, old_review, new_prepared, new_review = sealed_predictions()
    physical = bridge.read_json(bridge.read_fixed("physical_receipt"))
    panel = bridge.read_json(bridge.read_fixed("panel"))
    result = compare(old_prepared, old_review, new_prepared, new_review, physical, panel)
    require(hashes == {str(p.relative_to(ROOT)): bridge.digest(p.read_bytes()) for p in own}, "source changed")
    result["consumer_closure"] = hashes
    return result


def target(name):
    require(Path(name).name == name and name.startswith(NAME + "_") and name.endswith(".json")
            and RESULTS.resolve() == RESULTS and not RESULTS.is_symlink(), "output namespace")
    return RESULTS / name


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=["write", "verify"])
    ap.add_argument("--name", required=True)
    ap.add_argument("--sha256")
    args = ap.parse_args()
    path = target(args.name)
    data = bridge.canonical(build())
    if args.action == "write":
        require(args.sha256 is None, "unexpected SHA")
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY | os.O_NOFOLLOW, 0o600)
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        old.runner.sync_dir(path.parent)
    else:
        require(args.sha256 is not None and old.checked_file(path, args.sha256, 16 * 1024 * 1024) == data,
                "exact rebuild")
    print(bridge.digest(data))


if __name__ == "__main__":
    main()
