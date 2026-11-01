"""Saved-only MAIN02 comparison against the explicitly open known physical panel."""
import argparse
import math
import os
from pathlib import Path

import numpy as np

from experiments.spectral_reachability import p5_rgb_instrumented_supervision_20261030_codex_v1 as wiring

bridge, runner, gate = wiring.abi.bridge, wiring.runner, wiring.gate
NAME = "p5_rgb_known_physical_compare_20261030_codex_v1"
ACCEPT_SHA = "a9d8a8aecf9b639d3df2d308c0fc8b59aa06fc0c012833657d96bf4a3df9d23e"
PRODUCER_SHA = "f1f3f1d0bc962b6f153ace297f12da0f6ff63eb990252b96a8a63d62d96ffee7"
ROOT = bridge.ROOT
RESULTS = ROOT / "results"
ACCEPT = RESULTS / (wiring.NAME + "_accept_MAIN02.json")
RUN = RESULTS / (wiring.NAME + "_run_MAIN02")
SUPERVISOR = RESULTS / (wiring.NAME + "_supervisor_MAIN02") / "supervisor.jsonl"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def checked_file(path, sha, limit):
    require(path.is_file() and not path.is_symlink() and path.stat().st_size <= limit, "file: " + str(path))
    raw = path.read_bytes()
    require(bridge.digest(raw) == sha, "sha: " + str(path))
    return raw


def seal():
    """Complete all prediction checks before allowing any physical receipt read."""
    accept = bridge.read_json(checked_file(ACCEPT, ACCEPT_SHA, 1024 * 1024))
    require(accept["status"] == "self_report_consistency_only_not_scientific"
            and accept["P5_admission"] is False and accept["producer_sha256"] == PRODUCER_SHA
            and wiring.source_sha() == PRODUCER_SHA and accept["baseline_sha256"] == wiring.BASE_SHA,
            "accept identity")
    checked_file(SUPERVISOR, accept["supervisor_sha256"], 1024 * 1024)
    base = wiring.preflight()
    require(wiring.review_run(RUN, base, PRODUCER_SHA) == accept["review"], "accepted review mismatch")
    prepared = runner.load_prepared()
    journal = checked_file(RUN / "events.jsonl", accept["review"]["journal_sha256"], 16 * 1024 * 1024)
    reviewed = gate.strict_review(prepared, journal, accept["review"]["journal_sha256"])
    require(reviewed["status"] == "complete_journal_verified", "incomplete predictions")
    return prepared, reviewed


def finite_vector(value, n):
    require(isinstance(value, list) and len(value) == n and
            all(type(x) in (float, int) and math.isfinite(x) for x in value), "finite vector")
    return [float(x) for x in value]


def compare(prepared, reviewed, physical, panel):
    require(physical["status"] == "observed_raw164_physical_recorded_not_admitted"
            and physical["fixed_denominator"] == 164 and len(physical["rows"]) == 164
            and panel["branch_denominator"] == 164 and panel["pair_denominator"] == 80
            and len(panel["rows"]) == 164 and len(panel["pairs"]) == 80
            and prepared["pairs"] == panel["pairs"] and len(reviewed["rows"]) == 164,
            "fixed cohort")
    norm = prepared["normalization_from_prior_strict_load"]
    mean, std = finite_vector(norm["mean"], 6), finite_vector(norm["std"], 6)
    require(all(x > 0 for x in std), "normalization")
    rows = []
    for i, (p, r, ph, binding) in enumerate(zip(prepared["rows"], reviewed["rows"],
                                                physical["rows"], panel["rows"], strict=True)):
        keys = ("index", "root_slot", "condition", "task_id", "selection_digest")
        require(all(p[k] == ph[k] == binding[k] for k in keys)
                and (p["index"], p["root_slot"], p["condition"]) ==
                (i, i // 82, ("off", "on")[(i // 41) % 2])
                and all(r[k] == p[k] for k in ("index", "root_slot", "condition"))
                and ph["status"] == "horizon_completed" and ph["own_returned"] == 10
                and ph["initial_exact"] is True, "branch identity")
        commands = np.asarray(ph["plan"]["commands"], dtype="<f4")
        require(commands.shape == (10, 2) and np.isfinite(commands).all()
                and bridge.digest(commands.tobytes()) == binding["raw10_sha256"]
                and bridge.digest(commands[5:].tobytes()) == p["actions_sha256"], "action identity")
        steps = []
        require(len(r["steps"]) == 5, "five predictions")
        for h, prediction in enumerate(r["steps"], 1):
            frames = [x for x in ph["trajectory"] if x["phase"] == "plan" and x["step"] == h + 5
                      and x["condition"] == p["condition"] and x["overlay_index"] == int(p["condition"] == "on")]
            require(len(frames) == 1 and prediction["step"] == h and prediction["status"] == "returned",
                    "target alignment")
            state = finite_vector(frames[0]["state"], 7)
            require(frames[0]["snapshot"]["state"] == frames[0]["state"], "state snapshot")
            values = finite_vector(prediction["output"]["values"], 6)
            raw = [values[k] * std[k] + mean[k] for k in range(6)]
            require(all(math.isfinite(x) for x in raw), "decoded finite")
            truth = state[:4] + [math.sin(state[4]), math.cos(state[4])]
            steps.append({"horizon": h, "plan_step": h + 5, "prediction": raw, "physical": truth,
                          "error": [raw[k] - truth[k] for k in range(6)]})
        rows.append({**{k: p[k] for k in keys}, "steps": steps})
    pairs = []
    for j, pair in enumerate(panel["pairs"]):
        slot, local = divmod(j, 20)
        minus, plus = slot * 41 + 1 + 2 * local, slot * 41 + 2 + 2 * local
        require(pair["slot"] == slot and pair["minus_index"] == minus and pair["plus_index"] == plus
                and pair["condition"] == rows[minus]["condition"] == rows[plus]["condition"]
                and pair["root_slot"] == rows[minus]["root_slot"] == rows[plus]["root_slot"], "pair identity")
        steps = []
        for h in range(5):
            m, p = rows[minus]["steps"][h], rows[plus]["steps"][h]
            pred = [p["prediction"][k] - m["prediction"][k] for k in range(6)]
            truth = [p["physical"][k] - m["physical"][k] for k in range(6)]
            steps.append({"horizon": h + 1, "predicted_plus_minus": pred, "physical_plus_minus": truth,
                          "error": [pred[k] - truth[k] for k in range(6)]})
        pairs.append({"identity": pair, "steps": steps})
    summary = {}
    for condition in ("off", "on"):
        selected = [r for r in rows if r["condition"] == condition]
        selected_pairs = [p for p in pairs if p["identity"]["condition"] == condition]
        require(len(selected) == 82 and len(selected_pairs) == 40, "condition counts")
        summary[condition] = {
            "branch_denominator": 82, "pair_denominator": 40,
            "horizons": [{"horizon": h + 1,
                          "branch_coordinate_mae": [math.fsum(abs(r["steps"][h]["error"][k])
                                                             for r in selected) / 82 for k in range(6)],
                          "pair_coordinate_mae": [math.fsum(abs(p["steps"][h]["error"][k])
                                                           for p in selected_pairs) / 40 for k in range(6)]}
                         for h in range(5)]}
    return {"schema": NAME + ".result", "status": "known_panel_descriptive_only",
            "accept_sha256": ACCEPT_SHA, "journal_sha256": reviewed["journal_sha256"],
            "physical_receipt_sha256": bridge.FILES["physical_receipt"][1],
            "panel_sha256": bridge.FILES["panel"][1], "prepared_sha256": runner.PREP_SHA,
            "coordinates": ["agent_x", "agent_y", "block_x", "block_y", "sin_angle", "cos_angle"],
            "denominators": {"branches": 164, "pairs": 80, "baselines": 4, "known_roots": 2,
                             "fresh_roots": 0}, "summary": summary, "rows": rows, "pairs": pairs,
            "P5_admission": False, "new_neural_calls": 0, "new_physics_calls": 0,
            "boundary": "retrospective known panel; off/on differ; RGB pose model is not JEPA; no fairness, causal, independent replication or scientific admission claim"}


def build():
    own = [Path(__file__), ROOT / "tests" / ("test_" + NAME + ".py"),
           ROOT / "innerdocs" / (NAME + ".md")]
    closure = {str(p.relative_to(ROOT)): bridge.digest(p.read_bytes()) for p in own}
    prepared, reviewed = seal()
    # The known physical receipt is opened only after complete accepted prediction verification.
    physical = bridge.read_json(bridge.read_fixed("physical_receipt"))
    panel = bridge.read_json(bridge.read_fixed("panel"))
    result = compare(prepared, reviewed, physical, panel)
    require(closure == {str(p.relative_to(ROOT)): bridge.digest(p.read_bytes()) for p in own},
            "consumer closure changed")
    result["consumer_closure"] = closure
    return result


def target(name):
    require(Path(name).name == name and name.startswith(NAME + "_") and name.endswith(".json"),
            "exclusive output name")
    require(RESULTS.resolve() == RESULTS and not RESULTS.is_symlink(), "results directory")
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
        runner.sync_dir(path.parent)
    else:
        require(args.sha256 is not None and checked_file(path, args.sha256, 16 * 1024 * 1024) == data,
                "rebuild mismatch")
    print(bridge.digest(data))


if __name__ == "__main__":
    main()
