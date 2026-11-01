"""Verify fixed three-branch saved journals, action seals, and actual RGB receipts."""
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

from . import codex_p1_diag32_cp5_live_20261101_exclusive as live
from . import p1_moving_live_feedback_analysis_20261027_v2 as analysis

ROOT = live.ROOT
CONTRACT = ROOT / "results/codex_p1_diag32_cp5_live_20261101_exclusive_PREP_CODEXLIVE01.json"
DIRECTORY = ROOT / "results/codex_p1_diag32_cp5_live_20261101_exclusive_run_CODEXLIVE01"
REPORT_SHA = "f8df21134c839ad5dcd0c3c2c3aa9e99601b7eae1e9446dc554a15b35a6a59be"
CONTRACT_SHA = "12c619e4b7b201610719772556eb03a45879353ccc49339c60c7ce302f224587"
require, sha = live.require, live.sha


def verify():
    require(sha(CONTRACT) == CONTRACT_SHA and sha(DIRECTORY / "report.json") == REPORT_SHA, "pinned files")
    require(json.loads(CONTRACT.read_bytes()) == live.build(), "contract/source drift")
    report = json.loads((DIRECTORY / "report.json").read_bytes())
    value = json.loads(live.PRESEAL.read_bytes())
    require(report["status"] == "recorded_development" and not report["errors"]
            and report["contract_sha256"] == CONTRACT_SHA and report["preseal_sha256"] == live.PRESEAL_SHA
            and report["model_before"] == report["model_after"]
            and report["model_before"]["original_state_dict"] == value["model_sha256"], "run binding")
    journal = DIRECTORY / "journal.jsonl"
    require(sha(journal) == report["journal_sha256"], "journal hash")
    events = analysis.journal(journal.read_bytes(), report["journal_chain_tail"])
    require(events[0]["kind"] == "run_started" and events[-1]["kind"] == "run_finished"
            and events[-1]["counts"] == report["counts"], "journal boundaries")
    pending = {}
    counts = {k: {"attempted": 0, "completed": 0} for k in report["counts"]}
    for event in events:
        if event["kind"] == "operation_attempted":
            ident = event["operation_id"]
            require(ident not in pending, "repeated physical attempt")
            pending[ident] = event
            counts[event["resource"]]["attempted"] += event["units"]
        elif event["kind"] == "operation_completed":
            ident = event["operation_id"]
            require(ident in pending and all(event[k] == pending[ident][k] for k in
                    ("resource", "units", "region", "episode_id")), "physical pair")
            counts[event["resource"]]["completed"] += event["units"]
            del pending[ident]
    require(not pending and counts == report["counts"] and not report["pending"], "physical meter")
    rows = report["rows"]
    require([r["role"] for r in rows] == value["roles"], "role roster")
    summaries = []
    for row in rows:
        role = row["role"]
        require(json.loads((DIRECTORY / (role + "_row.json")).read_bytes()) == row and row["initial_exact"]
                and row["warmup_returned"] == 10 and row["control_returned"] == len(row["transitions"]),
                "row source/physical count")
        goal = np.asarray(row["observations"][0]["goal"])
        n = row["control_returned"]
        dispatch = [e for e in events if e["kind"] == "branch_dispatch" and e["role"] == role]
        returned = [e for e in events if e["kind"] == "branch_returned" and e["role"] == role]
        seal_events = [e for e in events if e["kind"] == "suffix_sealed" and e["role"] == role]
        require(len(dispatch) == len(returned) == n and len(seal_events) == 1, "physical roster")
        suffix = row["suffix"]
        raw = np.asarray(suffix["actions"], dtype="<f4")
        require(raw.shape == (2, 5, 2) and np.isfinite(raw).all() and (np.abs(raw) <= 1).all()
                and hashlib.sha256(raw.tobytes()).hexdigest() == row["suffix_sha256"]
                == seal_events[0]["suffix_sha256"], "suffix bytes")
        plan = np.concatenate((np.asarray(value["prefixes"][role], np.float32), raw.reshape(10, 2)))
        require(np.array_equal(np.asarray(row["executed_raw"], np.float32), plan[:n])
                and dispatch[5]["index"] > seal_events[0]["index"], "authorized actions/seal order")
        for i, (d, ret, transition) in enumerate(zip(dispatch, returned, row["transitions"], strict=True), 1):
            require(d["step"] == ret["step"] == transition["step"] == i and
                    d["command"] == ret["command"] == transition["command"] == plan[i-1].tolist(), "action return")
            require(all(ret[k] == transition[k] for k in ("state", "reward", "terminated", "truncated",
                                                           "n_contacts", "metrics")), "returned data")
            metrics = analysis.native_metrics(transition["state"], goal)
            for key in ("native_success", "native_distance", "native_angle_term", "negative_native_angle",
                        "wrapped_allxy_success", "wrapped_object_angle_error", "object_position_error"):
                require(metrics[key] == transition["metrics"][key], "physical metric")
            if i < n:
                require(not transition["terminated"] and not transition["truncated"], "post-stop action")
        require(len(row["observations"]) == n + 1 and row["observations"][5] == row["cp5"]["receipt"],
                "cp5 observation")
        for i, receipt in enumerate(row["observations"]):
            require(receipt["step"] == i and receipt["phase"] == "plan", "frame sequence")
            path = Path(receipt["path"])
            require(path.parent == DIRECTORY and not path.is_symlink() and sha(path) == receipt["file_sha256"],
                    "frame file")
            with Image.open(path) as image:
                pixels = np.asarray(image)
            require(pixels.shape == (224, 224, 3) and live.engine.old.array_sha(pixels) == receipt["pixel_sha256"],
                    "frame pixels")
        saved_seal = json.loads((DIRECTORY / (role + "_suffix_seal.json")).read_bytes())
        require(saved_seal["cp5"] == row["cp5"] and saved_seal["winner"] == suffix
                and saved_seal["suffix_sha256"] == row["suffix_sha256"]
                and saved_seal["preseal_sha256"] == live.PRESEAL_SHA, "suffix seal")
        score_events = [e for e in events if e["kind"] == "candidate_score" and e["role"] == role]
        require(len(score_events) == 264 and [(e["candidate"], e["iteration"]) for e in score_events] ==
                [(c, i) for c in range(8) for i in range(33)], "score roster")
        winner = min(score_events, key=lambda e: (e["q2"], e["candidate"], e["iteration"]))
        require(all(winner[k] == suffix[k] for k in ("candidate", "iteration", "q2", "actions")),
                "argmin suffix")
        neural = [e for e in events if e["kind"] in ("neural_attempt", "neural_return") and e["role"] == role]
        expected = {"cp5_rgb_encode": 1, "action_encoder": 528, "predict": 528, "reverse": 256}
        require(row["neural"]["attempts"] == row["neural"]["returns"] == expected
                and row["neural"]["module_hooks"] == {"predict": 528, "action_encoder": 528}, "neural hooks")
        require(len(neural) == 2 * sum(expected.values()), "neural events")
        neural_counts = dict.fromkeys(expected, 0)
        for a, b in zip(neural[::2], neural[1::2], strict=True):
            require(a["kind"] == "neural_attempt" and b["kind"] == "neural_return"
                    and a["resource"] == b["resource"] and a["context"] == b["context"], "neural pair")
            neural_counts[a["resource"]] += 1
        require(neural_counts == expected and neural[0]["context"]["pixel_sha256"] ==
                row["cp5"]["receipt"]["pixel_sha256"], "cp5 encode/charges")
        summaries.append({"role": role, "status": row["status"], "returned": n,
                          "unexecuted_suffix_steps": 15 - n, "cp5_pixel_sha256": row["cp5"]["receipt"]["pixel_sha256"],
                          "suffix_sha256": row["suffix_sha256"]})
    return {"status": "verified_saved_artifacts_without_inference_or_replay", "rows": summaries,
            "report_sha256": REPORT_SHA, "contract_sha256": CONTRACT_SHA, "physical_counts": counts,
            "limits": "checks recorded pairing and byte identities, not independent neural or physics replay",
            "P1_admission": False}


if __name__ == "__main__":
    out = ROOT / "results/codex_p1_diag32_cp5_live_verify_20261101_exclusive_CODEXLIVE01.json"
    result = verify()
    print(json.dumps({"output": str(out), "sha256": live.fixed.blind.write_new(out, result)}))
