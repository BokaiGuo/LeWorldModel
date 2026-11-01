"""Verify saved diagnosis arithmetic, source bindings, and inference journal."""
import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np

from . import codex_p1_cp5_tail_rgb_diagnosis_20261101_exclusive as src


def close(a, b):
    src.require(np.isfinite([a, b]).all() and np.isclose(a, b, atol=2e-6, rtol=2e-6), "numeric mismatch")


def verify(prep, prep_sha, directory, report_sha):
    src.require(src.sha(prep) == prep_sha and src.sha(directory / "report.json") == report_sha, "pinned bytes")
    contract = json.loads(prep.read_bytes())
    src.require(contract == src.build(), "source/contract drift")
    report = json.loads((directory / "report.json").read_bytes())
    src.require(report["status"] == "complete" and report["failure"] is None
                and report["prepare_sha256"] == prep_sha and report["counts"] == report["returns"] == src.CAPS
                and report["model_before"] == report["model_after"] == contract["model_hash"]
                and report["physical_calls"] == 0, "run status/accounting")
    ledger = directory / "events.jsonl"
    src.require(src.sha(ledger) == report["events_sha256"], "ledger hash")
    events = [json.loads(line) for line in ledger.read_bytes().splitlines()]
    src.require(len(events) == 2 * sum(src.CAPS.values()), "event count")
    counts = Counter()
    for a, b in zip(events[::2], events[1::2], strict=True):
        src.require(a["event"] == "attempt" and b["event"] == "return" and
                    a["resource"] == b["resource"] and a["context"] == b["context"], "attempt return")
        counts[a["resource"]] += 1
    src.require(dict(counts) == src.CAPS, "ledger caps")
    goal = np.asarray(report["result"]["goal_latent"], np.float64)
    src.require(goal.shape == (192,) and np.array_equal(goal, np.asarray(contract["task"]["goal"])), "goal latent")
    arms = report["result"]["arms"]
    src.require(set(arms) == {"hold", "optimized"}, "arm roster")
    summaries = {}
    for arm in ("hold", "optimized"):
        item = arms[arm]
        frames = contract["arms"][arm]["frames"]
        rows = item["stages"]
        src.require([r["step"] for r in rows] == list(range(5, 16)), "step roster")
        for point, row in zip(frames, rows, strict=True):
            step = row["step"]
            actual = np.asarray(item["actual_latents"][str(step)], np.float64)
            src.require(actual.shape == (192,) and np.isfinite(actual).all()
                        and row["pixel_sha256"] == point["pixel_sha256"]
                        and row["physical"] == src.physical(point["state"], point["goal"]), "frame/physical")
            close(row["actual_rgb_goal_mse"], np.mean((actual-goal)**2))
            prediction = row["model_prediction"]
            if step in (10, 15):
                pred = np.asarray(item[f"predicted_cp{step}"], np.float64)
                src.require(pred.shape == (192,) and prediction["target_step"] == step
                            and prediction["mode"] == "cp5_open_loop", "prediction timestamp")
                close(prediction["goal_mse"], np.mean((pred-goal)**2))
                close(prediction["actual_latent_rmse"], np.sqrt(np.mean((pred-actual)**2)))
            else:
                src.require(prediction is None, "intermediate prediction fabricated")
            if step == 15:
                pred = np.asarray(item["own_cp10_predicted_cp15"], np.float64)
                close(row["own_cp10_refresh_prediction"]["goal_mse"], np.mean((pred-goal)**2))
                close(row["own_cp10_refresh_prediction"]["actual_latent_rmse"], np.sqrt(np.mean((pred-actual)**2)))
                close(row["model_prediction"]["goal_mse"], contract["recorded_cp5_q2"][arm])
        summaries[arm] = [r for r in rows if r["step"] in (5, 10, 15)]
    src.check_cp5(contract["arms"]["hold"]["frames"][0], contract["arms"]["optimized"]["frames"][0])
    src.require(not report["result"]["cp10_rgb_same"], "expected distinct own cp10")
    return {"status": "verified_saved_source_ledger_arithmetic_no_inference_replay",
            "prepared_sha256": prep_sha, "report_sha256": report_sha, "counts": dict(counts),
            "arms": summaries, "independent_confirmation": False, "P1_admission": False,
            "note": "arithmetic verifies saved latents, not independent re-execution of encoder or predictor"}


def main():
    parser = argparse.ArgumentParser()
    for key in ("prep", "directory", "output"):
        parser.add_argument("--" + key, type=Path, required=True)
    for key in ("prep-sha", "report-sha"):
        parser.add_argument("--" + key, required=True)
    args = parser.parse_args()
    result = verify(args.prep, args.prep_sha, args.directory, args.report_sha)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"output": str(args.output), "sha256": src.sha(args.output)}))


if __name__ == "__main__":
    main()
