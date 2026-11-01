"""Recompute the saved normalizer sensitivity scores and action-token binding."""
import argparse
import json
from pathlib import Path

import numpy as np

from . import p1_cp10_normalizer_rank_cuda_20261031 as run


def check_tokens(tokens, raw_tokens, mean, std):
    tokens = np.asarray(tokens, dtype=np.float64)
    expected = ((np.asarray(raw_tokens, np.float32) - np.asarray(mean, np.float32)) /
                np.asarray(std, np.float32)).reshape(3, 10)
    run.need(tokens.shape == (3, 10) and np.allclose(tokens, expected, rtol=1e-6, atol=1e-6),
             "raw action token binding")


def verify(path, digest):
    need, sha = run.need, run.sha
    need(sha(path) == digest, "result SHA")
    result = run.source.feedback_plan.source.physical.h.read(path)
    need(result["schema"] == run.NAME and result["status"] == "complete"
         and len(result["rows"]) == 2 and result["host"]["device"] == "cuda:0"
         and result["model_before"] == result["model_after"], "result identity")
    need(all(sha(p) == h for p, h in result["sources"].items()), "source drift")
    reader = run.source.feedback_plan.source.physical.h.read
    ref, old, new, bundle = (reader(run.REFERENCE), reader(run.source.feedback_plan.source.PREPARE),
                             reader(run.source.FEEDBACK_PREPARE),
                             reader(run.source.feedback_plan.source.physical.BUNDLE))
    summaries = []
    for i, (row, previous, old_item, new_item, b) in enumerate(zip(
            result["rows"], ref["rows"], old["items"], new["items"], bundle["rows"], strict=True)):
        need(row["root_slot"] == previous["root_slot"] == i, "root order")
        goal = np.asarray(row["goal"], dtype=np.float64)
        need(goal.shape == (192,) and np.isfinite(goal).all() and
             np.array_equal(goal, np.asarray(previous["goal"], dtype=np.float64)), "goal identity")
        blind = b["blind"]
        prefix = np.frombuffer(bytes.fromhex(blind["prefix_raw_f32_hex"]), "<f4").reshape(5, 2)
        old_raw = np.frombuffer(bytes.fromhex(old_item["raw10_float32_hex"]), "<f4").reshape(2, 5, 2)
        new_raw = np.frombuffer(bytes.fromhex(new_item["raw10_float32_hex"]), "<f4").reshape(2, 5, 2)
        need(old_raw[0].tobytes() == new_raw[0].tobytes(), "first packet identity")
        settings = {
            "deployed": (blind["action_mean"], blind["action_std"]),
            "archive": (result["normalizers"]["archive_mean"], result["normalizers"]["archive_std"]),
        }
        for label, raw in (("open", old_raw), ("feedback", new_raw)):
            actual = np.asarray(row["actual"][label], dtype=np.float64)
            need(actual.shape == (192,) and np.array_equal(actual, previous["arms"][label]["actual"]),
                 "actual latent binding")
            need(row["arms"][label]["action_sha256"] == previous["arms"][label]["action_sha256"],
                 "action SHA")
            raw_tokens = np.stack((prefix, old_raw[0], raw[1]))
            for name, (mean, std) in settings.items():
                value = row["arms"][label]["variants"][name]
                pred = np.asarray(value["prediction"], dtype=np.float64)
                need(pred.shape == (192,) and np.isfinite(pred).all(), "finite prediction")
                check_tokens(value["tokens"], raw_tokens, mean, std)
                for key, other in (("predicted_goal_mse", goal), ("prediction_actual_mse", actual)):
                    need(np.isclose(np.square(pred-other).mean(), value[key], rtol=1e-6, atol=1e-7),
                         "MSE arithmetic")
        summaries.append({"root_slot": i, **{
            name + "_feedback_minus_open":
                row["arms"]["feedback"]["variants"][name]["predicted_goal_mse"] -
                row["arms"]["open"]["variants"][name]["predicted_goal_mse"]
            for name in ("deployed", "archive")}})
    need(sha(path) == digest, "result drift")
    return {"schema": "p1_cp10_normalizer_rank_saved_20261031", "status": "saved_arithmetic_passed",
            "result_sha256": digest, "rows": summaries, "P1_admission": False,
            "limitations": "Saved vectors, not independent inference; archive normalizer is not certified."}


def main():
    parser = argparse.ArgumentParser()
    for option in ("input", "sha256", "output"):
        parser.add_argument("--" + option, required=True)
    args = parser.parse_args()
    out = Path(args.output)
    run.need(out.is_absolute() and out.parent == run.ROOT / "results" and
             out.name.startswith("p1_cp10_normalizer_rank_saved_20261031_") and not out.exists(), "exclusive")
    result = verify(args.input, args.sha256)
    print(json.dumps({"status": result["status"], "sha256": run.base.write(out, result), "rows": result["rows"]}))


if __name__ == "__main__":
    main()
