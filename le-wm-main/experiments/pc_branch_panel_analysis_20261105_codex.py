"""Summarize the preregistered panel and audit replan/logging parity."""
import argparse
import json
from pathlib import Path

from experiments.pc_branch_full_pool_analysis_20261105_codex import run


def main(offset5=False):
    root = Path(__file__).resolve().parents[1] / "results"
    output = root / ("pc_branch_panel_offset5_failure_decomposition_dev_20261106_codex.json" if offset5 else
                     "pc_branch_panel_failure_decomposition_dev_20261105_codex.json")
    if output.exists():
        raise FileExistsError(output)
    summary = {"tasks": [5, 15, 25, 35, 45] if offset5 else [0, 10, 20, 30, 40], "arms": {}}
    for mode in ("lookahead", "effect"):
        stem = (f"pc_branch_panel_offset5_{mode}_seed42_20261106_codex" if offset5 else
                f"pc_branch_panel_{mode}_seed42_20261105_codex")
        original = f"pc_branch_effect_{mode}_task42_solver42_20261105_codex"
        same_trace = (root / f"{stem}.trace.jsonl").read_bytes() == (root / f"{original}.trace.jsonl").read_bytes()
        if not same_trace:
            raise ValueError("capture changed full execution trace")
        metrics = json.loads((root / stem / "eval_metrics.json").read_text())
        original_metrics = json.loads((root / original / "eval_metrics.json").read_text())
        if metrics != original_metrics or len(metrics["episode_successes"]) != 50:
            raise ValueError("capture changed metrics")
        replans = [json.loads(line) for line in (root / f"{stem}.capture/replanning.jsonl").read_text().splitlines()]
        if len(replans) != 2 or replans[0]["warm_start"] is not None or replans[1]["warm_start"] is not None:
            raise ValueError("unexpected replan contract")
        tasks = [run(mode, task, True, offset5) for task in summary["tasks"]]
        summary["arms"][mode] = {
            "execution_trace_byte_identical": True, "closed_loop_success_50": sum(metrics["episode_successes"]),
            "replan_count": len(replans[1]["selected_actions"]), "warm_start_absent": True,
            "tasks": [{"task": t["task"], "pool_successes": t["successes"],
                       "best_success_rank": t["first_success_score_rank_1based"],
                       "best_model_succeeds": t["lowest_score_succeeds"],
                       "selected_first_success": t["selected_first_success"],
                       "closed_loop_success": metrics["episode_successes"][t["task"]],
                       "rounds": t["rounds"]} for t in tasks],
        }
    with output.open("x", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2, allow_nan=False)
    print(json.dumps(summary, indent=2, allow_nan=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--offset5", action="store_true")
    main(parser.parse_args().offset5)
