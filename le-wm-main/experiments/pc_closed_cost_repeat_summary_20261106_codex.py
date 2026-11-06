"""Validate repeated cost traces and summarize nested timing samples."""
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "results"
ARMS = ("cem", "prefix", "cem300x10", "cem300x30")
PRIOR = {"cem": "cem_trace_r2", "prefix": "prefix_trace",
         "cem300x10": "cem300x10_trace", "cem300x30": "cem300x30_trace"}


def read_lines(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def sorted_trace(path):
    return sorted(read_lines(path), key=lambda x: (x["reset_seed"], x["step"]))


def main():
    output = ROOT / "pc_cost_split_repeated_summary_20261106_codex.json"
    if output.exists():
        raise FileExistsError(output)
    summary = {}
    initial = None
    for arm in ARMS:
        original = sorted_trace(ROOT / f"pc_closed_cost_{PRIOR[arm]}_20261106_codex.jsonl")
        samples = []
        for repeat in (1, 2, 3):
            stem = f"pc_cost_split_{arm}_r{repeat}"
            profile = json.loads((ROOT / f"{stem}.profile.json").read_text())["records"]
            gpu = json.loads((ROOT / f"{stem}.gpu.json").read_text())["events"]
            trace = sorted_trace(ROOT / f"{stem}.trace.jsonl")
            metrics = json.loads((ROOT / f"{stem}.run/eval_metrics.json").read_text())
            starts = {r["reset_seed"]: r["before"] for r in trace if r["step"] == 1}
            if len(starts) != 10:
                raise ValueError(f"{stem}: expected 10 initial states")
            if initial is None:
                initial = starts
            if starts != initial:
                raise ValueError(f"{stem}: initial state mismatch")
            if any(abs(v) > 1.000001 for r in trace for v in r["action"]):
                raise ValueError(f"{stem}: illegal action")
            samples.append({"successes": sum(metrics["episode_successes"]),
                            "trace_exact_prior": trace == original,
                            "trace_rows": len(trace), "records": profile, "gpu_events": gpu})
        def median(label):
            return statistics.median(s["records"][label]["wall_s"] for s in samples)
        summary[arm] = {
            "samples": samples,
            "median_wall_s": {key: median(key) for key in (
                "whole_evaluation", "solver", "observation_preprocessing",
                "env_pool_step_including_render", "pusht_step_including_observation_reward",
                "pymunk_physics_step", "pusht_render_frame")},
        }
    with output.open("x") as stream:
        json.dump({"runs": summary, "initial_physical_snapshots_equal": True,
                   "scope": "three sequential repeats on the same ten tasks and fixed solver seed",
                   "limitations": "nested intervals, trace/video overhead, non-independent repeats; GPU events contain stream idle"},
                  stream, indent=2, allow_nan=False)


if __name__ == "__main__":
    main()
