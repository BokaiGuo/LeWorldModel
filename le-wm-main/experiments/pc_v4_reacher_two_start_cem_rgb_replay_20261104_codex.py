"""Recompute saved baseline predictions or replay its physical actions."""
import sys

from experiments import pc_v4_reacher_two_start_cem_rgb_20261104_codex as run
from experiments import pc_response_temporal_v3_reacher_prediction_replay_20261104_codex as model_replay
from experiments import pc_response_temporal_v3_reacher_rgb_physical_replay_20261104_codex as physical_replay


def main():
    if len(sys.argv) != 2 or sys.argv[1] not in ("model", "physical"):
        raise SystemExit("choose model or physical")
    run.configure()
    replay = model_replay if sys.argv[1] == "model" else physical_replay
    suffix = "_prediction_replay.json" if sys.argv[1] == "model" else "_physical_replay.json"
    replay.OUT = run.PREFIX.with_name(run.PREFIX.name + suffix)
    replay.AUDIT = run.PREFIX.with_name(run.PREFIX.name + "_audit.json")
    replay.main()


if __name__ == "__main__":
    main()
