"""Reference PointMaze RGB and model replays for the own-carry CEM arm."""
import sys

from experiments import pc_v7_pointmaze_rgb_replay_20261104_codex as replay
from experiments import pc_v7_pointmaze_matched_warm_cem_dev_20261104_codex as configured

if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in ("prediction", "frames"):
        raise SystemExit("choose prediction or frames")
    replay.configured = configured
    replay.main(sys.argv[1])
