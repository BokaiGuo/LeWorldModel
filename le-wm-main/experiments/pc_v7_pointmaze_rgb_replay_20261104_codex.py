"""Run established PointMaze reference replays on v7 artifacts."""
import sys

from experiments import pc_v5_pointmaze_prediction_replay_20261104_codex as prediction
from experiments import pc_v5_pointmaze_rgb_frame_replay_20261104_codex as frames
from experiments import pc_v7_pointmaze_rgb_live_dev_20261104_codex as configured


def main(mode):
    if mode == "prediction":
        prediction.configured = configured
        prediction.PROTOCOL = configured.PROTOCOL
        prediction.AUDIT = configured.PREFIX.with_name(configured.PREFIX.name + "_audit.json")
        prediction.OUTPUT = configured.PREFIX.with_name(configured.PREFIX.name + "_prediction_replay.json")
        prediction.main()
    else:
        frames.configured = configured
        frames.PROTOCOL = configured.PROTOCOL
        frames.OUT = configured.PREFIX.with_name(configured.PREFIX.name + "_rgb_frame_replay.json")
        frames.main()


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in ("prediction", "frames"):
        raise SystemExit("choose prediction or frames")
    main(sys.argv[1])
