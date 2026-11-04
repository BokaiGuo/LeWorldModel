"""Replay each arm's frozen batches using the explicit-packet reference."""
from types import SimpleNamespace

from experiments import pc_v22_pointmaze_live_feedback_v2_20261104_codex as live
from experiments import pc_v5_pointmaze_prediction_replay_20261104_codex as reference

PROTOCOL = live.ROOT.parent / "docs/PC_V22_POINTMAZE_PREDICTION_REPLAY_DEV_PROTOCOL_20261104_CODEX.md"


class ArmView:
    def __init__(self, arm):
        self.arm = arm

    def paths(self, _unused):
        return live.paths(self.arm)

    def __getattr__(self, name):
        return getattr(live, name)


def main():
    outputs = [live.PREFIX.with_name(live.PREFIX.name + f"_{arm}_prediction_replay.json")
               for arm in live.ARMS]
    if any(path.exists() for path in outputs):
        raise FileExistsError("prediction replay already exists or is incomplete")
    for arm, output in zip(live.ARMS, outputs, strict=True):
        reference.configured = SimpleNamespace(configure=lambda: None)
        reference.run = ArmView(arm)
        reference.PROTOCOL = PROTOCOL
        reference.AUDIT = live.PREFIX.with_name(live.PREFIX.name + "_audit.json")
        reference.OUTPUT = output
        print(f"Explicit-packet replay: {arm}", flush=True)
        reference.main()


if __name__ == "__main__":
    main()
