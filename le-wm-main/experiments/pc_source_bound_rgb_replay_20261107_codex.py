"""Apply loaded-source binding to the existing RGB closed-loop replay."""
import importlib

from experiments import pc_bound_rgb_closed_loop_replay_20261107_codex as replay
from experiments.pc_loaded_source_binding_20261107_codex import SourceBoundCheckpointRGB


class BoundForReplay(SourceBoundCheckpointRGB):
    def __init__(self, model, audit, weights, config):
        super().__init__(model, audit, weights, config, [replay.source.ROOT])


def main():
    # Explicitly load runtime modules observed in the failed first integration.
    # They remain inside the snapshot; any further lazy imports still fail.
    for name in ("einops._torch_specific", "stable_worldmodel.envs.two_room",
                 "stable_worldmodel.envs.two_room.env",
                 "stable_worldmodel.envs.two_room.expert_policy"):
        importlib.import_module(name)
    replay.OUT = replay.source.ROOT / "results/pc_source_bound_rgb_replay_20261107_codex.json"
    replay.PROTOCOL = replay.source.ROOT.parent / "docs/PC_SOURCE_BOUND_RGB_REPLAY_PROTOCOL_20261107_CODEX.md"
    replay.CheckpointBoundRGB = BoundForReplay
    replay.main()


if __name__ == "__main__":
    main()
