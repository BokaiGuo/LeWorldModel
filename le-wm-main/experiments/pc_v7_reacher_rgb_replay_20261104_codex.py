"""Reference prediction or same-backend physical replay for v7 Reacher."""

import sys

from experiments import pc_v7_reacher_rgb_live_20261104_codex as run


def main(mode):
    run.configure()
    if mode == "prediction":
        from experiments import pc_response_temporal_v3_reacher_prediction_replay_20261104_codex as replay
        replay.PROTOCOL = run.PROTOCOL
        replay.OUT = run.PREFIX.with_name(run.PREFIX.name + "_prediction_replay.json")
    else:
        from experiments import pc_response_temporal_v3_reacher_rgb_physical_replay_20261104_codex as replay
        replay.OUT = run.PREFIX.with_name(run.PREFIX.name + "_physical_replay.json")
    replay.AUDIT = run.PREFIX.with_name(run.PREFIX.name + "_audit.json")
    replay.main()


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in ("prediction", "physical"):
        raise SystemExit("choose prediction or physical")
    main(sys.argv[1])
