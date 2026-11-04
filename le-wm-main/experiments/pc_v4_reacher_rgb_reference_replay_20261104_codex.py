"""Reference-model replay of every saved v4 RGB feedback prediction."""
from experiments import pc_response_temporal_v3_reacher_prediction_replay_20261104_codex as replay
from experiments import pc_v4_reacher_rgb_live_20261104_codex as run


if __name__ == "__main__":
    run.configure()
    replay.OUT = run.PREFIX.with_name(run.PREFIX.name + "_prediction_replay.json")
    replay.AUDIT = run.PREFIX.with_name(run.PREFIX.name + "_audit.json")
    replay.PROTOCOL = run.PROTOCOL
    replay.main()
