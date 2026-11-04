"""Reference-model or same-backend replay for the matched warm CEM."""

import sys

from experiments import pc_v7_reacher_matched_warm_cem_dev_20261104_codex as configured

if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in ("prediction", "physical"):
        raise SystemExit("choose prediction or physical")
    configured.configure()
    if sys.argv[1] == "prediction":
        from experiments import pc_response_temporal_v3_reacher_prediction_replay_20261104_codex as replay
        replay.PROTOCOL = configured.PROTOCOL
        replay.OUT = configured.PREFIX.with_name(configured.PREFIX.name + "_prediction_replay.json")
    else:
        from experiments import pc_response_temporal_v3_reacher_rgb_physical_replay_20261104_codex as replay
        replay.OUT = configured.PREFIX.with_name(configured.PREFIX.name + "_physical_replay.json")
    replay.AUDIT = configured.PREFIX.with_name(configured.PREFIX.name + "_audit.json")
    replay.main()
