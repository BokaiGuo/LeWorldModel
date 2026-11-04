"""Verify policy provenance then replay sealed multi-nominal RGB trajectories."""
import json
from pathlib import Path

from experiments import pc_fixed_terminal_fair_feedback_physical_20261103_codex as replay
from experiments import pc_multi_nominal_rgb_dev_20261103_codex as policy


def main():
    run = replay.run
    run.PROTOCOL = policy.PROTOCOL
    run.PREFIX = policy.PREFIX
    run.ARMS = policy.ARMS
    replay.OUTPUT = policy.PREFIX.with_name(policy.PREFIX.name + "_physical.json")
    sha = run.old.prior.sha
    for root in range(4):
        for arm in policy.ARMS:
            decisions, _, _ = run.paths(root, arm)
            for path in decisions:
                result = json.loads(path.read_text())["result"]
                if (result["policy_sha256"] != sha(Path(policy.__file__))
                        or result["multi_generator_sha256"] != sha(Path(policy.generator.__file__))):
                    raise ValueError("multi-nominal source hash mismatch")
    replay.main()


if __name__ == "__main__":
    main()
