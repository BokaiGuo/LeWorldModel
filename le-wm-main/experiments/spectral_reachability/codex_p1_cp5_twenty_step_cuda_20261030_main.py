"""Exploratory 20-step cp5 planning on the fixed 24-row known-root panel.

This reuses the exact two-step runner's input, model, initialization, and
objective, changing only the iteration budget. It is not an independent
confirmation, a physical result, or an equal-total-compute comparison.
"""
import argparse
from pathlib import Path

from . import codex_p1_cp5_exploratory_cuda_20261030_v1 as base

NAME = "codex_p1_cp5_twenty_step_cuda_20261030_main"
STEPS = 20
CONFIG = {"steps": STEPS, "learning_rate": 0.05, "history_size": 3}
_base_contract = base.make_contract
_base_bindings = base.bindings


def install():
    """Pin the adapter itself and install the higher-iteration contract."""
    base.NAME = NAME
    base.CONFIG = dict(CONFIG)

    def bindings():
        result = _base_bindings()
        result[str(Path(__file__).resolve())] = base.sha(Path(__file__).resolve())
        return result

    def make_contract(row, sources):
        policy, contract = _base_contract(row, sources)
        caps = contract["levels"][0]["caps"]
        # Two model rollout packets per gradient step and two in final replay.
        for axis in ("action_encode_calls", "predict_calls", "predict_samples"):
            caps[axis] = 2 * STEPS + 2
        for axis in ("backward_calls", "vjp_rows"):
            caps[axis] = STEPS
        contract["levels"][1]["caps"] = {key: value + 1 for key, value in caps.items()}
        return policy, contract

    base.bindings = bindings
    base.make_contract = make_contract


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print("Exploratory fixed-panel comparison: 20 vs 2 gradient steps; no physical claim.", flush=True)
    install()
    return base.run(args.output)


if __name__ == "__main__":
    raise SystemExit(main())
