"""Controlled seeded evaluation with per-environment action/state tracing."""
import json
import os
import sys
from pathlib import Path

import hydra
import numpy as np
from stable_worldmodel.envs.pusht.env import PushT

import eval as native_eval
from experiments.pc_eval_seeded_reset_20261105_codex import install_reset_seed
from experiments.pc_pusht_reset_hidden_audit_20261105_codex import snapshot


def main():
    reset_seed = int(os.environ["PC_PHYSICS_RESET_SEED"])
    trace_path = Path(os.environ["PC_PHYSICS_TRACE_PATH"])
    trace_path.open("x", encoding="utf-8").close()
    install_reset_seed(reset_seed)
    original_reset, original_step = PushT.reset, PushT.step

    def traced_reset(self, seed=None, options=None):
        observation = original_reset(self, seed=seed, options=options)
        self._pc_trace_seed = seed
        self._pc_trace_step = 0
        return observation

    def traced_step(self, action):
        before = snapshot(self)
        result = original_step(self, action)
        observation, _, terminated, truncated, _ = result
        self._pc_trace_step += 1
        record = {
            "reset_seed": self._pc_trace_seed, "step": self._pc_trace_step,
            "action": np.asarray(action, dtype=np.float64).tolist(),
            "before": before.tolist(), "after": snapshot(self).tolist(),
            "observed_state": np.asarray(observation["state"]).tolist(),
            "terminated": bool(terminated), "truncated": bool(truncated),
        }
        with trace_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, allow_nan=False) + "\n")
        return result

    PushT.reset, PushT.step = traced_reset, traced_step
    print(f"Controlled trace: reset seed={reset_seed}, trace={trace_path}", flush=True)
    config_dir = str(Path(native_eval.__file__).resolve().parent / "config" / "eval")
    with hydra.initialize_config_dir(version_base=None, config_dir=config_dir):
        config = hydra.compose(config_name="pusht", overrides=sys.argv[1:])
        native_eval.run.__wrapped__(config)


if __name__ == "__main__":
    main()
