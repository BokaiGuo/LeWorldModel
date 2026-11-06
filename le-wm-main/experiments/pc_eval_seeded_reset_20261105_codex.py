"""Explicitly seed only otherwise-unseeded world resets for controlled evaluation."""
import os
import sys
from pathlib import Path

import hydra
import stable_worldmodel as swm

import eval as native_eval


def install_reset_seed(seed):
    original_reset = swm.World.reset

    def reset_with_fallback(self, seed=None, options=None):
        return original_reset(self, seed=seed if seed is not None else reset_seed, options=options)

    reset_seed = seed
    swm.World.reset = reset_with_fallback


if __name__ == "__main__":
    requested_seed = int(os.environ["PC_PHYSICS_RESET_SEED"])
    print(f"Controlled non-native protocol: missing world reset seed -> {requested_seed}", flush=True)
    install_reset_seed(requested_seed)
    config_dir = str(Path(native_eval.__file__).resolve().parent / "config" / "eval")
    with hydra.initialize_config_dir(version_base=None, config_dir=config_dir):
        configuration = hydra.compose(config_name="pusht", overrides=sys.argv[1:])
        native_eval.run.__wrapped__(configuration)
