"""Run the guarded best-iterate midpoint planner with zero trust penalty."""
import argparse
from pathlib import Path

from . import codex_p1_adam_best_iterate_20261030_main as planner

NAME = "codex_p1_unreg_best_20261030_main"


def run(output):
    if not output.is_absolute() or output.parent != planner.base.ROOT / "results" or not output.name.startswith(NAME + "_") or output.exists() or output.is_symlink():
        raise ValueError("exclusive output")
    planner.LAMBDA = 0.0
    planner.NAME = NAME
    planner.run(output)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", required=True)
    print("Run fixed 24-row unregularized best-iterate ablation on host CUDA.", flush=True)
    run(Path(ap.parse_args().output))
