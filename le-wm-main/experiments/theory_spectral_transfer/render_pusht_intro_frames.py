"""Render two deterministic PushT simulator frames for the Introduction figure."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import cv2
import numpy as np

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

from stable_worldmodel.envs.pusht import PushT


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        default="results/theory_operator_alignment_iclr_oral_20260829/figures",
    )
    parser.add_argument("--resolution", type=int, default=256)
    return parser.parse_args()


def render_state(
    env: PushT,
    *,
    state: np.ndarray,
    goal_state: np.ndarray,
    seed: int,
) -> np.ndarray:
    env.reset(
        seed=seed,
        options={
            "state": state.astype(np.float64),
            "goal_state": goal_state.astype(np.float64),
        },
    )
    return env.render()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    env = PushT(
        resolution=args.resolution,
        render_mode="rgb_array",
        with_target=True,
        render_action=False,
    )
    goal = np.array([256, 420, 256, 218, np.pi / 2, 0, 0], dtype=np.float64)
    state_t = np.array([150, 390, 285, 315, 0.18, 0, 0], dtype=np.float64)
    state_t1 = np.array([188, 350, 271, 278, 0.43, 0, 0], dtype=np.float64)

    try:
        frames = {
            "pusht_intro_ot.png": render_state(env, state=state_t, goal_state=goal, seed=13),
            "pusht_intro_ot1.png": render_state(env, state=state_t1, goal_state=goal, seed=13),
        }
    finally:
        env.close()

    for name, frame in frames.items():
        path = out_dir / name
        ok = cv2.imwrite(str(path), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        if not ok:
            raise RuntimeError(f"Failed to write {path}")
        print(path)


if __name__ == "__main__":
    main()
