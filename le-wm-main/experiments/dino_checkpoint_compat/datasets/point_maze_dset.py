"""Reuse the vendored PointMaze loader with subset filename remapping.

The repository stores a 1800--1899 PointMaze tail subset whose tensors are
reindexed to zero but whose observation filenames retain their global episode
ids.  The released loader assumes zero-based filenames; this wrapper preserves
the tensor indexing while resolving the corresponding serialized image file.
"""

from pathlib import Path

import torch
from einops import rearrange

from app.plan_common.datasets.point_maze_dset import PointMazeDataset as _PointMazeDataset
from app.plan_common.datasets.point_maze_dset import load_point_maze_slice_train_val


class PointMazeDataset(_PointMazeDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._obs_files = sorted((Path(self.data_path) / "obses").glob("episode_*.pth"))

    def get_frames(self, idx, frames):
        obs_dir = Path(self.data_path) / "obses"
        direct = obs_dir / f"episode_{int(idx):03d}.pth"
        if direct.exists():
            image_path = direct
        elif int(idx) < len(self._obs_files):
            image_path = self._obs_files[int(idx)]
        else:
            raise FileNotFoundError(
                f"no observation file for episode {idx}; sparse exports must include the requested global episode id"
            )
        image = torch.load(image_path)
        proprio = self.proprios[idx, frames]
        act = self.actions[idx, frames]
        state = self.states[idx, frames]
        image = image[frames]
        image = image / 255.0
        image = rearrange(image, "T H W C -> T C H W")
        if self.transform:
            image = self.transform(image)
        return {"visual": image, "proprio": proprio}, act, state, None, {}

__all__ = ["PointMazeDataset", "load_point_maze_slice_train_val"]
