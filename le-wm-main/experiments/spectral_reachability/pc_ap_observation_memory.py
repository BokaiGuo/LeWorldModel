"""Observation-only AP retrieval core, based on arXiv:2609.30036v4.

This is not the authors' implementation or a complete AP-CEM controller.
It accepts precomputed frozen image features, never actions or simulator state.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class Retrieval:
    episode_id: str
    start: int
    horizon: int
    target: np.ndarray
    squared_distance: float


class ObservationMemory:
    def __init__(self, episodes: Mapping[str, np.ndarray], excluded_query_ids: Sequence[str], block: int = 5):
        if not isinstance(block, int) or block < 1:
            raise ValueError("block must be a positive integer")
        if not episodes or not excluded_query_ids:
            raise ValueError("memory and explicit query exclusion are required")
        if any(not isinstance(k, str) or not k for k in episodes):
            raise ValueError("episode IDs must be nonempty strings")
        excluded = set(excluded_query_ids)
        if not all(isinstance(x, str) and x for x in excluded):
            raise ValueError("query IDs must be nonempty strings")
        if excluded.intersection(episodes):
            raise ValueError("query episode present in retrieval memory")
        self.block = block
        self._episodes = {}
        dimension = None
        for episode_id, features in sorted(episodes.items()):
            value = np.asarray(features, dtype=np.float64)
            if value.ndim != 2 or value.shape[0] <= block or value.shape[1] == 0 or not np.isfinite(value).all():
                raise ValueError("invalid episode features")
            if dimension is None:
                dimension = value.shape[1]
            if value.shape[1] != dimension:
                raise ValueError("inconsistent feature dimension")
            value = value.copy()
            value.setflags(write=False)
            self._episodes[episode_id] = value
        self._dimension = dimension
        self._indices = {}

    def _index(self, horizon: int):
        if horizon not in self._indices:
            keys, targets, locations = [], [], []
            for episode_id, value in self._episodes.items():
                count = len(value) - max(horizon, self.block)
                if count <= 0:
                    continue
                starts = value[:count]
                ends = value[horizon:horizon + count]
                keys.append(np.concatenate((starts, ends, ends - starts), axis=1))
                targets.append(value[self.block:self.block + count])
                locations.extend((episode_id, i) for i in range(count))
            if not keys:
                raise ValueError("no eligible memory records at retrieval horizon")
            keys = np.concatenate(keys)
            self._indices[horizon] = (
                keys,
                np.concatenate(targets),
                np.maximum(keys.std(axis=0, ddof=0), 1e-4),
                locations,
            )
        return self._indices[horizon]

    def retrieve(self, current: np.ndarray, goal: np.ndarray, goal_offset: int, elapsed: int) -> Retrieval:
        if (not isinstance(goal_offset, int) or not isinstance(elapsed, int)
                or goal_offset < 0 or elapsed < 0):
            raise ValueError("offset and elapsed must be nonnegative integers")
        current = np.asarray(current, dtype=np.float64)
        goal = np.asarray(goal, dtype=np.float64)
        if (current.shape != (self._dimension,) or goal.shape != current.shape
                or not np.isfinite(current).all() or not np.isfinite(goal).all()):
            raise ValueError("invalid query features")
        horizon = max(self.block, goal_offset - elapsed)
        keys, targets, std, locations = self._index(horizon)
        query = np.concatenate((current, goal, goal - current))
        distances = np.sum(((keys - query) / std) ** 2, axis=1)
        index = int(np.argmin(distances))
        episode_id, start = locations[index]
        return Retrieval(episode_id, start, horizon, targets[index].copy(), float(distances[index]))
