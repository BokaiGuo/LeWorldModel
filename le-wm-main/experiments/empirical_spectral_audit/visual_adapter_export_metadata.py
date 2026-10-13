"""Dependency-light metadata helpers for visual-adapter exports."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any


def architecture_name(saved: dict[str, Any], checkpoint: Path) -> str:
    """Recover the architecture from both final and snapshot checkpoint paths."""
    if checkpoint.parent.name == "snapshots":
        name = checkpoint.parent.parent.parent.name
    else:
        name = checkpoint.parent.parent.name
    return f"causal_visual_{name}"


def training_stage(checkpoint: Path, saved: dict[str, Any], explicit: int | None) -> int:
    """Prefer an explicit stage, then snapshot filename, then final args."""
    if explicit is not None:
        if explicit < 1:
            raise ValueError("training-stage must be positive")
        return int(explicit)
    match = re.search(r"epoch[_-](\d+)", checkpoint.name)
    if match:
        return int(match.group(1))
    return int(saved.get("epochs", 100))
