"""Materialize RGB calibration frames for fresh PointMaze state roots.

The state/action extension deliberately shares only the original RGB archive
for fixed-renderer evaluation.  A subsequent experiment may need disjoint RGB
calibration roots beyond that archive's final episode.  This utility creates
only those calibration observations from a fixed renderer fit on declared
reference RGB roots; it neither changes trajectories nor reads planning
outcomes.  States/actions remain symlinks to the immutable source extension.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from experiments.run_live_pointmaze_planning import (
    DatasetMatchedRenderer,
    fit_visual_calibration,
    load_episode,
)


SCHEMA = "pointmaze.renderer_rgb_calibration_extension.v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_roots(value: str, label: str) -> list[int]:
    try:
        roots = [int(token.strip()) for token in value.split(",") if token.strip()]
    except ValueError as error:
        raise ValueError(f"{label} must be comma-separated integer root IDs") from error
    if not roots or len(roots) != len(set(roots)) or any(root < 0 for root in roots):
        raise ValueError(f"{label} must be a non-empty duplicate-free non-negative root list")
    return roots


def materialize(
    *,
    source_root: Path,
    reference_root: Path,
    output: Path,
    render_roots: list[int],
    reference_roots: list[int],
) -> dict[str, object]:
    """Create one immutable RGB-only calibration view over fresh states."""

    source_root = source_root.resolve()
    reference_root = reference_root.resolve()
    output = output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite RGB calibration extension: {output}")
    for path in (source_root / "states.pth", source_root / "actions.pth", source_root / "seq_lengths.pth"):
        if not path.is_file():
            raise FileNotFoundError(f"missing source tensor: {path}")
    for root in reference_roots:
        episode = reference_root / "obses" / f"episode_{root:03d}.pth"
        if not episode.is_file():
            raise FileNotFoundError(f"missing reference RGB episode: {episode}")
    states = torch.load(source_root / "states.pth", map_location="cpu", weights_only=True)
    if states.ndim != 3 or states.shape[1:] != (100, 4):
        raise ValueError("source states must have shape [roots,100,4]")
    if any(root >= len(states) for root in render_roots):
        raise ValueError("requested rendered root lies outside source states")

    calibration = fit_visual_calibration(
        reference_root, reference_roots, max_frames=1600,
    )
    reference_images, _ = load_episode(reference_root, reference_roots[0])
    renderer = DatasetMatchedRenderer.from_episode(reference_images, calibration)

    output.mkdir(parents=True, exist_ok=False)
    for name in ("states.pth", "actions.pth", "seq_lengths.pth"):
        (output / name).symlink_to(source_root / name)
    image_dir = output / "obses"
    image_dir.mkdir()
    rendered_paths: list[Path] = []
    states_np = states.detach().cpu().numpy().astype(np.float64)
    for root in render_roots:
        images = np.stack([renderer.render(state) for state in states_np[root]], axis=0)
        destination = image_dir / f"episode_{root:03d}.pth"
        torch.save(torch.from_numpy(images), destination)
        rendered_paths.append(destination)
    report = {
        "schema": SCHEMA,
        "source_root": str(source_root),
        "reference_root": str(reference_root),
        "source_states_sha256": _sha256(source_root / "states.pth"),
        "source_actions_sha256": _sha256(source_root / "actions.pth"),
        "reference_roots": [int(root) for root in reference_roots],
        "rendered_roots": [int(root) for root in render_roots],
        "rendered_episode_count": int(len(rendered_paths)),
        "renderer": "DatasetMatchedRenderer calibrated on declared reference RGB roots only",
        "planning_outcomes_read": False,
        "stored_state_action_mode": "immutable_symlink_to_source_extension",
        "rgb_frame_sha256": {path.name: _sha256(path) for path in rendered_paths},
    }
    (output / "rgb_calibration_manifest.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8",
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--render-roots", required=True)
    parser.add_argument("--reference-roots", required=True)
    args = parser.parse_args()
    report = materialize(
        source_root=args.source_root,
        reference_root=args.reference_root,
        output=args.output,
        render_roots=parse_roots(args.render_roots, "render roots"),
        reference_roots=parse_roots(args.reference_roots, "reference roots"),
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
