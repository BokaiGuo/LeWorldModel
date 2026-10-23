"""Audit whether the PointMaze terminal target is geometrically well posed.

This is a protocol audit, not a controller or a replacement task definition.
The legacy evaluator measures Euclidean distance to a qpos target, while the
particle has a non-zero collision radius and all maze walls live in world
coordinates.  A target at a wall contact can make the terminal distance an
unstable objective even when its success ball contains some free states.
"""

from __future__ import annotations

import argparse
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.empirical_spectral_audit.evaluate_pointmaze_adapter_external_evidence import (
    _legacy_pointmaze_xml,
)
from experiments.run_live_pointmaze_planning import GOAL_STATE


SCHEMA = "pointmaze_goal_geometry_audit.v1"


def _vec(value: str, expected: int) -> np.ndarray:
    result = np.fromstring(value, sep=" ", dtype=np.float64)
    if result.shape != (expected,):
        raise ValueError(f"expected {expected} coordinates, received {value!r}")
    return result


def legacy_collision_geometry() -> dict[str, Any]:
    """Return the world-frame collision geometry encoded by the legacy XML."""

    root = ET.fromstring(_legacy_pointmaze_xml())
    particle = root.find("./worldbody/body[@name='particle']")
    if particle is None:
        raise ValueError("legacy XML does not contain the particle body")
    particle_pos = _vec(str(particle.attrib["pos"]), 3)[:2]
    particle_geom = particle.find("./geom[@name='particle_geom']")
    if particle_geom is None:
        raise ValueError("legacy XML does not contain particle_geom")
    particle_radius = float(_vec(str(particle_geom.attrib["size"]), 1)[0])
    walls: list[dict[str, Any]] = []
    for geom in root.findall("./worldbody/geom"):
        name = str(geom.attrib.get("name", ""))
        if not name.startswith("wall_"):
            continue
        if geom.attrib.get("type") != "box":
            raise ValueError(f"legacy wall {name} is not a box")
        walls.append({
            "name": name,
            "center": _vec(str(geom.attrib["pos"]), 3)[:2],
            "half_extent": _vec(str(geom.attrib["size"]), 3)[:2],
        })
    if not walls:
        raise ValueError("legacy XML contains no walls")
    return {
        "particle_body_world_offset": particle_pos,
        "particle_radius": particle_radius,
        "walls": walls,
    }


def clearance_to_box(center: np.ndarray, wall_center: np.ndarray, half_extent: np.ndarray) -> float:
    """Signed disc-to-axis-aligned-box clearance in world qpos units."""

    delta = np.maximum(np.abs(center - wall_center) - half_extent, 0.0)
    return float(np.linalg.norm(delta))


def audit_goal_geometry(
    target_qpos: np.ndarray = GOAL_STATE,
    *,
    success_radius: float = 0.25,
    robust_clearance: float = 0.05,
) -> dict[str, Any]:
    """Measure target clearance without changing the legacy task or results."""

    target = np.asarray(target_qpos, dtype=np.float64).reshape(-1)
    if target.shape != (2,) or not np.isfinite(target).all():
        raise ValueError("target_qpos must be a finite [2] vector")
    if success_radius <= 0.0 or robust_clearance < 0.0:
        raise ValueError("success_radius must be positive and robust_clearance non-negative")
    geometry = legacy_collision_geometry()
    world_target = target + geometry["particle_body_world_offset"]
    radius = float(geometry["particle_radius"])
    candidates = []
    for wall in geometry["walls"]:
        center_distance = clearance_to_box(world_target, wall["center"], wall["half_extent"])
        candidates.append({
            "wall": wall["name"],
            "center_to_box_distance": center_distance,
            "disc_clearance": center_distance - radius,
        })
    nearest = min(candidates, key=lambda item: (float(item["disc_clearance"]), str(item["wall"])))
    clearance = float(nearest["disc_clearance"])
    # The evaluator's success disk reaches a wall whenever its radius exceeds
    # this clearance.  That is not itself an error, but it makes unmasked
    # Euclidean goal ranking geometry-sensitive rather than purely task-based.
    return {
        "schema": SCHEMA,
        "target_qpos": target.tolist(),
        "target_world_position": world_target.tolist(),
        "particle_radius": radius,
        "success_radius": float(success_radius),
        "robust_clearance_requirement": float(robust_clearance),
        "nearest_wall": nearest,
        "target_is_collision_free": bool(clearance > 0.0),
        "target_has_robust_clearance": bool(clearance >= robust_clearance),
        "success_disk_intersects_collision_boundary": bool(clearance < success_radius),
        "interpretation": (
            "legacy_target_near_or_at_wall_contact"
            if clearance < robust_clearance
            else "legacy_target_has_declared_clearance"
        ),
        "protocol_action": (
            "do_not_relabel_legacy_results; report terminal distance as a wall-sensitive legacy metric, "
            "and predeclare a free-region target objective before any new comparison"
            if clearance < robust_clearance
            else "legacy terminal target passes the declared clearance audit"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--success-radius", type=float, default=0.25)
    parser.add_argument("--robust-clearance", type=float, default=0.05)
    args = parser.parse_args()
    report = audit_goal_geometry(
        success_radius=args.success_radius,
        robust_clearance=args.robust_clearance,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
