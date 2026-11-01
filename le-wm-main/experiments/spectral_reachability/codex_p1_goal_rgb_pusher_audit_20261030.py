"""Inspect sealed RGB goal/pusher visibility without changing a planner or running a model."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from . import p1_backtrack_physical_runner_20261030 as prior


def blue_summary(image):
    image = np.asarray(image)
    prior.require(image.dtype == np.uint8 and image.shape == (224, 224, 3), "RGB contract")
    mask = np.all(image == [65, 105, 225], axis=-1)
    yy, xx = np.nonzero(mask)
    return dict(count=int(len(xx)), centroid_xy=[float(xx.mean()), float(yy.mean())] if len(xx) else None,
                bounds_xyxy=[int(xx.min()), int(yy.min()), int(xx.max()), int(yy.max())] if len(xx) else None,
                pixel_sha256=prior.native.moving.image_sha(image))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output
    prior.require(output.is_absolute() and output.parent == prior.ROOT / "results"
                  and output.name.startswith("codex_p1_goal_rgb_pusher_audit_20261030_")
                  and not output.exists() and not output.is_symlink(), "exclusive output")
    prepared, data = prior.material()
    rows = []
    for root in data["roots"]:
        for condition in ("off", "on"):
            packet = root["packets"][condition]
            rows.append(dict(root_slot=root["slot"], condition=condition,
                             goal_state=prior.native.source.vector(root["goal"]).tolist(),
                             goal=blue_summary(packet["goal_rgb"]),
                             current=blue_summary(packet["rgb_history"][-1])))
    prior.require(all(prior.sha(p) == h for p, h in prepared["bindings"].items()), "source drift")
    result = dict(schema="p1_goal_rgb_pusher_audit.v1", rows=rows,
                  interpretation="exact-color visibility only; not encoder sensitivity or planner input",
                  source_join_sha256=prior.JOIN_SHA, new_neural_calls=0, P1_admission=False)
    raw = (json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    with output.open("xb") as stream:
        prior.require(stream.write(raw) == len(raw), "full output")
    print(json.dumps(dict(output=str(output), sha256=hashlib.sha256(raw).hexdigest())))


if __name__ == "__main__":
    main()
