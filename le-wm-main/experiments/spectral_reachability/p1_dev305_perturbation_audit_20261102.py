"""Inspect pixel effects of the already sealed current-RGB perturbations."""
import argparse
from pathlib import Path

import numpy as np

from . import p1_dev305_component_tangent_vectors_20261102 as source

base, need, ROOT = source.base, source.need, source.ROOT
CONTRACT_SHA = "37f245aa3ea2d5b73f6de82d931d3ca3acd6268ab3b4fd7336c1bfc70cf3bebb"


def centroid(mask):
    y, x = np.nonzero(mask)
    if not len(x):
        raise ValueError("empty mask")
    return np.array([x.mean(), y.mean()])


def pixel_effects(pixels, axis, amount):
    masks = source.visibility.masks(pixels)
    target = "object" if axis.startswith("object") else "pusher"
    other = "pusher" if target == "object" else "object"
    changed = source.tangent.perturb(pixels, axis, amount)
    new_masks = source.visibility.masks(changed)
    difference = np.any(changed != pixels, axis=-1)
    old_center, new_center = centroid(masks[target]), centroid(new_masks[target])
    return {"axis": axis, "amount": amount,
            "target_pixel_count_before": int(masks[target].sum()),
            "target_pixel_count_after": int(new_masks[target].sum()),
            "target_centroid_delta_xy": (new_center - old_center).tolist(),
            "other_entity_changed_pixels": int((difference & masks[other]).sum()),
            "other_entity_pixels_before": int(masks[other].sum()),
            "other_entity_pixels_after": int(new_masks[other].sum()),
            "changed_pixels_outside_original_target": int((difference & ~masks[target]).sum()),
            "changed_pixels_total": int(difference.sum())}


def run():
    spec = base.read(ROOT / "results" / f"{source.NAME}_PREP_DEV01.json", CONTRACT_SHA)
    need(spec == source.protocol(), "protocol binding")
    rows = []
    for role in spec["roles"]:
        pixels = source.run.coherent.base.frame(role["images"][-1])
        need(source.visibility.inspect(pixels)["eligible"], "visibility")
        rows.append({"arm": role["arm"],
                     "effects": [pixel_effects(pixels, axis, sign)
                                 for axis in source.tangent.AXES for sign in (-1, 0, 1)]})
    return {"schema": "p1_dev305_perturbation_audit_20261102.v1", "contract_sha256": CONTRACT_SHA,
            "rows": rows, "scope": "RGB mask effects only; does not certify physical pose or occlusion",
            "independent_confirmation": False}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    print(base.write_new(parser.parse_args().output, run()))
