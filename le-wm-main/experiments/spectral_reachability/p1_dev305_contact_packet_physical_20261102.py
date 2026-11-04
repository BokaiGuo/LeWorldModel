"""Evaluate a fixed RGB contact-direction candidate family at real feedback states."""
import argparse
import hashlib

import numpy as np

from . import p1_dev305_selected_bank_first_feedback_20261102 as rgb_source
from . import p1_dev305_selected_bank_first_feedback_physical_20261102 as previous
from . import p1_dev305_triprefix_midfeedback_physical_20261102 as engine

base, need, ROOT = previous.base, previous.need, previous.ROOT
NAME = "p1_dev305_contact_packet_physical_20261102"
AUTHORITY = "DEV305_CONTACT_PACKET_PHYSICAL"
PHYS_PREP_SHA = "b813af00ba8fe5472c9b9a4066d4d7dd9317897b788e512cfcb435db21cb4d7c"
RGB_PREP_SHA = previous.PREP_SHA


def candidate_packets(direction):
    direction = np.asarray(direction, dtype=np.float64)
    if direction.shape != (2,) or not np.isfinite(direction).all():
        raise ValueError("direction")
    norm = float(np.linalg.norm(direction))
    if not np.isclose(norm, 1.0, atol=1e-5):
        raise ValueError("unit direction")
    return {name: np.repeat(np.asarray(scale * direction, dtype="<f4")[None, :], 5, axis=0)
            for name, scale in (("toward", .5), ("away", -.5), ("zero", 0.0))}


def protocol():
    rgb = base.read(ROOT / "results" / f"{rgb_source.NAME}_PREP_DEV01.json", RGB_PREP_SHA)
    phys = base.read(ROOT / "results" / f"{previous.NAME}_PREP_DEV01.json", PHYS_PREP_SHA)
    need(rgb == rgb_source.protocol() and phys == previous.protocol(), "sources")
    roles = []
    for visual, replay in zip(rgb["roles"], phys["roles"], strict=True):
        need(visual["arm"] == replay["arm"], "role")
        image = rgb_source.run.coherent.base.frame(visual["images"][-1])
        need(rgb_source.visibility.inspect(image)["eligible"], "visibility")
        direction = rgb_source.run.coherent.base.first.engine._geometry(image)["contact_direction_raw_xy"]
        need(np.allclose(direction, visual["contact_direction_raw_xy"], rtol=0, atol=1e-12), "geometry binding")
        for name, action in candidate_packets(direction).items():
            roles.append({"arm": visual["arm"] + "__" + name, "history": replay["history"],
                          "states": replay["states"], "checkpoint": replay["checkpoint"],
                          "tail": action.tolist(), "tail_sha256": hashlib.sha256(action.tobytes()).hexdigest(),
                          "candidate_origin": name})
    need(len(roles) == 9, "roster")
    return {"schema": NAME + ".protocol.v1", "code_sha256": base.sha(__file__),
            "engine_sha256": base.sha(engine.__file__), "rgb_prep_sha256": RGB_PREP_SHA,
            "physical_replay_sha256": PHYS_PREP_SHA, "seed": phys["seed"], "roles": roles,
            "policy": "fixed RGB nearest-boundary direction, opposite and zero; five steps, magnitude 0.5",
            "selection_limit": "no model-based selection, simulator state used only for exact replay and evaluation",
            "classification": "post_outcome_known_root_contact_coverage_development",
            "p1_admission": False, "independent_confirmation": False}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "execute"))
    parser.add_argument("--tag", required=True)
    parser.add_argument("--contract-sha")
    parser.add_argument("--authority")
    args = parser.parse_args()
    need(args.tag.isascii() and args.tag.isalnum(), "tag")
    path = ROOT / "results" / f"{NAME}_PREP_{args.tag}.json"
    if args.command == "prepare":
        print(base.write_new(path, protocol()))
    else:
        engine.protocol = protocol
        engine.AUTHORITY = AUTHORITY
        print(engine.execute(path, args.contract_sha, ROOT / "results" / f"{NAME}_RUN_{args.tag}", args.authority))
