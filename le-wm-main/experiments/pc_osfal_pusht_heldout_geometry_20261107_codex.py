"""Root-disjoint geometry extension of the sealed PushT RGB observation audit."""
import argparse
import itertools
import json
from pathlib import Path

import numpy as np

from experiments import pc_geometric_contact_equivalence_dev_20261106_codex as physical
from experiments import pc_osfal_pusht_layer_dev_20261107_codex as layer
from experiments.pc_input_permissions import file_hash

ROOT = Path(__file__).resolve().parents[1]
PREFIX = ROOT / "results/pc_osfal_pusht_heldout_geometry_20261107_codex"
PROTOCOL = ROOT.parent / "docs/PC_OSFAL_PUSHT_HELDOUT_GEOMETRY_PROTOCOL_20261107_CODEX.md"
OLD_MODEL = ROOT / "results/pc_geometric_contact_equivalence_dev_20261106_codex_model.json"
OLD_LAYER = ROOT / "results/pc_osfal_pusht_layer_dev_20261107_codex.json"
EXPECTED_MODEL = "ce70853a275c1c957a021c65186abcdc690bf9dfa237186da5eecfbccf0a585f"


def tasks():
    return [{"position": list(pos), "angle": float(angle)}
            for pos, angle in itertools.product(
                ((240., 240.), (260., 240.), (240., 260.), (260., 260.)),
                (np.pi / 8, 3 * np.pi / 8, 5 * np.pi / 8, 7 * np.pi / 8))]


def configure():
    for module in (physical, layer):
        module.PROTOCOL = PROTOCOL
    physical.PREFIX = PREFIX
    physical.PHYSICAL = PREFIX.with_name(PREFIX.name + "_physical.json")
    physical.IMAGES = PREFIX.with_name(PREFIX.name + "_images.npz")
    physical.MODEL = PREFIX.with_name(PREFIX.name + "_model.json")
    physical.FEATURES = PREFIX.with_name(PREFIX.name + "_features.npz")
    physical.tasks = tasks
    layer.PHYSICAL, layer.IMAGES, layer.MODEL = physical.PHYSICAL, physical.IMAGES, physical.MODEL
    layer.OUTPUT = PREFIX.with_name(PREFIX.name + "_layer.json")


def audit_preconditions():
    old = json.loads(OLD_MODEL.read_text())
    layers = json.loads(OLD_LAYER.read_text())
    if old["model_sha256"] != EXPECTED_MODEL or layers["model_fingerprint"] != EXPECTED_MODEL:
        raise ValueError("old model fingerprint mismatch")
    if len(tasks()) != 16:
        raise ValueError("unexpected panel size")
    original = physical.tasks()
    if not set(tuple(t["position"]) for t in tasks()).isdisjoint(tuple(t["position"]) for t in original):
        raise ValueError("position overlap")
    if not set(round(t["angle"], 12) for t in tasks()).isdisjoint(round(t["angle"], 12) for t in original):
        raise ValueError("angle overlap")


def compare():
    target = PREFIX.with_name(PREFIX.name + "_comparison.json")
    if target.exists():
        raise FileExistsError(target)
    data = json.loads(layer.OUTPUT.read_text())
    if data["model_fingerprint"] != EXPECTED_MODEL:
        raise ValueError("new model differs from frozen old checkpoint")
    rows = []
    for row in data["rows"]:
        def average(name):
            return float(np.mean([c["regret"] for c in row["layers"][name]["conditions"]]))
        actual = row["layers"]["official_visual"]["conditions"]
        predicted = []
        for condition in actual:
            costs = np.asarray(condition["physical_costs"])
            predicted.append(float(costs[np.argmin(condition["prior_predicted_costs"])] - costs.min()))
        rows.append({"root": row["root"], "task": tasks()[row["root"]],
                     "block9_regret": average("block_09"),
                     "official_regret": average("official_visual"),
                     "predicted_regret": float(np.mean(predicted))})
    differences = np.array([r["block9_regret"] - r["official_regret"] for r in rows])
    report = {"scope": "new geometric development panel; real successors offline only; not independent confirmation",
              "rows": rows, "block9_mean_regret": float(np.mean([r["block9_regret"] for r in rows])),
              "official_mean_regret": float(np.mean([r["official_regret"] for r in rows])),
              "predicted_mean_regret": float(np.mean([r["predicted_regret"] for r in rows])),
              "block9_better_roots": int(np.sum(differences < -1e-9)),
              "block9_worse_roots": int(np.sum(differences > 1e-9)),
              "tied_roots": int(np.sum(np.abs(differences) <= 1e-9)),
              "runner_sha256": file_hash(__file__), "protocol_sha256": file_hash(PROTOCOL),
              "layer_sha256": file_hash(layer.OUTPUT), "physical_sha256": file_hash(physical.PHYSICAL),
              "old_layer_sha256": file_hash(OLD_LAYER), "checkpoint_fingerprint": EXPECTED_MODEL}
    with target.open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
    print(json.dumps(report, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("collect", "analyze", "layers", "compare"))
    args = parser.parse_args()
    audit_preconditions()
    configure()
    if args.stage == "collect":
        physical.collect()
    elif args.stage == "analyze":
        physical.analyze()
        if json.loads(physical.MODEL.read_text())["model_sha256"] != EXPECTED_MODEL:
            raise ValueError("new model fingerprint mismatch")
    elif args.stage == "layers":
        if json.loads(physical.MODEL.read_text())["model_sha256"] != EXPECTED_MODEL:
            raise ValueError("new model fingerprint mismatch")
        layer.main()
    else:
        compare()


if __name__ == "__main__":
    main()
