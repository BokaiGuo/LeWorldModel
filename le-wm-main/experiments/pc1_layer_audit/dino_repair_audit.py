"""Independently recompute numerical records from sealed arrays, without inference."""
import json
from pathlib import Path

import numpy as np

from experiments.pc1_layer_audit import dino_repair_run as base, dino_repair_v2_run as v2
from experiments.pc1_layer_audit.metrics import ranking_record
from experiments.pc_input_permissions import file_hash


def audit():
    base.OUT = v2.OUT
    manifest = base.read_record("manifest.json")
    summary = base.read_record("summary.json")
    if manifest["track"] != "official_state_assisted" or summary["track"] != manifest["track"]:
        raise ValueError("unexpected input track")
    counts, errors = 0, []
    with (np.load(base.BASE/"prepare.npz", allow_pickle=False) as prep,
          np.load(base.BASE/"replay.npz", allow_pickle=False) as replay):
        for index, root in enumerate(base.camera.ROOTS):
            decision = base.read_record(f"predict_r{root}.json")
            evaluation = base.read_record(f"evaluate_r{root}.json")
            with (np.load(base.OUT/f"predict_r{root}.npz", allow_pickle=False) as pred,
                  np.load(base.OUT/f"evaluate_r{root}.npz", allow_pickle=False) as actual):
                np.testing.assert_array_equal(pred["raw_actions"],
                    np.broadcast_to(base.camera.ACTIONS[:, None], (3, 25, 2)))
                if not np.isfinite(pred["raw_actions"]).all() or np.abs(pred["raw_actions"]).max() > 1:
                    raise ValueError("illegal action")
                for row in evaluation["rows"]:
                    view, steps = row["view"], row["steps"]
                    h = base.camera.HORIZONS.index(steps)
                    physical = np.linalg.norm(replay["traces"][index, :, view, steps, :2] -
                                               prep["states"][index, 1, :2], axis=-1)
                    np.testing.assert_allclose(physical, row["physical_costs"], atol=1e-12, rtol=0)
                    sums = {}
                    for name in ("predicted", "real_encoded"):
                        components = []
                        for key, weight in (("visual", 1.), ("proprio", .1)):
                            goal = pred[f"v{view}_goal_{key}"].reshape(1, -1)
                            value = (pred[f"v{view}_pred_{key}"][steps//5].reshape(3, -1)
                                     if name == "predicted" else
                                     actual[f"v{view}_real_{key}"][1+3*h:4+3*h].reshape(3, -1))
                            components.append(weight*((value-goal)**2).mean(-1))
                        score = sum(components)
                        errors.append(float(np.max(np.abs(score - row[name+"_costs"]))))
                        np.testing.assert_allclose(score, row[name+"_costs"], atol=1e-5, rtol=1e-5)
                        sums[name] = score
                        rating = ranking_record(score, physical)
                        field = "repaired_predicted" if name == "predicted" else "repaired_real"
                        assert rating == row[field]
                    np.testing.assert_allclose(sums["predicted"],
                        decision["rows"][view]["scores"]["repaired"][steps//5], atol=1e-5, rtol=1e-5)
                    counts += 1
    if counts != 32*2*3 or summary["source_choice_mismatches"] != 0:
        raise ValueError("incomplete or mismatched decisions")
    result = dict(track=manifest["track"], root_count=32, decision_count=counts,
                  maximum_score_recompute_error=max(errors), passed=True,
                  scope="same cached arrays and physical trajectories; not independent environment replication",
                  sources={str(p): file_hash(p) for p in (Path(__file__), base.OUT/"manifest.json",
                                                         base.OUT/"summary.json")})
    target = base.OUT/"array_audit.json"
    if target.exists():
        old = json.loads(target.read_text())
        if old != result:
            raise ValueError("audit drift")
    else:
        base.write_json(target, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    audit()
