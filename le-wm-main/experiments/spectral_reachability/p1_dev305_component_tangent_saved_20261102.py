"""Pair sealed exploratory score selections with already observed physical outcomes."""
import argparse
from pathlib import Path

import numpy as np

from . import p1_dev305_component_tangent_bank_20261102 as source
from . import p1_dev305_two_packet_metric_decompose_20261102 as previous

base, need, ROOT = source.base, source.need, source.ROOT
PREP_SHA = "674cd849d281c3864ce2ac89e2c576e3617c5e52f5d698645b2902e51450331a"
REPORT_SHA = "41459053e799137e7bbc2c81b0b4ae1f2767771933266f52f038a1fb729fa055"


def analyze():
    spec = base.read(ROOT / "results" / f"{source.NAME}_PREP_DEV01.json", PREP_SHA)
    report = base.read(ROOT / "results" / f"{source.NAME}_RUN_DEV01/report.json", REPORT_SHA)
    phys = base.read(ROOT / "results" / f"{previous.actual.physical.source.NAME}_RUN_DEV01/report.json",
                     previous.actual.physical.saved.REPORT_SHA)
    capture = base.read(ROOT / "results" / "p1_dev305_diverse_cem_20261102_CAPTURE_DEV01/report.json",
                        previous.CAPTURE_SHA)
    need(spec == source.protocol() and report["contract_sha256"] == PREP_SHA
         and report["model_before"] == report["model_after"] == spec["model_hash"], "source")
    goal = np.asarray(capture["goal_state_evaluation_only"], dtype=float)
    keys = [(x["prefix"], x["first"], x["second"]) for x in phys["rows"]]
    need(len(keys) == len(set(keys)) == 867 and not phys["errors"], "physical coverage")
    physical_index = dict(zip(keys, phys["rows"], strict=True))
    rows = []
    for role, entry in zip(spec["roles"], report["rows"], strict=True):
        need(entry["arm"] == role["arm"], "role")
        jac = np.asarray(entry["jacobian"])
        _, ridge, singular = source.component_scores(np.zeros(jac.shape[0]), jac)
        need(np.allclose(singular, entry["singular_values"]) and np.isclose(ridge, entry["ridge"]), "ridge")
        scores = {name: np.asarray(value) for name, value in entry["scores"].items()}
        need(set(scores) == {"object", "pusher", "joint"} and
             all(x.shape == (17, 17) and np.isfinite(x).all() for x in scores.values()) and
             np.allclose(scores["joint"], scores["object"] + scores["pusher"]), "scores")
        for name, matrix in scores.items():
            pair = list(map(int, np.unravel_index(np.argmin(matrix), matrix.shape)))
            need(pair == entry["selected_pair"][name], "argmin")
            physical = physical_index[(role["arm"], *pair)]
            need(physical["status"] == "fixed_horizon_completed" and len(physical["transitions"]) == 10, "physical")
            last = physical["transitions"][-1]
            state = np.asarray(last["state"], dtype=float)
            rows.append({"arm": role["arm"], "metric": name, "pair": pair,
                         "reward": last["reward"], "object_position_error": float(np.linalg.norm(state[2:4] - goal[2:4])),
                         "object_angle_error": abs(float((state[4] - goal[4] + np.pi) % (2*np.pi) - np.pi)),
                         "native_success": bool(last["terminated"]), "status": physical["status"]})
    return {"schema": "p1_dev305_component_tangent_saved_20261102.v1", "rows": rows,
            "scope": "post-outcome known-root; no new physical execution or independent confirmation",
            "limitation": "latent deltas not saved; scores cannot be independently recomputed without rerunning model",
            "p1_admission": False, "full_cost_certified": False}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    print(base.write_new(parser.parse_args().output, analyze()))
