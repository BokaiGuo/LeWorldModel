"""Recompute goal-relevant mode rankings from the sealed dev304 model output.

Post-outcome exploratory analysis only: this does not generate new actions.
"""

import argparse
from pathlib import Path

import numpy as np

from . import d2p_goal_mode_rank_20261102 as geometry
from . import p1_dev304_spectral_real_rgb_saved_20261102 as source


def analyze():
    source.analyze()  # validate original candidate, physics and RGB bindings
    model = source.base.read(source.ROOT / f"{source.model_run.NAME}_RUN_DEV01/report.json",
                             source.physical.MODEL_SHA)
    rgb = source.base.read(source.RUN / "report.json", source.REPORT_SHA)
    goal = np.asarray(rgb["goal_latent"], dtype=np.float64)
    rows = []
    for row in model["rows"]:
        jac = np.asarray(row["jacobian"], dtype=np.float64)
        nominal = np.asarray(row["anchor_terminal"], dtype=np.float64)
        ranking = geometry.rank_goal_modes(jac, goal - nominal, 0.01)
        old = [int(i) for i in row["mode_indices"]]
        recomputed_old = np.argsort(-np.abs(ranking["coefficients"]), kind="stable")[:2].tolist()
        source.need(old == recomputed_old, "sealed old mode ranking differs")
        new = ranking["order"][:2].tolist()
        values = ranking["contributions"]
        rows.append({
            "arm": row["arm"],
            "legacy_coefficient_rank_top2": old,
            "goal_contribution_rank_top2": new,
            "legacy_top2_contribution": float(values[old].sum()),
            "goal_top2_contribution": float(values[new].sum()),
            "singular_values": ranking["singular"].tolist(),
            "coefficients": ranking["coefficients"].tolist(),
            "contributions": values.tolist(),
        })
    return {"schema": "p1_dev304_mode_rank_saved_20261102.v1",
            "classification": "post_outcome_one_known_root", "damping": 0.01,
            "task_metric": "identity on encoded goal latent",
            "action_metric": "identity in saved action coordinates",
            "caveat": "unconstrained quadratic improvement, not clipped nonlinear or physical outcome",
            "rows": rows, "independent_confirmation": False, "p1_admission": False}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    print(source.base.write_new(parser.parse_args().output, analyze()))
