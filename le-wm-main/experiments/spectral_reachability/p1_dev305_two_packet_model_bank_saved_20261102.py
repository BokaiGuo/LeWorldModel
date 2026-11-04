"""Join same-bank model predictions to saved physical outcomes for analysis only."""
import argparse
from pathlib import Path

import numpy as np

from . import p1_dev305_two_packet_model_bank_20261102 as model
from . import p1_dev305_two_packet_oracle_saved_20261102 as oracle

base, need, ROOT = model.base, model.need, model.ROOT
CONTRACT_SHA = "40f18e3147e034512d737a1ca5e17adc91c72aab33134111555d856dd3a0a9a9"
REPORT_SHA = "53de268ce215f3edd62e1b3b9bcdd68bd6a03c4536f3323489b1ef5e59391f37"


def analyze():
    spec = base.read(ROOT / "results" / f"{model.NAME}_PREP_DEV01.json", CONTRACT_SHA)
    report = base.read(ROOT / "results" / f"{model.NAME}_RUN_DEV01/report.json", REPORT_SHA)
    need(spec == model.protocol() and report["contract_sha256"] == CONTRACT_SHA
         and report["model_before"] == report["model_after"] == spec["model_hash"], "source")
    physical_report = base.read(ROOT / "results" / f"{model.physical.NAME}_RUN_DEV01/report.json",
                                oracle.REPORT_SHA)
    oracle_summary = oracle.analyze()
    rows = []
    for role, predicted, ceiling in zip(spec["roles"], report["rows"], oracle_summary["rows"], strict=True):
        need(role["arm"] == predicted["arm"] == ceiling["prefix"], "role binding")
        scores = np.asarray(predicted["terminal_scores"], dtype=float)
        need(scores.shape == (17, 17) and np.isfinite(scores).all()
             and list(np.unravel_index(np.argmin(scores), scores.shape)) == predicted["selected_pair"],
             "score selection")
        first, second = predicted["selected_pair"]
        physical = next(x for x in physical_report["rows"]
                        if x["prefix"] == role["arm"] and x["first"] == first and x["second"] == second)
        need(physical["status"] == "fixed_horizon_completed", "paired physical outcome")
        reward = physical["transitions"][-1]["reward"]
        rows.append({"arm": role["arm"], "model_pair": [first, second],
                     "oracle_pair": ceiling["oracle_pair"], "model_physical_reward": reward,
                     "oracle_physical_reward": ceiling["oracle_terminal_reward"],
                     "model_regret_reward_units": ceiling["oracle_terminal_reward"] - reward,
                     "greedy_physical_reward": ceiling["greedy_terminal_reward"],
                     "model_final_state": physical["transitions"][-1]["state"],
                     "oracle_final_state": ceiling["oracle_final_state"]})
    return {"schema": "p1_dev305_two_packet_model_bank_saved_20261102.v1", "rows": rows,
            "scope": "post-outcome same-root same-bank physical comparison; no independent confirmation",
            "p1_admission": False}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    print(base.write_new(parser.parse_args().output, analyze()))
