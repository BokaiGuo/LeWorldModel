"""Use the sealed spectral executor with an unchanged CEM candidate pool."""
from __future__ import annotations

import json
import sys
from pathlib import Path

from experiments.spectral_reachability import run_reacher_pc_spectral_tail_carryover as experiment
from experiments.spectral_reachability.reacher_pc_fixed_pool_tail import cem_fixed_pool_with_tail


def main():
    # The imported executor owns provenance checks and six physical arms.
    # Substitution is local to this process and always restored.
    original = experiment.cem_with_carried_tail
    try:
        experiment.cem_with_carried_tail = cem_fixed_pool_with_tail
        experiment.main()
    finally:
        experiment.cem_with_carried_tail = original
    output_index = sys.argv.index("--output") + 1
    output = Path(sys.argv[output_index])
    receipt = {
        "wrapper_sha256": experiment.setup._sha256_file(Path(__file__)),
        "fixed_pool_implementation_sha256": experiment.setup._sha256_file(
            Path(sys.modules[cem_fixed_pool_with_tail.__module__].__file__)),
        "physical_report_sha256": experiment.setup._sha256_file(output / "report.json"),
        "contract": "original CEM population plus one separate old-tail evaluation per physical branch",
        "nominal_suffix_rollouts_per_branch": 33,
        "nominal_suffix_model_predict_samples_per_branch": 66,
        "budget_warning": "one additional candidate plus singleton encoding/goal evaluation per branch",
    }
    (output / "wrapper_receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
