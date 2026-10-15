"""Run the fixed transfer-sufficient feature-family audit over a panel.

Each panel member is evaluated independently with the same predeclared
feature families and train-only ridge protocol.  This runner never selects a
checkpoint or feature family from held-out outcomes; it only aggregates the
individual diagnostic reports.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiments.gdd_wm.audit_transfer_sufficient_feature_families import audit


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--case",
        action="append",
        required=True,
        help="name|candidate_dataset|arrays; repeat for each panel member",
    )
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    panel = {}
    for specification in args.case:
        parts = specification.split("|")
        if len(parts) != 3 or not all(parts):
            raise ValueError("each --case must be name|candidate_dataset|arrays")
        name, dataset, arrays = parts
        if name in panel:
            raise ValueError(f"duplicate panel name: {name}")
        print(f"[feature-family-panel] auditing {name}: train-only feature-family transfer")
        panel[name] = audit(Path(dataset), Path(arrays))
    report = {
        "schema": "gdd_wm_transfer_sufficient_feature_family_panel.v0.1",
        "status": "completed_diagnostic",
        "claim_boundary": (
            "cross-panel train-only feature-family controls; not a D-JEPA teacher, "
            "not preference-stability evidence, and not closed-loop planning"
        ),
        "panel_member_order": list(panel),
        "panel": panel,
        "decision": {
            "panel_member_or_family_selection": "not_performed",
            "d1_ready": False,
            "d2_ready": False,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
