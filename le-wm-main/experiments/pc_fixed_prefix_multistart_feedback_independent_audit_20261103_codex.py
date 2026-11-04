"""Audit sealed multistart plans without mutating their recorded dependencies."""
import json
import tempfile
from pathlib import Path

from experiments import pc_fixed_prefix_multistart_feedback_diagnostic_20261103_codex as exp


def verify():
    recorded = json.loads(exp.PLANS.read_text())
    hashes = exp.dependencies()
    if recorded["dependencies"] != hashes:
        raise ValueError("sealed dependency drift")
    existing = json.loads(exp.AUDIT.read_text())
    base = exp.base
    original = base.PLANS, base.AUDIT, base.dependencies
    with tempfile.TemporaryDirectory(prefix="pc-multistart-audit-") as tmp:
        regenerated = Path(tmp) / "audit.json"
        try:
            base.PLANS, base.AUDIT = exp.PLANS, regenerated
            base.dependencies = lambda: hashes
            base.audit()
        finally:
            base.PLANS, base.AUDIT, base.dependencies = original
        if json.loads(regenerated.read_text()) != existing:
            raise ValueError("sealed audit differs from fresh replay")
    print("Existing audit matches fresh replay; sealed artifacts unchanged.", flush=True)


if __name__ == "__main__":
    verify()
