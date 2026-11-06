"""Run the existing fixed-terminal RGB harness with the v25 adapter.

The harness is configured only in this process at entry, not on import. Its
source remains part of the sealed dependencies; this wrapper is added too.
"""
import sys
from pathlib import Path

from experiments import pc_v23_pointmaze_live_feedback as harness
from experiments import pc_v25_pointmaze_feedback_adapter as feedback

PROTOCOL = harness.ROOT.parent / "docs/PC_V25_POINTMAZE_LIVE_FEEDBACK_CONTRACT_20261104_CODEX.md"


def validate_contract(contract, expected_goal):
    name = contract.get("output_name")
    if not isinstance(name, str) or not name.startswith("pc_v25_"):
        raise ValueError("v25 output name required")
    translated = dict(contract, output_name="pc_v23_" + name[len("pc_v25_"):])
    start, root_id, _ = harness.validate_contract(translated, expected_goal)
    return start, root_id, name


def main():
    original_dependencies = harness.dependencies
    original_validate = harness.validate_contract
    original_feedback, original_protocol = harness.feedback, harness.PROTOCOL

    def dependencies(contract_path):
        records = original_dependencies(contract_path)
        sources = (Path(__file__), Path(feedback.compiler.v13.__file__),
                   Path(feedback.compiler.core.pc_shared_multistart_cem.__file__))
        records.update({str(path.resolve()): harness.old.prior.sha(path) for path in sources})
        return records

    # validate_contract delegates to the original validator, so use a local
    # copy rather than recursively calling the patched module attribute.
    def validate(contract, expected_goal):
        name = contract.get("output_name")
        if not isinstance(name, str) or not name.startswith("pc_v25_"):
            raise ValueError("v25 output name required")
        translated = dict(contract, output_name="pc_v23_" + name[len("pc_v25_"):])
        start, root_id, _ = original_validate(translated, expected_goal)
        return start, root_id, name

    try:
        harness.feedback = feedback
        harness.PROTOCOL = PROTOCOL
        harness.dependencies = dependencies
        harness.validate_contract = validate
        harness.main()
    finally:
        harness.feedback, harness.PROTOCOL = original_feedback, original_protocol
        harness.dependencies, harness.validate_contract = original_dependencies, original_validate


if __name__ == "__main__":
    sys.exit(main())
