"""Fail-closed check before planning from a physically observed midpoint.

This is an additional guard for future callers; it does not change archived
plans or certify the authenticity of a journal. The caller must authenticate
the journal and bind its actions and midpoint RGB separately.
"""


def require_replan_allowed(transitions, expected_steps=5):
    """Reject missing, reordered, malformed, terminated or truncated steps."""
    if type(expected_steps) is not int or expected_steps <= 0:
        raise ValueError("invalid expected step count")
    if not isinstance(transitions, list) or len(transitions) < expected_steps:
        raise ValueError("incomplete prefix")
    for step, event in enumerate(transitions[:expected_steps], start=1):
        if not isinstance(event, dict) or type(event.get("step")) is not int or event["step"] != step:
            raise ValueError("prefix step identity")
        if type(event.get("terminated")) is not bool or type(event.get("truncated")) is not bool:
            raise ValueError("missing or invalid native stop flag")
        if event["terminated"] or event["truncated"]:
            raise ValueError("native stop forbids midpoint replan")


def verified_saved_inputs():
    """Load the established fixed roster and additionally check its stop flags.

    This adapter does not replace the historical planner automatically: new
    experiment entry points must explicitly call it and bind this source.
    """
    from . import codex_p1_adam_feedback_plan_20261030_main as source

    entries = source.inputs()
    report = source.gap.prior.gap.read(
        source.gap.PHYSICAL / "run_report.json", source.gap.REPORT_SHA
    )[0]
    if len(entries) != 24 or len(report["shards"]) != len(entries):
        raise ValueError("fixed roster mismatch")
    for index, (entry, shard) in enumerate(zip(entries, report["shards"], strict=True)):
        if entry[0]["index"] != index or shard["index"] != index:
            raise ValueError("shard identity")
        transitions = source.traces.transitions(
            source.gap.PHYSICAL / f"row_{index:02d}", shard["report_sha256"]
        )
        require_replan_allowed(transitions)
    return entries
