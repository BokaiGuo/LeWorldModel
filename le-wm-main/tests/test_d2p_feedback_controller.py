import numpy as np
import pytest

from experiments.spectral_reachability.d2p_feedback_controller import (
    Execution,
    Plan,
    compiler_plan,
    run_compiled_feedback,
    run_feedback,
)


def rgb(value):
    return np.full((2, 3, 3), value, dtype=np.uint8)


def test_replans_from_real_feedback_and_keeps_full_plans():
    plans = [
        Plan(np.arange(6, dtype=float).reshape(3, 2), {"predicted_terminal": "p0"}),
        Plan(np.ones((2, 2)), {"predicted_terminal": "p1"}),
        Plan(np.full((2, 2), 2.0), {"predicted_terminal": "p2"}),
    ]
    contexts = []
    observed = []
    packets = []

    def plan(context):
        contexts.append(context)
        return plans[len(contexts) - 1]

    def execute(packet):
        packets.append(packet.copy())
        return Execution(packet + 10, rgb(len(packets)), False, False)

    def observe(old, actions, image):
        observed.append((old, actions.copy(), image.copy()))
        return {"rgb": image, "last_action": actions, "prediction_must_not_leak": old.get("bad", None)}

    result = run_feedback({"rgb": rgb(0)}, plan=plan, execute=execute,
                          observe=observe, packet_size=2, max_replans=2)

    assert result["status"] == "replan_limit_reached"
    assert len(contexts) == len(observed) == len(packets) == 3
    assert np.array_equal(contexts[1]["rgb"], rgb(1))
    assert np.array_equal(contexts[2]["last_action"], plans[1].actions + 10)
    assert len([e for e in result["events"] if e["kind"] == "plan_return"]) == 3
    assert result["events"][0]["kind"] == "plan_attempt"
    assert result["events"][1]["kind"] == "plan_return"
    assert result["events"][1]["actions"].shape == (3, 2)
    assert "predicted_terminal" not in result["context"]
    assert result["counts"]["packets_complete"] == 3


@pytest.mark.parametrize("bad", [
    {"reward": 1},
    {"nested": {"simulator-state": [1]}},
    {"last_reward": 1},
    {"raw_state": [0]},
])
def test_forbidden_planning_fields_fail_closed(bad):
    called = []

    def plan(context):
        called.append(context)
        return Plan(np.zeros((1, 1)), None)

    with pytest.raises(ValueError):
        run_feedback(bad, plan=plan, execute=lambda _: None, observe=lambda *_: {},
                     packet_size=1, max_replans=0)
    assert called == []


def test_prediction_is_not_used_as_context_and_callback_mutation_is_isolated():
    prediction = {"terminal": ["not feedback"]}
    received = []

    def plan(context):
        received.append({"rgb": context["rgb"].copy()})
        context["rgb"][0, 0, 0] = 99
        return Plan(np.array([[1.0]]), prediction)

    def execute(packet):
        packet[0, 0] = 42
        return Execution(np.array([[2.0]]), rgb(7), True, False)

    def observe(old, actions, image):
        old["rgb"][0, 0, 0] = 88
        actions[0, 0] = 77
        image[0, 0, 0] = 66
        return {"rgb": rgb(8)}

    result = run_feedback({"rgb": rgb(1)}, plan=plan, execute=execute,
                          observe=observe, packet_size=1, max_replans=0)
    assert result["status"] == "terminated"
    assert received[0]["rgb"][0, 0, 0] == 1
    assert result["context"]["rgb"][0, 0, 0] == 8
    assert result["events"][1]["prediction"] == prediction
    actual_event = next(e for e in result["events"] if e["kind"] == "actual_actions")
    assert actual_event["actions"][0, 0] == 2


def test_partial_execution_preserves_reported_count_and_does_not_observe():
    observed = []

    result = run_feedback(
        {},
        plan=lambda _: Plan(np.zeros((4, 2)), {"terminal": "fake"}),
        execute=lambda _: Execution(np.ones((2, 2)), rgb(2), False, True),
        observe=lambda *args: observed.append(args) or {},
        packet_size=4,
        max_replans=3,
    )
    assert result["status"] == "truncated"
    assert result["counts"]["executed_actions_reported"] == 2
    assert result["counts"]["packets_complete"] == 0
    assert result["executed_action_count_fully_known"] is True
    assert observed == []


def test_execute_exception_marks_action_count_unknown():
    def execute(_):
        raise RuntimeError("backend disconnected")

    result = run_feedback({}, plan=lambda _: Plan(np.zeros((2, 1)), None),
                          execute=execute, observe=lambda *_: {},
                          packet_size=2, max_replans=0)
    assert result["status"] == "failed"
    assert result["error"]["stage"] == "execute"
    assert result["executed_action_count_fully_known"] is False
    assert result["counts"]["execute_returned"] == 0


def test_partial_error_keeps_actions_and_skips_observe():
    seen = []
    result = run_feedback(
        {},
        plan=lambda _: Plan(np.zeros((3, 1)), None),
        execute=lambda _: Execution(np.ones((1, 1)), rgb(1), False, False, "motor fault"),
        observe=lambda *args: seen.append(args) or {},
        packet_size=3,
        max_replans=0,
    )
    assert result["status"] == "failed"
    assert result["counts"]["executed_actions_reported"] == 1
    assert seen == []


def test_complete_terminal_packet_is_observed_once():
    calls = []
    result = run_feedback(
        {},
        plan=lambda _: calls.append("plan") or Plan(np.zeros((1, 1)), None),
        execute=lambda _: Execution(np.ones((1, 1)), rgb(3), True, False),
        observe=lambda old, action, image: calls.append("observe") or {"rgb": image},
        packet_size=1,
        max_replans=5,
    )
    assert result["status"] == "terminated"
    assert calls == ["plan", "observe"]


def test_actual_action_may_differ_from_plan_and_is_passed_to_observe():
    seen = []
    result = run_feedback(
        {},
        plan=lambda _: Plan(np.array([[1.0, 2.0]]), None),
        execute=lambda packet: Execution(np.array([[8.0, 9.0]]), rgb(1), False, False),
        observe=lambda old, action, image: seen.append(action.copy()) or {},
        packet_size=1,
        max_replans=0,
    )
    assert result["status"] == "replan_limit_reached"
    assert np.array_equal(seen[0], [[8.0, 9.0]])


@pytest.mark.parametrize("execution", [
    Execution(np.zeros((1, 1)), np.zeros((2, 2, 3), dtype=float), False, False),
    Execution(np.zeros((1, 1)), rgb(1), False, False, 3),
    Execution(np.zeros((2, 1)), rgb(1), False, False),
])
def test_invalid_execution_metadata_fails_closed(execution):
    result = run_feedback(
        {},
        plan=lambda _: Plan(np.zeros((1, 1)), None),
        execute=lambda _: execution,
        observe=lambda *_: {},
        packet_size=1,
        max_replans=0,
    )
    assert result["status"] == "failed"


def test_invalid_plan_and_observe_results_fail_closed():
    bad_plan = run_feedback({}, plan=lambda _: {"actions": []},
                            execute=lambda _: None, observe=lambda *_: {},
                            packet_size=1, max_replans=0)
    assert bad_plan["status"] == "failed"

    bad_observe = run_feedback({}, plan=lambda _: Plan(np.zeros((1, 1)), None),
                               execute=lambda _: Execution(np.zeros((1, 1)), rgb(1), False, False),
                               observe=lambda *_: {"state": 1},
                               packet_size=1, max_replans=0)
    assert bad_observe["status"] == "failed"
    assert bad_observe["counts"]["packets_complete"] == 1


def test_event_sink_receives_copies_and_sink_failure_is_classified():
    received = []

    def sink(event):
        received.append(event)
        event["kind"] = "mutated"

    result = run_feedback({}, plan=lambda _: Plan(np.zeros((1, 1)), None),
                          execute=lambda _: Execution(np.zeros((1, 1)), rgb(1), True, False),
                          observe=lambda *_: {},
                          packet_size=1, max_replans=0, event_sink=sink)
    assert result["status"] == "terminated"
    assert result["events"][0]["kind"] == "plan_attempt"
    assert received[0]["kind"] == "mutated"

    failing = run_feedback({}, plan=lambda _: Plan(np.zeros((1, 1)), None),
                           execute=lambda _: Execution(np.zeros((1, 1)), rgb(1), False, False),
                           observe=lambda *_: {},
                           packet_size=1, max_replans=0,
                           event_sink=lambda _: (_ for _ in ()).throw(OSError("journal full")))
    assert failing["status"] == "failed"
    assert failing["error"]["stage"] == "event_sink"


def test_configuration_and_numeric_validation():
    with pytest.raises(ValueError):
        run_feedback({}, plan=lambda _: None, execute=lambda _: None, observe=lambda *_: {},
                     packet_size=0, max_replans=0)
    result = run_feedback(
        {},
        plan=lambda _: Plan(np.array([[np.nan]]), None),
        execute=lambda _: None, observe=lambda *_: {},
        packet_size=1, max_replans=0,
    )
    assert result["status"] == "failed"


def test_compiler_plan_exposes_actions_and_keeps_full_program_in_prediction():
    class Compiler:
        def compile(self, context):
            return {
                "schema": "d2p_local_compiler.v1",
                "selected_index": 0,
                "operation_counts": {"rollout_attempted": 3},
                "programs": [{
                    "candidate": "reconfiguration",
                    "actions": [[1.0, 2.0]],
                    "output_context": {"predicted": True},
                }],
            }

    plan = compiler_plan(Compiler(), {"rgb": rgb(0)})
    assert np.array_equal(plan.actions, [[1.0, 2.0]])
    assert plan.prediction["selected_candidate"] == "reconfiguration"
    assert plan.prediction["selected_program"]["output_context"]["predicted"] is True


def test_compiled_feedback_rebuilds_from_observed_context():
    seen = []

    class Compiler:
        def compile(self, context):
            seen.append(context["rgb"][0, 0, 0])
            return {
                "schema": "d2p_local_compiler.v1",
                "selected_index": 0,
                "programs": [{"candidate": "p", "actions": [[1.0]]}],
            }

    result = run_compiled_feedback(
        {"rgb": rgb(0)},
        compiler_factory=lambda context: Compiler(),
        execute=lambda packet: Execution(packet.copy(), rgb(int(packet[0, 0]) + 1), False, False),
        observe=lambda old, actions, image: {"rgb": image},
        packet_size=1,
        max_replans=1,
    )
    assert result["status"] == "replan_limit_reached"
    assert seen == [0, 2]
