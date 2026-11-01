"""Synthetic CPU-only source/context/math reconstruction; not native evidence."""
from copy import deepcopy
import hashlib

import numpy as np
import pytest

import agent_p3_fullcommit_live_20261028_v1_tests as fixture
import test_p3_feedback_reverse_control_20261030_v1 as feedback
import test_p3_matched_order_feedback_joint_v2_codex_20261029 as joint
import test_p3_six_terminal_native_saved_20261030_v1 as terminal_fixture
import test_p3_six_durable_constructor_20261030_v1 as store_fixture
from experiments.spectral_reachability import p3_six_strict_rebuild_20261030_v1 as m
from experiments.spectral_reachability import p3_six_terminal_20261030_v1 as terminal


def generated(policy):
    root, prior = feedback.setup()
    warm = fixture.m.old.kernel.moving.warmup_commands()
    store = fixture.Events()
    sink = m.old_saved.host.prior.ArmSink(store, 0)
    def backend(task, sink):
        return m.old_saved.host.prior.RecordingBackend(joint.BufferedRAM(task, sink), task, sink, lambda: None)
    result = m.control.execute_slot(root, policy, prior, feedback.Session(), backend, sink, permit_ram=True)
    assert result["status"] == "callback_feedback_complete_not_certified", result["errors"]
    observed = [dict(boundary=n, latent=np.full(192, n / 100, np.float32),
                     rgb_sha256=hashlib.sha256(f"frame{n}".encode()).hexdigest()) for n in (5, 10, 15, 20)]
    return root, warm, result, store.events, observed


@pytest.mark.parametrize("policy", m.control.POLICIES)
def test_rebuild_three_policy_math_and_context(policy):
    root, warm, result, wrapped, observed = generated(policy)
    receipt = m.rebuild(root, warm, result, wrapped, observed)
    assert receipt["mathematical_rebuild"] and receipt["contexts"] == 4
    assert receipt["paid_native_calls"] == 22
    assert not receipt["physical_certified"] and not receipt["scientific_pass"]


def test_context_and_warm_tamper_rejected():
    root, warm, result, wrapped, observed = generated("reverse_signal_order")
    altered = deepcopy(observed)
    altered[0]["latent"][0] += 1
    with pytest.raises(ValueError, match="context digest"):
        m.rebuild(root, warm, result, wrapped, altered)
    warm = warm.copy(); warm[0, 0] += np.float32(0.01)
    with pytest.raises(ValueError, match="warm10"):
        m.rebuild(root, warm, result, wrapped, observed)


def test_missing_native_records_and_changed_actions_rejected():
    root, warm, result, wrapped, observed = generated("reverse_signal_order")
    missing = [x for x in wrapped if x["call"] is None]
    with pytest.raises((ValueError, AssertionError, KeyError, IndexError)):
        m.rebuild(root, warm, result, missing, observed)
    changed = deepcopy(wrapped)
    event = next(x["payload"] for x in changed if x["payload"]["kind"] == "packet_intent")
    event["raw"][0][0] += 0.1
    with pytest.raises(ValueError, match="packet context"):
        m.rebuild(root, warm, result, changed, observed)


def test_tampered_signal_or_bank_rejected_with_rebound_terminal():
    root, warm, result, wrapped, observed = generated("reverse_signal_order")
    changed = deepcopy(result)
    changed["blocks"][0]["responses"][0]["horizon"] = 3
    modified = deepcopy(wrapped)
    modified[-1]["payload"]["result"] = changed
    with pytest.raises(ValueError, match="paid signal"):
        m.rebuild(root, warm, changed, modified, observed)
    changed = deepcopy(result)
    candidate = (changed["blocks"][0]["winner_index"] + 1) % 4
    changed["blocks"][0]["bank"]["actions"][candidate][0][0] += 0.1
    modified = deepcopy(wrapped)
    modified[-1]["payload"]["result"] = changed
    with pytest.raises(ValueError, match="endpoint bank"):
        m.rebuild(root, warm, changed, modified, observed)


def test_reverse_horizon_tampering_rejected():
    root, warm, result, wrapped, observed = generated("reverse_signal_order")
    changed = deepcopy(result)
    changed["blocks"][0]["horizon"] = 4 - changed["blocks"][0]["horizon"]
    modified = deepcopy(wrapped)
    modified[-1]["payload"]["result"] = changed
    with pytest.raises(ValueError, match="policy and gate"):
        m.rebuild(root, warm, changed, modified, observed)


def test_binding_must_bracket_complete_lifecycle():
    contract = dict(roots=[{"root": 0}, {"root": 1}], expected_model_hash="a" * 64,
                    precision={"mode": "fixture"})
    sha = m.core.digest(contract)
    detail = dict(source_contract_sha256=sha, task_roots=contract["roots"],
                  expected_model_hash=contract["expected_model_hash"], precision=contract["precision"])
    def e(index, resource, d):
        return dict(index=index, origin="physical", kind="record", resource=resource, detail=d)
    events = [e(0, "six_strict_source_begin", detail), e(1, "six_denominator", {})]
    events += [e(i + 2, "six_slot_return", {}) for i in range(6)]
    events += [e(8, "six_strict_source_end", deepcopy(detail))]
    m.check_binding(events, contract, sha)
    with pytest.raises(ValueError, match="boundaries"):
        m.check_binding(events[:-1], contract, sha)
    altered = deepcopy(events); altered[-1]["detail"]["precision"] = {}
    with pytest.raises(ValueError, match="content"):
        m.check_binding(altered, contract, sha)
    with pytest.raises(ValueError, match="pinned"):
        m.check_binding(events, contract, "0" * 64)


def test_missing_source_and_incomplete_output_fail_closed(tmp_path):
    store = store_fixture.new_store(tmp_path)
    terminal_fixture.start(store)
    hashes = terminal.publish(store)
    with pytest.raises(ValueError, match="six complete"):
        m.consume(store.output, hashes["report_sha256"], hashes["completion_sha256"], "0" * 64)
