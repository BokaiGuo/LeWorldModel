"""Offline synthetic six-slot regression; no simulator or neural execution."""
from copy import deepcopy
import json
import numpy as np
import pytest

from experiments.spectral_reachability import codex_p3_equalroot_physical_saved_20261030_v1 as consumer

h, base, k = consumer.host, consumer.base, consumer.kernel


@pytest.fixture
def batch():
    p = h.helper.read(h.ROOT / "results/codex_p3_six_physical_host_20261030_v1_PREP_01.json")
    p["schema"] = h.NAME + ".prepare"
    rows, events = [], [dict(kind="run_started", prepare=p)]
    state = [0.] * 7
    image = np.zeros((224, 224, 3), dtype=np.uint8)
    for i, plan in enumerate(p["plans"]):
        row = dict(branch_index=i, root_slot=plan["root_slot"], arm=plan["arm"],
                   selected_plan=plan, initial_exact=True, first_error=None, secondary_errors=[],
                   warmup_returned=10, own_attempted=10, own_returned=10,
                   status="horizon_completed", goal=state, root_state=state, final_state=state,
                   initial_metrics=k.metric(state, state), final_metrics=k.metric(state, state),
                   native_success=True, full_horizon=True, PNG=3)
        rows.append(row)
        commands = list(k.moving.warmup_commands()) + list(h.spec.raw(plan["commands"]))
        for j, command in enumerate(commands):
            ack = dict(ordinal=j, sent_dtype="float32", actual_dtype="float32",
                       sent_shape=[2], actual_shape=[2], sent_hex=command.tobytes().hex(),
                       actual_hex=command.tobytes().hex(), terminated=False, truncated=False)
            events.append(dict(kind="step_ack_observed", branch=i, receipt=ack))
            if j >= 10:
                events.append(dict(kind="returned_transition", branch=i, phase="plan", step=j-9,
                                   command=plan["commands"][j-10], returned_goal_state=state,
                                   terminated=False, truncated=False, state=state))
        for step in (0, 5, 10):
            receipt = dict(phase="warmup", step=step, png=f"{i}_{step}.png", file_sha256="fixture",
                           pixel_sha256=k.moving.image_sha(image), state=state,
                           observation_state=state, snapshot={"state": state})
            events.extend([dict(kind="fresh_observation", branch=i, receipt=receipt),
                           dict(kind="guard_receipt", branch=i, accepted=True,
                                render_preserved_dynamics=True)])
        events.append(dict(kind="branch_closed", receipt=row))
    expected = dict(environment_construct=6, environment_close=6, reset=12, set_state=24,
                    goal_render=12, goal_bind=6, public_step=120, physics_step=1224,
                    explicit_render=18)
    counts = {key: dict(attempted=expected[key], completed=expected[key]) for key in p["contract"]["caps"]}
    for key, count in expected.items():
        for _ in range(count):
            events.extend([dict(kind="operation_attempted", resource=key),
                           dict(kind="operation_completed", resource=key)])
    report = dict(schema=h.NAME + ".report", status="recorded_not_admitted", errors=[],
                  fixed_denominator=6, scientific_pass=False, new_neural_calls=0, rows=rows, counts=counts)
    events.append(dict(kind="run_finished", status=report["status"], counts=counts))
    return p, report, events, image


def test_six_slot_roundtrip(batch):
    p, report, events, image = batch
    result = base.validate(p, report, events, lambda *_: image)
    assert len(result["slots"]) == 6 and len(result["frames"]) == 18
    assert result["scientific_pass"] is False


@pytest.mark.parametrize("tamper", ["ack", "goal", "count", "stop", "order", "missing", "frame"])
def test_tamper_rejected(batch, tamper):
    p, report, events, image = deepcopy(batch)
    if tamper == "ack":
        next(e for e in events if e["kind"] == "step_ack_observed")["receipt"]["actual_hex"] = "00"
    elif tamper == "goal":
        next(e for e in events if e["kind"] == "returned_transition")["returned_goal_state"] = [1] * 5
    elif tamper == "count":
        report["counts"]["physics_step"]["completed"] -= 1
    elif tamper == "stop":
        next(e for e in events if e["kind"] == "step_ack_observed")["receipt"]["terminated"] = True
    elif tamper == "order":
        report["rows"][0], report["rows"][1] = report["rows"][1], report["rows"][0]
    elif tamper == "missing":
        report["rows"].pop()
    else:
        image[0, 0, 0] = 1
    with pytest.raises((ValueError, KeyError, IndexError)):
        base.validate(p, report, events, lambda *_: image)


def test_external_prepare_copy_is_semantic_and_frame_hash_checked(batch, tmp_path, monkeypatch):
    p, report, events, image = batch
    report["journal_sha256"], report["journal_chain_tail"] = "journal", "tail"
    (tmp_path / "prepare.json").write_text(json.dumps(p))
    (tmp_path / "report.json").write_text(json.dumps(report))
    (tmp_path / "journal.jsonl").write_text("fixture")
    from PIL import Image
    for e in events:
        if e["kind"] == "fresh_observation":
            Image.fromarray(image).save(tmp_path / e["receipt"]["png"])
            e["receipt"]["file_sha256"] = h.sha(tmp_path / e["receipt"]["png"])
    refs = [dict(goal=np.zeros(7), root_sha256=k.source.root_hash(np.zeros(7), np.zeros(7)),
                 history=[image]*3, states=[np.zeros(7)]*3,
                 dynamics=[{"state": [0.]*7}]*3) for _ in range(2)]
    monkeypatch.setattr(h, "verify", lambda *args: (p, {"references": refs}))
    monkeypatch.setattr(k, "journal", lambda *args: events)
    real_sha = h.sha
    monkeypatch.setattr(h, "sha", lambda path: "journal" if str(path).endswith("journal.jsonl") else real_sha(path))
    assert len(consumer.consume(tmp_path, real_sha(tmp_path / "report.json"), "original", "original-sha")["slots"]) == 6
    (tmp_path / "0_0.png").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="frame SHA"):
        consumer.consume(tmp_path, real_sha(tmp_path / "report.json"), "original", "original-sha")
