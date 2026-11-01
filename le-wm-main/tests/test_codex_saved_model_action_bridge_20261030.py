import copy
import importlib.util
import json
from pathlib import Path

import pytest

PATH = Path(__file__).resolve().parents[1] / "experiments/spectral_reachability/codex_saved_model_action_bridge_20261030.py"
spec = importlib.util.spec_from_file_location("saved_bridge", PATH)
b = importlib.util.module_from_spec(spec)
spec.loader.exec_module(b)
jmod = b.load_journal_module()
base = jmod.load_base()
ROOT_ID = "23b8de7837c39fb4e0fcffe7678732a9d840c57de7a0c8376e22076770ef6050"
H = "a" * 64


def sra_row():
    report = json.loads((b.ROOT / b.SOURCES["sra_h2_saved"][0]).read_bytes())
    return next(r for r in report["rows"] if r["root_sha256"] == ROOT_ID)


def p4_event():
    path = b.ROOT / b.SOURCES["p4_suffix_calibration"][0]
    return next(r for r in (json.loads(x) for x in path.read_bytes().splitlines())
                if r.get("kind") == "selection" and r.get("arm") == "actual" and r.get("boundary") == 5)


def config():
    identity = {k: H for k in base.IDENTITY}
    identity["root_sha256"] = ROOT_ID
    arms = base.ARMS
    return {"identities": {a: copy.deepcopy(identity) for a in arms},
            "initial_frames": {a: [H]*3 for a in arms},
            "initial_actions": {a: [[[0., 0.]]*5, [[0., 0.]]*5] for a in arms},
            "horizon": 10, "packet_steps": 5,
            "caps": {a: {k: 100 for k in base.AXES} for a in arms}}


def test_pinned_real_schema_and_saved_winners():
    sra = b.load_selection("sra_h2_saved", ROOT_ID, 0)
    p4 = b.load_selection("p4_suffix_calibration", ROOT_ID, 5)
    assert sra["candidate_id"] == "joint_h2_no_composition_sra_38"
    assert p4["candidate_id"] == "suffix_6"
    assert len(sra["commands"]) == len(sra["remaining_saved_commands"]) == len(p4["commands"]) == 5
    assert sra["step"] == 0 and p4["step"] == 5
    assert "not Full D2P" in p4["meaning"] and "not live" in sra["meaning"]


def test_sra_saved_to_journal_consumer_fixture(tmp_path):
    saved = b.load_selection("sra_h2_saved", ROOT_ID, 0)
    ctx = sra_row()["root"]["context"]
    journal = jmod.Journal(tmp_path / "journal.jsonl", config())
    arm = "sra_no_composition"
    s = journal.protocol.state(arm)
    req = journal.call("observe", arm, 0, s["frames"], s["actions"])
    bound = b.bind_saved(journal, arm, req, saved, ctx, tmp_path / "receipt.json")
    assert len(bound["selection_sha256"]) == 64
    journal.call("stop", arm, "fixture_no_dispatch")
    journal.call("stop", "full_d2p", "fixture_not_started")
    sha = journal.finish()
    result = b.consume_bound(tmp_path / "journal.jsonl", sha, [(tmp_path / "receipt.json", bound["receipt_sha256"])])
    assert result["receipts_verified"] == 1 and result["full_d2p_verified"] is False
    assert result["online_sra_verified"] is False and result["production_authorized"] is False
    assert result["journal_consume"]["new_physical_steps"] == 0
    with pytest.raises(ValueError, match="census"):
        b.consume_bound(tmp_path / "journal.jsonl", sha, [])
    with pytest.raises(ValueError, match="receipt hash"):
        b.consume_bound(tmp_path / "journal.jsonl", sha, [(tmp_path / "receipt.json", "0"*64)])


@pytest.mark.parametrize("change", ["root", "goal", "actions", "winner", "candidate_missing"])
def test_real_sra_tampered_rejected(change):
    row = sra_row()
    if change == "root":
        row["root"]["root_id"] = H
    elif change == "goal":
        row["result"]["witness"]["edges"][0]["input_context"]["latent_history"][0][0] += 1
    elif change == "actions":
        row["result"]["witness"]["edges"][0]["actions"][0][0] = 3
    elif change == "winner":
        row["result"]["selected_candidate_id"] = "wrong"
    else:
        row["result"]["attempts"].pop()
    with pytest.raises(ValueError):
        b.normalize_sra(row)


@pytest.mark.parametrize("change", ["context", "winner", "commands", "digest", "imagined"])
def test_real_p4_tampered_rejected(change):
    event = p4_event()
    if change == "context":
        event["compiler_plan"]["context"]["latent_history"][0][0] += 1
    elif change == "winner":
        event["compiler_plan"]["selected_candidate_id"] = "wrong"
    elif change == "commands":
        event["commands"][0][0] = 1
    elif change == "digest":
        event["selection_sha256"] = H
    else:
        event["arm"] = "imagined"
    with pytest.raises(ValueError):
        b.normalize_p4(event)


def test_wrong_request_or_context_does_not_write(tmp_path):
    saved = b.load_selection("sra_h2_saved", ROOT_ID, 0)
    journal = jmod.Journal(tmp_path / "journal", config())
    arm = "sra_no_composition"
    s = journal.protocol.state(arm)
    req = journal.call("observe", arm, 0, s["frames"], s["actions"])
    ctx = copy.deepcopy(sra_row()["root"]["context"])
    ctx["latent_history"][0][0] += 1
    with pytest.raises(ValueError, match="context mismatch"):
        b.bind_saved(journal, arm, req, saved, ctx, tmp_path / "bad")
    assert not (tmp_path / "bad").exists()
    journal.close_partial()


def test_p4_real_selection_attaches_after_fixture_prefix(tmp_path):
    saved = b.load_selection("p4_suffix_calibration", ROOT_ID, 5)
    event = p4_event()
    cfg = config()
    first_frame = saved["observed_frames"][0]
    cfg["initial_frames"] = {arm: [first_frame]*3 for arm in base.ARMS}
    journal = jmod.Journal(tmp_path / "p4_journal", cfg)
    arm = "full_d2p"  # Compatibility slot only; not a Full D2P claim.
    s = journal.protocol.state(arm)
    req = journal.call("observe", arm, 0, s["frames"], s["actions"])
    token = journal.call("select", arm, req, event["compiler_plan"]["prefix"], H)
    for i in range(5):
        intent = journal.call("begin_step", arm, token, True)
        pixel = saved["observed_frames"][-1] if i == 4 else H
        journal.call("finish_step", arm, intent, pixel, {"state7": [0.]*7, "reward": 0.})
    s = journal.protocol.state(arm)
    req = journal.call("observe", arm, 5, s["frames"], s["actions"])
    bound = b.bind_saved(journal, arm, req, saved, event["compiler_plan"]["context"], tmp_path / "p4_receipt")
    assert len(bound["receipt_sha256"]) == 64
    journal.call("stop", arm, "fixture_stop")
    journal.call("stop", "sra_no_composition", "fixture_not_started")
    consumed = jmod.consume(tmp_path / "p4_journal", journal.finish())
    assert consumed["snapshot"]["actual_execution_verified"] is False
    # The fixture prefix did not have a bridge receipt: full bridge consumption must reject.
    with pytest.raises(ValueError, match="census"):
        b.consume_bound(tmp_path / "p4_journal", consumed["journal_sha256"],
                        [(tmp_path / "p4_receipt", bound["receipt_sha256"])])


def test_source_drift_rejected(tmp_path, monkeypatch):
    (tmp_path / "report").write_text("{}")
    monkeypatch.setattr(b, "ROOT", tmp_path)
    monkeypatch.setattr(b, "SOURCES", {"sra_h2_saved": ("report", "0"*64)})
    with pytest.raises(ValueError, match="source drift"):
        b.load_selection("sra_h2_saved", ROOT_ID, 0)
