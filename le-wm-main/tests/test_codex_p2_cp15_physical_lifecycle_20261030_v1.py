from copy import deepcopy
import hashlib
from pathlib import Path

import numpy as np
import pytest

from experiments.spectral_reachability import codex_p2_cp15_physical_lifecycle_20261030_v1 as run
from experiments.spectral_reachability import codex_p2_cp15_physical_partial_20261030_v1 as partial
from test_agent_p2_full30_control15_feedback_physical_20261029_v1 import fixture
from test_agent_allmode_physical_20261028_v1 import Env, fake_place, h, reseal


@pytest.fixture(scope="module")
def prepared():
    return run.material(0)[0]


def test_real_source_plans_and_caps(prepared):
    run.validate_execution(prepared)
    assert [p["policy"] for p in prepared["plans"]] == ["joint", "suffix"]
    assert run.contract()["caps"]["public_step"] == 80
    assert run.contract()["caps"]["physics_step"] == 808
    assert run.contract()["equal_total_compute"] is False


@pytest.mark.parametrize("field", ["action", "witness", "feedback", "reference", "root", "order"])
def test_tampered_prepare_rejected(prepared, field):
    p = deepcopy(prepared)
    if field == "action":
        p["plans"][0]["commands"][15][0] = 0.0
    elif field == "witness":
        p["plans"][0]["new_model_plan"]["cuda_report_sha256"] = "bad"
    elif field == "feedback":
        p["observed_context"] = {}
    elif field == "reference":
        p["prefix_reference"].pop()
    elif field == "root":
        p["plans"][0]["root_slot"] = 123
    else:
        p["plans"].reverse()
    # Recompute the superficial digest: semantic checks must still reject it.
    for plan in p["plans"]:
        plan["selection_digest"] = run.digest({k:v for k,v in plan.items() if k != "selection_digest"})
    p["planner_authority"]["plans_sha256"] = run.digest(p["plans"])
    with pytest.raises((ValueError, KeyError)):
        run.validate_execution(p)


def test_policy_denominator(prepared):
    roster = run.policy_roster(prepared, [{"status": "x"}, {"status": "y"}])
    assert len(roster) == 8
    assert [r["policy"] for r in roster] == ["joint", "suffix"] * 4


def test_wrong_authorization_before_factory():
    called = []
    with pytest.raises(ValueError):
        run.run("unused", "unused", "unused", lambda: called.append(1), authorization="wrong")
    assert not called


class MemoryStore(run.Store):
    def __init__(self):
        self.path=Path("/RAM_ONLY");self.files={};self.events=[];self.prefix_events={}
        self.previous=None;self.index=0;self.sizes={};self.bytes=0;self.finalizing=False

    def check(self, size=4096, final=False):
        run.require(self.bytes+size<=run.CAP-(0 if final or self.finalizing else run.FINAL), "RAM cap")

    def write(self, name, value, append=False, final=False):
        self.check(len(value), final)
        run.require(append or name not in self.files, "exclusive")
        self.files[name]=self.files.get(name,b"")+value if append else value
        self.bytes+=len(value);self.sizes[name]=len(self.files[name])
        return hashlib.sha256(self.files[name]).hexdigest()

    def emit(self, kind, **fields):
        start=len(self.files.get("journal.jsonl",b""))
        super().emit(kind, **fields)
        self.events.append(run.native.decode(self.files["journal.jsonl"][start:]))


def synthetic_run(fault=None, at=11, mismatch=False):
    upstream,data,_,_=fixture()
    parent=upstream["plans"][1]
    raw=run.raw30(parent["commands"]).tobytes()
    entries=[]
    for arm in ("joint","suffix"):
        entries.append(dict(base=0,arm=arm,task_id=parent["task_id"],root=parent["root_slot"],
            prefix_sha256=hashlib.sha256(raw[:120]).hexdigest(),
            suffix_raw_sha256=hashlib.sha256(raw[120:]).hexdigest(),
            suffix_raw_float32_hex=raw[120:].hex(),full_raw_float32_hex=raw.hex(),
            feedback_context_sha256="fixture"))
    b=dict(schema=run.cpbinding.NAME+".prepare",base=0,plans=entries,
        observed_context=deepcopy(upstream["observed_context"]),
        prefix_reference=deepcopy(upstream["prefix_reference"]),original_bindings=deepcopy(upstream["bindings"]),
        cuda_report_sha256=run.cpbinding.REPORT_SHA,cuda_prepare_sha256=run.cpbinding.PREP_SHA,
        cuda_receipt_sha256=run.cpbinding.RECEIPT_SHA,physical_ready=False,scientific_pass=False)
    p=run.cp15_assemble(upstream,b);store=MemoryStore();envs=[]
    if mismatch:
        event=next(e for e in p["prefix_reference"] if e["kind"]=="fresh_observation"
                   and e["receipt"]["phase"]=="plan" and e["receipt"]["step"]==15)
        event["receipt"]["pixel_sha256"]=h("wrong")
        p["cp15_binding"]["prefix_reference"]=deepcopy(p["prefix_reference"])
    def factory():
        env=Env(fault=fault,at=at);envs.append(env);return env
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(run,"place",fake_place);mp.setattr(run,"validate_env",lambda _:None)
        mp.setattr(run,"sha",lambda path:hashlib.sha256(store.files[Path(path).name]).hexdigest())
        result=run.execute(p,data,store,factory,authorization=run.AUTH,
                           prepared="/RAM/prepare.json",prepare_sha256=h("prepare"))
    return result,store,p,envs


def test_ram_complete_lifecycle_and_saved_resealed_tamper():
    report,store,p,envs=synthetic_run()
    assert report["status"]==run.RECORDED, report["errors"]
    assert len(run.validate_saved(report,store.events,p))==2
    assert all(row["own_returned"]==30 and row["feedback_guard"] for row in report["rows"])
    assert all(e.closed for e in envs)
    events=deepcopy(store.events);bad=deepcopy(report)
    next(e for e in events if e["kind"]=="feedback_prefix_guard")["guard"]["new_witness_sha256"]="bad"
    reseal(events,bad)
    with pytest.raises(ValueError):
        run.validate_saved(bad,events,p)


def test_ram_prefix_mismatch_no_suffix_dispatch():
    report,_,_,envs=synthetic_run(mismatch=True)
    assert report["status"]=="partial_or_failed"
    assert all(len(e.commands)==25 and e.closed for e in envs)
    assert all(row["feedback_guard"] is None for row in report["rows"])


def test_ram_native_stop_no_padding():
    report,store,p,envs=synthetic_run(fault="truncate",at=20)
    assert all(row["own_returned"]==10 and row["status"]=="native_stopped" for row in report["rows"])
    assert all(len(e.commands)==20 and e.closed for e in envs)
    assert len(run.validate_saved(report,store.events,p))==2


def test_saved_consumer_ram_and_png_tamper(monkeypatch):
    _,store,p,_=synthetic_run()
    files=deepcopy(store.files)
    monkeypatch.setattr(run,"saved_artifacts",lambda _:deepcopy(files))
    monkeypatch.setattr(run,"verify",lambda *_:(deepcopy(p),None))
    report_sha=hashlib.sha256(files["report.json"]).hexdigest()
    completion_sha=hashlib.sha256(files["completion.json"]).hexdigest()
    receipt=run.consume("/RAM/physical",report_sha,completion_sha)
    assert receipt["saved_PNG_verified"]==252
    png=next(name for name in files if name.endswith(".png"))
    files[png]=b"tampered"
    with pytest.raises(ValueError):
        run.consume("/RAM/physical",report_sha,completion_sha)


def test_failed_attempt_partial_journal_no_retry():
    report,store,p,envs=synthetic_run(fault="throw",at=30)
    assert report["status"]=="partial_or_failed"
    rows=partial.audit_events(report,store.events,p)
    assert all(row["attempted"]==20 and row["returned"]==19 for row in rows)
    assert all(e.closed for e in envs)
