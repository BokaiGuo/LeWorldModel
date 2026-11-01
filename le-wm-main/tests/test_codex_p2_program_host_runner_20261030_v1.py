from copy import deepcopy
import json
from pathlib import Path
import fcntl
import os
import pytest
from experiments.spectral_reachability import codex_p2_program_host_runner_20261030_v1 as runner
from experiments.spectral_reachability import codex_p2_program_host_saved_20261030_v1 as saved
from test_codex_p2_program_independent_saved_20261030_v1 import Sim,scope


def test_real_source_envelope_and_no_production_call():
    value=runner.material()
    assert len(value["parent"]["cases"])==4
    assert value["parent"]["protocol"]["per_prefix_per_arm_caps"]["packet"]==36
    assert not value["physical"] and not value["scientific_pass"]


def test_child_refuses_missing_inherited_inode(tmp_path):
    lock=tmp_path/"lock";lock.touch()
    with pytest.raises(ValueError,match="inherited"):
        runner.inherited_lock(lock)


def test_child_recognizes_existing_locked_descriptor(tmp_path):
    lock=tmp_path/"lock";lock.touch()
    fd=os.open(lock,os.O_RDWR)
    try:
        fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
        assert runner.inherited_lock(lock)["inode"]==os.fstat(fd).st_ino
    finally:os.close(fd)


def test_saved_reader_rejects_missing_or_wrong_report(tmp_path,monkeypatch):
    monkeypatch.setattr(runner,"verify",lambda *_:{"parent":{}})
    (tmp_path/"report.json").write_text("{}")
    (tmp_path/"supervisor_events.jsonl").write_text("")
    with pytest.raises(ValueError):
        saved.consume(tmp_path,"0"*64,"0"*64,"unused","unused")


def test_output_rejected_before_supervisor(monkeypatch,tmp_path):
    monkeypatch.setattr(runner,"verify",lambda *_:{})
    monkeypatch.setattr(runner.supervisor,"supervise",lambda *_:pytest.fail("started"))
    with pytest.raises(ValueError):
        runner.run("/unused","bad",tmp_path/"wrong")


def test_synthetic_full_saved_batch_and_tamper(tmp_path,monkeypatch):
    """Structural test only; fake counters cannot authenticate real CUDA."""
    context={"latent_history":[[0.0]*192],"action_history":[]}
    cases=[dict(base=i,context=context,goal=[0.0]*192) for i in range(4)]
    panel=tmp_path/"panel.json";panel.write_text(json.dumps(dict(cases=cases)))
    fake=dict(parent=dict(panel_path=str(panel),gpu_uuid="fixture",expected_model_hash="hash"))
    monkeypatch.setattr(runner,"verify",lambda *_:fake)
    monkeypatch.setattr(runner.identity,"verify",lambda *_:None)
    proof=dict(device=1,inode=2,fd=9)
    out=tmp_path/"run";out.mkdir()
    child=runner.journal.Journal(out/"child_events.jsonl",runner.CAP,1024)
    child.event(dict(kind="child_start",lock_proof=proof,order=runner.budget.contract()["order"]))
    child.event(dict(kind="model_loaded",device=dict(uuid="fixture"),model_sha256="hash"))
    rows=[]
    for index,(base,arm) in enumerate(runner.budget.contract()["order"]):
        row=runner.budget.run_branch(cases[base],arm,Sim,child.event,model_scope=scope)
        name=f"branch_{index}.json"
        sha=runner.prep.original.previous.write_new(out/name,row)
        rows.append(dict(base=base,arm=arm,status="complete",file=name,sha256=sha,
                         unique_action_sequences=row["unique_action_sequences"],resource_counts=row["resource_counts"]))
        child.event(dict(kind="branch_file",index=index,file=name,sha256=sha))
    child.append(dict(kind="child_terminal",status="exploratory_complete",error=None),terminal=True)
    tail=child.tail;child.close()
    report=dict(schema=runner.NAME+".report",status="exploratory_complete",prepare_sha256="fixture",
                error=None,lock_proof=proof,device=dict(uuid="fixture"),model_before="hash",model_after="hash",
                physical=False,P2_admission=False,loaded_sources=[],journal_tail=tail,rows=rows)
    reportsha=runner.prep.original.previous.write_new(out/"report.json",report)
    sup=runner.journal.Journal(out/"supervisor_events.jsonl",1024*1024,1024)
    for event in [dict(kind="lock_acquired",device=1,inode=2),dict(kind="child_spawn_intent"),
                  dict(kind="child_spawned"),dict(kind="child_exited",returncode=0),
                  dict(kind="supervisor_terminal",returncode=0)]:sup.event(event)
    sup.close();supsha=runner.prep.original.previous.sha(out/"supervisor_events.jsonl")
    receipt=saved.consume(out,reportsha,supsha,"fixture","fixture")
    assert len(receipt["rows"])==8 and receipt["P2_admission"] is False
    (out/"branch_0.json").write_bytes(b"tampered")
    with pytest.raises(ValueError):
        saved.consume(out,reportsha,supsha,"fixture","fixture")
