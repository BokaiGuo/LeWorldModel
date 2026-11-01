"""Synthetic writer-repair attribution, never experimental evidence."""
import copy

import pytest

from experiments.spectral_reachability import audit_reacher_d2p_masked_program_writer_repair as audit


def fixture():
    sources = [{"seed":i,"source_episode":300+i,"source_start_row":i*100} for i in range(8)]
    protocol = {"root_seeds":list(range(8)),"source_roots":sources,"implementation_sha256":{"auditor":"old","compiler":"same"}}
    old = {"schema":"reacher_d2p_masked_program_pilot.v1","status":"fail","sealed_protocol":protocol,
        "gates":{"all8_roots":False,"source_disjoint":False,"cuda_only":True},"physical_branches":456,"actual_model_predict_samples":59568,
        "rows":[{**sources[-1],"tasks":{"all":[i,192]}} for i in range(8)]}
    new = copy.deepcopy(old)
    new["status"] = "completed"
    new["gates"] = {k:True for k in old["gates"]}
    new["sealed_protocol"]["implementation_sha256"]["auditor"] = "new"
    new["rows"] = [{**r,"tasks":{"all":[r["seed"],192]}} for r in sources]
    predicted = {"rows":[{**r,"tasks":{"all":{"planning_seconds":1.,"raw_actions":[1,2]}}} for r in sources]}
    timed = copy.deepcopy(predicted)
    for r in timed["rows"]: r["tasks"]["all"]["planning_seconds"] = 2.
    sidecars = {i:row for i,row in enumerate(old["rows"])}
    return old,new,predicted,timed,sidecars


def test_exact_writer_only_intervention():
    result = audit.compare(*fixture())
    assert result["all456_physical_branches_exact"] is True
    assert result["combined_cuda_predict_samples"] == 119136


@pytest.mark.parametrize("mutation",["extra_failure","compiler","protocol","predictions","physical","metadata","partial","compute"])
def test_any_other_change_rejects_metadata_only_attribution(mutation):
    old,new,pred,timed,sidecars = fixture()
    if mutation == "extra_failure": old["gates"]["cuda_only"] = False
    elif mutation == "compiler": new["sealed_protocol"]["implementation_sha256"]["compiler"] = "changed"
    elif mutation == "protocol": new["sealed_protocol"]["threshold"] = .1
    elif mutation == "predictions": timed["rows"][0]["tasks"]["all"]["raw_actions"] = [3,4]
    elif mutation == "physical": new["rows"][0]["tasks"]["all"][0] += 1
    elif mutation == "metadata": new["rows"][0]["source_episode"] = 999
    elif mutation == "partial": old["rows"].pop()
    else: new["actual_model_predict_samples"] -= 1
    with pytest.raises(ValueError): audit.compare(old,new,pred,timed,sidecars)
