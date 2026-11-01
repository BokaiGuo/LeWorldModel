from copy import deepcopy
import json
import pytest
from experiments.spectral_reachability import codex_p2_program_physical_binding_20261030_v1 as p


@pytest.fixture(scope="module")
def actual():
    case=json.loads(p.old.driver.PREP.read_bytes())["cases"][0]
    physical,_=p.old.old.material(0)
    branches=[json.loads((p.RUN/f"branch_{i}.json").read_bytes()) for i in (0,1)]
    return physical,case,branches


def test_actual_saved_handoff_has_literal_h25(actual):
    result=p.build(0)
    assert result["physical_ready"] is False
    assert len(result["prefix_reference"])==106
    assert [x["arm"] for x in result["plans"]]==["joint","library_chain"]
    for plan in result["plans"]:
        assert len(bytes.fromhex(plan["full25_raw_float32_hex"]))==200
        assert plan["horizon"]==25


@pytest.mark.parametrize("field",["context","task","bits","edge","arm"])
def test_tamper_refused(actual,field):
    physical,case,branches=deepcopy(actual)
    if field=="context":case["context"]={}
    elif field=="task":case["task_id"]="fake"
    elif field=="bits":branches[0]["result"]["winner"]["candidate"]["raw_float32_hex"]="00"*40
    elif field=="edge":branches[1]["result"]["winner"]["edges"][0]["raw_float32_hex"]="00"*40
    else:branches[1]["arm"]="joint"
    with pytest.raises(ValueError):
        p.assemble(0,physical,case,branches)
