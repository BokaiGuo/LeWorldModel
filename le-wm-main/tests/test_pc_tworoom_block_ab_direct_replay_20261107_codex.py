import json

import numpy as np

from experiments import pc_tworoom_block_ab_direct_replay_20261107_codex as exp
from experiments.pc_input_permissions import file_hash


def test_direct_replay_binding_choices_and_summary():
    record = json.loads(exp.OUTPUT.read_text())
    plan = json.loads(exp.source.PLAN.read_text())
    b_physical = json.loads(exp.source.PHYSICAL.read_text())
    assert record["plan_sha256"] == file_hash(exp.source.PLAN)
    assert record["pool_sha256"] == file_hash(exp.source.POOL)
    assert record["b_physical_sha256"] == file_hash(exp.source.PHYSICAL)
    assert len(record["rows"]) == len(plan["rows"]) == 10
    with np.load(exp.source.POOL, allow_pickle=False) as pool:
        for root, row in enumerate(record["rows"]):
            assert row["root"] == root
            assert row["row"] == plan["rows"][root]["row"]
            for bank in ("a", "b"):
                assert row["choices"][bank] == int(np.argmin(pool[f"r{root}_{bank}_cost"]))
            assert row["b"] == b_physical["rows"][root]["outcomes"][row["choices"]["b"]]
            assert len(row["a"]["trace"]) == 25
            assert row["a"]["ever"] == any(x["terminated"] for x in row["a"]["trace"])
    for bank in ("a", "b"):
        assert record["summary"][f"{bank}_ever"] == sum(row[bank]["ever"] for row in record["rows"])
