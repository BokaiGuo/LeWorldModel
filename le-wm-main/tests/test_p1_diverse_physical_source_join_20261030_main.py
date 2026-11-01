import copy

import pytest

from experiments.spectral_reachability import p1_diverse_physical_source_join_20261030_main as source


def inputs(monkeypatch):
    rows = [dict(slot=i // 4, arm=str(i // 2), role=("incumbent", "diverse_selected")[i % 2],
                 candidate_index=0, task_binding={"task_id": str(i // 4)},
                 raw15_le_float32_hex="00" * 120) for i in range(16)]
    observed = [dict(**r, exact={"a": True}) for r in rows]

    def fake_assemble(action, report, _prepared, _data):
        assert [r["role"] for r in action["rows"]] == ["incumbent", "model_selected"] * 8
        assert [r["role"] for r in report["rows"]] == ["incumbent", "model_selected"] * 8
        return {"rows": [dict(task_id=str(r["slot"]), raw15_float32_hex=r["raw15_le_float32_hex"])
                         for r in action["rows"]]}

    monkeypatch.setattr(source.physical, "assemble", fake_assemble)
    return rows, {"rows": observed}, {}, {}


def test_mapping_and_provenance(monkeypatch):
    result = source.assemble(*inputs(monkeypatch))
    assert result["diverse_source_roles"] == ["incumbent", "diverse_selected"] * 8
    assert result["source_replay_sha256"] == source.RESULT_SHA


@pytest.mark.parametrize("kind", ["role", "action", "task"])
def test_rejects_mismatched_identity(monkeypatch, kind):
    args = copy.deepcopy(inputs(monkeypatch))
    if kind == "role":
        args[0][1]["role"] = "model_selected"
        args[1]["rows"][1]["role"] = "model_selected"
    elif kind == "action":
        args[1]["rows"][1]["raw15_le_float32_hex"] = "00" * 119 + "01"
    else:
        args[0][1]["task_binding"]["task_id"] = "incorrect"
        args[1]["rows"][1]["task_binding"]["task_id"] = "incorrect"
    with pytest.raises(ValueError):
        source.assemble(*args)
