import copy
import json

import pytest

from experiments.spectral_reachability import reacher_response_svd_predicted_cem_20261102 as run
from experiments.spectral_reachability import reacher_response_svd_predicted_cem_saved_20261102 as saved


def load_inputs():
    root = run.ROOT / "results"
    load = lambda path: json.loads(path.read_text())
    return [load(root / "reacher_response_svd_predicted_cem_RUN_20261102_v1/report.json"),
            load(root / "reacher_response_svd_predicted_cem_PLAN_20261102_v2.json"),
            load(run.SOURCE), load(saved.PHYSICAL), load(saved.FEEDBACK)]


def test_selection_recomputes_and_rejects_false_winner():
    values = load_inputs()
    result = saved.validate(*values)
    assert [row["selected"] for row in result["rows"]] == [12, 12]
    changed = copy.deepcopy(values)
    changed[0]["rows"][0]["selection"]["selected_proposal_index"] = 1
    with pytest.raises(ValueError, match="selection recomputation"):
        saved.validate(*changed)


def test_rejects_missing_common_tail():
    values = load_inputs()
    values[3]["paired_rows"] = [row for row in values[3]["paired_rows"]
                                 if not (row["seed"] == 20263160 and row["prefix"] == 12
                                         and row["tail"] == 4)]
    with pytest.raises(ValueError, match="prior physical roster"):
        saved.validate(*values)
