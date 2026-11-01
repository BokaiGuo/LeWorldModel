from copy import deepcopy
import pytest
from experiments.spectral_reachability import codex_p2_h3_physical_saved_20261030_v1 as m


def test_budget_reconstruction_and_tamper():
    import json
    d = m.producer.ROOT/"results/codex_p2_h3_rechart_physical_20261030_v1_RUN01"
    report = json.loads((d/"report.json").read_text())
    binding = m.producer.build()
    events = [json.loads(s) for s in (d/"physical_budget.events.jsonl").read_text().splitlines()]
    counts = m.check_budget(events, report, binding)
    assert counts["selected"]["step"] == {"attempts":120, "returns":120}
    corrupted = deepcopy(events)
    index = next(i for i,e in enumerate(corrupted) if e.get("kind") == "env_step" and e.get("phase") == "selected")
    corrupted[index]["command"][0] += .1
    with pytest.raises(ValueError): m.check_budget(corrupted, report, binding)
