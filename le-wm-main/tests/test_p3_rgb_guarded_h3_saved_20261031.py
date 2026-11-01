from copy import deepcopy
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image
import pytest

from experiments.spectral_reachability import p3_rgb_guarded_h3_saved_20261031 as saved
from experiments.spectral_reachability import p3_rgb_guarded_h3_physical_20261031 as runner


@pytest.fixture(scope="module")
def recorded():
    directory = runner.ROOT / "results/p3_rgb_guarded_h3_physical_20261031_batch16_RUN01"
    prepare = runner.host.helper.read(directory / "prepare.json")
    report = runner.host.helper.read(directory / "report.json")
    events = saved.kernel.journal(directory / "journal.jsonl", report["journal_chain_tail"])
    _, _, _, data = runner.old.prepared.h1.prepare()
    def reader(name, sha):
        path = directory / name
        assert runner.host.sha(path) == sha
        with Image.open(BytesIO(path.read_bytes())) as img:
            return np.asarray(img.convert("RGB")).copy()
    return prepare, report, events, data["references"], reader


def test_real_abstentions_and_tampering(recorded):
    prepare, report, events, refs, reader = recorded
    result = saved.validate(prepare, report, events, refs, reader)
    assert len(result["slots"]) == 6
    assert all(slot["status"] == "rgb_preflight_abstain" and slot["returned"] == 15
               and slot["endpoint"] is None for slot in result["slots"])
    altered = deepcopy(events)
    event = next(e for e in altered if e["kind"] == "rgb_edge_preflight")
    event["decision"]["admitted"] = True
    with pytest.raises(ValueError, match="RGB decision binding"):
        saved.validate(prepare, report, altered, refs, reader)


def test_no_sixteenth_action_after_abstention(recorded):
    prepare, report, events, refs, reader = recorded
    altered = deepcopy(events)
    insert_at = next(i for i, e in enumerate(altered) if e["kind"] == "branch_closed" and e["receipt"]["branch_index"] == 0)
    altered.insert(insert_at, {"kind": "operation_attempted", "resource": "public_step",
                               "branch": 0, "phase": "plan", "step": 16})
    with pytest.raises(ValueError, match="operation census"):
        saved.validate(prepare, report, altered, refs, reader)
