from copy import deepcopy
from io import BytesIO

import numpy as np
from PIL import Image
import pytest

from experiments.spectral_reachability import p3_rgb_fallback_h3_saved_20261031 as saved
from experiments.spectral_reachability import p3_rgb_fallback_h3_physical_20261031 as runner


@pytest.fixture(scope="module")
def recorded():
    directory = runner.ROOT / "results/p3_rgb_fallback_h3_physical_20261031_run_BATCH16_RUN01"
    prepare = runner.host.helper.read(directory / "prepare.json")
    report = runner.host.helper.read(directory / "report.json")
    events = saved.kernel.journal(directory / "journal.jsonl", report["journal_chain_tail"])
    _, _, _, data = runner.old.prepared.h1.prepare()
    def reader(name, sha):
        path = directory / name
        assert runner.host.sha(path) == sha
        with Image.open(BytesIO(path.read_bytes())) as image:
            return np.asarray(image.convert("RGB")).copy()
    return prepare, report, events, data["references"], reader


def test_saved_fallback_and_decision_tampering(recorded):
    prepare, report, events, refs, reader = recorded
    result = saved.validate(prepare, report, events, refs, reader)
    assert all(x["status"] == "horizon_completed" and x["returned"] == 20
               for x in result["slots"])
    altered = deepcopy(events)
    event = next(e for e in altered if e["kind"] == "fallback_selection")
    event["selection"] = "planned"
    with pytest.raises(ValueError, match="RGB decision binding"):
        saved.validate(prepare, report, altered, refs, reader)


def test_rejects_wrong_executed_suffix(recorded):
    prepare, report, events, refs, reader = recorded
    altered = deepcopy(events)
    transition = next(e for e in altered if e["kind"] == "returned_transition"
                      and e["phase"] == "plan" and e["step"] == 16)
    transition["actual_bits"] = "00"
    with pytest.raises(ValueError, match="transition action"):
        saved.validate(prepare, report, altered, refs, reader)
