from copy import deepcopy
import hashlib

import pytest

from experiments.spectral_reachability import p2_sra_duplicate_replay_check_20261102_exclusive as check


def fixture_data(tmp_path):
    journal = tmp_path / "journal"
    image = tmp_path / "image"
    journal.write_bytes(b"journal")
    image.write_bytes(b"rgb")
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    a, b = [], []
    for i, root in enumerate(check.ROOTS):
        trace = [{"primitive_step": n, "kind": "observation", "state": [n]}
                 for n in range(1, 11)]
        row = dict(root_sha256=root, root_index=i, status="completed",
                   full_horizon_completed=True, failure=None, planned_commands=10,
                   completed_commands=10, attempted_commands=10, physical_env_step_attempts=10,
                   physical_env_step_returns=10, trace=trace, final_state=[10],
                   journal_path=str(journal), journal_sha256=sha(journal),
                   terminal_rgb_path=str(image), terminal_rgb_sha256=sha(image),
                   context_identity=dict(state=[0], goal=[1], history_rgb_sha256="h",
                                         goal_rgb_sha256="g", raw_history=[2]))
        a.append(dict(deepcopy(row), method="joint_h2_no_composition"))
        row["context_identity"]["history"] = row["context_identity"].pop("raw_history")
        b.append(dict(row, method="joint_h2_no_composition_sra"))
    return {"composition": {"rows": a}, "sra": {"rows": b}}


def test_duplicate_and_different_trace(tmp_path):
    data = fixture_data(tmp_path)
    assert all(p["identical_trace_including_commands_rewards_states_termination"]
               for p in check.compare(data)["pairs"])
    data["sra"]["rows"][0]["trace"][0]["state"] = [999]
    assert not check.compare(data)["pairs"][0]["identical_trace_including_commands_rewards_states_termination"]


@pytest.mark.parametrize("mutation", ["partial", "missing", "context", "journal"])
def test_fail_closed(tmp_path, mutation):
    data = fixture_data(tmp_path)
    row = data["sra"]["rows"][0]
    if mutation == "partial":
        row["physical_env_step_returns"] = 9
    elif mutation == "missing":
        data["sra"]["rows"].pop()
    elif mutation == "context":
        row["context_identity"]["goal"] = [7]
    else:
        row["journal_sha256"] = "0" * 64
    with pytest.raises(ValueError):
        check.compare(data)
