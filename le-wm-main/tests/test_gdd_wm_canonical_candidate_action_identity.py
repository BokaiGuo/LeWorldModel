import json

from experiments.gdd_wm.canonicalize_candidate_action_identity import canonicalize


def test_context_prefixed_ids_become_shared_action_identity_without_reordering(tmp_path):
    source = tmp_path / "source.json"
    source.write_text(json.dumps({"metadata": {}, "records": [
        {"instance_id": "r", "teacher_set_id": "context0", "candidate_id": "context0:a", "action_sequence": [[0.1, 0.0]]},
        {"instance_id": "r", "teacher_set_id": "context1", "candidate_id": "context1:a", "action_sequence": [[0.1, 0.0]]},
    ]}), encoding="utf-8")
    output = tmp_path / "canonical.json"
    report = canonicalize(source, output)
    rows = json.loads(output.read_text(encoding="utf-8"))["records"]
    assert report["status"] == "built"
    assert rows[0]["candidate_id"] == rows[1]["candidate_id"]
    assert rows[0]["source_candidate_id"] == "context0:a"
    assert rows[1]["source_candidate_id"] == "context1:a"
