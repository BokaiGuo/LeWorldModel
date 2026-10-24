import json

from experiments.spectral_reachability.audit_sra_pointmaze_fresh_root_capacity import audit


def _episode(root, episode):
    path = root / "obses" / f"episode_{episode}.pth"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"rgb")


def test_fresh_root_capacity_blocks_when_a_receipt_consumes_the_archive(tmp_path):
    dataset = tmp_path / "dataset"
    for episode in (10, 11, 12):
        _episode(dataset, episode)
    receipt = tmp_path / "receipt.json"
    receipt.write_text(json.dumps({"protocol": {"roots": [10, 11]}, "rows": [{"episode_id": 12}]}))

    result = audit(dataset_root=dataset, receipts=[receipt], required_fresh_roots=2)

    assert result["status"] == "blocked_fresh_root_capacity"
    assert result["consumed_rgb_roots"] == [10, 11, 12]
    assert result["unconsumed_rgb_roots"] == []


def test_fresh_root_capacity_keeps_non_root_numbers_out_of_provenance(tmp_path):
    dataset = tmp_path / "dataset"
    for episode in (10, 11, 12, 13):
        _episode(dataset, episode)
    receipt = tmp_path / "receipt.json"
    receipt.write_text(json.dumps({"protocol": {"roots": [10, 11], "action_block": 5}}))

    result = audit(dataset_root=dataset, receipts=[receipt], required_fresh_roots=2)

    assert result["status"] == "pass"
    assert result["unconsumed_rgb_roots"] == [12, 13]
