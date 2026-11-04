import json

from experiments import pc_joint_carry_posthoc_rank_20261103 as audit


def test_ranking_recomputed_and_ties_excluded():
    report = json.loads(audit.SOURCE.read_text())
    rows = audit.analyze(report)
    assert [(r["root"], r["reversed"], r["comparable"], r["pairs"]) for r in rows] == [
        (8343, 3, 3, 6), (8345, 2, 6, 6), (8349, 6, 10, 10), (8351, 2, 5, 6)
    ]
    saved = json.loads((audit.ROOT / "results/pc_joint_carry_posthoc_rank_20261103_DEV01.json").read_text())
    assert rows == saved["rows"]
