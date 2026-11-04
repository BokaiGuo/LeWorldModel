"""Read-only independent rank and lineage audit of one saved diagnostic run."""
import hashlib
import json
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parents[2]
RUN = BASE / "results/pc_pointmaze_velocity_score_crosstab_RUN_20261103_codexexclusive_02"
OUT = BASE / "results/pc_pointmaze_velocity_score_crosstab_SAVED_AUDIT_20261103_codexexclusive_01.json"
PLAN_SHA = "922a5a711ed3edd39b6be01991545d8b623dc1fb5ab03f1e6386e8cbeaf9fadf"


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def ranks(x, y):
    x, y = np.asarray(x), np.asarray(y)
    assert x.shape == y.shape == (48,) and np.isfinite(x).all() and np.isfinite(y).all()
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    upper = np.triu(np.ones((48, 48), dtype=bool), 1)
    return dict(pairs=1128, comparable=int((upper & (dx != 0) & (dy != 0)).sum()),
                reversed=int((upper & (np.sign(dx) != np.sign(dy)) & (dx != 0) & (dy != 0)).sum()),
                score_ties=int((upper & (dx == 0)).sum()),
                reference_ties=int((upper & (dy == 0)).sum()))


def main():
    report = json.loads((RUN / "report.json").read_text())
    plan = json.loads((RUN / "plan.json").read_text())
    assert sha(RUN / "plan.json") == PLAN_SHA == report["plan_sha256"]
    for name, digest in plan["files"].items():
        assert sha(Path(name)) == digest, name
    assert sha(RUN / "evaluation.npz") == report["evaluation_sha256"]
    events, previous = [], None
    for index, line in enumerate((RUN / "events.jsonl").read_text().splitlines()):
        event = json.loads(line)
        digest = event.pop("event_sha256")
        assert event["index"] == index and event["previous_sha256"] == previous
        assert hashlib.sha256(json.dumps(event, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest() == digest
        previous = digest
        events.append(event["payload"])
    assert events[0]["kind"] == "run_started" and events[-1]["kind"] == "run_finished"
    assert events[-1]["report_sha256"] == sha(RUN / "report.json")
    assert [e["kind"] for e in events] == (["run_started"] + ["group_started", "group_returned"] * 8
                                           + ["all_scores_verified", "run_finished"])
    assert report["model_before"] == report["model_after"] == plan["model_parameter_hash"]
    totals = {}
    errors = {}
    with np.load(BASE / "results/a0_pointmaze_rgb_initial_candidate_oracle_20261103_DEV01.npz", allow_pickle=False) as bank, \
         np.load(RUN / "evaluation.npz", allow_pickle=False) as evaluation:
        for i, row in enumerate(report["rows"]):
            key = f'r{row["root"]}_{row["method"]}'
            assert [row["root"], row["method"]] == plan["groups"][i]
            assert sha(RUN / f"{key}.npz") == report["group_sha256"][key] == events[2 + 2*i]["arrays_sha256"]
            distance = evaluation[f"{key}_physical_distance"]
            assert np.array_equal(distance, bank[f"{key}_distance"])
            with np.load(RUN / f"{key}.npz", allow_pickle=False) as data:
                assert np.array_equal(data["actions"], bank[f"{key}_plans"])
                real = data["actual_rgb_cost"]
                checks = [("actual_rgb_vs_physical", real, distance, row["actual_rgb_vs_physical"])]
                for mode in ("zero", "causal", "privileged"):
                    for score in ("visual", "proprio"):
                        record = row["conditions"][mode][score]
                        for ref, target in (("vs_actual_rgb", real), ("vs_physical_distance", distance)):
                            checks.append((f"{mode}_{score}_{ref}", data[f"{mode}_{score}"], target, record[ref]))
                        choice = int(np.argmin(data[f"{mode}_{score}"]))
                        assert choice == record["posthoc_score_argmin"]
                        assert float(distance[choice]) == record["posthoc_distance_at_score_argmin"]
                for label, left, right, expected in checks:
                    actual = ranks(left, right)
                    assert actual == expected, (key, label)
                    total = totals.setdefault(label, {k: 0 for k in actual})
                    for k, value in actual.items():
                        total[k] += value
            for mode, e in events[2 + 2*i]["visual_errors"].items():
                for metric, value in e.items():
                    errors[f"{mode}_{metric}"] = max(errors.get(f"{mode}_{metric}", 0), value)
            errors["real_rgb_cost_max_abs"] = max(errors.get("real_rgb_cost_max_abs", 0),
                                                   events[2 + 2*i]["real_rgb_cost_max_abs"])
    result = {"run": str(RUN), "report_sha256": sha(RUN / "report.json"),
              "journal_sha256": sha(RUN / "events.jsonl"), "event_count": len(events),
              "counts": report["counts"], "ranking_totals_within_group_only": totals,
              "max_receipt_absolute_errors": errors, "source_closure_complete": plan["source_closure_complete"],
              "independent_outcome": False}
    with OUT.open("x") as f:
        json.dump(result, f, indent=2, sort_keys=True, allow_nan=False)
        f.write("\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
