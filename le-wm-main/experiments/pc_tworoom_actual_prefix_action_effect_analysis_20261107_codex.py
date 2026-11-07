"""Audit action-effect measurements and report root-level development summaries."""
import json

import numpy as np

from experiments import pc_tworoom_actual_prefix_action_effect_20261107_codex as run
from experiments.pc_input_permissions import file_hash


def signed(values, tolerance):
    array = np.asarray(values)
    return np.where(array > tolerance, 1, np.where(array < -tolerance, -1, 0))


def effects(arm):
    distances = np.asarray([trace[-1]["distance"] for trace in arm["trace"]])
    predicted = np.asarray(arm["predicted_cost"])
    encoded = np.asarray(arm["encoded_cost"])
    return distances[0] - distances[1:], predicted[0] - predicted[1:], encoded[0] - encoded[1:]


def agreement(first, second, tol_first, tol_second):
    a, b = signed(first, tol_first), signed(second, tol_second)
    keep = (a != 0) & (b != 0)
    return int((a[keep] == b[keep]).sum()), int(keep.sum()), int(len(a) - keep.sum())


def ranking_flip(a, b, tolerance=1e-5):
    a, b = np.asarray(a), np.asarray(b)
    return any(signed([a[i] - a[j]], tolerance)[0] * signed([b[i] - b[j]], tolerance)[0] == -1
               for i in range(len(a)) for j in range(i + 1, len(a)))


def main():
    data = json.loads(run.OUT.read_text())
    if (data["source_sha256"] != file_hash(run.source.OUTPUT) or data["protocol_sha256"] != file_hash(run.PROTOCOL)
            or data["runner_sha256"] != file_hash(run.__file__) or data["feature_sha256"] != file_hash(run.FEATURES)):
        raise ValueError("artifact drift")
    rows = [r for r in data["rows"] if "ineligible_reason" not in r]
    assert len(data["rows"]) == 40 and len(rows) == 35
    num_probes = 0
    with np.load(run.FEATURES, allow_pickle=False) as features:
        for row in rows:
            for name in run.source.ARMS:
                arm = row["arms"][name]
                key = f"e{row['episode']}_{name}"
                p, a, g = features[key + "_predicted"], features[key + "_actual"], features[key + "_goal"]
                np.testing.assert_allclose(np.square(p - g).sum(-1), arm["predicted_cost"], atol=1e-3, rtol=0)
                np.testing.assert_allclose(np.square(a - g).sum(-1), arm["encoded_cost"], atol=1e-3, rtol=0)
                assert p.shape == a.shape and len(arm["trace"]) == len(row["probes"])
                for trace in arm["trace"]:
                    assert len(trace) == 5 and all(x["success"] == (x["distance"] < 16) for x in trace)
                num_probes += len(arm["trace"])
    print("eligible roots", len(rows), "physical probe rollouts", num_probes)
    for arm in ("actual", "zero", "lookahead", "random"):
        real, pred, enc = zip(*(effects(row["arms"][arm]) for row in rows))
        # Only four fixed cardinal probes are common across every root.
        real = np.concatenate([v[:4] for v in real])
        pred = np.concatenate([v[:4] for v in pred])
        enc = np.concatenate([v[:4] for v in enc])
        print(arm, "pred_vs_physical", agreement(pred, real, 1e-6, 1e-5),
              "encoded_vs_physical", agreement(enc, real, 1e-6, 1e-5),
              "pred_vs_encoded", agreement(pred, enc, 1e-6, 1e-6))
    eligible = [r for r in rows if not any(s["success"] for s in r["arms"]["actual"]["prefix_trace"])]
    flips = sum(ranking_flip(effects(r["arms"]["actual"])[0][:4], effects(r["arms"]["zero"])[0][:4])
                for r in eligible)
    print("actual prefix not successful", len(eligible), "physical action-ranking flips vs zero", flips)
    tail_rows = [r for r in eligible if len(r["probes"]) == 6]
    actual_pos_zero_nonpos = sum(effects(r["arms"]["actual"])[0][-1] > 1e-5
                                 and effects(r["arms"]["zero"])[0][-1] <= 1e-5 for r in tail_rows)
    print("available actual 5-step suffix after unsuccessful prefix", len(tail_rows),
          "positive after actual and nonpositive after zero", actual_pos_zero_nonpos)


if __name__ == "__main__":
    main()
