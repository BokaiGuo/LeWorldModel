"""Replay lookahead selection, suffix optimizers and metrics from saved calls."""
import json

import numpy as np

from experiments import pc_retention_mechanism_next_dev as source


def main():
    out = source.OUT
    target = out / "interventions_algorithm_audit.json"
    if target.exists():
        raise FileExistsError(target)
    manifest = json.loads((out / "manifest.json").read_text())
    data = json.loads((out / "interventions.json").read_text())
    plans = json.loads((out / "plans.json").read_text())
    source.seal.verify(manifest["dependencies"])
    for row, plan in zip(data["rows"], plans["rows"], strict=True):
        if row["status"] != "completed":
            continue
        root, arm = row["root"], row["arm"]
        calls = iter(source.old.previous.fixed.read_calls(out / f"intervention_{root:02d}_{arm}_calls"))
        tokens = 0

        def predict(actions):
            nonlocal tokens
            recorded, positions, scores = next(calls)
            raw = np.asarray(actions, np.float32)
            np.testing.assert_array_equal(raw, recorded)
            tokens += len(raw) * raw.shape[1] // 5
            return positions, scores

        layer = plan["result"]["child"]["layers"][0]
        bank = [np.asarray(x["chain"], np.float32) for x in layer["candidates"]]
        bank.append(np.zeros((5, 2), np.float32))
        values = []
        for prefix in bank:
            zero = np.concatenate((prefix, np.zeros((10, 2), np.float32)))[None]
            initial = float(predict(zero)[1][0])
            _, score, _ = source.cem.cem(np.zeros((10, 2)), prefix, 119,
                np.random.default_rng(2032600000 + root), predict, initial, 4)
            values.append(score)
        if values != row["lookahead_scores"] or int(np.argmin(values)) != row["lookahead_index"]:
            raise ValueError("lookahead replay mismatch")
        if tokens != row["lookahead_selection_tokens"]:
            raise ValueError("lookahead selection budget mismatch")
        chain = np.asarray(plan["result"]["selected"], np.float32)
        greedy = int(np.argmin([x["score"] for x in layer["candidates"]]))
        rng = np.random.default_rng(2032200000 + root)
        random = np.clip(source.cem.interpolate(rng.normal(size=(1, 4, 2)), 5)[0], -1, 1).astype(np.float32)
        controls = {"selected": chain[:5], "zero": np.zeros((5, 2), np.float32),
                    "greedy": bank[greedy], "lookahead": bank[row["lookahead_index"]], "random": random}
        tails = [np.tile(x, (10, 1)) for x in ((0., 0.), (.5, 0.), (-.5, 0.), (.5, .5), (.5, -.5))]
        tails.append(chain[5:])
        for name, prefix in controls.items():
            saved = row["suffix"][name]
            np.testing.assert_array_equal(prefix, np.asarray(saved["prefix"], np.float32))
            common = np.asarray([np.concatenate((prefix, tail)) for tail in tails], np.float32)
            _, scores = predict(common)
            np.testing.assert_array_equal(scores, saved["common_distances"])
            np.testing.assert_array_equal(scores - scores[0], saved["relative_zero"])
            for repeat in range(3):
                zero = np.concatenate((prefix, np.zeros((10, 2), np.float32)))[None]
                initial = float(predict(zero)[1][0])
                best, score, _ = source.cem.cem(np.zeros((10, 2)), prefix, 480,
                    np.random.default_rng(2032300000 + root * 10 + repeat), predict, initial, 4)
                if score != saved["runs"][repeat]["score"]:
                    raise ValueError("replan score mismatch")
                np.testing.assert_array_equal(best, saved["runs"][repeat]["chain"])
        for name, prefix in controls.items():
            for eps in (.01, .05):
                request = source.response.prepare(chain, prefix, epsilon=eps)
                positions, scores = predict(request["actions"])
                saved = row["measurements"][name][str(eps)]
                np.testing.assert_array_equal(positions, saved["endpoint_positions"])
                np.testing.assert_array_equal(scores, saved["scores"])
        _, prefix_scores = predict(chain[:5][None])
        nonprogress = bool(prefix_scores[0] >= np.linalg.norm(np.asarray(plan["task"]["start"][:2]) -
                                                               plan["task"]["goal"]) - 1e-10)
        if nonprogress != row["nonprogress"] or tokens != row["tokens"] or next(calls, None) is not None:
            raise ValueError("nonprogress or consumption mismatch")
        for name, comp in row["comparisons"].items():
            selected, other = row["suffix"]["selected"], row["suffix"][name]
            common = selected["relative_zero"][-1] < other["relative_zero"][-1] - 1e-10
            replan = np.mean([x["score"] for x in selected["runs"]]) < np.mean(
                [x["score"] for x in other["runs"]]) - 1e-10
            if bool(common) != comp["common_gain"] or bool(replan) != comp["replan_gain"]:
                raise ValueError("gain predicate mismatch")
        print(f"replayed root={root} arm={arm}", flush=True)
    source.seal.verify(manifest["dependencies"])
    source.journal.write_json(target, {"status": "passed",
        "interventions_sha256": source.seal.digest(out / "interventions.json"),
        "scope": "same-code deterministic algorithm replay, not independent replication"})


if __name__ == "__main__":
    main()
