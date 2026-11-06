"""Fixed-opportunity post-hoc comparison of response and random prefix banks."""
import json
import sys
import traceback
from pathlib import Path

import numpy as np

from experiments import pc_v25_beam_resource_curve_dev_20261104_codex as panel
from experiments import pc_v25_same_chain_search_loss_posthoc_20261104_codex as attribution
from experiments.spectral_reachability import pc_shared_multistart_cem as cem

OUT = panel.ROOT / "results/pc_v25_response_random_fixed_opportunity_posthoc_20261104_codex"
PROTOCOL = panel.ROOT.parent / "docs/PC_V25_RESPONSE_RANDOM_FIXED_OPPORTUNITY_POSTHOC_PROTOCOL_20261104_CODEX.md"
ARMS = ("response", "random")
SAMPLES, TAILS = 49, 8


def make_banks(result, root):
    layer = result["child"]["layers"][0]
    nominal = np.asarray(result["config"]["nominals"], np.float32)
    response = np.asarray([item["chain"] for item in layer["candidates"]], np.float32)
    random = []
    for index, (candidate, item) in enumerate(zip(response, layer["candidates"], strict=True)):
        anchor = nominal[item["path"][0]["nominal"]]
        norm = np.linalg.norm((candidate - anchor).astype(np.float64))
        if norm == 0:
            random.append(anchor.copy())
            continue
        rng = np.random.default_rng(np.random.SeedSequence([2031401000, root, index]))
        direction = rng.normal(size=anchor.shape)
        random.append(np.clip(anchor + direction * (norm / np.linalg.norm(direction)), -1, 1))
    return response, np.asarray(random, np.float32)


def shared_tails(root):
    rng = np.random.default_rng(np.random.SeedSequence([2031402000, root]))
    return np.concatenate((np.zeros((1, 10, 2), np.float32),
                           rng.uniform(-1, 1, (TAILS, 10, 2)).astype(np.float32)))


def initial_tail(root):
    rng = np.random.default_rng(np.random.SeedSequence([2031403000, root]))
    return rng.uniform(-1, 1, (1, 10, 2)).astype(np.float32)


def choose(scores, lookahead):
    order = np.argsort(scores, kind="stable")
    best = int(order[0])
    second = next(int(i) for i in np.argsort(lookahead, kind="stable") if i != best)
    return {"greedy": [int(i) for i in order[:2]], "lookahead": [best, second]}


def score_and_choose(predict, bank, root):
    _, scores = predict(bank)
    tails = shared_tails(root)
    lookahead = []
    for prefix in bank:
        actions = np.concatenate((np.broadcast_to(prefix, (len(tails), *prefix.shape)), tails), axis=1)
        _, values = predict(actions)
        lookahead.append(float(np.min(values)))
    return {"immediate": scores.tolist(), "lookahead": lookahead,
            "selections": choose(scores, lookahead)}


def optimize(predict, bank, selections, root):
    scores = []
    for prefix in bank:
        result = cem.optimize(lambda actions: predict(actions)[1], prefix, initial_tail(root),
                              SAMPLES, seed=2031404000 + root, batch_size=12, elites=4)
        scores.append(float(result["score"]))
    best = min(scores)
    return {"scores": scores, "all_bank_best": best,
            "rules": {name: {"indices": indices, "best_score": min(scores[i] for i in indices)}
                      for name, indices in selections.items()}}


def generation_cost(root, row):
    calls = iter(panel.source.prior.read_calls(panel.OUT / f"root{root:02d}_1920_narrow_calls"))
    stages = {}

    def predict(actions):
        source = attribution.caller_source(sys._getframe(1))
        saved, features, scores = next(calls)
        np.testing.assert_array_equal(actions, saved)
        attribution.add_batch(stages, source, saved, scores)
        return features, scores

    result = panel.plan(predict, np.asarray(row["task"]["goal"]),
                        panel.config_for(root, "narrow", 1920), "narrow")
    if next(calls, None) is not None or result["selected"] != row["result"]["selected"]:
        raise ValueError("source replay mismatch")
    return stages


def run():
    if OUT.exists():
        raise FileExistsError(OUT)
    manifest_path, audit_path = panel.OUT / "manifest.json", panel.OUT / "audit.json"
    manifest, audit = json.loads(manifest_path.read_text()), json.loads(audit_path.read_text())
    panel.seal.verify(manifest["dependencies"])
    if audit["status"] != "passed" or len(audit["rows"]) != 96:
        raise ValueError("source panel not audited")
    files = [Path(__file__), PROTOCOL, Path(panel.__file__), Path(attribution.__file__),
             Path(cem.__file__), manifest_path, audit_path]
    files += [panel.OUT / f"root{root:02d}_1920_narrow.json" for root in range(len(panel.tasks()))]
    deps = panel.seal.capture(files)
    OUT.mkdir()
    panel.journal.write_json(OUT / "manifest.json", {"dependencies": deps, "samples": SAMPLES,
        "tails": TAILS, "arms": ARMS, "scope": "exposed tasks; post-hoc fixed opportunity"})
    print("Comparing response and matched random first-layer banks under common suffix opportunity.", flush=True)
    for root, task in enumerate(panel.tasks()):
        source = json.loads((panel.OUT / f"root{root:02d}_1920_narrow.json").read_text())
        row = {"root": root, "task": task}
        try:
            if source["status"] != "completed":
                raise ValueError("source plan failed")
            banks = dict(zip(ARMS, make_banks(source["result"], root), strict=True))
            stages = generation_cost(root, source)
            row["source_stages"] = stages
            row["arms"] = {}
            for arm in ARMS:
                bank = banks[arm]
                score_journal = panel.journal.PredictionJournal(
                    OUT / f"root{root:02d}_{arm}_selection", panel.source.prior.predictor(task),
                    action_dim=2, packet_length=5, token_cap=len(bank) * (1 + (TAILS + 1) * 3))
                selection = score_and_choose(score_journal, bank, root)
                path = OUT / f"root{root:02d}_{arm}_selection.json"
                panel.journal.write_json(path, {"bank": bank.tolist(), **selection,
                                                "tokens": score_journal.tokens})
                digest = panel.seal.digest(path)
                recorder = panel.journal.PredictionJournal(
                    OUT / f"root{root:02d}_{arm}_suffix", panel.source.prior.predictor(task),
                    action_dim=2, packet_length=5, token_cap=len(bank) * SAMPLES * 3)
                result = optimize(recorder, bank, selection["selections"], root)
                if panel.seal.digest(path) != digest:
                    raise ValueError("selection changed")
                row["arms"][arm] = {"selection_sha256": digest, "selection_tokens": score_journal.tokens,
                                     "suffix_tokens": recorder.tokens, "diagnostic": result}
            row["status"] = "completed"
        except Exception:
            row.update(status="failed", error=traceback.format_exc())
        panel.journal.write_json(OUT / f"root{root:02d}.json", row)
        print(f"root={root} status={row['status']}", flush=True)
    panel.seal.verify(deps)


def audit():
    output = OUT / "audit.json"
    if output.exists():
        raise FileExistsError(output)
    manifest = json.loads((OUT / "manifest.json").read_text())
    panel.seal.verify(manifest["dependencies"])
    rows = []
    for root, task in enumerate(panel.tasks()):
        row = json.loads((OUT / f"root{root:02d}.json").read_text())
        if (row["root"], row["task"]) != (root, task):
            raise ValueError("root mismatch")
        if row["status"] != "completed":
            rows.append(row)
            continue
        source = json.loads((panel.OUT / f"root{root:02d}_1920_narrow.json").read_text())
        if generation_cost(root, source) != row["source_stages"]:
            raise ValueError("source generation cost mismatch")
        banks = dict(zip(ARMS, make_banks(source["result"], root), strict=True))
        summary = {"root": root, "status": "completed", "source_stages": row["source_stages"], "arms": {}}
        for arm in ARMS:
            path = OUT / f"root{root:02d}_{arm}_selection.json"
            if panel.seal.digest(path) != row["arms"][arm]["selection_sha256"]:
                raise ValueError("selection digest mismatch")
            saved = json.loads(path.read_text())
            np.testing.assert_array_equal(banks[arm], np.asarray(saved["bank"], np.float32))
            for stage in ("selection", "suffix"):
                directory = OUT / f"root{root:02d}_{arm}_{stage}"
                info = panel.source.prospective.journal_audit.audit_journal(directory)
                if info["tokens"] != row["arms"][arm][stage + "_tokens"]:
                    raise ValueError("journal token mismatch")
                calls = list(panel.source.prior.read_calls(directory))
                for actions, positions, scores in calls:
                    path_geo, _ = panel.OLD.previous.independent(task["start"], actions)
                    np.testing.assert_allclose(path_geo[:, -1, :2], positions, atol=1e-12, rtol=0)
                    np.testing.assert_allclose(scores, np.linalg.norm(positions - task["goal"], axis=1),
                                               atol=1e-12, rtol=0)
                replay = iter(calls)

                def predict(actions):
                    recorded, positions, scores = next(replay)
                    np.testing.assert_array_equal(actions, recorded)
                    return positions, scores

                if stage == "selection":
                    result = score_and_choose(predict, banks[arm], root)
                    for key in ("immediate", "lookahead", "selections"):
                        if result[key] != saved[key]:
                            raise ValueError("selection replay mismatch")
                else:
                    result = optimize(predict, banks[arm], saved["selections"], root)
                    if result != row["arms"][arm]["diagnostic"]:
                        raise ValueError("suffix replay mismatch")
                    if info["tokens"] != len(banks[arm]) * SAMPLES * 3:
                        raise ValueError("unequal suffix samples")
                if next(replay, None) is not None:
                    raise ValueError("unconsumed calls")
            summary["arms"][arm] = {"selection_tokens": saved["tokens"],
                                    "suffix_tokens": row["arms"][arm]["suffix_tokens"],
                                    **row["arms"][arm]["diagnostic"]}
        rows.append(summary)
        print(f"audited root={root}", flush=True)
    panel.seal.verify(manifest["dependencies"])
    panel.journal.write_json(output, {"status": "passed", "rows": rows,
        "scope": "exposed tasks; alternate same-environment geometry and deterministic replay"})


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in ("run", "audit"):
        raise SystemExit("choose run or audit")
    {"run": run, "audit": audit}[sys.argv[1]]()
