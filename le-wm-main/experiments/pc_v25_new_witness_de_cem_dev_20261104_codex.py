"""Fixed-budget conditional suffix optimizer diagnostic on new E0 tasks."""
import itertools
import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np

from experiments import pc_v25_scale_feasibility_posthoc_20261104_codex as feasibility
from experiments import pc_v25_witness_suffix_expression_dev_20261104_codex as old

source, cem = old.source, old.cem
OUT = source.ROOT / "results/pc_v25_new_witness_de_cem_dev_20261104_codex"
PROTOCOL = source.ROOT.parent / "docs/PC_V25_NEW_WITNESS_DE_CEM_DEV_PROTOCOL_20261104_CODEX.md"
ARMS = ("cem", "de")
SAMPLES, RESTARTS, POP = 121, 2, 12


def tasks():
    return [{"start": [-.5, y, theta], "goal": [.5, gy]}
            for y, theta, gy in itertools.product((-.145, .145), (-1.12, 1.12), (-.10, .10))]


def initial(root, index, restart):
    rng = np.random.default_rng(np.random.SeedSequence([2031101000, root, index, restart]))
    return rng.uniform(-1, 1, (10, 2)).astype(np.float32)


def de_optimize(predict, prefix, start, seed):
    rng = np.random.default_rng(seed)
    pop = rng.uniform(-1, 1, (POP, 10, 2)).astype(np.float32)
    pop[0] = start

    def evaluate(tails):
        chains = np.concatenate((np.broadcast_to(prefix, (len(tails), 5, 2)), tails), axis=1)
        scores = np.asarray(predict(chains)[1], dtype=np.float64)
        if scores.shape != (len(tails),) or not np.isfinite(scores).all():
            raise ValueError("invalid predictor scores")
        return scores

    scores = evaluate(pop)
    used = POP
    while used < SAMPLES:
        size = min(POP, SAMPLES - used)
        trials = []
        for i in range(size):
            available = np.delete(np.arange(POP), i)
            a, b, c = rng.choice(available, 3, replace=False)
            mutant = pop[a].astype(np.float64) + .8 * (pop[b] - pop[c])
            mask = rng.random((10, 2)) < .9
            mask.flat[int(rng.integers(mask.size))] = True
            trials.append(np.clip(np.where(mask, mutant, pop[i]), -1, 1))
        trials = np.asarray(trials, np.float32)
        values = evaluate(trials)
        accepted = values < scores[:size]
        pop[:size][accepted] = trials[accepted]
        scores[:size][accepted] = values[accepted]
        used += size
    winner = int(np.argmin(scores))
    return {"score": float(scores[winner]),
            "actions": np.concatenate((prefix, pop[winner])).tolist(), "samples": used}


def optimize(predict, prefixes, root, arm):
    records = []
    for index, prefix in enumerate(np.asarray(prefixes, np.float32)):
        runs = []
        for restart in range(RESTARTS):
            start = initial(root, index, restart)
            seed = 2031102000 + root * 100 + index * 10 + restart
            if arm == "cem":
                result = cem.optimize(lambda chains: predict(chains)[1], prefix, start[None],
                                      SAMPLES, seed=seed, batch_size=12, elites=4)
                record = {"score": result["score"], "actions": result["actions"].tolist(),
                          "samples": result["samples"]}
            elif arm == "de":
                record = de_optimize(predict, prefix, start, seed)
            else:
                raise ValueError("invalid arm")
            runs.append(record)
        records.append({"runs": runs, "best_score": min(r["score"] for r in runs)})
    return records


def run():
    if OUT.exists():
        raise FileExistsError(OUT)
    existing = (*source.prior.prior_panels(), source.prior.tasks(), source.prospective.tasks(),
                source.prospective.cost.tasks(), source.tasks())
    if any(task in panel for task in tasks() for panel in existing):
        raise ValueError("overlap with enumerated prior development panel")
    deps = source.seal.capture([Path(__file__), PROTOCOL, Path(feasibility.__file__),
        Path(cem.__file__), Path(source.prior.predictor.__code__.co_filename),
        Path(source.OLD.exact.__file__), Path(source.OLD.exact.wall.__file__),
        Path(source.OLD.exact.wall.physics.__file__),
        Path(source.OLD.previous.independent.__code__.co_filename)])
    OUT.mkdir()
    source.journal.write_json(OUT / "manifest.json", {"tasks": tasks(), "arms": ARMS,
        "dependencies": deps, "scope": "new development; conditional privileged prefix diagnostic"})
    print("Comparing DE and CEM on new witnessed prefixes at equal complete-chain budget.", flush=True)
    for root, task in enumerate(tasks()):
        witnesses = feasibility.calculate(task)
        valid = [w for w in witnesses if w.get("success")]
        source.journal.write_json(OUT / f"root{root:02d}_witness.json",
                                  {"root": root, "task": task, "witnesses": witnesses})
        if len(valid) < 2:
            print(f"root={root} insufficient witnessed prefixes; unknown", flush=True)
            continue
        prefixes = np.asarray([w["actions"][:5] for w in valid[:2]], np.float32)
        for arm in ARMS:
            recorder = source.journal.PredictionJournal(
                OUT / f"root{root:02d}_{arm}_calls", source.prior.predictor(task),
                action_dim=2, packet_length=5, token_cap=2 * RESTARTS * SAMPLES * 3)
            started = time.monotonic()
            try:
                result = optimize(recorder, prefixes, root, arm)
                row = {"root": root, "task": task, "arm": arm, "status": "completed",
                       "prefixes": result}
            except Exception:
                row = {"root": root, "task": task, "arm": arm, "status": "failed",
                       "error": traceback.format_exc()}
            row.update(tokens=recorder.tokens, seconds_with_journal=time.monotonic() - started)
            source.journal.write_json(OUT / f"root{root:02d}_{arm}.json", row)
            print(f"root={root} arm={arm} status={row['status']} tokens={recorder.tokens}", flush=True)
    source.seal.verify(deps)


def audit():
    output = OUT / "audit.json"
    if output.exists():
        raise FileExistsError(output)
    manifest = json.loads((OUT / "manifest.json").read_text())
    source.seal.verify(manifest["dependencies"])
    if manifest["tasks"] != tasks() or manifest["arms"] != list(ARMS):
        raise ValueError("manifest mismatch")
    rows = []
    for root, task in enumerate(tasks()):
        saved = json.loads((OUT / f"root{root:02d}_witness.json").read_text())
        if saved != {"root": root, "task": task, "witnesses": feasibility.calculate(task)}:
            raise ValueError("witness replay mismatch")
        valid = [w for w in saved["witnesses"] if w.get("success")]
        if len(valid) < 2:
            rows.append({"root": root, "status": "unknown_insufficient_witnesses"})
            continue
        prefixes = np.asarray([w["actions"][:5] for w in valid[:2]], np.float32)
        for arm in ARMS:
            row = json.loads((OUT / f"root{root:02d}_{arm}.json").read_text())
            if (row["root"], row["task"], row["arm"]) != (root, task, arm):
                raise ValueError("identity mismatch")
            directory = OUT / f"root{root:02d}_{arm}_calls"
            info = source.prospective.journal_audit.audit_journal(directory)
            if info["tokens"] != row["tokens"]:
                raise ValueError("token mismatch")
            if row["status"] != "completed":
                rows.append(row)
                continue
            calls = list(source.prior.read_calls(directory))
            for actions, positions, scores in calls:
                path, _ = source.OLD.previous.independent(task["start"], actions)
                np.testing.assert_allclose(path[:, -1, :2], positions, atol=1e-12, rtol=0)
                np.testing.assert_allclose(scores, np.linalg.norm(positions - task["goal"], axis=1),
                                           atol=1e-12, rtol=0)
            replay = iter(calls)

            def predict(actions):
                recorded, positions, scores = next(replay)
                np.testing.assert_array_equal(actions, recorded)
                return positions, scores

            result = optimize(predict, prefixes, root, arm)
            if next(replay, None) is not None or result != row["prefixes"]:
                raise ValueError("optimizer replay mismatch")
            if info["tokens"] != 2 * RESTARTS * SAMPLES * 3:
                raise ValueError("budget mismatch")
            rows.append({"root": root, "arm": arm, "status": "completed",
                         "scores": [item["best_score"] for item in result],
                         "tokens": info["tokens"], "seconds_with_journal": row["seconds_with_journal"]})
            print(f"audited root={root} arm={arm}", flush=True)
    source.seal.verify(manifest["dependencies"])
    source.journal.write_json(output, {"status": "passed", "rows": rows,
        "scope": "same-environment alternate geometry and proposal replay; not independent replication"})


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in ("run", "audit"):
        raise SystemExit("choose run or audit")
    {"run": run, "audit": audit}[sys.argv[1]]()
