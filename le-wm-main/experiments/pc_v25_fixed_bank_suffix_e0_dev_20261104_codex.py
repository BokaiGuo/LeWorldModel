"""Prospective v25 fixed-first-layer-bank, common suffix diagnostic."""
import itertools
import json
from pathlib import Path
import sys
import traceback

import numpy as np

from experiments import pc_prediction_journal_audit_v1 as journal_audit
from experiments import pc_prediction_journal_v1 as journal
from experiments import pc_v25_retention_e0_dev_20261104_codex as prior
from experiments.spectral_reachability import pc_fixed_bank_suffix_diagnostic_v1 as suffix
from experiments.spectral_reachability import pc_global_reachability_v25 as compiler
from experiments.spectral_reachability import pc_recursive_reachability_v25 as core
from experiments.spectral_reachability import pc_shared_multistart_cem as cem
from experiments.spectral_reachability import pc_timed_program_execution_v25 as timed
from experiments.spectral_reachability import pc_v3_dependency_seal as seal

ROOT = prior.ROOT
OUT = ROOT / "results/pc_v25_fixed_bank_suffix_e0_dev_20261104_codex"
PROTOCOL = ROOT.parent / "docs/PC_V25_FIXED_BANK_SUFFIX_E0_DEV_PROTOCOL_20261104_CODEX.md"
OLD = prior.previous.prior.prior
SAMPLES, RESTARTS, LOOKAHEAD = 121, 2, 24


def tasks():
    return [{"start": [-.5, y, theta], "goal": [.5, .16]}
            for y, theta in itertools.product((-.21, .21), (-1.19, 1.19))]


def prior_panels():
    p = prior.previous.prior
    return (prior.tasks(), prior.panel24.tasks(), prior.panel23.tasks(),
            prior.prospective.tasks(), prior.modecount.tasks(), prior.v22_panel.tasks(),
            prior.retention_panel.tasks(), prior.v22_mechanism.tasks(), prior.v21_mechanism.tasks(),
            prior.v21_panel.tasks(), prior.v20_panel.tasks(), prior.v19_mechanism.tasks(),
            prior.v19_panel.tasks(), prior.prior_panel.tasks(), prior.previous.tasks(), p.tasks(),
            p.v15.tasks(), prior.previous.v16.tasks(), p.fixed.tasks(), *p.v15.source_panels())


def make_selections(layer, lookahead):
    selections = {key: list(value) for key, value in layer["retention_alternatives"].items()}
    scores = np.array([item["score"] for item in layer["candidates"]])
    count = len(selections["greedy"])
    selected = np.argsort(scores, kind="stable")[:2].tolist()
    for index in np.argsort(lookahead, kind="stable"):
        if int(index) not in selected:
            selected.append(int(index))
        if len(selected) == count:
            break
    selections["lookahead"] = selected
    return selections


def predictor(task):
    goal = np.asarray(task["goal"])
    def evaluate(actions):
        positions = OLD.exact.endpoints(task, actions)
        return positions, np.linalg.norm(positions - goal, axis=1)
    return evaluate


def run():
    if OUT.exists():
        raise FileExistsError(OUT)
    if any(task in panel for task in tasks() for panel in prior_panels()):
        raise ValueError("overlap with enumerated panels")
    dependencies = seal.capture([Path(__file__), PROTOCOL, Path(prior.__file__),
        Path(compiler.__file__), Path(core.__file__), Path(timed.__file__),
        Path(suffix.__file__), Path(cem.__file__), Path(journal.__file__),
        Path(journal_audit.__file__), Path(journal.write_json.__code__.co_filename),
        Path(OLD.exact.__file__), Path(OLD.exact.wall.__file__),
        Path(OLD.exact.wall.physics.__file__), Path(OLD.previous.independent.__code__.co_filename),
        Path(compiler.v13.__file__)])
    nominals = tuple(tuple(tuple(float(x) for x in point) for point in nominal)
                     for nominal in OLD.previous.prior.core.NOMINALS)
    OUT.mkdir()
    OLD.writer(OUT / "manifest.json", {"tasks": tasks(), "dependencies": dependencies,
        "scope": "fixed bank development; wider provenance unknown", "samples": SAMPLES,
        "restarts": RESTARTS, "lookahead": LOOKAHEAD})
    print("Evaluating equal suffix opportunities on a frozen v25 first-layer bank.", flush=True)
    for root, task in enumerate(tasks()):
        row = {"root": root, "task": task}
        try:
            predict = predictor(task)
            config = compiler.Config(action_dim=2, segment_length=5, segments=3,
                model_packet_length=5, nominals=nominals, token_cap=1920,
                seed=2030401000 + root, max_modes=2, retention_policy="response")
            gen = journal.PredictionJournal(OUT / f"root{root:02d}_generation", predict,
                                            action_dim=2, packet_length=5, token_cap=1920)
            compiled = timed.execute(gen, np.asarray(task["goal"]), config, arm="v25")
            compiled.pop("traces")
            if compiled["resource"]["prediction_tokens"] != gen.tokens:
                raise ValueError("generation meter mismatch")
            layer = compiled["result"]["child"]["layers"][0]
            prefixes = np.asarray([item["chain"] for item in layer["candidates"]], np.float32)
            scout = journal.PredictionJournal(OUT / f"root{root:02d}_lookahead", predict,
                                              action_dim=2, packet_length=5, token_cap=10000)
            lookahead = []
            # Shared tails avoid conflating retention quality with different random samples.
            tails = np.random.default_rng(2030402000 + root).uniform(
                -1, 1, (LOOKAHEAD, 10, 2)).astype(np.float32)
            tails = np.concatenate((np.zeros((1, 10, 2), np.float32), tails))
            for prefix in prefixes:
                actions = np.concatenate((np.broadcast_to(prefix, (len(tails), 5, 2)), tails), axis=1)
                _, scores = scout(actions)
                lookahead.append(float(scores.min()))
            selections = make_selections(layer, lookahead)
            selection_path = OUT / f"root{root:02d}_selection.json"
            OLD.writer(selection_path, {"root": root, "task": task, "layer": layer,
                "lookahead": lookahead, "selections": selections,
                "generation_tokens": gen.tokens, "lookahead_tokens": scout.tokens,
                "generation_calls": gen.calls, "lookahead_calls": scout.calls})
            selection_hash = seal.digest(selection_path)
            evaluation = journal.PredictionJournal(OUT / f"root{root:02d}_suffix", predict,
                                                   action_dim=2, packet_length=5, token_cap=50000)
            results = suffix.evaluate_bank(evaluation, prefixes, selections, tail_steps=10,
                samples=SAMPLES, restarts=RESTARTS, seed=2030403000 + root,
                batch_size=12, elites=4)
            if seal.digest(selection_path) != selection_hash:
                raise ValueError("selection changed after freezing")
            row.update(status="completed", selection=selection_path.name,
                       selection_sha256=selection_hash, suffix_tokens=evaluation.tokens,
                       suffix_calls=evaluation.calls, diagnostic=results)
        except Exception:
            row.update(status="failed", error=traceback.format_exc())
        OLD.writer(OUT / f"root{root:02d}_result.json", row)
        print(f"root={root} status={row['status']}", flush=True)
    seal.verify(dependencies)


def read_calls(directory):
    info = journal_audit.audit_journal(directory)
    if any(call["status"] != "returned" for call in info["calls"]):
        raise ValueError("journal has a nonreturned call")
    for call in info["calls"]:
        index = call["index"]
        with np.load(directory / f"{index:06d}_actions.npz", allow_pickle=False) as a:
            actions = a["actions"].copy()
        with np.load(directory / f"{index:06d}_result.npz", allow_pickle=False) as r:
            yield actions, r["features"].copy(), r["scores"].copy()


def audit():
    if (OUT / "audit.json").exists():
        raise FileExistsError(OUT / "audit.json")
    manifest = json.loads((OUT / "manifest.json").read_text())
    seal.verify(manifest["dependencies"])
    if manifest["tasks"] != tasks():
        raise ValueError("manifest task mismatch")
    summaries = []
    for root, task in enumerate(tasks()):
        row = json.loads((OUT / f"root{root:02d}_result.json").read_text())
        if row["root"] != root or row["task"] != task:
            raise ValueError("root mismatch")
        if row["status"] != "completed":
            summaries.append({"root": root, "status": "failed", "error": row["error"]})
            continue
        selection_path = OUT / row["selection"]
        if seal.digest(selection_path) != row["selection_sha256"]:
            raise ValueError("selection hash mismatch")
        selected = json.loads(selection_path.read_text())
        layer = selected["layer"]
        prefixes = np.asarray([item["chain"] for item in layer["candidates"]], np.float32)
        directories = [OUT / f"root{root:02d}_{name}" for name in ("generation", "lookahead", "suffix")]
        counts = [journal_audit.audit_journal(d) for d in directories]
        if ([x["tokens"] for x in counts] != [selected["generation_tokens"], selected["lookahead_tokens"],
                                               row["suffix_tokens"]]):
            raise ValueError("token totals differ")
        all_calls = [list(read_calls(d)) for d in directories]
        # Recompute every recorded trajectory using alternate geometry.
        for calls in all_calls:
            for actions, positions, scores in calls:
                path, _ = OLD.previous.independent(task["start"], actions)
                np.testing.assert_allclose(path[:, -1, :2], positions, atol=1e-12, rtol=0)
                np.testing.assert_allclose(scores, np.linalg.norm(positions - task["goal"], axis=1),
                                           atol=1e-12, rtol=0)
        if len(all_calls[1]) != len(prefixes):
            raise ValueError("missing lookahead")
        expected_tails = np.random.default_rng(2030402000 + root).uniform(
            -1, 1, (LOOKAHEAD, 10, 2)).astype(np.float32)
        expected_tails = np.concatenate((np.zeros((1, 10, 2), np.float32), expected_tails))
        for i, (actions, _, scores) in enumerate(all_calls[1]):
            np.testing.assert_array_equal(actions[:, :5], np.broadcast_to(prefixes[i], (len(actions), 5, 2)))
            np.testing.assert_array_equal(actions[:, 5:], expected_tails)
            np.testing.assert_allclose(scores.min(), selected["lookahead"][i], atol=1e-12, rtol=0)
        if make_selections(layer, selected["lookahead"]) != selected["selections"]:
            raise ValueError("selection mismatch")
        # Replay the optimizer against the stored calls: detects changed proposals,
        # budgets, prefix mapping, and reported winners, without new physics calls.
        replay = iter(all_calls[2])
        def replay_predict(actions):
            recorded, features, scores = next(replay)
            np.testing.assert_array_equal(actions, recorded)
            return features, scores
        recomputed = suffix.evaluate_bank(replay_predict, prefixes, selected["selections"],
            tail_steps=10, samples=SAMPLES, restarts=RESTARTS, seed=2030403000 + root,
            batch_size=12, elites=4)
        if next(replay, None) is not None or recomputed != row["diagnostic"]:
            raise ValueError("suffix replay mismatch")
        if row["suffix_tokens"] != len(prefixes) * SAMPLES * RESTARTS * 3:
            raise ValueError("suffix sample count mismatch")
        summaries.append({"root": root, "status": "completed", "candidate_count": len(prefixes),
            "response_added": layer["response_added"], "response_removed": layer["response_removed"],
            "all_bank_best": recomputed["all_bank_best"], "rules": recomputed["rules"],
            "generation_tokens": counts[0]["tokens"], "lookahead_tokens": counts[1]["tokens"],
            "suffix_tokens": counts[2]["tokens"]})
    seal.verify(manifest["dependencies"])
    OLD.writer(OUT / "audit.json", {"status": "passed", "rows": summaries,
        "scope": "alternate same-environment geometry, not independent replication"})
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in ("run", "audit"):
        raise SystemExit("choose run or audit")
    {"run": run, "audit": audit}[sys.argv[1]]()
