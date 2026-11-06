"""Comparable response/greedy/diverse/random retention and CEM on E0."""
import itertools
import json
from pathlib import Path
import traceback

import numpy as np

from experiments import pc_v17_e0_multistart_baseline_dev_20261104_codex as previous
from experiments import pc_v18_reachability_e0_dev_20261104_codex as prior_panel
from experiments import pc_v19_retention_fix_e0_dev_20261104_codex as v19_panel
from experiments import pc_v19_e0_same_chain_mechanism_dev_20261104_codex as v19_mechanism
from experiments import pc_v20_trust_e0_dev_20261104_codex as v20_panel
from experiments import pc_v21_multichain_e0_dev_20261104_codex as v21_panel
from experiments import pc_v21_e0_same_chain_mechanism_dev_20261104_codex as v21_mechanism
from experiments import pc_v22_e0_same_chain_mechanism_dev_20261104_codex as v22_mechanism
from experiments import pc_v22_reserve_e0_dev_20261104_codex as v22_panel
from experiments import pc_v22_response_retention_fixedbank_e0_dev_20261104_codex as retention_panel
from experiments.spectral_reachability import pc_global_reachability_v18 as v18
from experiments.spectral_reachability import pc_recursive_reachability_v18 as core
from experiments.spectral_reachability import pc_global_reachability_v19 as v19
from experiments.spectral_reachability import pc_recursive_reachability_v19 as core19
from experiments.spectral_reachability import pc_global_reachability_v20 as v20
from experiments.spectral_reachability import pc_recursive_reachability_v20 as core20
from experiments.spectral_reachability import pc_global_reachability_v21 as v21
from experiments.spectral_reachability import pc_recursive_reachability_v21 as core21
from experiments.spectral_reachability import pc_global_reachability_v22 as v22
from experiments.spectral_reachability import pc_recursive_reachability_v22 as core22
from experiments.spectral_reachability import pc_global_reachability_v23 as v23
from experiments.spectral_reachability import pc_recursive_reachability_v23 as core23
from experiments.spectral_reachability import pc_global_shared_cem_v23 as baseline23
from experiments.spectral_reachability import pc_shared_multistart_cem as shared_cem
from experiments.spectral_reachability import pc_global_reachability_v24 as v24
from experiments.spectral_reachability import pc_recursive_reachability_v24 as core24
from experiments.spectral_reachability import pc_global_reachability_v25 as v25
from experiments.spectral_reachability import pc_recursive_reachability_v25 as core25
from experiments.spectral_reachability import pc_timed_program_execution_v25 as timed
from experiments import pc_prediction_journal_v1 as journal
from experiments import pc_v24_global_seed_e0_dev_20261104_codex as panel24
from experiments import pc_v23_shared_cem_e0_dev_20261104_codex as panel23
from experiments import pc_v22_two_mode_prospective_mechanism_dev_20261104_codex as prospective
from experiments import pc_v22_modecount_e0_dev_20261104_codex as modecount
from experiments.spectral_reachability import pc_v3_dependency_seal as seal

ROOT = previous.ROOT
OUT = ROOT / "results/pc_v25_retention_e0_dev_20261104_codex"
PROTOCOL = ROOT.parent / "docs/PC_V25_RETENTION_E0_DEV_PROTOCOL_20261104_CODEX.md"
ARMS = ("response", "diverse", "greedy", "random", "shared_cem_v23")
CAP = 1920


def tasks():
    return [{"start": [-.5, y, theta], "goal": [.5, gy]}
            for y, theta, gy in itertools.product((-.23, .23), (-1.17, 1.17), (-.12, .12))]


def run():
    if OUT.exists():
        raise FileExistsError(OUT)
    prior, old = previous.prior, previous.prior.prior
    panels = (panel24.tasks(), panel23.tasks(), prospective.tasks(), modecount.tasks(), v22_panel.tasks(), retention_panel.tasks(), v22_mechanism.tasks(), v21_mechanism.tasks(), v21_panel.tasks(), v20_panel.tasks(),
              v19_mechanism.tasks(), v19_panel.tasks(), prior_panel.tasks(), previous.tasks(), prior.tasks(),
              prior.v15.tasks(), previous.v16.tasks(),
              prior.fixed.tasks(), *prior.v15.source_panels())
    if any(task in panel for task in tasks() for panel in panels):
        raise ValueError("task overlap in checked panels")
    deps = seal.capture([Path(__file__), PROTOCOL, Path(v22_panel.__file__), Path(retention_panel.__file__),
        Path(timed.__file__), Path(v25.__file__), Path(core25.__file__), Path(journal.__file__),
        Path(journal.write_json.__code__.co_filename),
        Path(v24.__file__), Path(core24.__file__), Path(panel23.__file__), Path(panel24.__file__),
        Path(v23.__file__), Path(core23.__file__), Path(baseline23.__file__),
        Path(shared_cem.__file__), Path(prospective.__file__), Path(modecount.__file__), Path(v18.__file__),
        Path(core.__file__), Path(v19.__file__), Path(core19.__file__), Path(v20.__file__), Path(core20.__file__),
        Path(v21.__file__), Path(core21.__file__), Path(v22.__file__), Path(core22.__file__),
        Path(prior.compiler.__file__), Path(old.compiler.__file__),
        Path(old.compiler_child.__file__), Path(previous.baseline.__file__), Path(prior.v15.prior.cem.__file__),
        Path(old.exact.__file__), Path(old.exact.wall.__file__), Path(old.exact.wall.physics.__file__),
        Path(old.previous.independent.__code__.co_filename)])
    nominals = tuple(tuple(tuple(float(x) for x in point) for point in nominal)
                     for nominal in old.previous.prior.core.NOMINALS)
    OUT.mkdir()
    old.writer(OUT / "manifest.json", {"tasks": tasks(), "arms": ARMS, "cap": CAP,
                                       "dependencies": deps, "scope": "development; provenance unknown"})
    rows = []
    for root, task in enumerate(tasks()):
        goal = np.asarray(task["goal"])

        def predict(actions):
            positions = old.exact.endpoints(task, actions)
            return positions, np.linalg.norm(positions-goal, axis=1)

        for arm in ARMS:
            config_class = v23.Config if arm == "shared_cem_v23" else v25.Config
            policy = {} if arm == "shared_cem_v23" else {"retention_policy": arm}
            config = config_class(action_dim=2, segment_length=5, segments=3, model_packet_length=5,
                                nominals=nominals, token_cap=CAP, seed=2030301000 + root,
                                max_modes=2, **policy)
            recorder = journal.PredictionJournal(OUT / f"root{root:02d}_{arm}_calls", predict,
                                                  action_dim=2, packet_length=5, token_cap=CAP)
            try:
                result = timed.execute(recorder, goal, config,
                                       arm="shared_cem_v23" if arm == "shared_cem_v23" else "v25",
                                       baseline_seed=2030302000+root*10)
                trace = result.pop("traces")
                if result["resource"]["prediction_tokens"] != recorder.tokens:
                    raise ValueError("persistent journal meter mismatch")
                error = None
            except Exception:
                result, trace, error = None, [], traceback.format_exc()
            path = OUT / f"root{root:02d}_{arm}.npz"
            with path.open("xb") as stream:
                np.savez_compressed(stream, **{f"{name}_{i}": batch[name]
                    for i, batch in enumerate(trace) for name in ("actions", "features", "scores")})
            rows.append({"root": root, "task": task, "arm": arm, "status": "failed" if error else "completed",
                         "error": error, "result": result, "trace": path.name, "trace_sha256": seal.digest(path),
                         "journal": recorder.directory.name, "journal_calls": recorder.calls,
                         "journal_tokens": recorder.tokens})
            print(f"root={root} arm={arm} status={rows[-1]['status']}", flush=True)
    seal.verify(deps)
    old.writer(OUT / "plans.json", {"manifest_sha256": seal.digest(OUT / "manifest.json"), "rows": rows})


def audit():
    output = OUT / "audit.json"
    if output.exists():
        raise FileExistsError(output)
    manifest = json.loads((OUT / "manifest.json").read_text())
    plans = json.loads((OUT / "plans.json").read_text())
    seal.verify(manifest["dependencies"])
    if (plans["manifest_sha256"] != seal.digest(OUT / "manifest.json")
            or manifest["tasks"] != tasks() or manifest["arms"] != list(ARMS) or len(plans["rows"]) != 40):
        raise ValueError("source binding mismatch")
    old, summary = previous.prior.prior, []
    for root, task in enumerate(tasks()):
        banks = []
        for arm in ARMS:
            matches = [row for row in plans["rows"] if row["root"] == root and row["arm"] == arm]
            if len(matches) != 1:
                raise ValueError("missing arm")
            row = matches[0]
            journal_dir = OUT / row["journal"]
            attempts = sorted(journal_dir.glob("*_attempt.json"))
            if len(attempts) != row["journal_calls"]:
                raise ValueError("journal attempt count mismatch")
            journal_tokens = 0
            for index, attempt_path in enumerate(attempts):
                attempt = json.loads(attempt_path.read_text())
                if (attempt["index"] != index
                        or journal.digest(journal_dir / attempt["actions"]) != attempt["sha256"]):
                    raise ValueError("journal action mismatch")
                journal_tokens += attempt["cost"]
            if journal_tokens != row["journal_tokens"] or journal_tokens > CAP:
                raise ValueError("journal budget mismatch")
            if row["status"] != "completed":
                summary.append({"root": root, "arm": arm, "status": "failed", "error": row["error"]})
                continue
            if row["task"] != task or seal.digest(OUT / row["trace"]) != row["trace_sha256"]:
                raise ValueError("trace mismatch")
            record, tokens, best, found = row["result"], 0, float("inf"), False
            selected = np.asarray(record["result"]["selected"], np.float32)
            with np.load(OUT / row["trace"], allow_pickle=False) as saved:
                banks.append(saved["actions_0"].copy())
                for i, call in enumerate(record["calls"]):
                    actions, positions, scores = (saved[f"{name}_{i}"] for name in ("actions", "features", "scores"))
                    returned = json.loads((journal_dir / f"{i:06d}_returned.json").read_text())
                    if journal.digest(journal_dir / returned["result"]) != returned["sha256"]:
                        raise ValueError("journal result digest mismatch")
                    with np.load(journal_dir / returned["result"], allow_pickle=False) as persisted:
                        np.testing.assert_array_equal(persisted["features"], positions)
                        np.testing.assert_array_equal(persisted["scores"], scores)
                    with np.load(journal_dir / f"{i:06d}_actions.npz", allow_pickle=False) as persisted:
                        np.testing.assert_array_equal(persisted["actions"], actions)
                    if (actions.dtype != np.float32 or actions.shape[1] % 5 or not np.isfinite(actions).all()
                            or np.any(np.abs(actions) > 1)):
                        raise ValueError("invalid actions")
                    path, _ = old.previous.independent(task["start"], actions)
                    np.testing.assert_allclose(path[:, -1, :2], positions, atol=1e-12, rtol=0)
                    np.testing.assert_allclose(scores, np.linalg.norm(positions-task["goal"], axis=1),
                                               atol=1e-12, rtol=0)
                    cost = len(actions) * actions.shape[1] // 5
                    if cost != call["prediction_tokens"]:
                        raise ValueError("per-call token mismatch")
                    tokens += cost
                    if actions.shape[1] == 15:
                        best = min(best, float(scores.min()))
                        found |= bool(np.any(np.all(actions == selected, axis=(1, 2))
                                             & (np.abs(scores-record["result"]["selected_score"]) < 1e-12)))
            if (tokens != record["resource"]["prediction_tokens"] or tokens != journal_tokens or tokens > CAP or not found
                    or record["result"]["selected_score"] > best + 1e-12):
                raise ValueError("selection or budget mismatch")
            probes = (sum(len(layer["reachability"]) for layer in record["result"]["child"]["layers"])
                      if arm != "shared_cem_v23" else 0)
            candidates = ([len(layer["candidates"]) for layer in record["result"]["child"]["layers"]]
                          if arm != "shared_cem_v23" else None)
            changed = (sum(bool(layer["response_added"]) for layer in record["result"]["child"]["layers"])
                       if arm != "shared_cem_v23" else 0)
            summary.append({"root": root, "arm": arm, "status": "completed",
                            "score": record["result"]["selected_score"],
                            "success": record["result"]["selected_score"] < .05,
                            "tokens": tokens, "successor_probes": probes, "layer_candidates": candidates,
                            "same_bank_response_changed_layers": changed})
        for bank in banks[1:]:
            np.testing.assert_array_equal(bank, banks[0])
    seal.verify(manifest["dependencies"])
    old.writer(output, {"status": "passed", "plans_sha256": seal.digest(OUT / "plans.json"), "rows": summary,
                        "scope": "alternate same-environment geometry; not independent replication"})
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2 or sys.argv[1] not in ("run", "audit"):
        raise SystemExit("choose run or audit")
    {"run": run, "audit": audit}[sys.argv[1]]()
