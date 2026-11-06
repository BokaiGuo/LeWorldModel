"""Prospective final-chain mechanism test of the narrow v25 development candidate."""
import itertools
import json
import sys
import traceback
from pathlib import Path
from types import FunctionType

import numpy as np

from experiments import pc_v25_e0_prospective_same_chain_dev_20261104_codex as previous
from experiments import pc_v25_staged_refinement_e0_dev_20261104_codex as staged_panel

ROOT, seal, journal, OLD, compiler = previous.ROOT, previous.seal, previous.journal, previous.OLD, previous.compiler
OUT = ROOT / "results/pc_v25_narrow_prospective_same_chain_dev_20261104_codex"
PROTOCOL = ROOT.parent / "docs/PC_V25_NARROW_PROSPECTIVE_SAME_CHAIN_DEV_PROTOCOL_20261104_CODEX.md"


def tasks():
    return [{"start": [-.5, y, theta], "goal": [.5, gy]}
            for y, theta, gy in itertools.product((-.185, .185), (-1.03, 1.03),
                                                   (-.12, -.04, .04, .12))]


def plan():
    if OUT.exists():
        raise FileExistsError(OUT)
    p = staged_panel.prior
    known = (*p.source.prior.prior_panels(), p.source.prior.tasks(), p.source.prospective.tasks(),
             p.source.prospective.cost.tasks(), p.source.tasks(), p.witness_panel.tasks(),
             p.prefix_panel.tasks(), p.random_panel.tasks(), p.probe_panel.tasks(), p.tasks(),
             staged_panel.tasks(), previous.tasks())
    if any(task in panel for task in tasks() for panel in known):
        raise ValueError("overlap with enumerated stopped panel")
    deps = seal.capture([Path(__file__), PROTOCOL, Path(previous.__file__),
        Path(compiler.__file__), Path(compiler.core.__file__), Path(compiler.v13.__file__),
        Path(compiler.core.pc_shared_multistart_cem.__file__), Path(journal.__file__),
        Path(previous.journal_audit.__file__), Path(journal.write_json.__code__.co_filename),
        Path(previous.mechanism.__file__), Path(previous.mechanism.intervention.__file__),
        Path(previous.mechanism.intervention.local.__file__), Path(previous.mechanism.intervention.core.__file__),
        Path(OLD.exact.__file__), Path(OLD.exact.wall.__file__), Path(OLD.exact.wall.physics.__file__),
        Path(OLD.previous.independent.__code__.co_filename)])
    nominals = tuple(tuple(tuple(float(x) for x in point) for point in nominal)
                     for nominal in OLD.previous.prior.core.NOMINALS)
    OUT.mkdir()
    journal.write_json(OUT / "manifest.json", {"tasks": tasks(), "cap": 2880,
        "beam_width": 2, "greedy_slots": 1, "max_modes": 2,
        "seed": "2031601000+root", "dependencies": deps,
        "scope": "prospective development, broader provenance unknown"})
    print("Sealing all narrow-v25 final plans before any mechanism interventions.", flush=True)
    rows = []
    for root, task in enumerate(tasks()):
        recorder = journal.PredictionJournal(OUT / f"plan_root{root:02d}_calls",
            previous.fixed.predictor(task), action_dim=2, packet_length=5, token_cap=2880)
        cfg = compiler.Config(action_dim=2, segment_length=5, segments=3, model_packet_length=5,
            nominals=nominals, token_cap=2880, seed=2031601000 + root, max_modes=2,
            beam_width=2, greedy_slots=1)
        try:
            result = compiler.compile_program(recorder, np.asarray(task["goal"]), cfg,
                                              score_scale=1., score_target=0.)
            if result["resource"]["prediction_tokens"] != recorder.tokens:
                raise ValueError("prediction-token mismatch")
            row = {"root": root, "task": task, "status": "completed", "result": result}
        except Exception:
            row = {"root": root, "task": task, "status": "failed", "error": traceback.format_exc()}
        row.update(journal_calls=recorder.calls, journal_tokens=recorder.tokens)
        journal.write_json(OUT / f"plan_root{root:02d}.json", row)
        rows.append(row)
        print(f"plan root={root} status={row['status']}", flush=True)
    seal.verify(deps)
    journal.write_json(OUT / "plans.json", {"manifest_sha256": seal.digest(OUT / "manifest.json"),
                                             "rows": rows})


def invoke(function):
    namespace = dict(function.__globals__)
    namespace.update({"OUT": OUT, "tasks": tasks})
    clone = FunctionType(function.__code__, namespace, function.__name__,
                         function.__defaults__, function.__closure__)
    clone.__kwdefaults__ = function.__kwdefaults__
    return clone()


if __name__ == "__main__":
    stages = {"plan": plan, "audit_plans": lambda: invoke(previous.audit_plans),
              "intervene": lambda: invoke(previous.intervene),
              "audit_interventions": lambda: invoke(previous.audit_interventions)}
    if len(sys.argv) != 2 or sys.argv[1] not in stages:
        raise SystemExit(f"choose one of {tuple(stages)}")
    stages[sys.argv[1]]()
