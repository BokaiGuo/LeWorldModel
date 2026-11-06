"""Run a fixed development timing schedule; each result has a unique path."""
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MEAN = "+solver.action_mean=[-0.007812564379916172,0.006860687229453032]"
SCALE = "+solver.action_scale=[0.20846744284501714,0.20674862637362224]"
CONFIGS = {
    "cem": ["solver._target_=experiments.pc_feedback_cem_20261106_codex.FeedbackCEM",
            "+solver.selection=mean", "solver.num_samples=105", "solver.n_steps=6", "solver.topk=10"],
    "prefix": ["solver._target_=experiments.pc_explicit_prefix_program_20261106_codex.ExplicitPrefixProgram",
               "+solver.prefix_mode=lookahead", "solver.num_samples=100", "solver.n_steps=3", "solver.topk=10"],
    "cem300x10": ["solver._target_=experiments.pc_feedback_cem_20261106_codex.FeedbackCEM",
                  "+solver.selection=mean", "solver.num_samples=300", "solver.n_steps=10", "solver.topk=30"],
    "cem300x30": ["solver._target_=experiments.pc_feedback_cem_20261106_codex.FeedbackCEM",
                  "+solver.selection=mean", "solver.num_samples=300", "solver.n_steps=30", "solver.topk=30"],
}


def main():
    # Interleave configurations across repeated passes; the existing cem/r1
    # is explicitly skipped, never overwritten.
    for repeat in (1, 2, 3):
        for name, overrides in CONFIGS.items():
            if (name, repeat) == ("cem", 1):
                continue
            stem = f"pc_cost_split_{name}_r{repeat}"
            profile = ROOT / f"results/{stem}.profile.json"
            trace = ROOT / f"results/{stem}.trace.jsonl"
            audit = ROOT / f"results/{stem}.audit.jsonl"
            gpu = ROOT / f"results/{stem}.gpu.json"
            output = ROOT / f"results/{stem}.run"
            log = ROOT / f"results/{stem}.log"
            if any(p.exists() for p in (profile, trace, audit, gpu, output, log)):
                raise FileExistsError(stem)
            environment = os.environ.copy()
            environment.update(PC_PHYSICS_RESET_SEED="871001",
                               PC_PHYSICS_TRACE_PATH=str(trace),
                               PC_COST_PROFILE_PATH=str(profile),
                               PC_GPU_EVENTS_PATH=str(gpu))
            command = [sys.executable, "-m", "experiments.pc_closed_cost_decomposed_20261106_codex",
                       "policy=pusht/lewm", "seed=42", "solver.seed=42", "eval.num_eval=10",
                       *overrides, MEAN, SCALE, f"+solver.audit_path={audit}", f"output.dir={output}"]
            print(f"Running {stem}", flush=True)
            with log.open("x") as stream:
                subprocess.run(command, cwd=ROOT, env=environment, stdout=stream,
                               stderr=subprocess.STDOUT, check=True)
            print(f"Completed {stem}", flush=True)


if __name__ == "__main__":
    main()
