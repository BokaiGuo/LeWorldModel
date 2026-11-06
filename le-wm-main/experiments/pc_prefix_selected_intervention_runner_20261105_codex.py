"""Trace selected conditional plans, verify parity, then run interventions."""
import json
import os
import subprocess
import sys
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[1]
    results = root / "results"
    output = results / "pc_prefix_selected_trace_seed42_20261105_codex"
    chain = results / "pc_prefix_selected_chains_seed42_20261105_codex.jsonl"
    audit = results / "pc_prefix_selected_search_seed42_20261105_codex.jsonl"
    trace = results / "pc_prefix_selected_physics_seed42_20261105_codex.jsonl"
    compatibility = results / "pc_prefix_selected_parity_input_seed42_20261105_codex.jsonl"
    parity = results / "pc_prefix_selected_parity_seed42_20261105_codex.json"
    intervention = results / "pc_prefix_selected_intervention_seed42_20261105_codex.json"
    for path in (output, chain, audit, trace, compatibility, parity, intervention):
        if path.exists():
            raise FileExistsError(f"refusing to overwrite {path}")
    environment = os.environ.copy()
    environment.update(PC_PHYSICS_RESET_SEED="771001", PC_PHYSICS_TRACE_PATH=str(trace))
    print("Trace first and subsequent actually selected conditional plans.", flush=True)
    subprocess.run([
        sys.executable, "-m", "experiments.pc_eval_seeded_trace_20261105_codex",
        "policy=pusht/lewm", "seed=42",
        "solver._target_=experiments.pc_native_prefix_trace_20261105_codex.NativePrefixTrace",
        "+solver.mode=conditional", f"+solver.chain_path={chain}", f"+solver.audit_path={audit}",
        "solver.num_samples=300", "solver.n_steps=10", "solver.topk=30",
        f"output.dir={output}",
    ], cwd=root, env=environment, check=True)
    first = json.loads(chain.read_text().splitlines()[0])
    with compatibility.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps({"mode": "mean", "mean_actions": first["actions"]}) + "\n")
    print("Verify physical execution parity before interventions.", flush=True)
    subprocess.run([
        sys.executable, "-m", "experiments.pc_seeded_candidate_replay_20261105_codex",
        "--audit", str(compatibility), "--trace", str(trace), "--output", str(parity),
    ], cwd=root, check=True)
    print("Run interventions only if every physical replay passed.", flush=True)
    subprocess.run([
        sys.executable, "-m", "experiments.pc_prefix_selected_intervention_20261105_codex",
        "--chains", str(chain), "--parity", str(parity), "--output", str(intervention),
    ], cwd=root, check=True)


if __name__ == "__main__":
    main()
