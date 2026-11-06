"""Replay and intervene on actual first-plan chains, gated by physical parity."""
import json
import subprocess
import sys
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[1]
    results = root / "results"
    for mode in ("lookahead", "effect"):
        stem = f"pc_branch_effect_{mode}_task42_solver42_20261105_chaintrace"
        audit = results / f"{stem}.audit.jsonl"
        trace = results / f"{stem}.trace.jsonl"
        chain = results / f"{stem}.firstchains.jsonl"
        compatibility = results / f"{stem}.parity_input.jsonl"
        parity = results / f"{stem}.parity.json"
        intervention = results / f"{stem}.intervention.json"
        if any(p.exists() for p in (chain, compatibility, parity, intervention)):
            raise FileExistsError(f"refusing to overwrite {stem}")
        first = json.loads(audit.read_text().splitlines()[0])
        envs = first["environments"]
        if first["solve"] != 0 or len(envs) != 50 or [e["index"] for e in envs] != list(range(50)):
            raise ValueError("unexpected first-plan task order")
        actions = [e["selected_normalized_chain"] for e in envs]
        with chain.open("x", encoding="utf-8") as stream:
            stream.write(json.dumps({"solve": 0, "actions": actions}, allow_nan=False) + "\n")
        with compatibility.open("x", encoding="utf-8") as stream:
            stream.write(json.dumps({"mode": "mean", "mean_actions": actions}, allow_nan=False) + "\n")
        subprocess.run([sys.executable, "-m", "experiments.pc_seeded_candidate_replay_20261105_codex",
                        "--audit", str(compatibility), "--trace", str(trace), "--output", str(parity)],
                       cwd=root, check=True)
        subprocess.run([sys.executable, "-m", "experiments.pc_prefix_selected_intervention_20261105_codex",
                        "--chains", str(chain), "--parity", str(parity), "--output", str(intervention)],
                       cwd=root, check=True)


if __name__ == "__main__":
    main()
