"""Apply the same selected-chain intervention to recorded native means."""
import json
import subprocess
import sys
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[1]
    results = root / "results"
    source = results / "pc_seeded_trace_mean_seed42_20261105_codex.jsonl"
    parity = results / "pc_seeded_trace_parity_seed42_20261105_codex.json"
    chains = results / "pc_prefix_baseline_chains_seed42_20261105_codex.jsonl"
    output = results / "pc_prefix_baseline_intervention_seed42_20261105_codex.json"
    if chains.exists() or output.exists():
        raise FileExistsError("refusing to overwrite baseline intervention outputs")
    first = json.loads(source.read_text().splitlines()[0])
    if first["mode"] != "mean" or first["solve"] != 0:
        raise ValueError("unexpected native first plan")
    with chains.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps({"solve": 0, "actions": first["mean_actions"]}) + "\n")
    subprocess.run([
        sys.executable, "-m", "experiments.pc_prefix_selected_intervention_20261105_codex",
        "--chains", str(chains), "--parity", str(parity), "--output", str(output),
    ], cwd=root, check=True)


if __name__ == "__main__":
    main()
