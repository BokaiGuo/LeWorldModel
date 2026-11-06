"""Compare recorded JEPA feedback runs, including same-process dual rollouts."""
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "results"


def read(stem):
    return [json.loads(line) for line in (ROOT / f"{stem}.trace.jsonl").read_text().splitlines()]


def main():
    output = ROOT / "pc_jepa_feedback_parity_summary_20261106_codex.json"
    if output.exists():
        raise FileExistsError(output)
    cases = []
    for seed in (1, 2):
        template = f"pc_jepa_multi_unroll_{{}}_seed{seed}_r3_step3_20261106_codex"
        local, pinned = read(template.format("local")), read(template.format("pinned"))
        if len(local) != len(pinned):
            raise ValueError("trace lengths differ")
        different = [(i, a, b) for i, (a, b) in enumerate(zip(local, pinned)) if a != b]
        kind = Counter(a["kind"] for _, a, _ in different)
        unrolls = [(a, b) for a, b in zip(local, pinned) if a["kind"] == "unroll"]
        cases.append({
            "seed": seed, "row_count": len(local), "record_counts": dict(Counter(r["kind"] for r in local)),
            "different_rows": len(different), "different_kinds": dict(kind),
            "first_different_row_zero_based": different[0][0] if different else None,
            "all_unroll_inputs_equal": all(a["context"] == b["context"] and a["actions"] == b["actions"]
                                           for a, b in unrolls),
            "all_executed_steps_equal": all(a == b for a, b in zip(local, pinned) if a["kind"] == "step"),
        })
    dual = read("pc_jepa_multi_unroll_local_seed1_dual1_step3_20261106_codex")
    baseline = read("pc_jepa_multi_unroll_local_seed1_r3_step3_20261106_codex")
    checks = [r["dual_same_input"] for r in dual if r["kind"] == "unroll"]
    result = {"cases": cases, "same_process_dual": {
        "seed": 1, "checks": len(checks), "exact": sum(r["exact"] for r in checks),
        "max_abs": max(r["max_abs"] for r in checks), "max_rms": max(r["rms"] for r in checks),
        "all_inputs_unchanged": all(r["input_unchanged"] for r in checks),
        "executed_steps_match_original_local_run":
            [r for r in dual if r["kind"] == "step"] == [r for r in baseline if r["kind"] == "step"],
    }, "limitations": [
        "Prediction hashes alone do not quantify cross-process numerical differences.",
        "Only one-frame context and reduced 8x2 CEM tested.",
        "Other local modifications and local DINO shim remain.",
        "Expert and planner actions are both included in the step count.",
    ]}
    with output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)


if __name__ == "__main__":
    main()
