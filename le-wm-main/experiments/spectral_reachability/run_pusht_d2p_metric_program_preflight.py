"""Known-bank host-CUDA compiler preflight; no new physical outcomes or P1."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def prefix_parameters(rows):
    """Preserve every declared prefix; reject rather than clip/drop endpoints."""
    group = sorted(rows, key=lambda row: row["proposal_index"])
    if [r["proposal_index"] for r in group] != list(range(16)):
        raise ValueError("complete ordered 16-prefix bank required")
    source = np.asarray([r["prefix_action"] for r in group])
    if source.dtype.kind not in "fiu" or source.shape != (16, 2) or not np.isfinite(source).all() or np.any(np.abs(source) >= 1):
        raise ValueError("finite strictly interior live prefix commands required")
    if not np.any(np.all(source == 0, axis=1)):
        raise ValueError("literal zero control required")
    return np.arctanh(source.astype(np.float64))[:, None, :]


def run(*, intervention_report, policy, output):
    # Runtime imports do not allocate a model; all neural work follows the seal.
    from experiments.spectral_reachability import audit_pusht_d2p_goal_agent_intervention as base
    from experiments.spectral_reachability import pusht_d2p_metric_program_compiler as compiler
    from experiments.spectral_reachability import preflight_pusht_d2p_p1_reconfiguration as p1
    from experiments.spectral_reachability.audit_pusht_d2p_metric_program_arrays import audit
    from experiments.spectral_reachability.d2p_prerun_source_seal import create_seal, verify_seal
    from experiments.spectral_reachability.d2p_model_call_instrumentation import PredictInstrumentation
    import torch

    intervention_report, policy, output = map(Path, (intervention_report, policy, output))
    intervention = json.loads(intervention_report.read_text())
    if intervention.get("status") != "completed" or not intervention["gates"] or not all(v is True for v in intervention["gates"].values()):
        raise ValueError("complete goal intervention evidence required")
    bank_path = Path(intervention["protocol"]["bank_path"])
    if base.digest(bank_path) != intervention["protocol"]["bank_sha256"]:
        raise ValueError("bank binding changed")
    bank = json.loads(bank_path.read_text())
    base.summarize(bank)
    base.verify_stages(bank_path, bank)
    upstream_path = Path(bank["upstream_path"])
    upstream = base.physical._load_preflight(upstream_path)
    if (base.digest(policy/"weights.pt") != base.OFFICIAL_SHA
            or base.digest(policy/"config.json") != intervention["protocol"]["config_sha256"]):
        raise ValueError("original frozen checkpoint/config required")
    banks = [prefix_parameters([r for r in bank["rows"] if r["root_id"] == index]) for index in range(4)]
    bindings = {str(p.resolve()): base.digest(p) for p in
                (intervention_report, bank_path, upstream_path, policy/"weights.pt", policy/"config.json")}
    sources = [Path(__file__), Path(compiler.__file__), Path(p1.__file__),
               Path(compiler.scoring.__file__), Path(base.__file__), Path(base.p0.__file__),
               Path(base.physical.__file__), Path(base.load_compatible_pretrained.__code__.co_filename),
               Path(base.collect_nonterminal_boundary_contact_roots.__code__.co_filename)]
    sources += [Path(__file__).with_name(name) for name in (
        "audit_pusht_d2p_metric_program_arrays.py", "pusht_d2p_goal_nuisance_metric.py",
        "d2p_prerun_source_seal.py", "d2p_model_call_instrumentation.py")]
    sources = list(dict.fromkeys(p.resolve() for p in sources))
    protocol = {"bindings": bindings, "roots": bank["protocol"]["roots"],
        "fixed_prefixes_per_root": 16, "prefix_horizon": 1, "suffix_horizon": 1,
        "damping": .01, "beta": 1., "expected_predict_samples_per_root": 64,
        "scope": "known-bank model compilation only; no spectral prefix synthesis or physical efficacy",
        "physical_future_labels_used": False, "P1_admission": False}
    output.mkdir(exist_ok=False)
    seal = create_seal(files=sources, args_json=json.dumps(protocol), output_dir=output/"prerun_source_seal")
    base.write_new(output/"protocol.json", protocol)
    host = base.require_cuda_allocation("cuda")
    model = base.load_compatible_pretrained(policy, device="cuda").eval().requires_grad_(False)
    before = base.p0._parameter_hash(model)
    buffers = {k: v.clone() for k, v in model.named_buffers()}
    env = base.gym.make("swm/PushT-v1", render_mode="rgb_array")
    rows = []
    try:
        roots = base.collect_nonterminal_boundary_contact_roots(env, count=4,
            seed=int(upstream["protocol"]["root_seed"]), oversample_factor=4)
        for index, root in enumerate(roots):
            if base.root_fingerprint(root) != protocol["roots"][index]:
                raise ValueError("known root identity changed")
            frames, history, goal_rgb = base.physical._reset_and_warmup(env, root, history_len=3)
            if base.physical._sha256(goal_rgb) != upstream["rows"][index]["goal_rgb_sha256"]:
                raise ValueError("goal render changed")
            emb, actions = p1._encode_context(model, frames, history, base.p0.image_transform(), torch.device("cuda"))
            goals = [v["goal_vector"] for v in intervention["rows"][index]["variants"]]
            with PredictInstrumentation(model) as counter:
                with counter.phase("planning", method="fixed_bank_metric_compiler", horizon_tokens=2):
                    compiled = compiler.compile_programs(model, emb, actions, goals, banks[index], 1)
            verification = audit(compiled)
            receipt = counter.records[0]
            if not receipt["completed"] or receipt["predict_samples"] != 64:
                raise ValueError("compiler prediction count mismatch")
            row = {"root_index": index, "root_sha256": protocol["roots"][index],
                   "compiled": compiled, "independent_array_audit": verification, "predict": receipt}
            base.write_new(output/f"root_{index}.json", row)
            rows.append(row)
    finally:
        env.close()
    gates = {"four_roots": len(rows) == 4, "parameters_unchanged": before == base.p0._parameter_hash(model),
        "buffers_unchanged": set(buffers) == set(dict(model.named_buffers())) and all(torch.equal(buffers[k], v) for k, v in model.named_buffers()),
        "all_eval": all(not m.training for m in model.modules()),
        "inputs_unchanged": all(base.digest(Path(p)) == digest for p, digest in bindings.items())}
    verified = verify_seal(output_dir=seal["output_dir"], expected_manifest_sha256=seal["manifest_sha256"])
    base.write_new(output/"report.json", {"schema": "pusht_d2p_metric_program_preflight.v1",
        "status": "compiled_calibration" if all(gates.values()) else "integrity_fail",
        "protocol": protocol, "host": host, "gates": gates, "rows": rows,
        "source_seal": seal, "source_verification": verified, "P1_admission": False})
    if not all(gates.values()):
        raise RuntimeError("compiler preflight integrity failed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--intervention-report", type=Path, required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(**vars(args))
