"""H3 physical source join and raw-action provenance preflight; no physics."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import numpy as np

from experiments.spectral_reachability import codex_p2_symmetric_h3_saved_checks_20261030_v1 as checks
from experiments.spectral_reachability import p3_horizon_residual_physical_20261027_v4 as physical

NAME = "codex_p2_h3_physical_preflight_20261030_v1"
root = checks.run.ROOT
RUN = root/"results/codex_p2_symmetric_h3_run_20261030_v2_RUN01"
REPORT_SHA = "4d0a4f4399227609548678329df506e2f0f6e7e6fe0ce49b0690f0031521b90f"
PREP = root/"results/codex_p2_symmetric_h3_run_20261030_v2_PREP01.json"
PREP_SHA = "b20ac436c23d697fed53d3b78423f1ebd6861428fd512c7513743b53d2236c95"
REF = root/"results/p3_horizon_residual_physical_20261027_v4_source_prepared01"
REF_SHA = "c5eee78ded753d1c2b955335b97e218d420671e1d2d35636aae8303a8811622b"
need = checks.need


def source_join(cases, protocol, refs):
    packets = {p["root_sha256"]: p for p in refs["packets"]}
    original = {c["root_sha256"]: c for c in protocol["context"]["cases"]}
    need(len(packets) == len(original) == len(cases) == 4 and
         set(packets) == set(original) == set(refs["references"]), "four source roots")
    out = []
    for case in cases:
        root_id = case["root_sha256"]
        packet, ref, old = packets[root_id], refs["references"][root_id], original[root_id]
        need(packet["root_context"] == case["context"] and packet["goal"] == case["goal"] and
             packet["projection"] == case["projection"], "context/goal/projection mismatch")
        need(ref["root_sha256"] == root_id and ref["history_rgb_sha256"] == old["root"]["root_rgb_sha256"]
             and ref["goal_rgb_sha256"] == old["root"]["goal_rgb_sha256"], "root/goal RGB mismatch")
        out.append(dict(root_index=case["root_index"], root_sha256=root_id,
                        context_sha256=case["context_sha256"], goal_sha256=case["goal_sha256"],
                        projection_sha256=case["projection_sha256"], history_rgb_sha256=ref["history_rgb_sha256"],
                        goal_rgb_sha256=ref["goal_rgb_sha256"], action_domain=case["action_domain"]))
    return out


def inspect():
    checked = checks.check(RUN, REPORT_SHA, PREP, PREP_SHA)
    protocol, cases = checks.run.verify_prepare(PREP, PREP_SHA)
    refs = physical.validate_prepared(REF, REF_SHA)
    need(refs["root_seed"] == 202753010 and refs["root_oversample_factor"] == 4, "root recipe")
    roots = source_join(cases, protocol, refs)
    report = checks.run.bridge.host.frontier.read(RUN/"report.json")
    rows = []
    for row in report["rows"]:
        winner = row["winner"]
        # The producer emitted only these witness fields. Absence is not zero.
        need(set(winner) == {"tag", "logits", "contexts", "projected_terminal", "cost"}, "unexpected witness")
        rows.append(dict(root=row["root"], arm=row["arm"], status="raw_action_bits_missing",
                         planned_commands=15, raw_actions=None, physical_attempted=False))
    return dict(schema=NAME+".preflight", status="blocked_original_raw_float32_not_saved",
                report_sha256=REPORT_SHA, prepare_sha256=PREP_SHA, physical_reference_sha256=REF_SHA,
                saved_numeric_status=checked["status"], roots=roots, slots=rows,
                root_recipe=dict(seed=refs["root_seed"], oversample_factor=refs["root_oversample_factor"],
                                 warmup_public_steps_per_branch=15),
                early_stop="native terminated/truncated stops immediately; preserve partial, no padding",
                missing="producer did not save pre-normalization CUDA tanh raw float32 values or their bits",
                rationale="normalized float32 history is not injective and does not authenticate original raw bits",
                physical_executed=False, neural_calls=0, scientific_admission=False,
                P1_admission=False, P2_admission=False, total_resource_matched=False)


def save_cuda_raw_packet(tensor, *, index, sink):
    """Future producer hook: persist the actual CUDA raw packet before transition.

    The containing producer must itself bind call identity and event chronology;
    this function does not retroactively authenticate any saved H3 run.
    """
    need(getattr(getattr(tensor, "device", None), "type", None) == "cuda", "actual CUDA tensor required")
    need(type(index) is int and 0 <= index < 3 and callable(sink), "packet index/sink")
    a = np.asarray(tensor.detach().cpu().numpy())
    need(a.dtype == np.float32 and a.shape == (5, 2) and np.isfinite(a).all() and
         np.all(np.abs(a) <= 1), "literal bounded float32 raw packet")
    raw = a.astype("<f4", copy=False).tobytes(order="C")
    event = dict(kind="actual_cuda_raw_packet_before_transition", packet=index,
                 dtype="<f4", shape=[5, 2], raw_hex=raw.hex(), raw_sha256=hashlib.sha256(raw).hexdigest())
    sink(event)
    return event


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", required=True)
    a = p.parse_args()
    path = Path(a.output)
    need(path.is_absolute() and path.parent.resolve() == path.parent and
         path.name.startswith(NAME+"_") and not os.path.lexists(path), "exclusive output")
    result = inspect()
    checks.run.bridge.host.bank._write(path, result)
    print(json.dumps(dict(status=result["status"], sha256=checks.run.digest_file(path))))


if __name__ == "__main__":
    main()
