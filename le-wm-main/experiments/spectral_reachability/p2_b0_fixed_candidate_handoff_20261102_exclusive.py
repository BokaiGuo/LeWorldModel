"""Export all authenticated B0 candidate actions, not a physical authorization."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile

import numpy as np

from . import p2_b0_isolated_consume_20261102_exclusive as isolation

ROOT = isolation.ROOT
OUTPUT = ROOT / "results/p2_b0_fixed_candidate_handoff_20261102_exclusive_run01.json"
ARMS = ("joint_multistart", "prefix_chain")
CHILD = """
import hashlib, json
from experiments.spectral_reachability import p2_shared_capture_launch_20261102_exclusive as s
from experiments.spectral_reachability import p2_b0_isolated_consume_20261102_exclusive as iso
verified=s.audit(iso.RUN,iso.REPORT_SHA)
pair=json.loads((iso.RUN/'pair.json').read_bytes())
print(json.dumps({'verified':verified,
    'pair_sha256':hashlib.sha256((iso.RUN/'pair.json').read_bytes()).hexdigest(),
    'context_sha256':s.wire.kernel.old.digest(pair['context']),
    'goal_sha256':s.wire.kernel.old.digest(pair['goal']),
    'arms':[{'arm':b['arm'],'candidates':[{'index':c['index'],
              'raw_float32_hex':(c['raw_float32_hex'] if b['arm']=='joint_multistart'
                                 else c['first_raw_float32_hex']+c['second_raw_float32_hex'])}
                   for c in b['candidates']]} for b in pair['branches']]},sort_keys=True,allow_nan=False))
"""


def validate(value):
    proof = value["verified"]
    if (proof["status"] != "verified_saved_pair_known_root_not_admitted" or
            proof["report_sha256"] != isolation.REPORT_SHA or proof["P1_admission"] is not False or
            proof["P2_admission"] is not False):
        raise ValueError("strict B0 source proof required")
    arms = value["arms"]
    if [a["arm"] for a in arms] != list(ARMS):
        raise ValueError("fixed arm order")
    decoded = []
    for arm in arms:
        candidates = arm["candidates"]
        if [c["index"] for c in candidates] != list(range(16)):
            raise ValueError("fixed sixteen-candidate census")
        actions = []
        for candidate in candidates:
            raw = bytes.fromhex(candidate["raw_float32_hex"])
            if len(raw) != 80:
                raise ValueError("10x2 float32 action required")
            arr = np.frombuffer(raw, dtype="<f4")
            if not np.isfinite(arr).all() or not (np.abs(arr) <= 1).all():
                raise ValueError("bounded finite actions required")
            actions.append(raw)
        decoded.append(actions)
    return {
        "candidate_count_per_arm": 16,
        "same_full_action_at_same_index": sum(x == y for x, y in zip(*decoded, strict=True)),
        "same_first_packet_at_same_index": sum(x[:40] == y[:40] for x, y in zip(*decoded, strict=True)),
    }


def run():
    source = isolation.historical_bytes()
    before, error = isolation.recovery.checked_file(isolation.recovery.TARGET)
    if error:
        raise ValueError("current P0 source unreadable")
    with tempfile.TemporaryDirectory(prefix="p2_b0_candidates_") as directory:
        historical = Path(directory) / "historical_p0.py"
        with historical.open("xb") as stream:
            stream.write(source)
            stream.flush()
            os.fsync(stream.fileno())
        historical.chmod(0o444)
        command = isolation.command(historical)
        command[-1] = CHILD
        process = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                 timeout=300, check=False)
    after, error = isolation.recovery.checked_file(isolation.recovery.TARGET)
    if error or after != before:
        raise ValueError("current P0 source changed")
    if process.returncode != 0:
        raise RuntimeError("isolated consumer failed: " + process.stderr.decode("utf-8", "replace")[-4000:])
    value = json.loads(process.stdout)
    summary = validate(value)
    return {
        "schema": "p2_b0_fixed_candidate_handoff_20261102_exclusive_v1",
        "source": value, "comparison": summary,
        "historical_source_sha256": isolation.digest(source), "working_tree_sha256": before,
        "handoff_scope": "saved candidate bytes only; no optimizer or physical dispatch authorization",
        "budget": {"candidate_count_per_arm": 16, "common_optimizer_callbacks": None,
                   "initializer_and_total_compute_matched": False, "FLOPs": None,
                   "energy": None, "physical_execution": None},
        "P1_admission": False, "P2_admission": False, "independent_confirmation": False,
        "limitations": ["Already observed known B0; candidate actions need not match across arms.",
                        "No new optimizer, GPU, neural model or physical execution.",
                        "Same candidate count does not establish equal compute or common objective."],
    }


def main():
    if OUTPUT.exists() or OUTPUT.is_symlink():
        raise FileExistsError(OUTPUT)
    value = run()
    raw = (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    with OUTPUT.open("xb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    print(f"created {OUTPUT} sha256={hashlib.sha256(raw).hexdigest()}")


if __name__ == "__main__":
    main()
