"""Reconstruct paired CUDA feedback task identities from saved artifacts."""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import numpy as np

from . import agent_p3_same_feedback_physical20_20261028_v1 as p3
from . import p1_spare_budget_pair_audit_20261030 as pair

ROOT = Path(__file__).resolve().parents[2]
NAME = "p1_feedback_identity_rebuild_20261030"


def need(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path):
    return json.loads(path.read_bytes())


def normalized_receipt(value, directory):
    """Normalize only run path and path-dependent request hashes."""
    if isinstance(value, dict):
        return {key: normalized_receipt(item, directory) for key, item in value.items()
                if key != "request_sha256"}
    if isinstance(value, list):
        return [normalized_receipt(item, directory) for item in value]
    if isinstance(value, str) and value.startswith(str(directory) + "/"):
        return "$RUN/" + value[len(str(directory)) + 1:]
    return value


def reconstruct(directory, report, base, arm):
    row = next(row for row in report["rows"] if row["arm"] == arm)
    actions = np.asarray(row["executed_raw"], dtype=np.float32)
    need(actions.shape == (20, 2), "physical action roster")
    context = deepcopy(base["root_context"])
    initial_raw = None
    output = []
    for boundary in (0, 5, 10, 15):
        name = f"s0_{arm}_b{boundary:02}"
        seal = load(directory / (name + "_seal.json"))
        receipt = load(directory / (name + "_encoding.json"))
        observation = seal["identity"]["observation"]
        need(p3.moving.source.digest(receipt) == observation["encoding_sha256"], "encoding receipt hash")
        enc = receipt["response"]["latent"]
        latent_path = Path(enc["path"])
        need(latent_path == directory / (name + "_encode/native192.npy")
             and sha(latent_path) == enc["file_sha256"], "latent file binding")
        z = np.load(latent_path, allow_pickle=False)
        need(z.shape == (192,) and z.dtype == np.float32 and np.isfinite(z).all()
             and p3.old.packet_digest(z.tolist()) == observation["observed_latent_sha256"], "latent digest")
        png = Path(receipt["request"]["png"]["path"])
        need(png == directory / (name + "_plan_05.png")
             and sha(png) == receipt["request"]["png"]["file_sha256"] == observation["png_sha256"],
             "PNG binding")
        raw10 = np.asarray(seal["identity"]["raw10"], np.float32)
        need(raw10.shape == (10, 2), "raw history")
        if boundary == 0:
            initial_raw = raw10.copy()
            context["latent_history"][-1] = z.tolist()
            context["action_history"] = p3.moving.normalized(base, raw10)
        else:
            expected = np.concatenate((initial_raw, actions[:boundary]), axis=0)[-10:]
            need(np.array_equal(raw10, expected), "raw history/action continuity")
            context = p3.moving.roll(context, z, raw10, base)
        task = p3.local_task(base, context, seal["identity"])
        need(p3.digest(task) == seal["task_sha256"], "reconstructed task hash")
        output.append({"task": task, "seal": seal, "receipt": receipt,
                       "latent_bytes": z.tobytes(), "png_bytes": png.read_bytes()})
    return output


def analyze(old, new):
    pair.analyze(old, new)
    reports = [load(d / "report.json") for d in (old, new)]
    prepared = []
    for directory in (old, new):
        binding = load(directory / "execution_binding.json")
        path = Path(binding["prepared"])
        need(sha(path) == binding["prepare_sha256"], "prepared hash")
        prepared.append(load(path))
    bases = [item["source"]["tasks"][0] for item in prepared]
    need(bases[0] == bases[1], "base tasks mismatch")
    rows = []
    for arm in ("native", "random"):
        x = reconstruct(old, reports[0], bases[0], arm)
        y = reconstruct(new, reports[1], bases[1], arm)
        for boundary, a, b in zip((0, 5, 10, 15), x, y):
            identity_a, identity_b = deepcopy(a["seal"]["identity"]), deepcopy(b["seal"]["identity"])
            need(identity_a["observation"].pop("encoding_sha256") !=
                 identity_b["observation"].pop("encoding_sha256"), "expected distinct receipt hashes")
            need(identity_a == identity_b and a["latent_bytes"] == b["latent_bytes"]
                 and a["png_bytes"] == b["png_bytes"], "paired observation mismatch")
            need(normalized_receipt(a["receipt"], old) == normalized_receipt(b["receipt"], new),
                 "receipt differs beyond run paths/derived request hash")
            task_a, task_b = deepcopy(a["task"]), deepcopy(b["task"])
            # local_task changes these fields as a function of the receipt identity.
            for key in ("task_id", "task_sha256", "task_contract_sha256"):
                task_a["binding"].pop(key)
                task_b["binding"].pop(key)
            need(task_a == task_b, "task differs beyond derived identity fields")
            rows.append({"arm": arm, "boundary": boundary, "task_hashes_reconstructed": True,
                         "raw_history_and_input_arrays_equal": True,
                         "receipt_equal_after_run_path_and_request_hash_normalization": True})
    return {"schema": NAME, "rows": rows, "source_reports": [pair.OLD_SHA, pair.NEW_SHA],
            "scope": "saved-input reconstruction, not independent physical event-chain attestation",
            "P1_admission": False, "fairness_proven": False}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output)
    need(output.is_absolute() and output.parent == ROOT / "results"
         and output.name.startswith(NAME + "_") and not output.exists(), "exclusive output")
    result = analyze(ROOT / "results/p1_four_arm_hooked_cuda_20261030_main_RUN01",
                     ROOT / "results/p1_four_arm_spare_budget_cuda_20261030_RUN01")
    result["code_sha256"] = sha(Path(__file__))
    content = json.dumps(result, sort_keys=True, indent=2, allow_nan=False).encode() + b"\n"
    with output.open("xb") as stream:
        need(stream.write(content) == len(content), "short write")
    print(json.dumps({"rows": len(result["rows"]), "sha256": hashlib.sha256(content).hexdigest()}))


if __name__ == "__main__":
    main()
