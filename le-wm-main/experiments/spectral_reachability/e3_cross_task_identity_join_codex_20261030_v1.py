"""Read-only join of four existing E3 saved feedback identities; no launcher."""
import argparse
import json
from pathlib import Path
import sys

from experiments.spectral_reachability import agent_e3_chunked_identity_host_20261029_v1 as host
from experiments.spectral_reachability import agent_e3_frozen_model_task_resolver_20261029_v1 as resolver

NAME = "e3_cross_task_identity_join_codex_20261030_v1"
ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"
PREPARE_SHA = "ea8e4feae44e31876852c960db03ce4613184c55fc0745ef8f890f70485f8425"
SPECS = {
    0: ("agent_e3_chunked_identity_host_20261029_v1_MAIN06",
        "3ee72bd9b3ade29020083359f878b67accf4ec531dbccefa2acb3d94f1065bea",
        "9adfc093f3b6293c92e56ec245daeb431d0a0c59548805dcd3d67560f745806f",
        "agent_e3_observed_authority_20261029_v1_MAIN02.json",
        "ea2d7d747e02c0054eca90092ca947e136cdba10aca3a204dfa85d8b01ec64e9"),
    1: ("agent_e3_chunked_identity_host_20261029_v1_CELL01_MAIN01",
        "3136ec5292c9efc6821e8976af9af43b13bc89b2ae070285400e99535770e4db",
        "dc53d17f844e86c3dc1f34013c157d357d5aa059879790005c8c274a8d6537bc",
        "agent_e3_cell01_feedback_prepare_20261029_v1_MAIN01.json",
        "0e525436e311dc8ffcb4cf7ec04fd0ffcbccafedc3ea52391fe9e888bb5105f8"),
    8: ("agent_e3_chunked_identity_host_20261029_v1_CELL08_MAIN01",
        "04d11c71116212f158beab0c3b6be534109eada4fa3f47576d168740cfb7abbd",
        "577cab704f45e97adc1b10c8c9762c0bd375a8c78fe50fec2226d179f82fa665",
        "agent_e3_cell08_feedback_prepare_20261029_v1_MAIN01.json",
        "5319f6c942c2eefe08ed92811c7b3c6f646c3b01af6d29ee90f6afda716468e2"),
    9: ("agent_e3_chunked_identity_host_20261029_v1_CELL09_MAIN01",
        "bf55b30581fc70200432801e1e835959936e8935a39b3f6d4d9858d463b26d2e",
        "9c4649eca9ccbd3cbba2d21d5934b39ff608d5294f872a0eda0cee2b6b208c2d",
        "agent_e3_cell09_feedback_prepare_20261029_v1_MAIN01.json",
        "d06ce5207d600de135b19d0ab7cbcb43d6354fe2fb419712a5d0bdfd42137994"),
}


def check_inspection(index, result):
    """Do not promote partial, wrong-cell or missing identity records."""
    if (type(result) is not dict or result.get("binding_chunks_complete") is not True or
            type(result.get("base")) is not dict or result["base"].get("cell") != index or
            result["base"].get("status") != "completed20" or
            result.get("full_saved_strict") is not False or result.get("scientific_admission") is not False):
        raise ValueError("completed cell and chunked identity inspection required")
    identity = result.get("identity")
    if type(identity) is not dict or identity.get("pending") is not None:
        raise ValueError("complete identity required")
    binding = identity.get("binding")
    if (type(binding) is not dict or binding.get("model_sha256") != resolver.MODEL_SHA or
            binding.get("prepare_sha256") != PREPARE_SHA):
        raise ValueError("frozen checkpoint/prepare identity mismatch")
    rows = identity.get("encodes")
    if (type(rows) is not list or len(rows) != 2 or
            [(x.get("boundary"), x.get("outcome")) for x in rows if type(x) is dict] !=
            [(10, "return"), (15, "return")]):
        raise ValueError("both identity-bound feedback encodes required")
    return dict(state="saved_identity_joined", model_sha256=binding["model_sha256"],
                prepare_sha256=binding["prepare_sha256"], device_label=binding.get("device"),
                feedback_boundaries=[10, 15], source_guard_revalidated=True)


def inspect_cell(index, spec):
    directory, report, completion, authority, authority_sha = spec
    result = host.inspect(str(RESULTS / directory), report, completion, str(RESULTS / authority), authority_sha)
    return check_inspection(index, result)


def build():
    slots = resolver.roster()
    names = [f"r{r}.g{g}.h{h}.d{domain}" for r in range(2) for g in range(2)
             for h in (1, 3) for domain in ("Dxy", "Dx")]
    if (len(slots) != 16 or [s["index"] for s in slots] != list(range(16)) or
            [s["cell_id"] for s in slots] != names):
        raise ValueError("fixed 16-cell denominator")
    rows = []
    for slot in slots:
        i = slot["index"]
        if i not in SPECS:
            rows.append(dict(index=i, cell_id=slot["cell_id"], state="not_inspected", identity=None))
            continue
        try:
            identity = inspect_cell(i, SPECS[i])
            rows.append(dict(index=i, cell_id=slot["cell_id"], state=identity["state"], identity=identity))
        except (OSError, ValueError, KeyError, TypeError, RuntimeError) as exc:
            rows.append(dict(index=i, cell_id=slot["cell_id"], state="identity_join_failed",
                             identity=None, error_type=type(exc).__name__, error=str(exc)[:2048]))
    joined = [r["index"] for r in rows if r["state"] == "saved_identity_joined"]
    return dict(schema=NAME + ".report", fixed_denominator=16, inspected_cells=list(SPECS),
                rows=rows, joined_cells=joined,
                four_cell_saved_join_complete=joined == list(SPECS),
                same_model_object_across_processes="unknown", physical_gpu_uuid="unknown",
                neural_or_physical_parity="unknown", full_saved_strict=False,
                production_authorized=False, scientific_admission=False,
                interpretation="source-trusted saved hash/encoder-binding join; not hardware or process attestation")


def main():
    argparse.ArgumentParser(description=__doc__).parse_args()
    result = build()
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return 0 if result["four_cell_saved_join_complete"] else 1


if __name__ == "__main__":
    sys.exit(main())
