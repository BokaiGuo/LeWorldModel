"""Bind all new diverse CUDA-replayed actions to native physical source identities."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from . import p1_backtrack_physical_source_join_20261030 as physical
from . import p1_diverse_cuda_replay_consume_20261030_main as consumer

replay = consumer.replay
ROOT = replay.ROOT
NAME = "p1_diverse_physical_source_join_20261030_main"
RESULT = ROOT / "results/p1_diverse_cuda_replay_20261030_main_RUN01.json"
RESULT_SHA = "f862e2baacad7d17921348c4085fde830abe5b0aa6a3185d0d896844b02f14d3"
PREPARE = ROOT / "results/p1_diverse_cuda_replay_20261030_main_prepare_RUN01.json"
PREPARE_SHA = "0b8147a37b768650fa4686a185cf419c7436f154677ec679eeebd8c249a1ff0c"


def assemble(rows, result, prepared, data):
    if len(rows) != 16 or len(result["rows"]) != 16:
        raise ValueError("fixed roster")
    actions, replay_rows = [], []
    for i, (row, observed) in enumerate(zip(rows, result["rows"], strict=True)):
        expected_role = ("incumbent", "diverse_selected")[i % 2]
        if row["role"] != expected_role or any(observed[k] != v for k, v in row.items()):
            raise ValueError("new replay identity")
        raw = np.frombuffer(bytes.fromhex(row["raw15_le_float32_hex"]), dtype="<f4").reshape(15, 2)
        role = ("incumbent", "model_selected")[i % 2]
        actions.append(dict(**{k: row[k] for k in ("slot", "arm", "candidate_index")},
                            role=role, raw15=raw.tolist(), raw15_le_float32_hex=raw.tobytes().hex()))
        replay_rows.append(dict(**{k: row[k] for k in ("slot", "arm", "candidate_index")},
                                role=role, exact=observed["exact"]))
    joined = physical.assemble({"rows": actions}, {"rows": replay_rows}, prepared, data)
    joined.update(source_replay_sha256=RESULT_SHA, diverse_replay_prepare_sha256=PREPARE_SHA,
                  diverse_join_sha256=replay.JOIN_SHA,
                  diverse_source_roles=[r["role"] for r in rows],
                  role_mapping={"diverse_selected": "model_selected", "incumbent": "incumbent"},
                  diverse_source_join_code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  limitation="Known-root source join; CUDA replay checked; no live reset, physical dispatch or independent confirmation.")
    for i, row in enumerate(joined["rows"]):
        if (row["task_id"] != rows[i]["task_binding"]["task_id"]
                or row["raw15_float32_hex"] != rows[i]["raw15_le_float32_hex"]):
            raise ValueError("native task or literal action mismatch")
    return joined


def build():
    result = replay.prior.one.saved.read_verified(RESULT, RESULT_SHA)
    manifest = replay.prior.one.saved.read_verified(PREPARE, PREPARE_SHA)
    if result["manifest"] != str(PREPARE) or result["manifest_sha256"] != PREPARE_SHA:
        raise ValueError("replay manifest binding")
    rows, backtrack, report, prepared = replay.sources()
    check = consumer.verify(result, manifest, rows, backtrack, report, prepared)
    if not check["all_exact"]:
        raise ValueError("CUDA replay incomplete")
    data = physical.physical.source_material()
    joined = assemble(rows, result, prepared, data)
    for path, expected in data["bindings"].items():
        physical.physical.require(physical.physical.sha(path) == expected, "source drift")
    return joined


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    path = Path(parser.parse_args().output)
    if (not path.is_absolute() or path.parent != ROOT / "results" or path.resolve() != path
            or not path.name.startswith(NAME + "_") or path.exists()):
        raise ValueError("exclusive output path")
    joined = build()
    print(json.dumps({"path": str(path), "sha256": replay.prior.one.saved.write(path, joined),
                      "rows": len(joined["rows"]), "physical_authorized": False}))


if __name__ == "__main__":
    main()
