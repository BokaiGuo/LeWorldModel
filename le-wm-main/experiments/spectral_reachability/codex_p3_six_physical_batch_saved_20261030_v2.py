"""Saved six-slot adapter with external original prepare and source-root checks."""
from pathlib import Path

from experiments.spectral_reachability import codex_p3_six_physical_batch_saved_20261030_v1 as base

host, require, kernel = base.host, base.require, base.kernel


def consume(output, report_sha, original_prepare, prepare_sha):
    out = Path(output)
    require(host.sha(out / "report.json") == report_sha, "external report sha")
    p, data = host.verify(original_prepare, prepare_sha)
    require(host.helper.read(out / "prepare.json") == p, "semantic prepare copy")
    report = host.helper.read(out / "report.json")
    require(host.sha(out / "journal.jsonl") == report["journal_sha256"], "journal bytes")
    events = kernel.journal(out / "journal.jsonl", report["journal_chain_tail"])
    from PIL import Image
    from io import BytesIO
    import numpy as np
    def frame_reader(name, expected):
        require(Path(name).name == name and host.sha(out / name) == expected, "frame SHA")
        with Image.open(BytesIO((out / name).read_bytes())) as image:
            return np.asarray(image.convert("RGB")).copy()
    result = base.validate(p, report, events, frame_reader)
    for i, row in enumerate(report["rows"]):
        reference = data["references"][row["root_slot"]]
        require(kernel.source.root_hash(kernel.source.vector(row["root_state"]),
                                        kernel.source.vector(row["goal"])) == reference["root_sha256"],
                "actual root and goal identity")
        require(np.array_equal(np.asarray(row["goal"]), reference["goal"]), "actual goal identity")
        obs = [e["receipt"] for e in events if e.get("branch") == i and
               e["kind"] == "fresh_observation" and e["receipt"]["phase"] == "warmup" and
               e["receipt"]["step"] in (0, 5, 10)]
        require([x["step"] for x in obs] == [0, 5, 10] and
                all(x["pixel_sha256"] == kernel.moving.image_sha(reference["history"][j]) and
                    x["state"] == reference["states"][j].tolist() and
                    x["snapshot"] == reference["dynamics"][j] for j, x in enumerate(obs)),
                "runtime warmup identity")
    require({f.name for f in out.iterdir()} == set(result["frames"]) |
            {"prepare.json", "report.json", "journal.jsonl"}, "artifact census")
    result["runtime_source_root_rebuilt"] = True
    result["independent_model_or_physical_reexecution"] = False
    return result
