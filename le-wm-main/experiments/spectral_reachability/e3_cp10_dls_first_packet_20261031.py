"""Predict the fixed first packets of the two DLS development runs."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

from . import e3_cp10_first_packet_cuda_20261031 as original

ROOT, base, need = original.ROOT, original.base, original.need
ORIGINAL_COLLECT = original.collect
NAME = "e3_cp10_dls_first_packet_20261031"
PROTOCOL = ROOT.parent / "docs/D2P_E3_DLS_FIRST_PACKET_PROTOCOL_20261031.md"
INPUTS = (
    ("e3_cp10_dls_sourcefix_20261031_root0",
     "62dc2d6e8a5bd3265b0b92a9c01c87625c43cdcba243ae5fd5d24aa09e28def8",
     "e3_cp10_dls_sourcefix_saved_20261031_root0_CHECK01.json",
     "307f66cb841549ff4426c86b9168a26cbcc8bdb1225b8c713456003755a91691"),
    ("e3_cp10_goal_dls_20261031_root1",
     "b4847770b0941867104e6a417c4fcc25983a703191a2b5265b6186996e4bf899",
     "e3_cp10_goal_dls_saved_20261031_root1_CHECK01.json",
     "2ee6fabba57f1f739921c4a2e95c58ac9d5ae6840b182675a88f01bca41ce657"),
)


def collect():
    entries, sources = ORIGINAL_COLLECT()
    sources.update({str(PROTOCOL): base.sha(PROTOCOL),
                    str(Path(__file__).resolve()): base.sha(Path(__file__).resolve())})
    for root, (stem, report_sha, seal_name, seal_sha) in enumerate(INPUTS):
        directory = ROOT / "results" / f"{stem}_RUN01"
        report, prep = directory / "report.json", ROOT / "results" / f"{stem}_PREPARE01.json"
        seal = ROOT / "results" / seal_name
        need(base.sha(report) == report_sha and base.sha(seal) == seal_sha, "report/seal SHA")
        doc, prepared, checked = (json.loads(p.read_bytes()) for p in (report, prep, seal))
        need(doc["status"] == "recorded_known_root_not_admitted" and not doc["errors"]
             and doc["E3_admission"] is False and checked["report_sha256"] == report_sha
             and [row["arm"] for row in doc["rows"]] == list(original.ARMS), "report contract")
        journal = directory / "journal.jsonl"
        need(base.sha(journal) == doc["journal_sha256"], "journal")
        for path in (report, prep, seal, journal):
            sources[str(path)] = base.sha(path)
        for row in doc["rows"]:
            arm = row["arm"]
            item = entries[root * 4 + original.ARMS.index(arm)]
            checks = {c["step"]: c for c in row["checkpoints"]}
            need(row["control_returned"] >= 5 and 0 in checks and 5 in checks, "first packet")
            path = Path(checks[5]["path"])
            need(path.parent == directory and not path.is_symlink()
                 and base.sha(path) == checks[5]["file_sha256"], "image source")
            with Image.open(path) as image:
                pixels = np.asarray(image).copy()
            need(pixels.shape == (224, 224, 3) and pixels.dtype == np.uint8
                 and hashlib.sha256(pixels.tobytes()).hexdigest() == checks[5]["pixel_sha256"],
                 "actual pixels")
            encoding_path = directory / f"s{root}_{arm}_b00_encoding.json"
            node_path = directory / f"s{root}_{arm}_b00_seal.json"
            encoding, node = (json.loads(p.read_bytes()) for p in (encoding_path, node_path))
            need(encoding["actual_step"] == 0 and node["identity"]["boundary"] == 0
                 and node["identity"]["observation"]["pixel_sha256"] == checks[0]["pixel_sha256"],
                 "context identity")
            action = np.asarray(row["executed_raw"][:5], dtype="<f4")
            need(action.shape == (5, 2) and
                 action.tobytes() == np.asarray(node["receipt"]["commands"], dtype="<f4").tobytes(),
                 "action binding")
            history = np.asarray(node["identity"]["raw10"], dtype="<f4")
            need(history.shape == (10, 2) and np.isfinite(history).all(), "history")
            latent_meta = encoding["response"]["latent"]
            latent_path = Path(latent_meta["path"])
            need(latent_path.parent == directory / f"s{root}_{arm}_b00_encode"
                 and not latent_path.is_symlink() and base.sha(latent_path) == latent_meta["file_sha256"],
                 "initial latent file")
            latent = np.load(latent_path, allow_pickle=False)
            need(latent.shape == (192,) and latent.dtype == np.float32 and np.isfinite(latent).all()
                 and hashlib.sha256(latent.tobytes()).hexdigest() == latent_meta["array_sha256"],
                 "initial latent bytes")
            item.update(task=prepared["source"]["tasks"][root], initial_latent=latent,
                        history=history, action=action, pixels=pixels,
                        actual_state=checks[5]["state"], pixel_sha256=checks[5]["pixel_sha256"])
            for source in (path, encoding_path, node_path, latent_path):
                sources[str(source)] = base.sha(source)
    need(all(base.sha(path) == digest for path, digest in sources.items()), "source drift")
    return entries, sources


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    original.NAME = NAME
    original.collect = collect
    original.run(args.output)


if __name__ == "__main__":
    main()
