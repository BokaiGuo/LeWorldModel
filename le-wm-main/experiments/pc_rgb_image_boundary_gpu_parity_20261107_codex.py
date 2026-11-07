"""Compare legacy and image-shaped RGB boundaries using frozen PushT LeWM."""
import json
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import torch

from eval import img_transform
from experiments.checkpoint_compat import load_compatible_pretrained
from experiments.pc_input_permissions import RGB, RGBLeWM, file_hash, freeze_audit
from experiments.pc_prefix_novel_actual_encoding_20261105_codex import model_fingerprint
from experiments.pc_rgb_image_boundary_20261107_codex import RGBImageLeWM
from experiments import pc_pusht_first_plan_alignment_20261107_codex as prior

PROTOCOL = prior.ROOT.parent / "docs/PC_RGB_SHAPE_BOUNDARY_PROTOCOL_20261107_CODEX.md"
OUTPUT = prior.RESULTS / "pc_rgb_image_boundary_gpu_parity_20261107_codex.json"
CHECKPOINT = Path("/home/gbk/.stable_worldmodel/checkpoints/pusht/lewm")


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    source_path = prior.OUT.with_suffix(".json")
    source = json.loads(source_path.read_text())
    for path, expected in source["source_sha256"].items():
        if file_hash(path) != expected:
            raise ValueError(f"source drift: {path}")
    data_stat = prior.DATA.stat()
    if (data_stat.st_size, data_stat.st_mtime_ns) != (source["data_size"], source["data_mtime_ns"]):
        raise ValueError("dataset changed")
    model = load_compatible_pretrained("pusht/lewm", device="cuda").eval().requires_grad_(False)
    fingerprint = model_fingerprint(model)
    if fingerprint != source["model_sha256"]:
        raise ValueError("checkpoint changed")
    audit = freeze_audit(RGB, "lewm", [CHECKPOINT / "weights.pt", CHECKPOINT / "config.json",
                                       Path(__file__), PROTOCOL, source_path],
                         preprocessing="ImageNet 224, frozen LeWM",
                         clock="five physical actions per 10D packet")
    old, new = RGBLeWM(model, audit), RGBImageLeWM(model, audit)
    row = source["rows"][0]["data_row"]
    with h5py.File(prior.DATA, "r") as data:
        transform = img_transform(SimpleNamespace(eval=SimpleNamespace(img_size=224)))
        initial = transform(np.asarray(data["pixels"][row])).cuda()
        goal = transform(np.asarray(data["pixels"][row + 25])).cuda()
    ctx = {"pixels": initial[None, None, None], "goal": goal[None, None, None],
           "action": torch.empty((1, 1, 0, 10), device="cuda")}
    audits = [prior.load_first(stem) for stem in (prior.OLD, prior.NEW)]
    chains = torch.as_tensor(np.stack([a[0]["selected_actions"] for a in audits])[None],
                             dtype=torch.float32, device="cuda")
    print("Compare old and image-shaped boundaries on the same real RGB and candidate chains.", flush=True)
    with torch.inference_mode():
        old_cost, old_context = old.score(ctx, chains)
        new_cost, new_context = new.score(ctx, chains)
        torch.testing.assert_close(old_cost, new_cost, atol=0, rtol=0)
        torch.testing.assert_close(old_context["predicted_emb"], new_context["predicted_emb"], atol=0, rtol=0)
    if old.receipts != new.receipts:
        raise ValueError("runtime receipts differ")
    if model_fingerprint(model) != fingerprint:
        raise ValueError("model changed")
    with OUTPUT.open("x") as stream:
        json.dump({"data_row": row, "score": new_cost.cpu().tolist(), "model_sha256": fingerprint,
                   "input_audit": audit.fingerprint, "receipts": new.receipts,
                   "protocol_sha256": file_hash(PROTOCOL), "runner_sha256": file_hash(__file__),
                   "boundary_sha256": file_hash(Path(__file__).with_name("pc_rgb_image_boundary_20261107_codex.py")),
                   "source_sha256": file_hash(source_path),
                   "scope": "single exposed RGB input, two fixed candidate chains; no control performance claim"},
                  stream, indent=2, allow_nan=False)
        stream.write("\n")
    print("two chain scores and predicted embeddings are bitwise identical", flush=True)


if __name__ == "__main__":
    main()
