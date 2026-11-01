"""Known-root five-step witness prediction against actual cp5 RGB embeddings."""
import argparse
import json
from pathlib import Path

import numpy as np

from . import codex_p1_cp5_exploratory_cuda_20261030_v1 as base
from . import e3_initial_goal_qualification_20261031 as qualification

ROOT = base.ROOT
NAME = "e3_cp5_witness_prediction_20261031"
TARGET = ROOT / "results/e3_cp5_target_cuda_20261031_RUN01.json"
TARGET_SHA = "ff788213aff290ebd636fce22e55de64d3fddae128d8492bfdc4aa995c586518"
PROTOCOL = ROOT.parent / "docs/D2P_E3_CP5_WITNESS_PREDICTION_PROTOCOL_20261031.md"


def array(value, shape, name):
    x = np.asarray(value, dtype=np.float32)
    if x.shape != shape or not np.isfinite(x).all():
        raise ValueError(name)
    return x


def collect():
    if base.sha(qualification.INPUT) != qualification.INPUT_SHA or base.sha(TARGET) != TARGET_SHA:
        raise ValueError("input identity")
    data = json.loads(qualification.INPUT.read_bytes())
    target = json.loads(TARGET.read_bytes())
    if target["status"] != "complete" or target["model_before"] != base.MODEL_HASH or \
            target["model_after"] != base.MODEL_HASH or len(target["rows"]) != 2:
        raise ValueError("target receipt")
    qualification.analyze(data)
    result = []
    for i, item in enumerate(data["root_provenance"]):
        p = item["provenance"]
        original = p["original_task"]
        row = target["rows"][i]
        witness = array(p["prefix5"], (5, 2), "witness")
        if row["root"] != i or np.abs(witness).max() > 1 or \
                witness.tobytes() != array(row["prefix5"], (5, 2), "target witness").tobytes():
            raise ValueError("witness identity")
        goal = array(row["latent"], (192,), "target latent")
        if goal.tobytes() != array(p["actual_cp5_task"]["root_context"]["latent_history"][-1],
                                   (192,), "saved cp5 latent").tobytes():
            raise ValueError("actual target identity")
        mean = array(original["action_mean"], (2,), "mean")
        std = array(original["action_std"], (2,), "std")
        if (std <= 0).any():
            raise ValueError("normalizer")
        result.append(dict(root=i, states=array(original["root_context"]["latent_history"], (3, 192), "states"),
                           history=array(original["root_context"]["action_history"], (2, 10), "history"),
                           witness=witness, mean=mean, std=std, goal=goal))
    return result


def run(output):
    import torch
    from torch.nn.attention import SDPBackend, sdpa_kernel
    from experiments.checkpoint_compat import load_compatible_pretrained
    from experiments.gpu_preflight import require_cuda_allocation

    out = Path(output)
    if not out.is_absolute() or out.parent != ROOT / "results" or \
            not out.name.startswith(NAME + "_") or out.exists() or out.is_symlink():
        raise ValueError("exclusive output")
    entries = collect()
    sources = base.bindings()
    sources.update({str(qualification.INPUT): qualification.INPUT_SHA, str(TARGET): TARGET_SHA,
                    str(PROTOCOL): base.sha(PROTOCOL), str(Path(__file__).resolve()): base.sha(__file__)})
    with base.locking.gpu_lock(base.LOCK):
        host = require_cuda_allocation("cuda:0")
        device = torch.device("cuda:0")
        torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
        model = load_compatible_pretrained(str(base.MODEL), device=device).eval().requires_grad_(False)
        before = base.model_hash(model)
        if before != base.MODEL_HASH:
            raise ValueError("model identity")
        rows = []
        with torch.inference_mode(), sdpa_kernel(SDPBackend.MATH), torch.autocast(device_type="cuda", enabled=False):
            for item in entries:
                states = torch.as_tensor(item["states"][None], device=device)
                history = torch.as_tensor(item["history"], device=device)
                predictions = {}
                for label, action in (("witness", item["witness"]), ("zero", np.zeros((5, 2), np.float32))):
                    token = ((action - item["mean"]) / item["std"]).reshape(1, 10)
                    outgoing = torch.cat((history, torch.as_tensor(token, device=device)), dim=0)[None]
                    prediction = model.predict(states, model.action_encoder(outgoing))
                    if prediction.shape != states.shape or not bool(torch.isfinite(prediction).all()):
                        raise ValueError("prediction shape or finite")
                    z = prediction[0, -1].cpu().numpy().astype(np.float32)
                    predictions[label] = dict(latent=z.tolist(),
                                              mse_to_actual_cp5=float(np.mean((z.astype(np.float64) -
                                                                                item["goal"].astype(np.float64)) ** 2)))
                rows.append(dict(root=item["root"], witness=item["witness"].tolist(),
                                 actual_cp5_latent=item["goal"].tolist(), predictions=predictions))
        torch.cuda.synchronize(device)
        after = base.model_hash(model)
        if before != after or any(base.sha(path) != digest for path, digest in sources.items()):
            raise ValueError("model or source drift")
        digest = base.write(out, dict(schema=NAME, status="complete", host=host, model_before=before,
                                      model_after=after, rows=rows, sources=sources, predict_calls=4,
                                      action_encoder_calls=4, independent_confirmation=False,
                                      scientific_admission=False, physical_execution=False))
    print(json.dumps({"sha256": digest, "rows": [{"root": x["root"],
          "witness_mse": x["predictions"]["witness"]["mse_to_actual_cp5"],
          "zero_mse": x["predictions"]["zero"]["mse_to_actual_cp5"]} for x in rows]}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    print("Predict both fixed five-step witnesses and compare with actual cp5 RGB encoding.", flush=True)
    run(parser.parse_args().output)
