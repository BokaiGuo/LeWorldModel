"""Replay the complete response-diverse action roster on CUDA; no release authority."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from . import p1_backtrack_cuda_replay_20261030 as prior
from . import p1_response_diverse_backtrack_join_20261030_main as diverse

NAME = "p1_diverse_cuda_replay_20261030_main"
ROOT = prior.ROOT
JOIN = ROOT / "results/p1_response_diverse_backtrack_join_20261030_main_RUN01.json"
JOIN_SHA = "ad5b2df40dc14d73e37c1b1b4913d76c20f224090e705892acd29f7852140d67"


def sources():
    original, backtrack, report, prepared = prior.sources()
    saved = prior.one.saved.read_verified(JOIN, JOIN_SHA)
    selected = prior.one.saved.read_verified(
        ROOT / "results/p1_task_response_saved_panel_20261030_main_MAIN01.json",
        diverse.PANEL_SHA)
    expected = diverse.build(backtrack, report, prepared, selected)
    if {k: v for k, v in saved.items() if k != "code_sha256"} != expected:
        raise ValueError("diversity source reconstruction mismatch")
    rows = []
    for arm, chosen in enumerate(saved["rows"]):
        for role, index in (("incumbent", 0), ("diverse_selected", chosen["chosen_index"])):
            source = report["rows"][arm]["result"]["candidates"][index]
            result = backtrack["rows"][arm * 20 + index]
            raw = np.concatenate((np.asarray(source["prefix"], dtype="<f4"),
                                  np.asarray(result["final"]["raw"], dtype="<f4").reshape(10, 2)))
            if raw.shape != (15, 2) or not np.isfinite(raw).all() or np.any(np.abs(raw) > 1):
                raise ValueError("invalid action witness")
            if role == "diverse_selected" and (raw.tobytes().hex() != chosen["raw15_f32_hex"]
                    or hashlib.sha256(raw.tobytes()).hexdigest() != chosen["raw15_sha256"]):
                raise ValueError("diverse action witness mismatch")
            old = original["rows"][arm * 2]
            if (old["slot"], old["arm"], old["task_binding"]) != (
                    chosen["slot"], chosen["arm"], chosen["task_binding"]):
                raise ValueError("task identity mismatch")
            rows.append(dict(slot=chosen["slot"], arm=chosen["arm"], role=role,
                             candidate_index=index, task_binding=chosen["task_binding"],
                             raw15_le_float32_hex=raw.tobytes().hex()))
    return rows, backtrack, report, prepared


def bindings():
    paths = [Path(__file__).resolve(), Path(diverse.__file__).resolve(), JOIN]
    return {"prior": prior.files(), "additional": {str(p): prior.digest(p) for p in paths}}


def output_path(path):
    path = Path(path)
    if (not path.is_absolute() or path.parent != ROOT / "results" or path.resolve() != path
            or not path.name.startswith(NAME + "_") or path.exists()):
        raise ValueError("exclusive output path required")
    return path


def prepare(output):
    output = output_path(output)
    rows, _, report, _ = sources()
    value = dict(schema=NAME + ".prepare", join_sha256=JOIN_SHA,
                 source_report_sha256=prior.one.source.REPORT_SHA,
                 original_prepare_sha256=report["prepare_sha256"], files=bindings(),
                 rows=rows, predictor_calls=96, training=False, physical_release=False)
    return prior.one.saved.write(output, value)


def run(manifest_path, manifest_sha, output):
    output = output_path(output)
    manifest = prior.one.saved.read_verified(manifest_path, manifest_sha)
    rows, backtrack, report, prepared = sources()
    if manifest != dict(schema=NAME + ".prepare", join_sha256=JOIN_SHA,
                        source_report_sha256=prior.one.source.REPORT_SHA,
                        original_prepare_sha256=report["prepare_sha256"], files=bindings(),
                        rows=rows, predictor_calls=96, training=False, physical_release=False):
        raise ValueError("manifest mismatch")
    import os
    import torch
    from torch.nn.attention import SDPBackend, sdpa_kernel
    from experiments.checkpoint_compat import load_compatible_pretrained
    from experiments.gpu_preflight import require_cuda_allocation

    os.environ["HF_HUB_OFFLINE"] = os.environ["TRANSFORMERS_OFFLINE"] = "1"
    device_info = require_cuda_allocation("cuda:0")
    device = torch.device("cuda:0")
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    counts = dict(predict=0, action_encoder=0)
    results = []
    with torch.no_grad(), torch.autocast(device_type="cuda", enabled=False), sdpa_kernel(SDPBackend.MATH):
        model = load_compatible_pretrained(prepared["policy"], device=device).eval().requires_grad_(False)
        before = prior.one.host.signal.model_hash(model)
        if before != report["model_before"]:
            raise ValueError("model hash mismatch")
        import inspect
        for module in model.modules():
            if str(Path(inspect.getfile(type(module))).resolve()) not in prepared["bindings"]:
                raise ValueError("unbound model class")

        def hook(name):
            def count(_module, _args):
                counts[name] += 1
            return count

        handles = [model.predictor.register_forward_pre_hook(hook("predict")),
                   model.action_encoder.register_forward_pre_hook(hook("action_encoder"))]
        try:
            for i, row in enumerate(rows):
                arm = i // 2
                candidate = report["rows"][arm]["result"]["candidates"][row["candidate_index"]]
                saved = backtrack["rows"][20 * arm + row["candidate_index"]]
                task = prepared["requests"][arm // 2]["task"]
                raw = np.frombuffer(bytes.fromhex(row["raw15_le_float32_hex"]), dtype="<f4").copy().reshape(15, 2)
                mean = torch.tensor(np.tile(task["action_mean"], 5), dtype=torch.float32, device=device)
                std = torch.tensor(np.tile(task["action_std"], 5), dtype=torch.float32, device=device)
                tokens = [(torch.tensor(raw[5*k:5*k+5], device=device).reshape(10) - mean) / std for k in range(3)]
                z0 = torch.tensor(task["root_context"]["latent_history"], dtype=torch.float32, device=device)[None]
                h0 = torch.tensor(task["root_context"]["action_history"], dtype=torch.float32, device=device)[None]

                def predict(z, h):
                    return model.predict(z, model.action_encoder(h))

                whole = prior.inventory.verifier.producer.base.advance(z0, h0, tokens, predict)
                z, h = z0.clone(), h0.clone()
                segmented, contexts = [], []
                for token in tokens:
                    outgoing = torch.cat((h, token[None, None]), dim=1)
                    end = predict(z, outgoing)[:, -1:]
                    z = torch.cat((z[:, 1:], end), dim=1)
                    h = outgoing[:, 1:]
                    segmented.append(end[0, 0])
                    contexts.append(dict(latent_history=z[0].cpu().tolist(), action_history=h[0].cpu().tolist()))
                whole_np = np.stack([x.cpu().numpy() for x in whole])
                split_np = np.stack([x.cpu().numpy() for x in segmented])
                exact = dict(whole_segmented=bool(np.array_equal(whole_np, split_np)),
                             source_prefix=bool(np.array_equal(split_np[0], np.asarray(candidate["mid"]["latent_history"][-1], np.float32))),
                             source_endpoint=bool(np.array_equal(split_np[2], np.asarray(saved["final"]["endpoint"], np.float32))),
                             source_prefix_action_history=bool(np.array_equal(
                                 np.asarray(contexts[0]["action_history"], np.float32),
                                 np.asarray(candidate["mid"]["action_history"], np.float32))))
                results.append(dict(**row, exact=exact, whole_endpoints=whole_np.tolist(),
                                    segmented_endpoints=split_np.tolist(), contexts=contexts,
                                    action_tokens=[x.cpu().tolist() for x in tokens]))
        finally:
            for handle in handles:
                handle.remove()
        torch.cuda.synchronize(device)
        after = prior.one.host.signal.model_hash(model)
    prior.one.host.F.check_bindings(prepared["bindings"])
    if before != after or bindings() != manifest["files"] or counts != dict(predict=96, action_encoder=96):
        raise ValueError("model, source or call count changed")
    complete = all(all(row["exact"].values()) for row in results)
    value = dict(schema=NAME, manifest=str(Path(manifest_path).resolve()), manifest_sha256=manifest_sha,
                 model_before=before, model_after=after, device=device_info, calls=counts, rows=results,
                 complete=complete, training=False, physical_release=False, P1_admission=False)
    return prior.one.saved.write(output, value), complete


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--output", required=True)
    r = sub.add_parser("run")
    r.add_argument("--prepared", required=True)
    r.add_argument("--sha256", required=True)
    r.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        print(json.dumps({"sha256": prepare(args.output)}))
    else:
        print("Replay all fixed diverse and incumbent actions on CUDA; no physical release.", flush=True)
        digest, complete = run(args.prepared, args.sha256, args.output)
        print(json.dumps({"sha256": digest, "complete": complete}))
        if not complete:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
