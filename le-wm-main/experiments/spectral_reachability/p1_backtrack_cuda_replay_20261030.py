"""Fixed action-bank replay on host CUDA; not a physical-release authority."""
import argparse
import hashlib
import inspect
import json
import os
from pathlib import Path

import numpy as np
from experiments.spectral_reachability import p1_backtrack_action_inventory_20261030 as inventory

NAME = "p1_backtrack_cuda_replay_20261030"
ROOT = inventory.verifier.producer.base.one.source.ROOT
INPUT = ROOT / "results/p1_backtrack_action_inventory_20261030_MAIN01.json"
INPUT_SHA = "27b3f1016efa0affe5c8b38c2da0d48cf0ce0ef4db11de974e1c5dc30e1c863e"
PROTOCOL = ROOT.parent / "docs/D2P_P1_BACKTRACK_CUDA_REPLAY_PROTOCOL_20261030.md"
one = inventory.verifier.producer.base.one


def digest(path):
    path = Path(path)
    if not path.is_file() or path.is_symlink():
        raise ValueError("missing or symlink source")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def files():
    modules = (inventory, inventory.verifier, inventory.verifier.producer,
               inventory.verifier.producer.base, one, one.source, one.saved, one.host)
    paths = {Path(m.__file__).resolve() for m in modules}
    paths.update((Path(__file__).resolve(), PROTOCOL.resolve()))
    return {str(p): digest(p) for p in sorted(paths)}


def sources():
    action = one.saved.read_verified(INPUT, INPUT_SHA)
    result = one.saved.read_verified(
        ROOT / "results/p1_bounded_backtrack_host_20261030_MAIN01.json", inventory.SHA)
    report = one.saved.read_verified(one.source.REPORT, one.source.REPORT_SHA)
    receipt = one.saved.read_verified(one.source.RECEIPT, one.source.RECEIPT_SHA)
    if receipt["report_sha256"] != one.source.REPORT_SHA:
        raise ValueError("source receipt mismatch")
    prepared = one.saved.read_verified(report["prepared"], report["prepare_sha256"])
    one.saved.analyze(report, receipt, prepared)
    if action != inventory.build(result, report, prepared):
        raise ValueError("inventory mismatch")
    one.host.verify(report["prepared"], report["prepare_sha256"])
    return action, result, report, prepared


def prepare(output):
    output = Path(output)
    if (not output.is_absolute() or output.parent != ROOT / "results"
            or output.resolve() != output or not output.name.startswith(NAME + "_") or output.exists()):
        raise ValueError("exclusive output path")
    action, _, report, _ = sources()
    manifest = dict(schema=NAME + ".prepare", input_sha256=INPUT_SHA,
                    source_report_sha256=one.source.REPORT_SHA,
                    original_prepare_sha256=report["prepare_sha256"],
                    files=files(), roster=16, predictor_calls=96,
                    action_encoder_calls=96, training=False, P1_admission=False)
    return dict(path=str(output), sha256=one.saved.write(output, manifest))


def run(manifest_path, manifest_sha, output):
    manifest = one.saved.read_verified(manifest_path, manifest_sha)
    action, result, report, prepared = sources()
    if (manifest["schema"] != NAME + ".prepare" or manifest["input_sha256"] != INPUT_SHA
            or manifest["source_report_sha256"] != one.source.REPORT_SHA
            or manifest["original_prepare_sha256"] != report["prepare_sha256"]
            or manifest["files"] != files() or manifest["roster"] != 16
            or manifest["predictor_calls"] != 96 or manifest["action_encoder_calls"] != 96
            or manifest["training"] is not False or manifest["P1_admission"] is not False):
        raise ValueError("manifest mismatch")
    output = Path(output)
    if (not output.is_absolute() or output.parent != ROOT / "results"
            or output.resolve() != output or not output.name.startswith(NAME + "_") or output.exists()):
        raise ValueError("exclusive output path")
    import torch
    from torch.nn.attention import SDPBackend, sdpa_kernel
    from experiments.checkpoint_compat import load_compatible_pretrained
    from experiments.gpu_preflight import require_cuda_allocation
    os.environ["HF_HUB_OFFLINE"] = os.environ["TRANSFORMERS_OFFLINE"] = "1"
    device_info = require_cuda_allocation("cuda:0")
    device = torch.device("cuda:0")
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    counts = dict(predict=0, action_encoder=0)
    rows = []
    with torch.no_grad(), torch.autocast(device_type="cuda", enabled=False), sdpa_kernel(SDPBackend.MATH):
        model = load_compatible_pretrained(prepared["policy"], device=device).eval().requires_grad_(False)
        before = one.host.signal.model_hash(model)
        if before != report["model_before"]:
            raise ValueError("model hash")
        for module in model.modules():
            if str(Path(inspect.getfile(type(module))).resolve()) not in prepared["bindings"]:
                raise ValueError("unbound model class")

        def hook(name):
            def count(_module, _args):
                counts[name] += 1
            return count

        hooks = [model.predictor.register_forward_pre_hook(hook("predict")),
                 model.action_encoder.register_forward_pre_hook(hook("action_encoder"))]
        try:
            for index, item in enumerate(action["rows"]):
                arm = index // 2
                selected = item["candidate_index"]
                candidate = report["rows"][arm]["result"]["candidates"][selected]
                saved = result["rows"][20 * arm + selected]
                task = prepared["requests"][arm // 2]["task"]
                raw = np.asarray(item["raw15"], dtype="<f4")
                if raw.shape != (15, 2) or raw.tobytes().hex() != item["raw15_le_float32_hex"]:
                    raise ValueError("raw action bits")
                mean = torch.tensor(np.tile(task["action_mean"], 5), dtype=torch.float32, device=device)
                std = torch.tensor(np.tile(task["action_std"], 5), dtype=torch.float32, device=device)
                tokens = [(torch.tensor(raw[5*k:5*k+5], device=device).reshape(10) - mean) / std for k in range(3)]
                z0 = torch.tensor(task["root_context"]["latent_history"], dtype=torch.float32, device=device)[None]
                h0 = torch.tensor(task["root_context"]["action_history"], dtype=torch.float32, device=device)[None]

                def predict(z, h):
                    return model.predict(z, model.action_encoder(h))

                whole = inventory.verifier.producer.base.advance(z0, h0, tokens, predict)
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
                mid = np.asarray(candidate["mid"]["latent_history"][-1], np.float32)
                end = np.asarray(saved["final"]["endpoint"], np.float32)
                exact = dict(whole_segmented=bool(np.array_equal(whole_np, split_np)),
                             source_prefix=bool(np.array_equal(split_np[0], mid)),
                             source_endpoint=bool(np.array_equal(split_np[2], end)),
                             source_prefix_action_history=bool(np.array_equal(
                                 np.asarray(contexts[0]["action_history"], np.float32),
                                 np.asarray(candidate["mid"]["action_history"], np.float32))))
                rows.append(dict(slot=item["slot"], arm=item["arm"], role=item["role"],
                                 candidate_index=selected, exact=exact, endpoints=split_np.tolist(),
                                 contexts=contexts, action_tokens=[x.cpu().tolist() for x in tokens]))
        finally:
            for handle in hooks:
                handle.remove()
        torch.cuda.synchronize(device)
        after = one.host.signal.model_hash(model)
    one.host.F.check_bindings(prepared["bindings"])
    if before != after or files() != manifest["files"] or counts != dict(predict=96, action_encoder=96):
        raise ValueError("model, sources or call count changed")
    complete = all(all(row["exact"].values()) for row in rows)
    value = dict(schema=NAME, manifest=str(Path(manifest_path).resolve()), manifest_sha256=manifest_sha,
                 model_before=before, model_after=after, device=device_info, calls=counts, rows=rows,
                 complete=complete, training=False, physical_release=False, P1_admission=False)
    return dict(path=str(output), sha256=one.saved.write(output, value), complete=complete)


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
        print(json.dumps(prepare(args.output)))
    else:
        print("Replaying every frozen P1 action on host CUDA in two paths; no physical execution.", flush=True)
        outcome = run(args.prepared, args.sha256, args.output)
        print(json.dumps(outcome))
        if not outcome["complete"]:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
