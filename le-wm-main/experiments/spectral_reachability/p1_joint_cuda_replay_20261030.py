"""Independently replay every fixed joint winner and incumbent on CUDA."""
import argparse
import inspect
import json
import os
from pathlib import Path

import numpy as np

from . import p1_backtrack_cuda_replay_20261030 as old
from . import p1_joint_prefix_suffix_saved_check_20261030 as check

ROOT = old.ROOT
NAME = "p1_joint_cuda_replay_20261030"
JOINT = ROOT / "results/p1_joint_prefix_suffix_cuda_20261030_RUN01.json"
PROTOCOL = ROOT.parent / "docs/D2P_P1_JOINT_REPLAY_PROTOCOL_20261030_MAIN.md"


def sources():
    original, backtrack, report, prepared = old.sources()
    check.check(JOINT)
    joint = old.one.saved.read_verified(JOINT, check.REPORT_SHA)
    rows = []
    for i, winner in enumerate(joint["winners"]):
        incumbent = original["rows"][2 * i]
        row = joint["rows"][20 * i + winner["candidate_index"]]
        if (row["slot"], row["arm"], row["candidate_index"]) != (
                winner["slot"], winner["arm"], winner["candidate_index"]):
            raise ValueError("winner identity")
        raw = np.asarray(row["best_actions"], dtype="<f4").reshape(15, 2)
        if not np.isfinite(raw).all() or (np.abs(raw) > 1).any():
            raise ValueError("invalid action")
        rows.append(dict(slot=incumbent["slot"], arm=incumbent["arm"], role="incumbent",
                         candidate_index=incumbent["candidate_index"],
                         task_binding=incumbent["task_binding"],
                         raw15_le_float32_hex=incumbent["raw15_le_float32_hex"]))
        rows.append(dict(slot=row["slot"], arm=row["arm"], role="joint_selected",
                         candidate_index=row["candidate_index"], task_binding=incumbent["task_binding"],
                         raw15_le_float32_hex=raw.tobytes().hex()))
    return rows, joint, backtrack, report, prepared


def files():
    result = old.files()
    for path in (Path(__file__).resolve(), Path(check.__file__).resolve(), JOINT, PROTOCOL):
        result[str(path)] = old.digest(path)
    return result


def target(path):
    path = Path(path)
    if (not path.is_absolute() or path.parent != ROOT / "results" or path.resolve() != path
            or not path.name.startswith(NAME + "_") or os.path.lexists(path)):
        raise ValueError("exclusive output")
    return path


def prepare(output):
    rows, _, _, report, _ = sources()
    value = {"schema": NAME + ".prepare", "source_joint_sha256": check.REPORT_SHA,
             "source_report_sha256": old.one.source.REPORT_SHA, "rows": rows,
             "files": files(), "original_prepare_sha256": report["prepare_sha256"],
             "training": False, "physical_release": False}
    return old.one.saved.write(target(output), value)


def run(prepared_path, prepared_sha, output):
    output = target(output)
    manifest = old.one.saved.read_verified(prepared_path, prepared_sha)
    rows, joint, backtrack, report, prepared = sources()
    expected = {"schema": NAME + ".prepare", "source_joint_sha256": check.REPORT_SHA,
                "source_report_sha256": old.one.source.REPORT_SHA, "rows": rows,
                "files": files(), "original_prepare_sha256": report["prepare_sha256"],
                "training": False, "physical_release": False}
    if manifest != expected:
        raise ValueError("manifest drift")
    import torch
    from torch.nn.attention import SDPBackend, sdpa_kernel
    from experiments.checkpoint_compat import load_compatible_pretrained
    from experiments.gpu_preflight import require_cuda_allocation

    os.environ["HF_HUB_OFFLINE"] = os.environ["TRANSFORMERS_OFFLINE"] = "1"
    device_info = require_cuda_allocation("cuda:0")
    device = torch.device("cuda:0")
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    counts = {"predict": 0, "action_encoder": 0}
    results = []
    with torch.no_grad(), torch.autocast(device_type="cuda", enabled=False), sdpa_kernel(SDPBackend.MATH):
        model = load_compatible_pretrained(prepared["policy"], device=device).eval().requires_grad_(False)
        before = old.one.host.signal.model_hash(model)
        if before != report["model_before"]:
            raise ValueError("checkpoint mismatch")
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
            for i, row in enumerate(rows):
                arm = i // 2
                task = prepared["requests"][arm // 2]["task"]
                raw = np.frombuffer(bytes.fromhex(row["raw15_le_float32_hex"]), dtype="<f4").copy().reshape(15, 2)
                mean = torch.tensor(np.tile(task["action_mean"], 5), dtype=torch.float32, device=device)
                std = torch.tensor(np.tile(task["action_std"], 5), dtype=torch.float32, device=device)
                if not bool(torch.isfinite(std).all() and (std > 0).all()):
                    raise ValueError("normalizer")
                tokens = [(torch.tensor(raw[5*k:5*k+5], device=device).reshape(10) - mean) / std for k in range(3)]
                z0 = torch.tensor(task["root_context"]["latent_history"], dtype=torch.float32, device=device)[None]
                h0 = torch.tensor(task["root_context"]["action_history"], dtype=torch.float32, device=device)[None]
                def predict(z, h):
                    return model.predict(z, model.action_encoder(h))
                whole = old.inventory.verifier.producer.base.advance(z0, h0, tokens, predict)
                z, h = z0.clone(), h0.clone()
                ends, contexts = [], []
                for token in tokens:
                    outgoing = torch.cat((h, token[None, None]), dim=1)
                    end = predict(z, outgoing)[:, -1:]
                    z = torch.cat((z[:, 1:], end), dim=1)
                    h = outgoing[:, 1:]
                    ends.append(end[0, 0])
                    contexts.append({"latent_history": z[0].cpu().tolist(), "action_history": h[0].cpu().tolist()})
                a = np.stack([x.cpu().numpy() for x in whole])
                b = np.stack([x.cpu().numpy() for x in ends])
                candidate = report["rows"][arm]["result"]["candidates"][row["candidate_index"]]
                if row["role"] == "incumbent":
                    prefix = candidate["mid"]["latent_history"][-1]
                    terminal = backtrack["rows"][20 * arm + row["candidate_index"]]["final"]["endpoint"]
                else:
                    selected = joint["rows"][20 * arm + row["candidate_index"]]
                    record = selected["trajectory"][selected["best_iteration"]]
                    prefix, terminal = record["prefix_latent"], record["terminal_latent"]
                exact = {"whole_segmented": bool(np.array_equal(a, b)),
                         "source_prefix": bool(np.array_equal(b[0], np.asarray(prefix, np.float32))),
                         "source_endpoint": bool(np.array_equal(b[-1], np.asarray(terminal, np.float32)))}
                results.append({**row, "exact": exact, "whole_endpoints": a.tolist(),
                                "segmented_endpoints": b.tolist(), "contexts": contexts,
                                "action_tokens": [t.cpu().tolist() for t in tokens]})
        finally:
            for handle in hooks:
                handle.remove()
        torch.cuda.synchronize(device)
        after = old.one.host.signal.model_hash(model)
    old.one.host.F.check_bindings(prepared["bindings"])
    if before != after or files() != manifest["files"] or counts != {"predict": 96, "action_encoder": 96}:
        raise ValueError("model, source or call drift")
    complete = all(all(row["exact"].values()) for row in results)
    value = {"schema": NAME, "manifest": str(prepared_path), "manifest_sha256": prepared_sha,
             "device": device_info, "model_before": before, "model_after": after, "calls": counts,
             "rows": results, "complete": complete, "training": False, "physical_release": False,
             "P1_admission": False}
    return old.one.saved.write(output, value), complete


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("prepare", "run"))
    parser.add_argument("--output", required=True)
    parser.add_argument("--prepared")
    parser.add_argument("--sha256")
    args = parser.parse_args()
    if args.command == "prepare":
        print(json.dumps({"sha256": prepare(args.output)}))
    else:
        print("Replay all 16 incumbent and joint-selected actions on CUDA; no physics.", flush=True)
        digest, complete = run(args.prepared, args.sha256, args.output)
        print(json.dumps({"sha256": digest, "complete": complete}))
        if not complete:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
