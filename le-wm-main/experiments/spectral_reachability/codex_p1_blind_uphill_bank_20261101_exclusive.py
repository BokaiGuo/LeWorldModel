"""Fixed two-known-root, model-only first-packet screen of existing H3 banks."""
import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np

from . import codex_four_arm_common_rgb_launcher_20261030 as host
from . import codex_p1_two_root_saved_audit_20261101_exclusive as chain
from . import e3_p2_cooperative_gpu_lock_codex_20261030_v1 as locking

ROOT = host.ROOT
NAME = "codex_p1_blind_uphill_bank_20261101_exclusive"
ITEMS = (
    ("codex_p1_common_structured_cp10_host_20261101_exclusive_PREP_CODEX01.json",
     "8509f02776e9d6650e605d4128a76bce1148ebdc4f2e6dc3b068265de8b74b75",
     "codex_p1_common_structured_cp10_host_20261101_exclusive_RUN_CODEX01", "s0_structured",
     "209d795ea5db5f1a5d803669646bce6e7ebea2953c045780fa4bf0e19d5fb5ba",
     "2d1751880a0965fef85739a09c59c02e414a8be2cc6da74c571af22a3d844c16",
     "0ccccb7a13dcea7558040f3d2f22b9be8271738c86067dd1f3d4fb8d0e88c955"),
    ("codex_p1_structured_root1_cp10_20261101_exclusive_PREP_CODEXROOT1DEV01.json",
     "662b2abc564a4e5ff5e80627bfbdb4d2a3953bb5a8b52c13aec5fc3404b23fda",
     "codex_p1_structured_root1_cp10_20261101_exclusive_RUN_CODEXROOT1DEV01", "s1_structured",
     "6ee74e16c7a233fd59a0fb6d5ab769bb63238d984f842ee44825a97102ec9f39",
     "d9963fd35220dee0f931dbb81daf32a8f25b7349a194f69f3da60fba742e4bae",
     "d826c3f64fddfd5d2c8efa07807754da1834546e448787fd762776d5035a2af3"),
)


def choose(root_q, prefix_q, terminal_q):
    p, t = np.asarray(prefix_q, np.float64), np.asarray(terminal_q, np.float64)
    if p.shape != (8,) or t.shape != (8,) or not np.isfinite(p).all() or not np.isfinite(t).all() or not np.isfinite(root_q):
        raise ValueError("eight finite model scores required")
    feasible = [i for i in range(8) if p[i] >= root_q]
    return min(feasible, key=lambda i: (t[i], i)) if feasible else None


def load(slot):
    prep, prep_sha, run, episode, seal_sha, trace_sha, latent_sha = ITEMS[slot]
    paths = [ROOT / "results" / prep, ROOT / "results" / run / f"{episode}_b00_seal.json",
             ROOT / "results" / run / "common_structured_b00/candidates.jsonl",
             ROOT / "results" / run / f"{episode}_b00_encode/native192.npy"]
    for path, expected in zip(paths, (prep_sha, seal_sha, trace_sha, latent_sha), strict=True):
        host.require(path.is_file() and not path.is_symlink() and host.sha(path) == expected, "source SHA")
    prepared, seal = (json.loads(path.read_bytes()) for path in paths[:2])
    for name, expected in prepared["sources"].items():
        source = Path(name)
        host.require(source.is_file() and host.sha(source) == expected, "source closure drift: " + name)
    trace = chain.chain(paths[2], "ledger")
    events = {kind: [x["payload"] for x in trace if x["kind"] == kind]
              for kind in ("node_header", "optimized_candidates", "final_selection")}
    host.require(all(len(v) == 1 for v in events.values()), "candidate trace roster")
    bank = np.asarray(events["optimized_candidates"][0]["raw"], np.float32)
    host.require(bank.shape == (8, 3, 5, 2) and np.isfinite(bank).all() and
                 np.max(np.abs(bank)) <= 1, "bounded bank")
    base = prepared["source"]["tasks"][slot]
    latent = np.load(paths[3], allow_pickle=False)
    host.require(latent.shape == (192,) and latent.dtype == np.float32 and np.isfinite(latent).all(),
                 "actual root RGB latent")
    identity = seal["identity"]
    host.require(identity["episode_id"] == episode and identity["boundary"] == 0, "node identity")
    context = deepcopy(base["root_context"])
    context["latent_history"][-1] = latent.tolist()
    context["action_history"] = host.rgb.moving.normalized(base, np.asarray(identity["raw10"], np.float32))
    task = host.p3.local_task(base, context, identity)
    host.require(host.digest(task) == seal["task_sha256"] == events["node_header"][0]["task_sha256"],
                 "reconstructed actual RGB context")
    return task, bank, {str(p): host.sha(p) for p in paths}, prepared["source"]["policy"]


def write_new(path, value):
    payload = (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    with path.open("xb") as f:
        f.write(payload)
        f.flush()
        os.fsync(f.fileno())
    return hashlib.sha256(payload).hexdigest()


def protocol():
    inputs = []
    for slot in (0, 1):
        task, bank, sources, policy = load(slot)
        inputs.append({"slot": slot, "task_sha256": host.digest(task), "sources": sources,
                       "bank_float32_sha256": hashlib.sha256(bank.astype("<f4").tobytes()).hexdigest(),
                       "root_sha256": task["binding"]["root_sha256"], "goal_sha256": host.digest(task["goal"]),
                       "policy": policy})
    own = Path(__file__).resolve()
    dependencies = (host, chain, locking, host.p3, host.rgb.moving, host.checkpoint_compat, host.gpu_preflight)
    return {"schema": NAME + ".protocol.v1", "inputs": inputs, "code_sha256": host.sha(own),
            "dependency_sha256": {str(Path(x.__file__).resolve()): host.sha(Path(x.__file__).resolve())
                                  for x in dependencies},
            "rule": "first predicted MSE >= actual root RGB MSE; minimize predicted H3 MSE, tie by index; otherwise no selection",
            "candidate_count": 8, "horizon": 3, "no_physical_future_input": True,
            "physical_release": False, "development_only": True, "P1_admission": False,
            "limits": {"predict_calls": 48, "action_encode_calls": 48, "wall_seconds": 600},
            "note": "old banks predate protocol and have known outcomes; not prospective science"}


def run(prep, expected, output):
    import torch
    from torch.nn.attention import SDPBackend, sdpa_kernel
    from experiments.checkpoint_compat import load_compatible_pretrained
    from experiments.gpu_preflight import require_cuda_allocation

    host.require(host.sha(prep) == expected and json.loads(prep.read_bytes()) == protocol(),
                 "protocol/source drift")
    output.mkdir()
    started = time.monotonic()
    events_path = output / "events.jsonl"
    def emit(event):
        with events_path.open("ab") as f:
            f.write((json.dumps(event, sort_keys=True, allow_nan=False) + "\n").encode())
            f.flush()
            os.fsync(f.fileno())
    rows, calls = [], 0
    with locking.gpu_lock(ROOT / "results/e3_p2_host_cuda_cooperative.lock"):
        gpu = require_cuda_allocation("cuda:0")
        data = [load(i) for i in (0, 1)]
        host.require(data[0][3] == data[1][3], "same policy")
        torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
        model = load_compatible_pretrained(data[0][3], device="cuda:0").eval().requires_grad_(False)
        before = host.signal.model_hash(model)
        host.require(before["original_state_dict"] == "4b12c01f5859f4b46bea478f3a8d5e2d3f516df456a622a3feec31bde7e33546",
                     "model hash")
        with torch.inference_mode(), torch.autocast("cuda", enabled=False), sdpa_kernel(SDPBackend.MATH):
            for slot, (task, bank, _, _) in enumerate(data):
                tensor = lambda x: torch.tensor(x, dtype=torch.float32, device="cuda:0")
                states = tensor(task["root_context"]["latent_history"])[None]
                history = tensor(task["root_context"]["action_history"])[None]
                goal, mean, std = tensor(task["goal"])[None], tensor(task["action_mean"]), tensor(task["action_std"])
                root_q = float((states[:, -1] - goal).square().mean().item())
                prefix, terminal = [], []
                for index, plan in enumerate(bank):
                    z, past = states.clone(), history.clone()
                    for packet in range(3):
                        host.require(time.monotonic() - started < 600 and calls < 48, "deadline/call cap")
                        context = torch.cat((past, ((tensor(plan[packet]) - mean) / std).reshape(1, 1, 10)), 1)
                        emit({"event": "attempt", "slot": slot, "candidate": index, "packet": packet})
                        encoded = model.action_encoder(context)
                        pred = model.predict(z, encoded)
                        calls += 1
                        host.require(pred.shape == z.shape and bool(torch.isfinite(pred).all().item()), "model output")
                        z = torch.cat((z, pred[:, -1:]), 1)[:, -3:]
                        past = context[:, -2:]
                        score = float((z[:, -1] - goal).square().mean().item())
                        emit({"event": "return", "slot": slot, "candidate": index, "packet": packet, "mse": score})
                        if packet == 0:
                            prefix.append(score)
                    terminal.append(score)
                index = choose(root_q, prefix, terminal)
                rows.append({"slot": slot, "root_mse": root_q, "prefix_mse": prefix, "terminal_mse": terminal,
                             "eligible": [i for i in range(8) if prefix[i] >= root_q],
                             "selected_index": index, "selected_raw": None if index is None else bank[index].tolist()})
        torch.cuda.synchronize()
        after = host.signal.model_hash(model)
        host.require(before == after and calls == 48 and host.sha(prep) == expected and
                     json.loads(prep.read_bytes()) == protocol(), "post-run model/source drift")
        report = {"schema": NAME + ".report.v1", "status": "model_only_complete",
                  "protocol_sha256": expected, "rows": rows, "host": gpu,
                  "model_before": before, "model_after": after, "predict_calls": calls,
                  "action_encode_calls": calls, "events_sha256": host.sha(events_path),
                  "physical_future_used": False, "physical_execution": False,
                  "P1_admission": False, "independent_confirmation": False}
        return write_new(output / "report.json", report)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tag", required=True)
    p.add_argument("--run", action="store_true")
    args = p.parse_args()
    host.require(args.tag.isalnum(), "tag")
    pre = ROOT / "results" / f"{NAME}_PREP_{args.tag}.json"
    out = ROOT / "results" / f"{NAME}_RUN_{args.tag}"
    host.require(not pre.exists() and not out.exists() and not pre.is_symlink() and not out.is_symlink(),
                 "exclusive outputs")
    sha = write_new(pre, protocol())
    print(json.dumps({"protocol": str(pre), "sha256": sha}), flush=True)
    if args.run:
        print(json.dumps({"report_sha256": run(pre, sha, out), "output": str(out)}), flush=True)


if __name__ == "__main__":
    main()
