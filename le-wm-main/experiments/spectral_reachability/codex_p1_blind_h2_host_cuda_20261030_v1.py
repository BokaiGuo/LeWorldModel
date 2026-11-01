"""Blind RGB -> exploratory fixed-H2 CUDA planner. Never executes physics.

Only the blind bundle, local model files and runtime code are inputs. This is
not an isolation boundary or physical authorization; see the handoff.
"""
import argparse
from copy import deepcopy
import hashlib
import inspect
import json
import os
from pathlib import Path
import shutil
import time

import numpy as np

from experiments.spectral_reachability import codex_p1_h2_fixed_candidate_20261030_v1 as h2

NAME = "codex_p1_blind_h2_host_cuda_20261030_v1"
CAP = 128 * 1024**2
RESERVE = 2 * 1024**2
FLOOR = 200 * 1024**2
WALL = 3600
need = h2.blind.need
obs = h2.observed
base = obs.base
enc = obs.rgb.enc


def hash_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for part in iter(lambda: stream.read(1024*1024), b""):
            h.update(part)
    return h.hexdigest()


def bytes_json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode() + b"\n"


def model_hash(model):
    state, buffers = hashlib.sha256(), hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        a = value.detach().cpu().contiguous()
        state.update(name.encode()); state.update(str(a.dtype).encode())
        state.update(str(tuple(a.shape)).encode()); state.update(a.numpy().tobytes())
    for name, value in model.named_buffers():
        a = value.detach().cpu().contiguous()
        buffers.update(json.dumps([name, str(a.dtype), list(a.shape)], sort_keys=True,
                                  separators=(",", ":"), allow_nan=False).encode())
        buffers.update(a.numpy().tobytes())
    return dict(original_state_dict=state.hexdigest(), all_buffers=buffers.hexdigest())


def inputs(bundle_path, bundle_sha, model_dir):
    bundle, _ = h2.blind.read_verified(bundle_path, bundle_sha)
    model_dir = Path(model_dir)
    need(model_dir.is_absolute() and model_dir.is_dir() and model_dir.resolve() == model_dir,
         "absolute real local model directory required")
    files = {}
    for name in ("weights.pt", "config.json"):
        path = model_dir / name
        need(path.is_file() and not path.is_symlink() and path.resolve() == path, "local model file")
        files[name] = hash_file(path)
    need(files == bundle["model_files"], "blind task model file hashes")
    return bundle, files


class Store:
    def __init__(self, path):
        path = Path(path)
        need(path.is_absolute() and path.parent.is_dir() and path.parent.resolve() == path.parent and
             path.name.startswith(NAME+"_") and not os.path.lexists(path), "exclusive attempt path")
        need(shutil.disk_usage(path.parent).free >= FLOOR + CAP, "disk reserve")
        path.mkdir(mode=0o700)
        self.path, self.used, self.started = path, 0, time.monotonic()

    def put(self, name, value, append=False, final=False):
        need(Path(name).name == name, "flat filename")
        payload = bytes_json(value)
        need(self.used + len(payload) <= CAP-(0 if final else RESERVE), "attempt byte cap")
        need(shutil.disk_usage(self.path).free >= FLOOR+len(payload), "disk floor")
        if not final:
            need(time.monotonic()-self.started < WALL, "wall cap")
        flags = os.O_WRONLY | os.O_CREAT | os.O_NOFOLLOW | (os.O_APPEND if append else os.O_EXCL)
        fd = os.open(self.path/name, flags, 0o600)
        with os.fdopen(fd, "wb") as stream:
            need(stream.write(payload) == len(payload), "short write")
            stream.flush(); os.fsync(stream.fileno())
        self.used += len(payload)

    def event(self, event):
        self.put("events.jsonl", event, append=True)


class EncodeLedger:
    def __init__(self, torch, sync, sink):
        self.torch, self.sync, self.sink = torch, sync, sink
        self.entries, self.blocked, self.image = [], [], None

    def dispatch(self, kind, tensor, callback, *args, **kwargs):
        expected = [(i, k) for i in range(16) for k in ("encoder", "projector")]
        need(len(self.entries) < len(expected) and (self.image, kind) == expected[len(self.entries)] and
             not self.torch.is_grad_enabled() and all(e["completed"] for e in self.entries), "encode order/cap")
        shape = (1, 3, 224, 224) if kind == "encoder" else (1, 192)
        need(getattr(getattr(tensor, "device", None), "type", None) == "cuda" and
             tuple(tensor.shape) == shape, "encoder/projector CUDA shape")
        entry = dict(index=len(self.entries), image=self.image, resource=kind, completed=False)
        self.entries.append(entry)
        self.sink(dict(kind="encode_attempt", entry=deepcopy(entry)))
        value = callback(*args, **kwargs)
        self.sync()
        entry["completed"] = True
        self.sink(dict(kind="encode_completed", entry=deepcopy(entry)))
        return value

    def forbid(self, name):
        def denied(*args, **kwargs):
            self.blocked.append(name)
            raise ValueError("forbidden during encode: "+name)
        return denied

    def counts(self):
        return {key: dict(attempted=sum(e["resource"] == key for e in self.entries),
                          completed=sum(e["resource"] == key and e["completed"] for e in self.entries))
                for key in ("encoder", "projector")}


class GuardedBackend:
    def __init__(self, packet, ledger, model, device, torch, sink):
        self.ledger, self.model, self.device, self.torch = ledger, model, device, torch
        self.ledger.sync = lambda: torch.cuda.synchronize(device)
        self.ad = obs.OneAD(torch, base.ARMS[1], sink, self.ledger.sync)
        self.native = base.d2p.TorchBackend(model, device)

    def call(self, method, *args):
        self.ad.undo = []
        with self.ad, base.module_guard(self.model, self.ledger):
            return getattr(self.native, method)(*args)

    def nominal(self, context): return self.call("nominal", context)
    def jacobian(self, context, operation): return self.call("jacobian", context, operation)
    def step(self, context, commands): return self.call("step", context, commands)
    def replay(self, context, commands): return self.call("replay", context, commands)


def drive(bundle, encode_rgb, plan, sink, rows):
    """Single ordered pass; injected callables for test only or real CUDA wiring."""
    need(len(rows) == 16 and len(bundle["rows"]) == 16, "fixed denominator")
    for index, (row, source) in enumerate(zip(rows, bundle["rows"], strict=True)):
        need(index == source["index"] == row["index"], "row order")
        row["status"] = "attempted"
        rgb = h2.blind.decode(source["cp5_rgb"]["rgb_base64"], 224*224*3)
        need(h2.blind.sha(rgb) == source["cp5_rgb"]["pixel_sha256"], "cp5 pixel binding")
        pixels = np.frombuffer(rgb, dtype=np.uint8).reshape(224, 224, 3).copy()
        sink(dict(kind="cp5_encoding_started", index=index, pixel_sha256=source["cp5_rgb"]["pixel_sha256"]))
        latent = encode_rgb(index, pixels)
        enc.native(latent)
        latent_sha = h2.blind.sha(latent.astype("<f4", copy=False).tobytes())
        sink(dict(kind="cp5_encoded", index=index, latent_sha256=latent_sha))
        result, ad_counts = plan(index, latent)
        need(result["row_index"] == index and result["physical_authorized"] is False and
             len(ad_counts) == 2, "fixed H2 result")
        row.update(status="planned", cp5_pixel_sha256=source["cp5_rgb"]["pixel_sha256"],
                   cp5_latent_float32=latent.tolist(), cp5_latent_sha256=latent_sha,
                   h2_result=result, ad_counts=ad_counts)
        sink(dict(kind="row_planned", index=index, cp5_latent_sha256=latent_sha,
                  result_sha256=h2.blind.digest(result)))


def run(bundle_path, bundle_sha, model_dir, output, acknowledge):
    need(acknowledge == "ISOLATED_HOST_CUDA_COORDINATED", "explicit host/isolation coordination required")
    bundle, model_files = inputs(bundle_path, bundle_sha, model_dir)
    store = Store(output)
    rows = [dict(index=i, status="not_started") for i in range(16)]
    errors, before, after, host, enc_ledger = [], None, None, None, None
    store.put("binding.json", dict(bundle_sha256=bundle_sha, model_files=model_files,
              specification=h2.specification(), specification_sha256=h2.blind.digest(h2.specification()),
              runner_sha256=hash_file(Path(__file__).resolve()), training=False, physical_authorized=False,
              pollution="prior adapter tests read historical outcome/checkpoints; no preregistration guarantee"))
    try:
        need(os.environ.get("CUDA_VISIBLE_DEVICES") not in ("", "-1"), "CUDA disabled")
        os.environ["HF_HUB_OFFLINE"] = os.environ["TRANSFORMERS_OFFLINE"] = "1"
        import torch
        from torch.nn.attention import SDPBackend, sdpa_kernel
        from experiments.checkpoint_compat import load_compatible_pretrained
        from experiments.gpu_preflight import require_cuda_allocation
        from experiments.spectral_reachability import preflight_pusht_d2p_p0 as p0

        host = require_cuda_allocation("cuda:0")
        device = torch.device("cuda:0")
        torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
        model = load_compatible_pretrained(str(model_dir), device=device).eval().requires_grad_(False)
        enc.frozen(model)
        before = model_hash(model)
        need(before == bundle["model_before"], "saved model state/buffer identity")
        classes = {str(Path(inspect.getfile(type(m))).resolve()) for m in model.modules()}
        need(all("/results/" not in p and Path(p).is_file() for p in classes), "model class source")
        class_hashes = {p: hash_file(p) for p in sorted(classes)}
        code_paths = {str(Path(m.__file__).resolve()) for m in (h2, h2.blind, obs, base, enc, p0)}
        code_paths.add(str(Path(__file__).resolve()))
        code_hashes = {p: hash_file(p) for p in sorted(code_paths)}
        store.put("runtime_sources.json", dict(model_class_sha256=class_hashes, core_sha256=code_hashes,
                  limitation="Observed after model load; not a preapproved complete dynamic dependency census."))
        sync = lambda: torch.cuda.synchronize(device)
        transform = p0.image_transform()
        enc_ledger = EncodeLedger(torch, sync, store.event)
        restore_errors = []

        def encode_rgb(index, pixels):
            enc_ledger.image = index
            with enc.guard(model, torch, enc_ledger, restore_errors), torch.inference_mode():
                x = p0._image_tensor(pixels, transform=transform, device=device)
                need(x.device.type == "cuda" and tuple(x.shape) == (1, 1, 3, 224, 224), "RGB CUDA input")
                z = model.encode({"pixels": x})["emb"]
                need(z.device.type == "cuda" and z.dtype == torch.float32 and
                     tuple(z.shape) == (1, 1, 192) and bool(torch.isfinite(z).all()), "encoded cp5")
                result = z[0, 0].detach().cpu().numpy().copy()
            need(not restore_errors and [(e["resource"], e["completed"]) for e in enc_ledger.entries
                                         if e["image"] == index] == [("encoder", True), ("projector", True)],
                 "encoding receipts")
            return result

        def plan(index, latent):
            instances = []
            def sink(e): store.event(dict(row=index, event=e))
            def factory(packet, ledger):
                obj = GuardedBackend(packet, ledger, model, device, torch, sink)
                instances.append(obj)
                return obj
            result = h2.compile_row(bundle, index, latent, factory, sink)
            need(len(instances) == 2, "two packets")
            counts = [obj.ad.counts() for obj in instances]
            for c in counts:
                base.exact_counts(c, dict(grad=1, jacobian=1))
            return result, counts

        with torch.autocast(device_type="cuda", enabled=False), sdpa_kernel(SDPBackend.MATH):
            drive(bundle, encode_rgb, plan, store.event, rows)
        sync()
        enc.frozen(model)
        after = model_hash(model)
        need(before == after and enc_ledger.counts() ==
             {k: dict(attempted=16, completed=16) for k in ("encoder", "projector")},
             "model unchanged and exact encoding count")
        need({p: hash_file(p) for p in class_hashes} == class_hashes, "model class drift")
        need({p: hash_file(p) for p in code_hashes} == code_hashes, "core source drift")
        _, after_files = inputs(bundle_path, bundle_sha, model_dir)
        need(after_files == model_files, "input drift")
    except Exception as exc:
        errors.append(dict(type=type(exc).__name__, message=str(exc)[:1500]))
        try: store.event(dict(kind="failure", error=errors[-1]))
        except Exception as secondary: errors.append(dict(type=type(secondary).__name__, message=str(secondary)[:1500]))
    for row in rows:
        if row["status"] == "attempted":
            row["status"] = "failed_partial"
    complete = not errors and before is not None and before == after and all(r["status"] == "planned" for r in rows)
    report = dict(schema=NAME+".report", status="exploratory_plans_only" if complete else "partial_or_failed",
                  rows=rows, errors=errors, bundle_sha256=bundle_sha, model_before=before, model_after=after,
                  host=host, encoder_counts=enc_ledger.counts() if enc_ledger else None,
                  formal_conditions=24, greedy_unknown=8, training=False, physical_authorized=False,
                  P1_admission=False, preregistration_guarantee=False,
                  prior_outcome_read_contamination=True, total_budget_equality_proven=False)
    store.put("report.json", report, final=True)
    need(complete, "partial attempt retained; no automatic retry")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", required=True)
    parser.add_argument("--sha256", required=True)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--acknowledge", required=True)
    args = parser.parse_args()
    result = run(args.bundle, args.sha256, args.model_dir, args.output, args.acknowledge)
    print(json.dumps(dict(status=result["status"], rows=len(result["rows"]), physical_authorized=False)))


if __name__ == "__main__":
    main()
