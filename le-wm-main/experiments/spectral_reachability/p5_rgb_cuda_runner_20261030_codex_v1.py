"""Fixed 164 x 5 RGB rollout; CUDA execution requires explicit main-thread authorization."""
import argparse
import io
import os
import time
from pathlib import Path

import numpy as np

from experiments.spectral_reachability import p5_rgb_rollout_contract_20261030_codex_v1 as abi

NAME = "p5_rgb_cuda_runner_20261030_codex_v1"
PREP = abi.bridge.ROOT / "results/p5_rgb_rollout_contract_20261030_codex_v1_MAIN01.json"
PREP_SHA = "1d714552ff6758e7793a37b58c6f9ca65ed88e40e0e2039fc24d2b586c33f070"
CODE_PINS = {
    "experiments/spectral_reachability/p5_rgb_rollout_contract_20261030_codex_v1.py": "e74fdf13c7f22b502afd29f1ed8b602377cf44e68fd2eb33d153c62cbd428689",
    "experiments/spectral_reachability/p5_rgb_causal_panel_bridge_20261030_codex_v1.py": "552125b60a2d79d7d17d969c6a1faf31ae55edc2d899888daaf67e328d9b349b",
    "experiments/evaluate_causal_visual_checkpoint.py": "f73a3830be6c75f6ca7f81130107bf54245d6057ef33d0885fec8a80b2c2eeef",
}
AUTH = "MAIN_THREAD_APPROVED_FIXED_164_RGB_CUDA"


def load_prepared():
    raw = PREP.read_bytes()
    abi.check(abi.bridge.digest(raw) == PREP_SHA, "prepared SHA")
    p = abi.bridge.read_json(raw)
    abi.check(p["schema"] == abi.STEM and p["branch_denominator"] == 164 and
              p["pair_denominator"] == 80 and len(p["rows"]) == 164 and
              len(p["pairs"]) == 80 and p["execution_authorized"] is False, "prepared schema")
    return p


def verify_sources():
    for name, expected in CODE_PINS.items():
        path = abi.bridge.ROOT / name
        abi.check(path.is_file() and not path.is_symlink() and
                  abi.bridge.digest(path.read_bytes()) == expected, "source SHA " + name)
    return abi.bridge.read_fixed("checkpoint"), abi.bridge.read_fixed("model_source")


def sync_dir(path):
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


class Journal:
    def __init__(self, directory):
        self.directory = directory
        directory.mkdir()
        sync_dir(directory.parent)
        self.file = (directory / "events.jsonl").open("xb")
        sync_dir(directory)
        self.prev = "0" * 64
        self.seq = 0

    def emit(self, kind, **fields):
        body = {"seq": self.seq, "prev": self.prev, "kind": kind, **fields}
        digest = abi.bridge.digest(abi.bridge.canonical(body))
        self.file.write(abi.bridge.canonical({**body, "hash": digest}))
        self.file.flush()
        os.fsync(self.file.fileno())
        self.prev, self.seq = digest, self.seq + 1

    def close(self):
        self.file.close()


def vector(value):
    a = np.asarray(value, dtype="<f4")
    abi.check(a.shape == (1, 6) and np.isfinite(a).all(), "output finite float32 [1,6]")
    return {"values": a[0].tolist(), "sha256": abi.bridge.digest(a.tobytes())}


def run_protocol(prepared, backend, journal, max_seconds, clock=time.monotonic):
    """Journal interface calls only; caller has already checked all sources and inputs."""
    start = clock()
    journal.emit("start", prepared_sha256=PREP_SHA, max_seconds=max_seconds,
                 runner_source_sha256=abi.bridge.digest(Path(__file__).read_bytes()),
                 backend=backend.metadata())
    try:
        for row in prepared["rows"]:
            if clock() - start >= max_seconds:
                raise TimeoutError("between-call time cap")
            values = abi.materialize(row)
            identity = {"branch": row["index"], "condition": row["condition"],
                        "previous_sha256": row["previous_rgb"]["model_input_chw_float32_sha256"],
                        "image_sha256": row["current_rgb"]["model_input_chw_float32_sha256"],
                        "actions_sha256": row["actions_sha256"]}
            for step in range(6):
                if clock() - start >= max_seconds:
                    raise TimeoutError("between-call time cap")
                op = "encode_state" if step == 0 else "predict_next"
                journal.emit("attempt", **identity, op=op, step=step)
                try:
                    result = backend.encode(values) if step == 0 else backend.step(values, step - 1)
                    out = vector(result)
                except Exception as exc:
                    journal.emit("failed", **identity, op=op, step=step,
                                 error_type=type(exc).__name__, error=str(exc)[:500])
                    raise
                journal.emit("returned", **identity, op=op, step=step, output=out)
        journal.emit("end", status="complete", elapsed_seconds=clock() - start)
        return "complete"
    except Exception as exc:
        journal.emit("end", status="partial", elapsed_seconds=clock() - start,
                     error_type=type(exc).__name__, error=str(exc)[:500])
        return "partial"


class CudaBackend:
    def __init__(self, checkpoint, prepared, device_index):
        import torch
        from experiments.run_causal_visual_pusht_world_model import Normalization, VisualCausalWorldModel
        from experiments.evaluate_causal_visual_checkpoint import checkpoint_adapter_architecture, checkpoint_utility_mode

        abi.check(torch.cuda.is_available() and 0 <= device_index < torch.cuda.device_count(), "CUDA required")
        self.torch = torch
        self.device = torch.device("cuda", device_index)
        payload = torch.load(io.BytesIO(checkpoint), map_location="cpu", weights_only=True)
        args = payload["args"]
        flags = ("structured_readout", "contact_frame_dynamics", "mode_conditioned_dynamics",
                 "current_only_readout", "torque_features", "temporal_dynamics",
                 "previous_action_dynamics", "explicit_angle_effect", "explicit_angle_readout",
                 "explicit_goal_angle_readout", "rich_temporal_dynamics", "invariant_angle_readout",
                 "contact_interface_dynamics", "angle_readout_current_only")
        abi.check(set(payload) == {"model", "normalization", "args"} and
                  args["latent_dim"] == 96 and args["vision_encoder"] == "mask_moments" and
                  args["temporal_context"] is True and args["action_repeat"] == 1 and
                  args["image_size"] == 64 and all(args.get(k, False) is False for k in flags) and
                  checkpoint_utility_mode(payload) == "phase" and
                  checkpoint_adapter_architecture(payload) == "legacy" and
                  payload["normalization"] == prepared["normalization_from_prior_strict_load"], "checkpoint ABI")
        with torch.device("cpu"):
            model = VisualCausalWorldModel(96, "mask_moments", True, "phase", "legacy")
            status = model.load_state_dict(payload["model"], strict=True)
        abi.check(not status.missing_keys and not status.unexpected_keys and
                  len(payload["model"]) == 26 and sum(p.numel() for p in model.parameters()) == 93842,
                  "strict state dictionary")
        self.model = model.to(self.device).eval().requires_grad_(False)
        norm = Normalization(**payload["normalization"])
        self.model.set_state_normalization(*norm.tensors(str(self.device)))
        abi.check(not self.model.training and all(p.device == self.device and not p.requires_grad
                  for p in self.model.parameters()), "CUDA frozen model")
        self.state = None

    def metadata(self):
        return {"device": str(self.device), "device_name": self.torch.cuda.get_device_name(self.device),
                "torch_version": self.torch.__version__, "cuda_version": self.torch.version.cuda,
                "strict_load": True, "frozen": True, "eval": True,
                "checkpoint_sha256": abi.bridge.FILES["checkpoint"][1]}

    def output(self):
        t = self.state
        abi.check(t.device == self.device and t.dtype == self.torch.float32 and tuple(t.shape) == (1, 6),
                  "CUDA float32 output")
        self.torch.cuda.synchronize(self.device)
        return t.detach().cpu().numpy()

    def encode(self, values):
        t = self.torch
        with t.inference_mode():
            self.state = self.model.encode_state(t.from_numpy(values["image"]).to(self.device),
                                                 t.from_numpy(values["previous_image"]).to(self.device))
        return self.output()

    def step(self, values, index):
        t = self.torch
        with t.inference_mode():
            self.state = self.model.predict_next(self.state, t.from_numpy(values["actions"][:, index, :]).to(self.device))
        return self.output()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--authorization", required=True)
    ap.add_argument("--device-index", type=int, required=True)
    ap.add_argument("--max-seconds", type=int, required=True)
    ap.add_argument("--output-name", required=True)
    args = ap.parse_args()
    abi.check(args.authorization == AUTH, "main-thread authorization required")
    abi.check(1 <= args.max_seconds <= 3600 and args.device_index >= 0, "resource cap")
    abi.check(Path(args.output_name).name == args.output_name and
              args.output_name.startswith(NAME + "_"), "exclusive output name")
    directory = abi.bridge.ROOT / "results" / args.output_name
    abi.check(not directory.exists() and not directory.is_symlink(), "output already exists")
    prepared = load_prepared()
    checkpoint, _ = verify_sources()
    for row in prepared["rows"]:
        abi.materialize(row)
    backend = CudaBackend(checkpoint, prepared, args.device_index)
    journal = Journal(directory)
    try:
        status = run_protocol(prepared, backend, journal, args.max_seconds)
    finally:
        journal.close()
    print(f"status: {status}; journal: {directory / 'events.jsonl'}")
    if status != "complete":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
