"""Spawned RGB-only planner protocol; process separation is not an OS sandbox."""
import multiprocessing as mp
from types import SimpleNamespace

import numpy as np

FIELDS = {"pixels", "goal", "old_tail", "remaining_steps", "candidate_seed", "strategy_seed", "strategy"}


def validate_request(request):
    if not isinstance(request, dict) or set(request) != FIELDS:
        raise ValueError("planner request fields do not match RGB-only contract")
    for name in ("pixels", "goal"):
        value = request[name]
        if not isinstance(value, np.ndarray) or value.dtype != np.uint8 or value.shape != (224, 224, 3):
            raise ValueError(f"{name} must be uint8 224x224 RGB")
    remaining = request["remaining_steps"]
    if type(remaining) is not int or remaining not in (5, 10, 15, 20, 25):
        raise ValueError("invalid remaining clock")
    for key in ("candidate_seed", "strategy_seed"):
        if type(request[key]) is not int or not 0 <= request[key] < 2**63:
            raise ValueError("invalid seed")
    if request["strategy"] != "lookahead":
        raise ValueError("this frozen smoke only supports lookahead")
    old = request["old_tail"]
    if old is not None:
        if (not isinstance(old, np.ndarray) or old.dtype != np.float32 or old.shape != (remaining, 2)
                or not np.isfinite(old).all() or (np.abs(old) > 1).any()):
            raise ValueError("invalid legal old tail")
    return request


def _worker(connection, mean, scale, audit, expected_model):
    import torch
    from sklearn.preprocessing import StandardScaler
    from eval import img_transform
    from experiments.checkpoint_compat import load_compatible_pretrained
    from experiments.pc_input_permissions import RGBLeWM
    from experiments.pc_prefix_novel_actual_encoding_20261105_codex import model_fingerprint
    from experiments import pc_rgb_hold_retention_search_20261107_codex as search

    try:
        scaler = StandardScaler()
        scaler.mean_ = np.asarray(mean, dtype=np.float64)
        scaler.scale_ = np.asarray(scale, dtype=np.float64)
        scaler.var_ = scaler.scale_ ** 2
        scaler.n_features_in_ = 2
        if scaler.mean_.shape != (2,) or scaler.scale_.shape != (2,) or not np.all(scaler.scale_ > 0):
            raise ValueError("invalid normalizer")
        model = load_compatible_pretrained("tworoom/lewm", device="cuda").eval().requires_grad_(False)
        model.interpolate_pos_encoding = True
        fingerprint = model_fingerprint(model)
        if fingerprint != expected_model:
            raise ValueError("unexpected checkpoint")
        boundary = RGBLeWM(model, audit)
        transform = img_transform(SimpleNamespace(eval=SimpleNamespace(img_size=224)))
        connection.send({"status": "ready", "model_sha256": fingerprint})
        with torch.inference_mode():
            while True:
                request = connection.recv()
                if request is None:
                    break
                validate_request(request)
                context = {"pixels": transform(request["pixels"])[None, None, None].cuda(),
                           "goal": transform(request["goal"])[None, None, None].cuda(),
                           "action": torch.empty((1, 1, 0, 10), device="cuda")}
                before = len(boundary.receipts)
                result = search.compile_rgb(boundary, context, scaler=scaler, old_tail=request["old_tail"],
                    candidate_seed=request["candidate_seed"], strategy_seed=request["strategy_seed"],
                    strategy=request["strategy"], remaining_steps=request["remaining_steps"])
                torch.cuda.synchronize()
                choice = result["choice"]
                connection.send({"status": "ok", "actions": result["actions"],
                    "choice": {"source": choice.source, "index": choice.index, "cost": choice.cost},
                    "queries": result["model_queries"], "forward_calls": result["forward_calls"],
                    "receipts": boundary.receipts[before:]})
        connection.send({"status": "stopped", "model_sha256": model_fingerprint(model)})
    except BaseException as exc:
        try:
            connection.send({"status": "error", "error": f"{type(exc).__name__}: {exc}"})
        except (BrokenPipeError, EOFError, OSError):
            pass
    finally:
        connection.close()


class SpawnPlanner:
    def __init__(self, mean, scale, audit, expected_model):
        context = mp.get_context("spawn")
        self.connection, child = context.Pipe()
        self.process = context.Process(target=_worker, args=(child, mean, scale, audit, expected_model))
        self.process.start()
        child.close()
        self.ready = self._receive(60)

    def _receive(self, timeout):
        if not self.connection.poll(timeout):
            raise TimeoutError("planner process did not respond")
        reply = self.connection.recv()
        if reply.get("status") == "error":
            raise RuntimeError(reply["error"])
        return reply

    def plan(self, request):
        validate_request(request)
        self.connection.send(request)
        reply = self._receive(60)
        if reply.get("status") != "ok":
            raise RuntimeError("unexpected planner response")
        return reply

    def close(self):
        self.connection.send(None)
        reply = self._receive(60)
        self.process.join(10)
        if self.process.is_alive() or self.process.exitcode != 0 or reply.get("status") != "stopped":
            raise RuntimeError("planner did not stop cleanly")
        self.connection.close()
        return reply
