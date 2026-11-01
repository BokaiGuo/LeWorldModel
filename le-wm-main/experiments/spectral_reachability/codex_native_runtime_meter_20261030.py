"""Native PushT model entry metering; no model loading or physical dispatch.

The journal meter is deliberately not nested: each wrapped model entry makes
one durable reservation, then invokes the original method. Other operations
are not inferred from neural counts. This module does not identify Full D2P.
"""
from contextlib import contextmanager, nullcontext
from copy import deepcopy
from functools import wraps
import hashlib
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRA_SOURCE = HERE / "p2_no_composition_sra_20261027_v2.py"
SRA_SHA = "6e1402ac8abe3aabfffefd24aa016c341e6eb4f40d3f60e233152460ca70e45a"
ENCODER_SOURCE = HERE / "p4_feedback_matched_20261027_v2.py"
ENCODER_SHA = "60d39d39c6fc6292cfbe643d73e9a0ec5109f7bdae72c673afbe4af8918e212d"
FULL_SOURCE = HERE / "pusht_full_packet_compiler_20261027.py"
FULL_SHA = "f359e7b21d1970f95f065dbee206d2dc9359c2eaff7ddbcd7611e07686c6cf2a"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_files(records):
    if not isinstance(records, dict) or not records:
        raise ValueError("explicit nonempty source inventory required")
    for name, record in records.items():
        if (not isinstance(name, str) or not name or not isinstance(record, dict)
                or set(record) != {"path", "sha256"} or digest(record["path"]) != record["sha256"]):
            raise ValueError("source inventory mismatch")


@contextmanager
def reserve_native_stage(meter, label, charges):
    """Reserve the entire native stage; a failed stage leaves pending work."""
    state = meter.journal.protocol.state(meter.arm)
    if any(state["attempted"][k] + v > meter.journal.config["caps"][meter.arm][k]
           for k, v in charges.items()):
        raise ValueError("native stage budget exhausted")
    operation = f"{meter.decision_id}:native-stage:{label}"
    meter.journal.call("record_resource_attempt", meter.arm, operation, charges)
    yield
    meter.journal.call("record_resource_return", meter.arm, operation)


@contextmanager
def metered_native_model(model, meter):
    """Instrument the native model.predict and action_encoder.forward entrypoints.

    Batch size is read from the actual input shape. Forward methods are restored
    even after exceptions. Concurrent or reentrant use of this model is unsafe.
    """
    events, originals = [], []
    try:
        for owner, name, kind in ((model, "predict", "predict"),
                                  (model.action_encoder, "forward", "action_encoder")):
            original = getattr(owner, name)
            originals.append((owner, name, name in vars(owner), vars(owner).get(name)))

            def wrapper(original, kind):
                @wraps(original)
                def counted(*args, **kwargs):
                    if not args or len(args[0].shape) < 2:
                        raise ValueError("native model input shape missing")
                    batch, tokens = (int(args[0].shape[0]), int(args[0].shape[1]))
                    if batch < 1 or tokens < 1:
                        raise ValueError("empty native model batch")
                    charge = ({"predict_calls": 1, "predict_samples": batch}
                              if kind == "predict" else {"action_encoder": 1})
                    event = {"kind": kind, "batch": batch, "context_tokens": batch * tokens,
                             "returned": False}
                    events.append(event)
                    def invoke():
                        result = original(*args, **kwargs)
                        event["returned"] = True
                        return result
                    return meter.run("native_" + kind, charge, invoke)
                return counted
            setattr(owner, name, wrapper(original, kind))
        yield events
    finally:
        for owner, name, existed, value in reversed(originals):
            if existed:
                setattr(owner, name, value)
            else:
                delattr(owner, name)


def native_rgb_encoder(model, device, p0, transform, frame_shape, *, native=None):
    """Return an online encoder using the existing encode_cuda and raw RGB bytes."""
    if (not isinstance(frame_shape, tuple) or len(frame_shape) != 3
            or any(type(v) is not int or v <= 0 for v in frame_shape) or frame_shape[2] != 3):
        raise ValueError("explicit H,W,3 RGB shape required")
    if native is None:
        if digest(ENCODER_SOURCE) != ENCODER_SHA:
            raise ValueError("native encoder source drift")
        from experiments.spectral_reachability.p4_feedback_matched_20261027_v2 import encode_cuda
        native = encode_cuda
    def encode(frames, between):
        import numpy as np
        size = frame_shape[0] * frame_shape[1] * 3
        if len(frames) != 3 or any(not isinstance(frame, bytes) or len(frame) != size for frame in frames):
            raise ValueError("raw uint8 RGB frame size mismatch")
        arrays = [np.frombuffer(frame, dtype=np.uint8).reshape(frame_shape).copy() for frame in frames]
        return native(model, device, arrays, deepcopy(between), p0, transform)
    return encode


def sra_online_bank(model, device, context, compiled, replay, meter, *,
                    native=None, source_inventory):
    """Invoke native joint-H2 SRA compiler on the current context.

    This returns the *bank*, not a selected action; native run_bank still must
    evaluate it online and connect its result to the paired journal. Reservations
    for Jacobian/SVD/solve are conservative, not measured completed operations.
    """
    verify_files(source_inventory)
    if native is None:
        if digest(SRA_SOURCE) != SRA_SHA:
            raise ValueError("native SRA source drift")
        from experiments.spectral_reachability.p2_no_composition_sra_20261027_v2 import compile_cuda
        native = compile_cuda
    operations = []
    with reserve_native_stage(meter, "sra", {"jacobian": 1, "vjp_directions": 192, "svd": 2, "solve": 1}):
        with metered_native_model(model, meter) as entries:
            bank, compile_events = native(model, device, {"root": {"context": deepcopy(context)}},
                                          deepcopy(compiled), replay, lambda e: operations.append(deepcopy(e)))
    verify_files(source_inventory)
    return {"bank": bank, "compile_events": compile_events, "native_entries": entries,
            "operation_events": operations, "resources_are_complete": False,
            "equal_total_compute": False, "flops": None}


def _full_native(model, device, context, compiled):
    """Execute the existing full-coordinate packet compiler with a fresh root Jacobian."""
    if digest(FULL_SOURCE) != FULL_SHA:
        raise ValueError("native packet compiler source drift")
    import numpy as np
    import torch
    from experiments.spectral_reachability import pusht_full_packet_compiler_20261027 as compiler
    emb = torch.tensor(context["latent_history"], dtype=torch.float32, device=device)[None]
    history = torch.tensor(context["action_history"], dtype=torch.float32, device=device)[None]
    goal = compiler.finite(compiled["goal_vectors"][0], (192,))
    q = compiler.finite(compiled["scoring"]["metric"]["projection"], (192, 192))
    transition = compiler.make_transition(model, device)
    def terminal(u):
        return transition(emb, history, torch.tanh(u).reshape(5, 2))[0][0, -1]
    with torch.no_grad():
        origin = terminal(emb.new_zeros(10))
    jac = torch.autograd.functional.jacobian(terminal, emb.new_zeros(10).requires_grad_(True), vectorize=True)
    array = lambda x: x.detach().cpu().numpy().astype(np.float64)
    class Counter:
        def phase(self, *args, **kwargs):
            return nullcontext()
    return compiler.compile_arm(arm="full", emb=emb, history=history, goal=goal, q=q,
                                jac_full=array(jac), origin=array(origin), transition=transition,
                                model=model, device=device, counter=Counter())


def full_packet_online_compile(model, device, context, compiled, meter, *, source_inventory):
    """Fresh native composition-component compile, not a complete Full D2P method."""
    verify_files(source_inventory)
    with reserve_native_stage(meter, "packet", {"jacobian": 11, "vjp_directions": 2112, "svd": 12, "solve": 11}):
        with metered_native_model(model, meter) as entries:
            result = _full_native(model, device, deepcopy(context), deepcopy(compiled))
    verify_files(source_inventory)
    return {"native_result": result, "native_entries": entries, "resources_are_complete": False,
            "equal_total_compute": False, "flops": None}
