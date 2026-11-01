"""Observe model calls during a separate replay of the fixed P1 Adam planner.

This does not instrument the earlier sealed run or the prefix-generation cost.
"""
import argparse
import hashlib
import json
from pathlib import Path
from unittest.mock import patch

from . import p1_uphill_multistart_adam_20261030_main as planner

NAME = "p1_adam_runtime_counts_20261031"


def install_counters(model, counts):
    """Instrument public model entry points and the action encoder module."""
    encode, predict = model.encode, model.predict

    def observed_encode(*args, **kwargs):
        counts["encode"] += 1
        return encode(*args, **kwargs)

    def observed_predict(*args, **kwargs):
        counts["predict"] += 1
        return predict(*args, **kwargs)

    def observe_action(_module, _inputs):
        counts["action_encoder"] += 1

    model.encode = observed_encode
    model.predict = observed_predict
    return model.action_encoder.register_forward_pre_hook(observe_action)


def run(output, receipt):
    import torch
    from experiments import checkpoint_compat

    base = planner.adam.prior.base
    output, receipt = Path(output), Path(receipt)
    for path in (output, receipt):
        if (not path.is_absolute() or path.parent != base.ROOT / "results"
                or not path.name.startswith(NAME + "_") or path.exists() or path.is_symlink()):
            raise ValueError("exclusive result path required")
    if output == receipt:
        raise ValueError("result paths must differ")
    counts = dict(encode=0, predict=0, action_encoder=0, backward=0)
    original_load = checkpoint_compat.load_compatible_pretrained
    original_backward = torch.Tensor.backward
    hooks = []

    def load(*args, **kwargs):
        model = original_load(*args, **kwargs)
        hooks.append(install_counters(model, counts))
        return model

    def backward(tensor, *args, **kwargs):
        counts["backward"] += 1
        return original_backward(tensor, *args, **kwargs)

    planner.configure()
    planner.adam.NAME = NAME
    try:
        with patch.object(checkpoint_compat, "load_compatible_pretrained", load), \
                patch.object(torch.Tensor, "backward", backward):
            planner.adam.run(output)
    finally:
        for hook in hooks:
            hook.remove()

    expected = {"encode": 24, "predict": 4032, "action_encoder": 4032, "backward": 1920}
    source = Path(__file__).resolve()
    result = {
        "schema": NAME,
        "status": "observed",
        "counts": counts,
        "expected_code_path_counts": expected,
        "counts_match": counts == expected,
        "planner_output": str(output),
        "planner_output_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "observer_source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "scope": "Separate fixed 24-row cp5 suffix-planning replay only; excludes prefix generation.",
        "limitations": "Entry point and tensor-backward counts, not FLOPs or complete budget certification.",
        "P1_admission": False,
    }
    with receipt.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--receipt", required=True)
    args = parser.parse_args()
    run(args.output, args.receipt)
