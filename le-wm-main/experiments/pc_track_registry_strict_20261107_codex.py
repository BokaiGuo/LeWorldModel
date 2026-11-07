"""Additional fail-closed receipt checks for future dual-track registrations.

This verifies receipt structure, not the authenticity of unretained tensors.
Legacy eligibility remains unchanged so historical hashes remain reproducible.
"""
import re
from pathlib import Path

from experiments.pc_input_permissions import file_hash
from experiments.pc_track_registry import eligible

HEX = re.compile(r"^[0-9a-f]{64}$")
DTYPES = {
    "torch.float16", "torch.float32", "torch.float64", "torch.bfloat16",
    "torch.int8", "torch.int16", "torch.int32", "torch.int64",
    "torch.uint8", "torch.bool",
}


def validate_tensor_receipt(receipt):
    if not isinstance(receipt, dict) or set(receipt) != {"shape", "dtype", "sha256"}:
        raise ValueError("tensor receipt requires exactly shape, dtype and sha256")
    shape, dtype, digest = receipt["shape"], receipt["dtype"], receipt["sha256"]
    if (not isinstance(shape, list) or
            any(type(value) is not int or value < 0 for value in shape)):
        raise ValueError("invalid tensor shape")
    if (not isinstance(dtype, str) or dtype not in DTYPES
            or not isinstance(digest, str) or not HEX.fullmatch(digest)):
        raise ValueError("invalid tensor dtype or digest")


def strict_eligible(record, *, target_track, algorithm_version, protocol_path,
                    task_manifest_path, required_operations):
    """Verify real source files and receipt schema; never decide a scientific gate."""
    if not isinstance(required_operations, (set, frozenset)) or not required_operations:
        raise ValueError("explicit nonempty required operation set is required")
    if any(not isinstance(x, str) for x in required_operations):
        raise ValueError("invalid required operations")
    protocol_path, task_manifest_path = Path(protocol_path), Path(task_manifest_path)
    if not protocol_path.is_file() or not task_manifest_path.is_file():
        raise ValueError("protocol and task manifest must exist")
    eligible(record, target_track=target_track, algorithm_version=algorithm_version,
             protocol_hash=file_hash(protocol_path), task_manifest_hash=file_hash(task_manifest_path))
    operations = set()
    for receipt in record["input_receipts"]:
        operations.add(receipt["operation"])
        for tensor in receipt["inputs"].values():
            validate_tensor_receipt(tensor)
    if not required_operations.issubset(operations):
        raise ValueError("required runtime operations are missing")
    return True
