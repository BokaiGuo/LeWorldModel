"""Fail-closed UUID representation bridge; integration prototype, not a launcher.

The caller independently obtains nvidia-smi's physical UUID and retains the
original source/plan/authority gates. No CUDA, nvidia-smi, or model call occurs
on import. Do not patch the global torch module.
"""
import re
import uuid

GPU = re.compile(r"GPU-[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}\Z")


def require(value, message):
    if not value:
        raise ValueError(message)


def canonical_physical(text):
    require(type(text) is str and GPU.fullmatch(text) is not None, "canonical physical GPU UUID required")
    return text


def object_evidence(value, uuid_type, expected):
    """Interpret the exact runtime UUID type's 16 unsigned bytes in wire order."""
    canonical_physical(expected)
    require(isinstance(uuid_type, type) and type(value) is uuid_type, "unexpected CUDA UUID object type")
    data = value.bytes
    require(type(data) in (list, tuple, bytes) and len(data) == 16 and
            all(type(x) is int and 0 <= x <= 255 for x in data), "invalid CUDA UUID bytes")
    raw = bytes(data)
    canonical = "GPU-" + str(uuid.UUID(bytes=raw))
    require(canonical == expected, "CUDA UUID bytes differ from admitted physical GPU")
    return dict(physical_uuid=canonical, bytes_hex=raw.hex(),
                runtime_type=type(value).__module__ + "." + type(value).__qualname__)


class _Properties:
    def __init__(self, original, evidence):
        self._original = original
        self.uuid = evidence["physical_uuid"]

    def __getattr__(self, name):
        return getattr(self._original, name)


class _Cuda:
    def __init__(self, torch, expected):
        self._torch, self._cuda, self._expected = torch, torch.cuda, expected

    def get_device_properties(self, index):
        require(type(index) is int and index == 0, "only admitted cuda:0")
        props = self._cuda.get_device_properties(index)
        proof = object_evidence(props.uuid, self._torch._C._CUuuid, self._expected)
        return _Properties(props, proof)

    def __getattr__(self, name):
        return getattr(self._cuda, name)


class TorchView:
    """Delegate torch unchanged except checked cuda:0 property UUID formatting."""
    def __init__(self, torch, expected, smi_uuid, visible):
        canonical_physical(expected)
        require(smi_uuid == expected and visible == expected, "smi, visibility and plan UUID mismatch")
        require(torch.cuda.device_count() == 1, "exactly one visible CUDA device")
        self._torch = torch
        self.cuda = _Cuda(torch, expected)
        props = torch.cuda.get_device_properties(0)
        self.binding_evidence = object_evidence(props.uuid, torch._C._CUuuid, expected)
        self.binding_evidence.update(nvidia_smi_uuid=smi_uuid, cuda_visible_devices=visible,
                                     visible_device_count=1)

    def __getattr__(self, name):
        return getattr(self._torch, name)
