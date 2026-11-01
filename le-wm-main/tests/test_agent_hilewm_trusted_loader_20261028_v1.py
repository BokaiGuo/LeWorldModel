"""RAM metadata/NPY/owned memory-copy tests. No torch, foreign model or forward."""
from copy import deepcopy
import builtins
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT/"experiments/spectral_reachability/agent_hilewm_trusted_loader_20261028_v1.py"
spec = importlib.util.spec_from_file_location("owned_hilewm_trusted_loader", SOURCE)
m = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = m
spec.loader.exec_module(m)


@pytest.fixture(scope="module")
def metadata():
    return m.audit()


def test_actual_metadata_and_config(metadata):
    saved, graph, facts = metadata
    assert (graph["module_count"], graph["entry_count"], graph["storage_count"]) == (531, 491, 425)
    assert len(graph["state_keys"]) == 425
    assert sum(r["metadata"] is None for r in saved["nn"]["entries"]) == 66
    assert facts["low_pos"] == [1, 3, 192] and facts["high_pos"] == [1, 4, 192]
    assert all(v[1] == "BatchNorm1d" for v in facts["projector_norms"].values())
    assert facts["macro_input"] == [192, 10] and facts["macro_output"] == [32, 192]
    c = facts["encoder_config"]
    assert (c["image_size"], c["patch_size"], c["hidden_size"], c["num_attention_heads"]) == (224, 14, 192, 3)
    assert c["_attn_implementation_internal"] == "sdpa" and c["transformers_version"] is None
    assert c["id2label"] == {"literal_mapping_entries": [[0, "LABEL_0"], [1, "LABEL_1"]]}
    assert facts["runtime_blockers"]


@pytest.mark.parametrize("mutation", ["alias", "class", "symbol", "path", "entryname", "dtype", "offset",
    "stride", "shape", "numel", "None", "status", "incomplete", "child", "persistence", "hook"])
def test_metadata_mutations(metadata, mutation):
    saved = deepcopy(metadata[0])
    nn = saved["nn"]
    row = nn["entries"][0]
    if mutation == "alias":
        nn["storage_alias_paths"] = {"0": ["fake"]}
    elif mutation == "class":
        nn["modules"][0]["class_declaration"] = ["arbitrary_import", "HiJEPA"]
    elif mutation == "symbol":
        nn["modules"][0]["class_declaration"] = ["module", "MLP"]
    elif mutation == "path":
        nn["modules"][1]["path"] = ["__class__"]
    elif mutation == "entryname":
        row["name"] = "unsafe.name"
    elif mutation == "dtype":
        row["metadata"]["dtype_declaration"] = "float16"
    elif mutation == "offset":
        row["metadata"]["storage_offset"] = 1
    elif mutation == "stride":
        row["metadata"]["stride"][-1] = 0
    elif mutation == "shape":
        row["metadata"]["shape"][-1] = 0
    elif mutation == "numel":
        row["metadata"]["storage_numel"] += 1
    elif mutation == "None":
        row["metadata"] = None
    elif mutation == "status":
        row["status"] = "guessed"
    elif mutation == "incomplete":
        nn["complete"] = False
    elif mutation == "child":
        nn["modules"].pop()
    else:
        ir = m.LiteralIR(saved)
        fields = ir.state(nn["modules"][0]["node"])
        field = "_non_persistent_buffers_set" if mutation == "persistence" else "_forward_hooks"
        node = ir.nodes[fields[field]]
        if mutation == "persistence":
            node["kind"] = "unknown"
        else:
            # Use an actual nonempty literal map as a hook map: never executed.
            node.update(kind="dict", entries=ir.nodes[ir.nodes[nn["modules"][0]["node"]]["states"][0]]["entries"][:1])
    with pytest.raises(ValueError):
        m.graph_spec(saved)


def export_metadata():
    path = ROOT/"results/agent_hilewm_storage_npy_export_20261028_v1_MAIN01/report.json"
    return m.json_read(path, m.EXPORT_SHA)


def RAM_receipt_shape():
    """Synthetic unit input, NOT an actual persisted MAIN verify receipt."""
    return {"status": "saved_data_verified_not_model_equivalence", "report_sha256": m.EXPORT_SHA,
            "arrays_verified": 850, "ir_sha256": m.IR_SHA, "trusted_pickle_loading_allowed": False,
            "scientific_admission": False}


def test_actual_export_metadata_join_with_RAM_receipt_shape(metadata):
    m.check_export(metadata[0], metadata[1], export_metadata(), RAM_receipt_shape(), m.EXPORT_SHA)


@pytest.mark.parametrize("mutation", ["receipt", "None", "shape", "dtype", "alias", "keys", "crc", "storage", "journal"])
def test_export_metadata_mutations(metadata, mutation):
    report, receipt = export_metadata(), RAM_receipt_shape()
    if mutation == "receipt":
        receipt["arrays_verified"] -= 1
    elif mutation == "None":
        next(r for r in report["rows"] if r["array"] is None)["array"] = {}
    elif mutation == "shape":
        report["rows"][0]["array"]["shape"] = [192]
    elif mutation == "dtype":
        report["rows"][0]["array"]["descr"] = "<f8"
    elif mutation == "alias":
        report["storage_alias_paths"] = {"0": ["other"]}
    elif mutation == "keys":
        report["state_dict_paths"][0]["key"] = "other"
    elif mutation == "crc":
        report["storages"][0]["crc_verified"] = False
    elif mutation == "storage":
        report["storages"][0] = deepcopy(report["storages"][1])
    else:
        report["journal"].pop()
    with pytest.raises(ValueError):
        m.check_export(metadata[0], metadata[1], report, receipt, m.EXPORT_SHA)


def array_file(tmp_path, array):
    path = tmp_path/"entry_0000.npy"
    stream = io.BytesIO()
    np.save(stream, array, allow_pickle=False)
    raw = stream.getvalue()
    path.write_bytes(raw)
    return path, {"file": path.name, "file_bytes": len(raw), "file_sha256": m.sha(raw),
                  "value_bytes": array.nbytes, "value_sha256": m.sha(array.tobytes()),
                  "descr": array.dtype.str, "shape": list(array.shape), "fortran_order": False,
                  "value_hash_order": "C-index exact source-endian bytes"}


@pytest.mark.parametrize("kind", ["float", "int", "scalar", "signedzero_nanbits"])
def test_NPY_exact_payload_no_pickle(tmp_path, kind):
    arrays = {"float": np.arange(6, dtype="<f4").reshape(2, 3), "int": np.arange(3, dtype="<i8"),
              "scalar": np.array(5, dtype="<i8"),
              "signedzero_nanbits": np.array([0x80000000, 0x7fc01234], dtype="<u4").view("<f4")}
    array = arrays[kind]
    _, descriptor = array_file(tmp_path, array)
    loaded = m.read_array(tmp_path, descriptor)
    assert loaded.tobytes() == array.tobytes()


@pytest.mark.parametrize("mutation", ["file", "bytes", "valuehash", "dtype", "shape", "trailing", "symlink"])
def test_array_mutations(tmp_path, mutation):
    path, descriptor = array_file(tmp_path, np.arange(6, dtype="<f4").reshape(2, 3))
    if mutation == "file":
        descriptor["file"] = "../entry_0000.npy"
    elif mutation == "bytes":
        path.write_bytes(path.read_bytes()[:-1]+b"X")
    elif mutation == "valuehash":
        descriptor["value_sha256"] = "0"*64
    elif mutation == "dtype":
        descriptor["descr"] = "<f8"
    elif mutation == "shape":
        descriptor["shape"] = [6]
    elif mutation == "trailing":
        raw = path.read_bytes()+b"tail"
        path.write_bytes(raw)
        descriptor.update(file_bytes=len(raw), file_sha256=m.sha(raw))
    else:
        link = tmp_path/"entry_0001.npy"
        link.symlink_to(path)
        descriptor["file"] = link.name
    with pytest.raises((ValueError, OSError)):
        m.read_array(tmp_path, descriptor)


def test_authorization_before_any_IO_or_import(monkeypatch):
    monkeypatch.setattr(m, "authorities", lambda *args: pytest.fail("must not read input"))
    with pytest.raises(ValueError, match="authorization"):
        m.construct_cuda(None, None, None, None)


def test_missing_actual_receipt_is_failclosed():
    with pytest.raises(ValueError, match="actual export/receipt SHA"):
        m.authorities(None, m.EXPORT_SHA, None, None)


def test_JSON_duplicate_and_hash_cap(tmp_path):
    path = tmp_path/"duplicate.json"
    raw = b'{"x":1,"x":2}'
    path.write_bytes(raw)
    with pytest.raises(ValueError, match="duplicate"):
        m.json_read(path, m.sha(raw))
    with pytest.raises(ValueError, match="SHA"):
        m.json_read(path, "0"*64)
    with pytest.raises(ValueError, match="cap"):
        m.bytes_read(path, 1)


def test_fifo_refused_without_blocking(tmp_path):
    path = tmp_path/"not_a_regular_file"
    os.mkfifo(path)
    with pytest.raises(ValueError, match="regular file"):
        m.bytes_read(path, 100)


def test_large_header_rejected_before_allocation(tmp_path, monkeypatch):
    out = io.BytesIO()
    np.lib.format.write_array_header_1_0(out, {"descr": "<f4", "fortran_order": False, "shape": (10**12,)})
    raw = out.getvalue()
    (tmp_path/"entry_0000.npy").write_bytes(raw)
    monkeypatch.setattr(np, "load", lambda *args, **kwargs: pytest.fail("allocation must not occur"))
    descriptor = {"file": "entry_0000.npy", "file_bytes": len(raw), "file_sha256": m.sha(raw),
                  "descr": "<f4", "shape": [10**12]}
    with pytest.raises(ValueError, match="before allocation"):
        m.read_array(tmp_path, descriptor)


def test_cold_audit_blocks_all_foreign_imports(tmp_path):
    script = """
import builtins, runpy, sys
original = builtins.__import__
def deny(name, *args, **kwargs):
    if name.split('.')[0] in {'torch','transformers','hi_jepa','hi_module','module','_baseline_lewm_module'}:
        raise AssertionError('foreign or neural import: '+name)
    return original(name,*args,**kwargs)
builtins.__import__=deny
sys.argv=[sys.argv[1],'audit']
runpy.run_path(sys.argv[0],run_name='__main__')
"""
    result = subprocess.run([sys.executable, "-B", "-c", script, str(SOURCE)], cwd=ROOT,
                            text=True, capture_output=True, check=True)
    assert json.loads(result.stdout)["state_key_count"] == 425


def test_doc_delivered():
    assert (ROOT/"innerdocs/agent_hilewm_trusted_loader_20261028_v1.md").is_file()


def test_actual_construct_control_path_with_owned_memory_stub(tmp_path, monkeypatch):
    """Only copy/register/state_dict, no tensor arithmetic, no neural/torch import.

    Synthetic authority is injected locally and is never an on-disk MAIN receipt.
    This checks our control path, not actual CUDA or native module equivalence.
    """
    class Tensor:
        def __init__(self, a):
            self.a = np.array(a, copy=True)
            self.device = "RAM_copy_stub"
            self.requires_grad = False
            self.dtype = self.a.dtype
            self.shape = self.a.shape

        def to(self, *, device):
            self.device = device
            return self

        def detach(self):
            return self

        def cpu(self):
            return self

        def numpy(self):
            return self.a

    class Parameter(Tensor):
        def __init__(self, value, requires_grad):
            super().__init__(value.a)
            self.device = value.device
            self.requires_grad = requires_grad

    class Module:
        def __init__(self):
            self._modules, self._parameters, self._buffers, self._nonpersistent = {}, {}, {}, set()
            self.training = True

        def add_module(self, name, module):
            self._modules[name] = module

        def register_parameter(self, name, value):
            self._parameters[name] = value

        def register_buffer(self, name, value, persistent):
            self._buffers[name] = value
            if not persistent:
                self._nonpersistent.add(name)

        def train(self, mode):
            self.training = mode
            for child in self._modules.values():
                child.train(mode)
            return self

        def eval(self):
            return self.train(False)

        def named_modules(self):
            yield "", self
            for name, child in self._modules.items():
                for subname, value in child.named_modules():
                    yield name+("."+subname if subname else ""), value

        def state_dict(self):
            out = {k: v for k, v in self._parameters.items() if v is not None}
            out.update({k: v for k, v in self._buffers.items() if v is not None and k not in self._nonpersistent})
            for name, child in self._modules.items():
                out.update({name+"."+k: v for k, v in child.state_dict().items()})
            return out

        def parameters(self):
            yield from (v for v in self._parameters.values() if v is not None)
            for child in self._modules.values():
                yield from child.parameters()

    cuda = SimpleNamespace(is_available=lambda: True, current_device=lambda: 0, synchronize=lambda: None)
    torch = SimpleNamespace(cuda=cuda, device=lambda *args: "RAM_copy_stub", from_numpy=Tensor,
                            float32=np.dtype("float32"), int64=np.dtype("int64"),
                            nn=SimpleNamespace(Module=Module, Parameter=Parameter), __version__="RAM_STUB_not_CUDA")
    monkeypatch.setitem(sys.modules, "torch", torch)
    def module(path, kind):
        return dict(path=path, class_declaration=["hi_jepa",kind], attribute_nodes={}, literal_attributes={})
    graph = {"modules": [module([], "HiJEPA"), module(["projector"], "Linear")],
             "entries": [], "state_keys": ["counter", "projector.weight"], "module_count": 2}
    report = {"rows": []}
    values = [np.array(7, dtype="<i8"), np.arange(4, dtype="<f4").reshape(2, 2)]
    descriptors = []
    for i, (path, collection, name, value) in enumerate([
            ([], "_buffers", "counter", values[0]), (["projector"], "_parameters", "weight", values[1]),
            (["projector"], "_parameters", "bias", None)]):
        key = ".".join(path+[name])
        descriptor = None
        if value is not None:
            directory = tmp_path/str(i)
            directory.mkdir()
            _, descriptor = array_file(directory, value)
            descriptors.append((descriptor, value))
        report["rows"].append({"source_entry": {"module_path": path, "collection": collection, "name": name},
                               "array": descriptor})
        graph["entries"].append({"key": key, "nonpersistent_buffer": False})
    called = []
    def authority(*args):
        called.append("authority")
        return {}, graph, {}, report
    def read_array(directory, descriptor):
        return next(v.copy() for d, v in descriptors if d is descriptor)
    monkeypatch.setattr(m, "authorities", authority)
    monkeypatch.setattr(m, "read_array", read_array)
    root, receipt = m.construct_cuda(tmp_path/"report.json", "RAM_fixture_only", None, "RAM_fixture_only", main_authorized=True)
    assert called == ["authority", "authority"]
    assert receipt["tensor_copy_attempts"] == receipt["tensor_copy_returns"] == 2
    assert receipt["state_key_count"] == 2 and receipt["forward_calls"] == 0
    assert not receipt["baseline_runnable"] and not receipt["runtime_native_state_dict_equivalence"]
    assert root._modules["projector"]._parameters["bias"] is None
    assert root._buffers["counter"].numpy().dtype == np.int64
    assert all(not p.requires_grad for p in root.parameters())
    with pytest.raises(ValueError, match="cannot train"):
        root.train(True)
