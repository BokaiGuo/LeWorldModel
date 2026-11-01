"""Trusted tensor-only HiJEPA graph construction. No pickle or foreign imports.

The owned high-latent kernels are source transplants, not original-bytecode
equivalence. Pixel/ViT and macro encoding remain blocked. MAIN-only CUDA
construction is separate from metadata inspection and never performs forward.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import stat

import numpy as np

TAG = "agent_hilewm_trusted_loader_20261028_v1"
ROOT = Path(__file__).resolve().parents[2]
IR_PATH = ROOT / "results/agent_hilewm_inert_pickle_ir_20261028_v1_MAIN01.json"
IR_SHA = "709b31e28bee8039d577668cbd9613347be1d070a7619e3db3ca61e3674cabbe"
EXPORT_SHA = "3cd7dd6783129893f397ddef63bd253e447fbef93582dd5f4dc8d7144310efd3"
TEXTS = ROOT / "results/p1_hilewm_text_contract_20261027_v1/texts"
TEXT_HASHES = {
    "package__code__h_le_wm__models__jepa.py.source.txt": "bb5338bbb4f9e76b0bd15b94767624168883c994ab4c2ad8209337a3d70022a2",
    "package__code__h_le_wm__models__latent_action.py.source.txt": "d06aa5955c5fdc6e61ec2f7a06317d024d8c4ad6ec6ceba7802e3d203e518e58",
}
LOCAL_MODULE_SHA = "0b258a9e8dc24c29fcb1e8c50a09ec78b8ea85aeb79e21dd8adf712396646620"
JSON_CAP, ARRAY_CAP = 32 * 1024**2, 128 * 1024**2
PRIMITIVES = {
    "torch.nn.modules.activation": {"GELU", "MultiheadAttention", "SiLU", "Softmax"},
    "torch.nn.modules.batchnorm": {"BatchNorm1d"},
    "torch.nn.modules.container": {"ModuleList", "Sequential"},
    "torch.nn.modules.conv": {"Conv1d", "Conv2d"},
    "torch.nn.modules.dropout": {"Dropout"},
    "torch.nn.modules.linear": {"Identity", "Linear", "NonDynamicallyQuantizableLinear"},
    "torch.nn.modules.normalization": {"LayerNorm"},
    "torch.nn.modules.transformer": {"TransformerEncoder", "TransformerEncoderLayer"},
    "transformers.activations": {"GELUActivation"},
    "transformers.models.vit.modeling_vit": {"ViTAttention", "ViTEmbeddings", "ViTEncoder",
        "ViTIntermediate", "ViTLayer", "ViTModel", "ViTOutput", "ViTPatchEmbeddings",
        "ViTSelfAttention", "ViTSelfOutput"},
    "module": {"ARPredictor", "Attention", "ConditionalBlock", "Embedder", "FeedForward", "MLP", "Transformer"},
    "_baseline_lewm_module": {"ARPredictor", "Attention", "ConditionalBlock", "FeedForward", "Transformer"},
    "hi_jepa": {"HiJEPA"}, "hi_module": {"LatentActionEncoder"},
}


def need(ok, message):
    if not ok:
        raise ValueError(message)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def bytes_read(path, cap):
    """NOFOLLOW + bounded stable fd read; all production paths are explicit."""
    path = Path(path)
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(fd, "rb") as handle:
        before = os.fstat(handle.fileno())
        need(stat.S_ISREG(before.st_mode) and before.st_size <= cap, "regular file/byte cap")
        raw = handle.read(cap+1)
        after = os.fstat(handle.fileno())
        linked = path.stat()
        identity = lambda s: (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns)
        need(len(raw) <= cap and identity(before) == identity(after) == identity(linked), "read identity/cap")
    return raw


def json_read(path, expected):
    raw = bytes_read(path, JSON_CAP)
    need(sha(raw) == expected, "JSON SHA")
    def pairs(items):
        out = {}
        for key, value in items:
            need(key not in out, "duplicate JSON key")
            out[key] = value
        return out
    def reject(value):
        raise ValueError("nonfinite JSON: " + value)
    return json.loads(raw, object_pairs_hook=pairs, parse_constant=reject)


class LiteralIR:
    def __init__(self, saved):
        self.nodes = saved["ir"]["nodes"]
        need(len(self.nodes) <= 100000 and all(n["id"] == i for i, n in enumerate(self.nodes)), "IR node identity/cap")

    def mapping(self, index):
        n = self.nodes[index]
        need(n["kind"] == "dict" or n.get("container") == "ordered_dict", "literal mapping")
        out = {}
        for k, v in n["entries"]:
            key = self.nodes[k]
            need(key["kind"] == "str" and key["value"] not in out, "unique literal key")
            out[key["value"]] = v
        return out

    def state(self, index):
        n = self.nodes[index]
        need(n["kind"] in {"newobj", "newobj_ex"} and n.get("executed") is False
             and len(n["states"]) == 1, "inert object state only")
        s = self.nodes[n["states"][0]]
        parts = s["items"] if s["kind"] == "tuple" else n["states"]
        out = {}
        for part in parts:
            if self.nodes[part]["kind"] == "none":
                continue
            values = self.mapping(part)
            need(not out.keys() & values.keys(), "duplicate BUILD state key")
            out.update(values)
        return out

    def literal(self, index, depth=0):
        need(depth < 32, "literal depth")
        n = self.nodes[index]
        kind = n["kind"]
        if kind in {"str", "bool"}:
            return n["value"]
        if kind == "none":
            return None
        if kind == "int":
            return int(n["value"])
        if kind == "float":
            value = float.fromhex(n["hex"])
            need(math.isfinite(value), "finite configuration scalar")
            return value
        if kind in {"tuple", "list", "set", "frozenset"}:
            return [self.literal(i, depth+1) for i in n["items"]]
        if kind == "dict" or n.get("container") == "ordered_dict":
            if any(self.nodes[k]["kind"] != "str" for k, _ in n["entries"]):
                return {"literal_mapping_entries": [[self.literal(k, depth+1), self.literal(v, depth+1)]
                                                    for k, v in n["entries"]]}
            return {k: self.literal(v, depth+1) for k, v in self.mapping(index).items()}
        if kind == "global":
            return {"inert_global_not_callable": [n["module"], n["name"]]}
        return {"opaque_node_not_executed": index, "kind": kind}

    def persistence(self, index):
        n = self.nodes[index]
        if n["kind"] == "reduce":
            symbol, args = self.nodes[n["callable"]], self.nodes[n["args"]]
            need(symbol["kind"] == "global" and symbol["module"] in {"__builtin__", "builtins"}
                 and symbol["name"] == "set" and args["kind"] == "tuple" and len(args["items"]) == 1
                 and n["executed"] is False and not n["states"], "literal persistence reducer")
            n = self.nodes[args["items"][0]]
        need(n["kind"] in {"list", "tuple", "set", "frozenset"}, "persistence set")
        values = [self.literal(i) for i in n["items"]]
        need(all(type(v) is str for v in values) and len(set(values)) == len(values), "persistence names")
        return values


def contiguous(shape):
    stride, result = 1, []
    for size in reversed(shape):
        result.insert(0, stride)
        stride *= size
    return result


def graph_spec(saved):
    need(saved["ir"]["complete"] is True and saved["nn"]["complete"] is True
         and saved["ir"]["errors"] == saved["nn"]["errors"] == [], "complete IR/nn required")
    nn = saved["nn"]
    need(all(nn[k] == {} for k in ("module_alias_paths", "node_alias_paths", "storage_alias_paths")),
         "alias layout unsupported by this exact nonalias checkpoint loader")
    ir = LiteralIR(saved)
    modules, entries, state_keys, groups = [], [], [], {}
    paths = {}
    for row in nn["modules"]:
        path = tuple(row["path"])
        need(path not in paths and all(re.fullmatch(r"[A-Za-z0-9_]+", p) and not p.startswith("_") for p in path),
             "safe unique module path")
        declaration = row["class_declaration"]
        need(declaration[1] in PRIMITIVES.get(declaration[0], set()), "unrecognized declared module")
        state = ir.state(row["node"])
        symbol = ir.nodes[ir.nodes[row["node"]]["callable"]]
        need(symbol["kind"] == "global" and [symbol["module"], symbol["name"]] == declaration, "class symbol join")
        for key, value in state.items():
            if "hook" in key and key != "_is_full_backward_hook":
                need(ir.mapping(value) == {}, "nonempty native hooks refused")
        module = {**row, "nonpersistent": ir.persistence(state["_non_persistent_buffers_set"]),
                  "attribute_nodes": {k: v for k, v in state.items() if k not in {"_modules", "_parameters", "_buffers"}},
                  "literal_attributes": {k: ir.literal(v) for k, v in state.items()
                                         if not k.startswith("_") or k == "_freeze_flags"},
                  "children": ir.mapping(state["_modules"])}
        paths[path] = module
        modules.append(module)
    need(() in paths and paths[()]["node"] == saved["ir"]["root_node"], "root identity")
    for path, module in paths.items():
        for key, node in module["children"].items():
            need(path+(key,) in paths and paths[path+(key,)]["node"] == node, "exact child graph")
        if path:
            need(path[:-1] in paths and paths[path[:-1]]["children"].get(path[-1]) == module["node"], "parent graph")
    for index, row in enumerate(nn["entries"]):
        path = tuple(row["module_path"])
        name = row["name"]
        need(path in paths and re.fullmatch(r"[A-Za-z0-9_]+", name) and not name.startswith("_"), "safe entry name")
        need(row["collection"] in {"_parameters", "_buffers"}, "parameter/buffer collection")
        key = ".".join((*path, name))
        nonpersistent = row["collection"] == "_buffers" and name in paths[path]["nonpersistent"]
        meta = row["metadata"]
        state = ir.state(paths[path]["node"])
        need(ir.mapping(state[row["collection"]]).get(name) == row["node"], "entry IR collection join")
        if meta is None:
            need(row["status"] == "explicit_none" and ir.nodes[row["node"]]["kind"] == "none", "explicit None only")
        else:
            need(row["status"] == "literal_tensor_metadata" and meta["dtype_declaration"] in {"float32", "int64"}, "dtype declaration")
            shape = meta["shape"]
            need(all(type(d) is int and d > 0 for d in shape) and len(shape) <= 64
                 and meta["stride"] == contiguous(shape) and meta["storage_offset"] == 0
                 and meta["storage_numel"] == math.prod(shape), "only exact whole contiguous nonalias entries")
            groups.setdefault(meta["storage_key"], []).append(index)
        entries.append({"index": index, "key": key, "nonpersistent_buffer": nonpersistent,
                        "include_in_literal_state_dict": meta is not None and not nonpersistent})
        if entries[-1]["include_in_literal_state_dict"]:
            state_keys.append(key)
    need(len(state_keys) == len(set(state_keys)), "unique state keys")
    need(set(groups) == set(nn["storage_definitions"]) and all(len(g) == 1 for g in groups.values()), "one entry per storage")
    return {"modules": modules, "entries": entries, "state_keys": state_keys,
            "module_count": len(modules), "entry_count": len(entries), "storage_count": len(groups)}


def audit():
    saved = json_read(IR_PATH, IR_SHA)
    for path, expected in saved["own_sha256"].items():
        need(Path(path).parent in {ROOT/"experiments/spectral_reachability", ROOT/"tests", ROOT/"innerdocs"}
             and "agent_hilewm_inert_pickle_ir_20261028_v1" in Path(path).name, "IR source path")
        need(sha(bytes_read(path, JSON_CAP)) == expected, "IR source closure")
    for name, expected in TEXT_HASHES.items():
        need(sha(bytes_read(TEXTS/name, JSON_CAP)) == expected, "official text SHA")
    need(sha(bytes_read(ROOT/"module.py", JSON_CAP)) == LOCAL_MODULE_SHA, "local template SHA")
    spec = graph_spec(saved)
    need((spec["module_count"], spec["entry_count"], spec["storage_count"], len(spec["state_keys"])) == (531, 491, 425, 425),
         "fixed complete checkpoint denominator")
    ir = LiteralIR(saved)
    bypath = {tuple(r["path"]): r for r in spec["modules"]}
    encoder_state = ir.state(bypath[("encoder",)]["node"])
    config = {k: ir.literal(v) for k, v in ir.state(encoder_state["config"]).items()}
    shapes = {".".join(r["module_path"]+[r["name"]]): r["metadata"]["shape"]
              for r in saved["nn"]["entries"] if r["metadata"] is not None}
    facts = {"encoder_config": config, "low_pos": shapes["low_predictor.pos_embedding"],
             "high_pos": shapes["high_predictor.pos_embedding"],
             "projector_norms": {k: bypath[(k,"net","1")]["class_declaration"]
                                 for k in ("projector", "low_pred_proj", "high_pred_proj")},
             "macro_input": shapes["latent_action_encoder.input_proj.weight"],
             "macro_output": shapes["latent_action_encoder.output_proj.weight"],
             "macro_position": shapes["latent_action_encoder.pos_embedding"],
             "runtime_blockers": [
                 "IR encoder config transformers_version is None; original ViT forward bytecode/version not authenticated",
                 "installed ViT uses q_proj/k_proj/v_proj/o_proj, IR uses nested attention.attention.query/key/value",
                 "pixel ViT and macro encoder forward kernels are deliberately not implemented in this loader",
                 "owned high latent source transplants have no CUDA forward parity receipt yet",
                 "frozen raw-action normalizer/history/bounds/public RGB-latent contract is separate and not certified here"]}
    return saved, spec, facts


def check_export(saved, spec, report, receipt, report_sha):
    need(receipt == {"status": "saved_data_verified_not_model_equivalence", "report_sha256": report_sha,
                    "arrays_verified": 850, "ir_sha256": IR_SHA, "trusted_pickle_loading_allowed": False,
                    "scientific_admission": False}, "actual MAIN exporter verify receipt required")
    need(report["schema"] == "agent_hilewm_storage_npy_export_20261028_v1"
         and report["status"] == "literal_nn_data_exported_not_model_equivalence"
         and report["ir_sha256"] == IR_SHA and report["errors"] == []
         and report["byteorder"] == "little" and report["all_storage_crc_verified"] is True,
         "complete data-only little-endian export required")
    need(report["zip_sha256"] == saved["input_sha256"] and
         report["static_report_sha256"] == saved["static_report_sha256"] and
         report["byteorder_crc_verified"] is True and report["pickle_executed"] is False
         and report["foreign_code_executed"] is False, "original export provenance")
    expected_paths = {str(ROOT/"experiments/spectral_reachability/agent_hilewm_storage_npy_export_20261028_v1.py"),
                      str(ROOT/"tests/test_agent_hilewm_storage_npy_export_20261028_v1.py"),
                      str(ROOT/"innerdocs/agent_hilewm_storage_npy_export_20261028_v1.md")}
    need(set(report["own_sha256"]) == expected_paths and all(
        sha(bytes_read(p, JSON_CAP)) == h for p, h in report["own_sha256"].items()), "exporter source closure")
    need(report["modules"] == saved["nn"]["modules"] and len(report["rows"]) == len(spec["entries"])
         and report["state_dict_paths"] == spec["entries"], "export graph/key/persistence join")
    for key in ("module_alias_paths", "node_alias_paths", "storage_alias_paths"):
        need(report[key] == saved["nn"][key] == {}, "export alias join")
    groups = {}
    for index, row in enumerate(report["rows"]):
        need(row["index"] == index and row["source_entry"] == saved["nn"]["entries"][index], "export entry identity")
        meta = row["source_entry"]["metadata"]
        need((row["array"] is None) == (meta is None), "None cannot become zero tensor")
        if meta is not None:
            need(row["array"]["shape"] == meta["shape"] and row["array"]["descr"] ==
                 {"float32": "<f4", "int64": "<i8"}[meta["dtype_declaration"]], "entry dtype/shape")
            groups[meta["storage_key"]] = index
    need(len(report["storages"]) == len(groups) == 425, "complete storage denominator")
    need({s["storage_key"] for s in report["storages"]} == set(groups), "unique complete storage keys")
    for storage in report["storages"]:
        key = storage["storage_key"]
        need(storage["definition"] == saved["nn"]["storage_definitions"][key]
             and storage["aliases"] == [groups[key]] and storage["crc_verified"] is True,
             "storage definition/alias/CRC join")
    need(report["journal"] == [{"storage_key": s["storage_key"], "operation": "storage_crc_and_arrays", "phase": phase}
                               for s in report["storages"] for phase in ("attempt", "return")], "complete storage attempt-return ledger")


def read_array(directory, descriptor):
    name = descriptor["file"]
    need(re.fullmatch(r"entry_[0-9]{4,5}\.npy", name), "exact entry NPY name")
    raw = bytes_read(Path(directory)/name, ARRAY_CAP)
    need(len(raw) == descriptor["file_bytes"] and sha(raw) == descriptor["file_sha256"], "NPY file SHA/size")
    stream = io.BytesIO(raw)
    need(np.lib.format.read_magic(stream) == (1, 0), "exact exporter NPY v1 header")
    shape, fortran, dtype = np.lib.format.read_array_header_1_0(stream)
    need(len(shape) <= 64 and all(type(d) is int and d >= 0 for d in shape)
         and list(shape) == descriptor["shape"] and dtype.str == descriptor["descr"]
         and dtype.str in {"<f4", "<i8"} and fortran is False
         and math.prod(shape)*dtype.itemsize == len(raw)-stream.tell() <= ARRAY_CAP,
         "bounded exact NPY header before allocation")
    stream.seek(0)
    array = np.load(stream, allow_pickle=False)
    need(stream.tell() == len(raw) and not array.dtype.hasobject and array.flags.c_contiguous
         and array.dtype.str == descriptor["descr"] and list(array.shape) == descriptor["shape"]
         and descriptor["fortran_order"] is False and descriptor["value_bytes"] == array.nbytes
         and descriptor["value_hash_order"] == "C-index exact source-endian bytes"
         and sha(array.tobytes(order="C")) == descriptor["value_sha256"], "NPY dtype/shape/bytes/order")
    return array


def authorities(report_path, report_sha, receipt_path, receipt_sha):
    need(report_sha == EXPORT_SHA and re.fullmatch(r"[0-9a-f]{64}", receipt_sha or ""), "actual export/receipt SHA required")
    saved, spec, facts = audit()
    report = json_read(report_path, report_sha)
    receipt = json_read(receipt_path, receipt_sha)
    check_export(saved, spec, report, receipt, report_sha)
    return saved, spec, facts, report


def construct_cuda(report_path, report_sha, receipt_path, receipt_sha, *, main_authorized=False):
    """Construct only. No forward, random initialization, training or native class dispatch."""
    need(main_authorized is True, "MAIN authorization required before torch import/data reads")
    saved, spec, facts, report = authorities(report_path, report_sha, receipt_path, receipt_sha)
    import torch
    need(torch.cuda.is_available(), "HOST CUDA required; no CPU model fallback")
    device = torch.device("cuda", torch.cuda.current_device())

    class TrustedTensorNode(torch.nn.Module):
        def __init__(self, row):
            super().__init__()
            self.declared_class_not_imported = tuple(row["class_declaration"])
            self.ir_attribute_nodes = dict(row["attribute_nodes"])
            self.literal_source_attributes = dict(row["literal_attributes"])

        def train(self, mode=True):
            need(mode is False, "frozen construction-only graph cannot train")
            return super().train(False)

        def forward(self, *args, **kwargs):
            need(not kwargs and len(args) in {1, 2}, "explicit latent subtree positional API")
            x = args[0]
            need(torch.is_tensor(x) and x.device == device and not self.training, "frozen HOST CUDA inputs only")
            f = torch.nn.functional
            kind = self.declared_class_not_imported[1]
            a = self.literal_source_attributes
            with torch.inference_mode():
                if kind in {"Linear", "NonDynamicallyQuantizableLinear"}:
                    return f.linear(x, self.weight, self.bias)
                if kind == "LayerNorm":
                    return f.layer_norm(x, a["normalized_shape"], self.weight, self.bias, a["eps"])
                if kind == "BatchNorm1d":
                    need(a["track_running_stats"] is True, "stored running stats required")
                    return f.batch_norm(x, self.running_mean, self.running_var, self.weight, self.bias,
                                        training=False, momentum=a["momentum"], eps=a["eps"])
                if kind == "GELU":
                    return f.gelu(x, approximate=a["approximate"])
                if kind == "SiLU":
                    return f.silu(x)
                if kind in {"Identity", "Dropout"}:
                    return x
                if kind == "Sequential":
                    for child in self._modules.values():
                        x = child(x)
                    return x
                if kind in {"MLP", "FeedForward"}:
                    return self.net(x)
                if kind == "Attention":
                    norm = self.norm(x)
                    q, k, v = self.to_qkv(norm).chunk(3, dim=-1)
                    heads = a["heads"]
                    def split(t):
                        return t.reshape(t.shape[0], t.shape[1], heads, -1).transpose(1, 2)
                    need(a["scale"] == (q.shape[-1]//heads)**-0.5, "explicit source attention scale")
                    y = f.scaled_dot_product_attention(split(q), split(k), split(v), dropout_p=0.0, is_causal=True)
                    return self.to_out(y.transpose(1, 2).reshape(x.shape[0], x.shape[1], -1))
                if kind == "ConditionalBlock":
                    need(len(args) == 2 and args[1].device == device, "CUDA conditioning required")
                    sm, cm, gm, sf, cf, gf = self.adaLN_modulation(args[1]).chunk(6, dim=-1)
                    x = x + gm*self.attn(self.norm1(x)*(1+cm)+sm)
                    return x + gf*self.mlp(self.norm2(x)*(1+cf)+sf)
                if kind == "Transformer":
                    need(len(args) == 2, "conditioned transformer required")
                    x, c = self.input_proj(x), self.cond_proj(args[1])
                    for block in self.layers._modules.values():
                        x = block(x, c)
                    return self.output_proj(self.norm(x))
                if kind == "ARPredictor":
                    need(len(args) == 2 and x.shape[1] <= self.pos_embedding.shape[1], "predictor context bound")
                    return self.transformer(self.dropout(x+self.pos_embedding[:, :x.shape[1]]), args[1])
            raise ValueError("uncertified forward kind: " + kind + "; pixel ViT/macro transformer execution remains blocked")

    class TrustedHiJEPADataGraph(TrustedTensorNode):
        """Owned root and explicit high-latent subset; not an imported class."""

        def predict_high(self, emb, macro_actions):
            need(emb.device == macro_actions.device == device and not self.training, "frozen CUDA high inputs")
            with torch.inference_mode():
                preds = self.high_predictor(emb, self.macro_to_condition(macro_actions))
                return self.high_pred_proj(preds.reshape(-1, preds.shape[-1])).reshape(preds.shape)

        def rollout_high(self, z_init, latent_actions):
            need(z_init.device == latent_actions.device == device and not self.training, "frozen CUDA rollout inputs")
            with torch.inference_mode():
                if latent_actions.ndim == 3:
                    latent_actions = latent_actions.unsqueeze(1)
                need(latent_actions.ndim == 4, "macro B,S,T,L")
                b, s, t, width = latent_actions.shape
                if z_init.ndim == 2:
                    z_init = z_init[:, None].expand(-1, s, -1)
                need(z_init.shape[:2] == (b, s), "z_init B,S")
                history = z_init.reshape(b*s, 1, -1)
                act_history, predictions = [], []
                for step in range(t):
                    act_history.append(latent_actions[:, :, step].reshape(b*s, width))
                    sequence = torch.stack(act_history, dim=1)  # saved official quantize is identity
                    context = min(self.high_predictor.pos_embedding.shape[1], history.shape[1], sequence.shape[1])
                    nxt = self.predict_high(history[:, -context:], sequence[:, -context:])[:, -1:]
                    history = torch.cat((history, nxt), dim=1)
                    predictions.append(nxt[:, 0])
                return torch.stack(predictions, dim=1).reshape(b, s, t, -1)

        def get_cost(self, info, candidates):
            need(set(info) == {"planner_level", "z_init", "z_goal"} and info["planner_level"] == "high",
                 "high-latent-only contract; low/flat/pixel path not implemented")
            need(candidates.device == device and candidates.ndim == 4, "CUDA high candidates B,S,H,D")
            with torch.inference_mode():
                b, s, h, width = candidates.shape
                latent_dim = self.latent_action_encoder.output_proj.weight.shape[0]
                need(width % latent_dim == 0, "grouped macro width")
                actions = candidates.reshape(b, s, h*(width//latent_dim), latent_dim)
                z0, zg = info["z_init"], info["z_goal"]
                if z0.ndim == 2:
                    z0 = z0[:, None].expand(-1, s, -1)
                if zg.ndim == 2:
                    zg = zg[:, None].expand(-1, s, -1)
                prediction = self.rollout_high(z0, actions)[:, :, -1]
                return (prediction-zg).square().sum(dim=-1)

    nodes = {}
    for row in spec["modules"]:
        path = tuple(row["path"])
        node = (TrustedHiJEPADataGraph if not path else TrustedTensorNode)(row)
        nodes[path] = node
        if path:
            need(not hasattr(nodes[path[:-1]], path[-1]), "module name collides with owned class API")
            nodes[path[:-1]].add_module(path[-1], node)
    hashes, attempted, returned = {}, 0, 0
    for record, entry in zip(report["rows"], spec["entries"]):
        original = record["source_entry"]
        owner = nodes[tuple(original["module_path"])]
        value = None
        if record["array"] is not None:
            attempted += 1
            array = read_array(Path(report_path).parent, record["array"])
            value = torch.from_numpy(array.copy()).to(device=device)
            need(value.dtype == ({"<f4": torch.float32, "<i8": torch.int64}[array.dtype.str])
                 and tuple(value.shape) == tuple(array.shape), "no dtype/shape coercion")
            copied = value.detach().cpu().numpy()
            need(sha(copied.tobytes(order="C")) == record["array"]["value_sha256"], "CUDA copy must preserve original bytes")
            hashes[entry["key"]] = record["array"]["value_sha256"]
            returned += 1
        need(not hasattr(owner, original["name"]), "entry name collides with owned class API")
        if original["collection"] == "_parameters":
            owner.register_parameter(original["name"], None if value is None else torch.nn.Parameter(value, requires_grad=False))
        else:
            owner.register_buffer(original["name"], value, persistent=not entry["nonpersistent_buffer"])
    root = nodes[()]
    root.eval()
    state = root.state_dict()
    need(set(state) == set(spec["state_keys"]) and len(list(root.named_modules())) == spec["module_count"], "runtime exact key/module denominator")
    for key, value in state.items():
        need(value.device == device and not value.requires_grad
             and sha(value.detach().cpu().numpy().tobytes(order="C")) == hashes[key], "post-registration tensor bytes/freeze")
    need(all(not p.requires_grad for p in root.parameters()), "all parameters frozen")
    torch.cuda.synchronize()
    authorities(report_path, report_sha, receipt_path, receipt_sha)  # no changing authority during construction
    receipt = {"schema": TAG+".construction", "status": "owned_tensor_graph_constructed_not_native_forward_verified",
               "own_sha256": {str(p): sha(bytes_read(p, JSON_CAP)) for p in (
                   Path(__file__), ROOT/"tests"/("test_"+TAG+".py"), ROOT/"innerdocs"/(TAG+".md"))},
               "ir_sha256": IR_SHA, "export_report_sha256": report_sha, "export_receipt_sha256": receipt_sha,
               "module_count": len(nodes), "entry_count": len(spec["entries"]), "state_key_count": len(state),
               "tensor_copy_attempts": attempted, "tensor_copy_returns": returned, "tensor_value_hashes": hashes,
               "torch": str(torch.__version__), "device": str(device), "forward_calls": 0, "physics": 0,
               "all_parameters_frozen": True, "native_module_types_recreated": False,
               "high_latent_kernels": "owned_source_transplant_not_forward_verified",
               "pixel_and_macro_encoding_allowed": False,
               "runtime_native_state_dict_equivalence": False, "low_high_pairing_verified": False,
               "baseline_runnable": False, "scientific_admission": False}
    return root, receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("audit", "check-export", "construct"))
    parser.add_argument("--export-report", type=Path)
    parser.add_argument("--report-sha256")
    parser.add_argument("--export-receipt", type=Path)
    parser.add_argument("--receipt-sha256")
    parser.add_argument("--main-authorized", action="store_true")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    if args.command == "audit":
        _, spec, facts = audit()
        result = {"schema": TAG+".audit", "ir_sha256": IR_SHA, "facts": facts,
                  "module_count": spec["module_count"], "entry_count": spec["entry_count"],
                  "state_key_count": len(spec["state_keys"]), "forward_allowed": False}
    else:
        need(all((args.export_report, args.report_sha256, args.export_receipt, args.receipt_sha256)), "actual report and strict receipt paths/SHAs required")
        values = (args.export_report, args.report_sha256, args.export_receipt, args.receipt_sha256)
        if args.command == "check-export":
            authorities(*values)
            result = {"metadata_and_actual_receipt_bound": True, "forward_allowed": False}
        else:
            need(args.out is not None and not args.out.exists(), "new exclusive --out receipt required")
            _, result = construct_cuda(*values, main_authorized=args.main_authorized)
    raw = (json.dumps(result, sort_keys=True, indent=2, allow_nan=False)+"\n").encode()
    if args.out is not None:
        with args.out.open("xb") as handle:
            handle.write(raw)
    print(raw.decode(), end="")


if __name__ == "__main__":
    main()
