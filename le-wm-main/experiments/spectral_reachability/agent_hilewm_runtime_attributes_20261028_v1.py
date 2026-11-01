"""Saved inert-state runtime attributes and static source mapping, not a loader."""
from __future__ import annotations

import argparse
import ast
import os
from pathlib import Path
import re
import shutil

from experiments.spectral_reachability import agent_hilewm_storage_npy_export_20261028_v1 as data

base, ir, ROOT = data.base, data.ir, data.ROOT
NAME = "agent_hilewm_runtime_attributes_20261028_v1"
PARENT_SHA = "1e88a27e5c1cff0d4c852c2f75e7f804cdc69a23b3c956a0b0ef2163560eacce"
LOCAL_SHA = "0b258a9e8dc24c29fcb1e8c50a09ec78b8ea85aeb79e21dd8adf712396646620"
SOURCE_MEMBERS = (
    "package/code/h_le_wm/models/jepa.py",
    "package/code/h_le_wm/models/latent_action.py",
    "package/code/h_le_wm/eval/hierarchical.py",
    "package/code/h_le_wm/baseline/adapter.py",
    "package/checkpoints/pusht/main/config.yaml",
)
CAP = 24 * 1024**2
STRUCTURAL = {"_modules", "_parameters", "_buffers"}
NORMALIZER = re.compile(r"normaliz|scaler|(?:^|_)(?:action_mean|action_std|dataset_stats|mean_|scale_|var_)", re.I)
SPECIAL_METHODS = {"__getstate__", "__setstate__", "get_extra_state", "set_extra_state",
                   "_save_to_state_dict", "_load_from_state_dict", "state_dict", "load_state_dict"}


def need(ok, message):
    base.need(ok, message)


def origin(nodes, index):
    n = nodes[index]
    return dict(ir_sha256=data.IR_SHA, node=index, json_pointer=f"/ir/nodes/{index}",
                opcode_position=n.get("position"))


def state_fields(nodes, index):
    n = nodes[index]
    need(n["kind"] in {"newobj", "newobj_ex"} and len(n["states"]) == 1,
         "exact one saved BUILD state")
    state = nodes[n["states"][0]]
    parts = state["items"] if state["kind"] == "tuple" else n["states"]
    need(state["kind"] != "tuple" or len(parts) == 2, "BUILD tuple state layout")
    out = {}
    for part in parts:
        p = nodes[part]
        if p["kind"] == "none":
            continue
        need(p["kind"] == "dict" or p.get("container") == "ordered_dict", "literal state mapping")
        for key, value in p["entries"]:
            k = nodes[key]
            need(k["kind"] == "str" and k["value"] not in out, "unique literal state field")
            out[k["value"]] = dict(key_node=key, value_node=value, state_node=part)
    return out


class Literals:
    """Our builtin-only values; object/call nodes remain explicitly inert records."""
    def __init__(self, nodes, module_nodes=(), tensor_nodes=(), cap=100000):
        self.nodes, self.modules, self.tensors = nodes, set(module_nodes), set(tensor_nodes)
        self.cache, self.count, self.cap = {}, 0, cap

    def decode(self, index, ancestors=frozenset()):
        need(type(index) is int and 0 <= index < len(self.nodes), "IR node index")
        if index in self.cache:
            return self.cache[index]
        self.count += 1
        need(self.count <= self.cap and len(ancestors) <= 64, "attribute decode count/depth cap")
        node = self.nodes[index]
        result = dict(origin=origin(self.nodes, index), kind=node["kind"], status="opaque")
        if index in ancestors:
            return dict(result, reason="cyclic symbolic reference; no object construction")
        child = lambda i: self.decode(i, ancestors | {index})
        kind = node["kind"]
        if index in self.modules:
            result.update(status="module_reference")
        elif index in self.tensors:
            result.update(status="tensor_reference")
        elif kind in {"str", "bool", "int", "none", "float"}:
            if kind == "float":
                float.fromhex(node["hex"])
                result.update(status="literal", hex=node["hex"])
            else:
                value = None if kind == "none" else node["value"]
                need((kind != "str" or type(value) is str) and
                     (kind != "bool" or type(value) is bool), "literal type")
                if kind == "int":
                    need(type(value) is str and re.fullmatch(r"-?[0-9]{1,1024}", value), "bounded decimal int")
                result.update(status="literal", value=value)
        elif kind in {"list", "tuple", "set", "frozenset"}:
            items = [child(i) for i in node["items"]]
            result.update(items=items, status="literal" if all(x["status"] == "literal" for x in items) else "opaque")
        elif kind == "dict" or node.get("container") == "ordered_dict":
            entries = [dict(key=child(k), value=child(v)) for k,v in node["entries"]]
            result.update(kind="ordered_dict" if node.get("container") else "dict", entries=entries,
                          status="literal" if all(x[k]["status"] == "literal" for x in entries for k in ("key","value")) else "opaque")
            if result["status"] == "literal":
                try:
                    literal_value(result)
                except (ValueError, TypeError):
                    result.update(status="opaque", reason="non-unique or unhashable literal mapping key")
            if node.get("states"):
                result.update(status="opaque", reason="container BUILD states not interpreted", state_nodes=node["states"])
        elif kind == "global":
            result.update(symbol=[node["module"], node["name"]], reason="symbol, not a callable import")
        elif kind == "reduce":
            symbol, args = self.nodes[node["callable"]], self.nodes[node["args"]]
            if (symbol["kind"] == "global" and symbol["module"] in {"__builtin__", "builtins"}
                    and symbol["name"] == "set" and args["kind"] == "tuple" and not node["states"]
                    and len(args["items"]) <= 1):
                values = child(args["items"][0]) if args["items"] else dict(status="literal", kind="list", items=[])
                if values["status"] == "literal" and values["kind"] in {"list","tuple","set","frozenset"}:
                    result.update(kind="set", items=values["items"], status="literal", reducer_executed=False)
                else:
                    result["reason"] = "nonliteral set syntax"
            else:
                result.update(callable_node=node["callable"], args_node=node["args"], reason="REDUCE never executed")
        elif kind in {"newobj", "newobj_ex"}:
            cls = self.nodes[node["callable"]]
            try:
                fields = state_fields(self.nodes, index)
                need(cls["kind"]=="global","literal class symbol required")
                result.update(status="inert_object_state", class_declaration=[cls["module"],cls["name"]],
                              args_node=node["args"], kwargs_node=node.get("kwargs"),
                              fields={k: dict(**v, decoded=child(v["value_node"])) for k,v in fields.items()},
                              reason="saved state only; no constructor or __setstate__ interpretation")
            except ValueError as exc:
                result.update(reason=str(exc),args_node=node.get("args"),class_node=node.get("callable"))
        else:
            result["reason"] = "unknown/encoding-sensitive syntax retained at origin"
        self.cache[index] = result
        return result


def literal_value(record):
    """Public constructor-input helper: only builtin literals, refuses every opaque/ref/object."""
    need(record["status"] == "literal", "constructor field is not a literal")
    kind = record["kind"]
    if kind in {"str", "bool", "none"}:
        return record["value"]
    if kind == "int":
        return int(record["value"])
    if kind == "float":
        return float.fromhex(record["hex"])
    if kind in {"tuple", "list", "set", "frozenset"}:
        values = [literal_value(x) for x in record["items"]]
        return {"tuple":tuple, "list":list, "set":set, "frozenset":frozenset}[kind](values)
    need(kind in {"dict", "ordered_dict"}, "unknown literal kind")
    result = {}
    for e in record["entries"]:
        k, v = literal_value(e["key"]), literal_value(e["value"])
        need(k not in result, "duplicate literal dict key; no overwrite")
        result[k] = v
    return result


def source_ast(text, pointer):
    """Parse source only; no compile/eval/import of text or default expressions."""
    need(len(text.encode()) <= 2*1024**2, "source cap")
    tree, classes = ast.parse(text), {}
    for cls in tree.body:
        if not isinstance(cls, ast.ClassDef):
            continue
        methods, attributes, constructor = [], {}, []
        for method in cls.body:
            if not isinstance(method, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            methods.append(dict(name=method.name, first_line=method.lineno, last_line=method.end_lineno))
            if method.name == "__init__":
                pos = method.args.posonlyargs + method.args.args
                defaults = [None]*(len(pos)-len(method.args.defaults)) + method.args.defaults
                constructor = [dict(name=a.arg, kind="positional", default_source=ast.get_source_segment(text,d)
                                    if d is not None else None) for a,d in zip(pos,defaults) if a.arg != "self"]
                constructor += [dict(name=a.arg, kind="keyword_only", default_source=ast.get_source_segment(text,d)
                                     if d is not None else None) for a,d in zip(method.args.kwonlyargs, method.args.kw_defaults)]
                for arg in (method.args.vararg, method.args.kwarg):
                    if arg:
                        constructor.append(dict(name=arg.arg, kind="variadic", default_source=None))
            for node in ast.walk(method):
                if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id == "self":
                    attributes.setdefault(node.attr, []).append(dict(method=method.name, line=node.lineno,
                        access="write" if isinstance(node.ctx, ast.Store) else "read"))
        classes[cls.name] = dict(source=pointer, first_line=cls.lineno, last_line=cls.end_lineno,
                                constructor=constructor, self_attributes=attributes, methods=methods,
                                special_methods=[m for m in methods if m["name"] in SPECIAL_METHODS],
                                defaults_are_not_used=True)
    calls = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = ast.unparse(node.func)
            if "StandardScaler" in func or func.endswith(".fit"):
                calls.append(dict(source=pointer, line=node.lineno, expression=ast.get_source_segment(text,node)))
    return dict(classes=classes, normalizer_calls=calls)


def released_scalars(text, pointer):
    """Only document scalar leaves. No YAML library, interpolation/eval or defaults."""
    stack, out = [], []
    for line_no, line in enumerate(text.splitlines(),1):
        match = re.fullmatch(r"( *)([A-Za-z_][A-Za-z0-9_]*):(?: +(.*))?", line)
        if not match:
            continue
        spaces, key, raw = match.groups()
        while stack and stack[-1][0] >= len(spaces):
            stack.pop()
        path = [x[1] for x in stack]+[key]
        if raw is None:
            stack.append((len(spaces),key))
            continue
        status, value = "opaque_source_expression", None
        if raw in {"true","false","null"}:
            status, value = "released_scalar", {"true":True,"false":False,"null":None}[raw]
        elif re.fullmatch(r"-?[0-9]+",raw):
            status, value = "released_scalar", int(raw)
        elif re.fullmatch(r"-?(?:[0-9]+\.[0-9]+)(?:e[-+]?[0-9]+)?",raw,re.I):
            status, value = "released_scalar", float(raw)
        elif re.fullmatch(r"[A-Za-z_][A-Za-z0-9_./-]*", raw):
            status, value = "released_scalar", raw
        out.append(dict(path=path, raw=raw, value=value, status=status,
                        source=pointer, line=line_no, saved_object_value=False))
    return out


def read_sources():
    contract = base.small(base.TEXT_BASE/"contract.json", base.CONTRACT_SHA)
    result = []
    for member in SOURCE_MEMBERS:
        record = contract["sources"][member]
        need(record["archive_member"] == member and record["local_text"].startswith("texts/") and
             ".." not in Path(record["local_text"]).parts, "source locator")
        path = base.TEXT_BASE/record["local_text"]
        with base.reader(path) as f:
            before = os.fstat(f.fileno()); raw = f.read(2*1024**2+1)
            need(len(raw) <= 2*1024**2 and base.sha(raw) == record["sha256"], "official source SHA/cap")
            base.unchanged(f,path,before)
        result.append(dict(text=raw.decode(), pointer=dict(path=str(path), sha256=record["sha256"],
                           archive_member=member, role="released_official_source_not_checkpoint_code_equivalence")))
    path = ROOT/"module.py"
    with base.reader(path) as f:
        before = os.fstat(f.fileno()); raw=f.read(2*1024**2+1)
        need(len(raw) <= 2*1024**2 and base.sha(raw) == LOCAL_SHA, "local LeWM candidate SHA")
        base.unchanged(f,path,before)
    result.append(dict(text=raw.decode(), pointer=dict(path=str(path), sha256=LOCAL_SHA,
                       archive_member=None, role="local_candidate_not_official_checkpoint_source_equivalence")))
    return result


def extract(saved, sources):
    nodes, nn = saved["ir"]["nodes"], saved["nn"]
    need(nn["complete"] is True and nn["errors"] == [] and len(nodes) <= 100000, "complete bounded IR")
    decoder = Literals(nodes, [m["node"] for m in nn["modules"] if not m.get("explicit_none")],
                       [e["node"] for e in nn["entries"] if e["metadata"] is not None])
    source_indexes, scalars, source_rows = {}, [], []
    for source in sources:
        p = source["pointer"]
        if p["path"].endswith(".yaml.source.txt"):
            scalars += released_scalars(source["text"],p)
        else:
            index = source_ast(source["text"],p)
            source_rows.append(index)
            for name, value in index["classes"].items():
                source_indexes.setdefault(name, []).append(value)
    modules, configs, hook_risks, normalizer_hits = [], {}, [], []
    module_by_path = {}
    for module in nn["modules"]:
        if module.get("explicit_none"):
            modules.append(dict(path=module["path"], node=module["node"], explicit_none=True))
            continue
        fields = state_fields(nodes,module["node"])
        class_name = module["class_declaration"]
        candidates = source_indexes.get(class_name[1], []) if class_name[0] in {
            "hi_jepa","hi_module","module","_baseline_lewm_module"} else []
        need(len(candidates) <= 1, "ambiguous source class mapping")
        source_class = candidates[0] if candidates else None
        attributes = {}
        for name, field in fields.items():
            record = decoder.decode(field["value_node"]) if name not in STRUCTURAL else dict(
                status="structural_reference", kind=nodes[field["value_node"]]["kind"],
                origin=origin(nodes,field["value_node"]))
            attributes[name] = dict(**field, decoded=record, source_accesses=(
                source_class["self_attributes"].get(name,[]) if source_class else []))
            if record["status"] == "inert_object_state":
                configs[str(field["value_node"])] = record
            if "hook" in name.lower() or "extra_state" in name.lower():
                empty = False
                try:
                    value = literal_value(record)
                    empty = value is None or isinstance(value,(dict,list,tuple,set,frozenset)) and len(value)==0
                except (ValueError,TypeError):
                    pass
                if not empty:
                    hook_risks.append(dict(module_path=module["path"], field=name, decoded=record,
                                           risk="nonempty_or_opaque_saved_hook_or_extra_state"))
            if NORMALIZER.search(name):
                normalizer_hits.append(dict(module_path=module["path"], field=name, decoded=record,
                    candidate_is_not_dataset_normalizer_certification=True))
        registered = {}
        for collection in sorted(STRUCTURAL):
            need(collection in fields, "explicit registered collection required")
            container = nodes[fields[collection]["value_node"]]
            need(container["kind"] == "dict" or container.get("container") == "ordered_dict", "registered mapping")
            for key,value in container["entries"]:
                k=nodes[key]
                need(k["kind"] == "str" and k["value"] not in registered and k["value"] not in fields,
                     "registered runtime name collision")
                registered[k["value"]]=dict(saved_in=collection, key_node=key, value_node=value,
                    decoded=decoder.decode(value), source_accesses=(
                        source_class["self_attributes"].get(k["value"],[]) if source_class else []))
        missing_writes = sorted(set(a for a, uses in source_class["self_attributes"].items()
                                    if any(x["access"]=="write" for x in uses))-set(fields)-set(registered)) if source_class else None
        row = dict(path=module["path"], node=module["node"], class_declaration=class_name,
                   origin=origin(nodes,module["node"]), attributes=attributes, registered_fields=registered,
                   source_class=source_class,
                   source_self_write_fields_absent_in_saved_state=missing_writes,
                   module_source_identity_verified=False)
        modules.append(row)
        module_by_path[tuple(module["path"])] = row
    tensor_by_path = {tuple(e["module_path"]+[e["name"]]):e for e in nn["entries"] if e["metadata"] is not None}
    def attr(path,name):
        field = module_by_path[tuple(path)]["attributes"][name]
        return literal_value(field["decoded"]), dict(module_path=path,field=name,**field["decoded"]["origin"])
    def shape(path,axis):
        entry = tensor_by_path[tuple(path)]
        return entry["metadata"]["shape"][axis], dict(entry_path=path,axis=axis,
            json_pointer=f'/nn/entries/{nn["entries"].index(entry)}/metadata/shape/{axis}', ir_sha256=data.IR_SHA)
    def only_dim(path):
        value,ref=attr(path,"normalized_shape")
        need(type(value) in (tuple,list) and len(value)==1,"exact one-dimensional normalization")
        return value[0],ref
    def uniform(paths,field):
        values=[attr(p,field) for p in paths]
        need(values and all(type(x[0]) is type(values[0][0]) and x[0]==values[0][0] for x in values),
             "saved layers differ; no first-layer defaults")
        return values[0][0],[x[1] for x in values]
    def head_dim(paths):
        heads,refs=uniform([p+["attn"] for p in paths],"heads")
        width,more=uniform([p+["attn","to_qkv"] for p in paths],"out_features")
        need(type(heads) is int and heads>0 and type(width) is int and width%(3*heads)==0,"exact qkv/head division")
        return width//(3*heads),refs+more
    derived = []
    def derive(name,fn):
        try:
            value, refs = fn()
            derived.append(dict(name=name,status="saved_metadata_derived",value=value,evidence=refs,
                                not_an_original_saved_attribute=True))
        except (KeyError,ValueError,IndexError,TypeError) as exc:
            derived.append(dict(name=name,status="unknown",value=None,reason=str(exc),evidence=[]))
    for prefix in ("low_predictor","high_predictor"):
        derive(prefix+".num_frames", lambda p=prefix: shape([p,"pos_embedding"],1))
        derive(prefix+".input_dim", lambda p=prefix: shape([p,"pos_embedding"],2))
        layer_paths = [p for p in module_by_path if len(p)==4 and p[:3]==(prefix,"transformer","layers")]
        try:
            numbers = sorted(int(p[-1]) for p in layer_paths)
            need(numbers == list(range(len(numbers))) and numbers, "contiguous nonempty saved layers")
            derived.append(dict(name=prefix+".depth", status="saved_metadata_derived", value=len(numbers),
                                evidence=[dict(module_path=list(p),node=module_by_path[p]["node"]) for p in layer_paths],
                                not_an_original_saved_attribute=True))
        except ValueError as exc:
            derived.append(dict(name=prefix+".depth",status="unknown",value=None,reason=str(exc),evidence=[]))
        blocks=[list(p) for p in sorted(layer_paths)]
        derive(prefix+".hidden_dim",lambda p=prefix: only_dim([p,"transformer","norm"]))
        derive(prefix+".heads",lambda b=blocks: uniform([p+["attn"] for p in b],"heads"))
        derive(prefix+".dim_head",lambda b=blocks: head_dim(b))
        derive(prefix+".mlp_dim",lambda b=blocks: uniform([p+["mlp","net","1"] for p in b],"out_features"))
        derive(prefix+".dropout",lambda b=blocks: uniform([p+["attn"] for p in b],"dropout"))
        derive(prefix+".emb_dropout",lambda p=prefix: attr([p,"dropout"],"p"))
        def output_dim(p=prefix):
            child=module_by_path[(p,"transformer","output_proj")]
            if child["class_declaration"]==["torch.nn.modules.linear","Identity"]:
                return only_dim([p,"transformer","norm"])
            return attr([p,"transformer","output_proj"],"out_features")
        derive(prefix+".output_dim",output_dim)
    for name,path,field in (("latent_action.input_dim",["latent_action_encoder","input_proj"],"in_features"),
          ("latent_action.model_dim",["latent_action_encoder","input_proj"],"out_features"),
          ("latent_action.latent_dim",["latent_action_encoder","output_proj"],"out_features"),
          ("latent_action.max_seq_len",["latent_action_encoder"],"max_seq_len"),
          ("latent_action.num_layers",["latent_action_encoder","encoder"],"num_layers"),
          ("action_encoder.input_dim",["action_encoder","patch_embed"],"in_channels"),
          ("action_encoder.smoothed_dim",["action_encoder","patch_embed"],"out_channels"),
          ("action_encoder.emb_dim",["action_encoder","embed","2"],"out_features"),
          ("macro_to_condition.input_dim",["macro_to_condition"],"in_features"),
          ("macro_to_condition.output_dim",["macro_to_condition"],"out_features")):
        derive(name,lambda p=path,f=field: attr(p,f))
    def action_scale():
        width,ref=attr(["action_encoder","embed","0"],"out_features")
        dim,other=attr(["action_encoder","embed","2"],"out_features")
        need(type(width) is int and type(dim) is int and dim>0 and width%dim==0,"exact action MLP scale division")
        return width//dim,[ref,other]
    derive("action_encoder.mlp_scale",action_scale)
    for prefix in ("projector","low_pred_proj","high_pred_proj"):
        for name,path,field in (("input_dim",[prefix,"net","0"],"in_features"),
                                ("hidden_dim",[prefix,"net","0"],"out_features"),
                                ("output_dim",[prefix,"net","3"],"out_features")):
            derive(prefix+"."+name,lambda p=path,f=field:attr(p,f))
        for name,index in (("norm_class","1"),("activation_class","2")):
            def declaration(p=prefix,i=index):
                child=module_by_path[(p,"net",i)]
                return child["class_declaration"],dict(module_path=child["path"],node=child["node"],
                                                      ir_sha256=data.IR_SHA)
            derive(prefix+"."+name,declaration)
    latent_layers=[list(p) for p in sorted(module_by_path) if len(p)==4 and
                   p[:3]==("latent_action_encoder","encoder","layers")]
    def latent_uniform(child,field):
        count,count_ref=attr(["latent_action_encoder","encoder"],"num_layers")
        need(type(count) is int and count>0 and sorted(int(p[-1]) for p in latent_layers)==list(range(count)),
             "saved latent encoder layers/count not closed")
        value,refs=uniform([p+[child] for p in latent_layers],field)
        return value,[count_ref]+refs
    for name,child,field in (("num_heads","self_attn","num_heads"),
                             ("mlp_dim","linear1","out_features"),("dropout","dropout","p")):
        derive("latent_action."+name,lambda c=child,f=field:latent_uniform(c,f))
    declarations = [d for d in saved["ir"]["declarations"] if NORMALIZER.search(d["module"]+"."+d["name"])]
    return dict(schema=NAME, status="saved_attributes_extracted_not_constructor_approval", modules=modules,
                object_states=list(configs.values()), derived_constructor_fields=derived,
                released_config_scalars=scalars, static_source_indexes=source_rows,
                normalizer=dict(saved_named_field_candidates=normalizer_hits, declared_symbol_candidates=declarations,
                    predicate=NORMALIZER.pattern, scan_scope="all saved nn module state field names and IR declarations",
                    external_fit_source_calls=[x for row in source_rows for x in row["normalizer_calls"]],
                    frozen_normalizer_values_verified=False, absence_is_not_proof_of_external_runtime_absence=True),
                hooks=dict(nonempty_or_opaque_saved_fields=hook_risks,
                    reviewed_module_count=len(module_by_path), inherited_or_unsourced_class_methods="unknown",
                    no_saved_hooks_does_not_prove_class_hooks_absent=True),
                ir_sha256=data.IR_SHA, defaults_applied=False, pickle_executed=False, foreign_code_executed=False,
                constructor_approved=False, model_equivalence=False, low_high_pairing_verified=False,
                derived_constructor_rules_are_conditional_on_reviewed_source_mapping=True,
                scientific_admission=False, gpu_calls=0, physics_calls=0)


def constructor_values(report, prefix, required):
    """Return explicit fields only. Does NOT instantiate a class or authorize its defaults."""
    need(len(set(required)) == len(required),"duplicate constructor request")
    rows={r["name"]:r for r in report["derived_constructor_fields"]}
    out={}
    for name in required:
        row=rows[prefix+"."+name]
        need(row["status"]=="saved_metadata_derived", "unknown required constructor field: "+name)
        out[name]=row["value"]
    return out


def object_literal_fields(record):
    need(record["status"]=="inert_object_state", "literal object state required")
    return {name:literal_value(field["decoded"]) for name,field in record["fields"].items()}


def own_closure():
    return {str(p):base.sha(p.read_bytes()) for p in (Path(__file__), ROOT/"tests"/("test_"+NAME+".py"),
                                                    ROOT/"innerdocs"/(NAME+".md"))}


def bound_inputs():
    with base.reader(Path(data.__file__)) as f:
        need(base.hash_stream(f)[1] == PARENT_SHA, "frozen parent source SHA")
    saved, _ = data.binding()
    return saved, read_sources()


def build():
    saved, sources = bound_inputs()
    own = own_closure()
    result = extract(saved,sources)
    need(bound_inputs() == (saved,sources) and own_closure()==own, "input/source closure changed")
    result["own_sha256"] = own
    result["source_closure"] = [s["pointer"] for s in sources]
    need(len(base.encoded(result)) <= CAP, "attribute report cap")
    return result


def save(path,result):
    path=Path(path)
    need(path.is_absolute() and path.parent == ROOT/"results" and
         re.fullmatch(re.escape(NAME)+r"_[A-Za-z0-9_-]+\.json",path.name), "exclusive attribute output")
    raw=base.encoded(result)
    need(len(raw)<=CAP,"attribute report cap")
    used=data.budget_bytes()
    for p in (ROOT/"results").glob(NAME+"_*.json"):
        with base.reader(p) as f:
            used+=os.fstat(f.fileno()).st_size
    base.capacity(used,shutil.disk_usage(path.parent).free,len(raw))
    fd=base.dir_fd(path.parent)
    try:
        with base.new_file(fd,path.name) as f:
            need(f.write(raw)==len(raw),"short attribute report write"); f.flush(); os.fsync(f.fileno())
        os.fsync(fd)
    finally:
        os.close(fd)
    return dict(path=str(path),sha256=base.sha(raw),bytes=len(raw),status=result["status"])


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest="command",required=True)
    sub.add_parser("plan")
    cmd=sub.add_parser("extract"); cmd.add_argument("--output",required=True,type=Path)
    cmd=sub.add_parser("verify"); cmd.add_argument("--input",required=True,type=Path); cmd.add_argument("--sha256",required=True)
    a=p.parse_args(argv)
    if a.command=="plan":
        saved,sources=bound_inputs()
        result=dict(schema=NAME+".plan",ir_sha256=data.IR_SHA,modules=len(saved["nn"]["modules"]),
                    sources=[s["pointer"] for s in sources],production_attributes_extracted=False)
    elif a.command=="extract":
        result=save(a.output,build())
    else:
        expected=build(); actual=data.read_json(a.input,a.sha256,CAP)
        need(actual==expected,"saved attributes differ from fixed IR/source recomputation")
        result=dict(status="saved_attribute_data_verified_not_constructor_approval",sha256=a.sha256)
    print(base.encoded(result).decode(),end="")


if __name__=="__main__":
    main()
