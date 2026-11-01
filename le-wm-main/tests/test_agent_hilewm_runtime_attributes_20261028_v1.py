"""Synthetic saved IR/AST only; no checkpoint, NPY or foreign imports."""
import copy
import json
import math
from pathlib import Path
import subprocess
import sys

import pytest

from experiments.spectral_reachability import agent_hilewm_runtime_attributes_20261028_v1 as m


class Fixture:
    def __init__(self):
        self.nodes=[]; self.modules=[]; self.entries=[]
    def node(self,kind,**kw):
        index=len(self.nodes)
        self.nodes.append(dict(id=index,kind=kind,position=index,**kw))
        return index
    def literal(self,v):
        if v is None: return self.node("none")
        if type(v) is bool: return self.node("bool",value=v)
        if type(v) is int: return self.node("int",value=str(v))
        if type(v) is float: return self.node("float",hex=v.hex())
        if type(v) is str: return self.node("str",value=v)
        if type(v) in (list,tuple,set):
            return self.node({list:"list",tuple:"tuple",set:"set"}[type(v)],items=[self.literal(x) for x in v])
        if type(v) is dict:
            return self.node("dict",entries=[[self.literal(k),self.literal(x)] for k,x in v.items()])
        raise AssertionError(type(v))
    def obj(self,fields,cls=("foreign","NeverImport")):
        state=self.node("dict",entries=[[self.literal(k),v] for k,v in fields.items()])
        return self.node("newobj",callable=self.node("global",module=cls[0],name=cls[1]),
                         args=self.literal(()),states=[state],executed=False)
    def module(self,path,attrs=None,params=None,children=None,cls=("fixture","Module")):
        params=params or {}
        pnodes={k:self.node("reduce",executed=False) for k in params}
        fields={"_modules":self.node("dict",entries=[[self.literal(k),v] for k,v in (children or {}).items()]),
                "_parameters":self.node("dict",entries=[[self.literal(k),v] for k,v in pnodes.items()]),
                "_buffers":self.literal({}),"training":self.literal(False)}
        fields.update(attrs or {})
        index=self.obj(fields,cls)
        self.modules.append(dict(path=path,node=index,class_declaration=list(cls)))
        for name,shape in params.items():
            self.entries.append(dict(module_path=path,name=name,node=pnodes[name],
                collection="_parameters",metadata=dict(shape=shape)))
        return index
    def saved(self):
        return dict(ir=dict(nodes=self.nodes,declarations=[]),nn=dict(complete=True,errors=[],
                    modules=self.modules,entries=self.entries))


def sources():
    return [dict(pointer=dict(path="ram.py",sha256="RAM",role="local_candidate"),text='''
class HiJEPA:
    def __init__(self, encoder, history=999):
        self.encoder = encoder
        self.history = history
    def get_extra_state(self):
        raise AssertionError("must not execute")
class LatentActionEncoder:
    def __init__(self,input_dim=888,latent_dim=999):
        self.input_dim = input_dim
        self.latent_dim = latent_dim
''')]


def small_saved():
    f=Fixture()
    config=f.obj({"image_size":f.literal(224),"patch_size":f.literal(14),
                  "id2label":f.literal({0:"a",1:"b"}),"eps":f.literal(1e-12),
                  "return_dict":f.literal(True)},("transformers.models.vit.configuration_vit","ViTConfig"))
    enc=f.module(["encoder"],{"config":config},cls=("transformers.models.vit.modeling_vit","ViTModel"))
    low=f.module(["low_predictor"],params={"pos_embedding":[1,3,192]})
    high=f.module(["high_predictor"],params={"pos_embedding":[1,4,192]})
    action=f.module(["latent_action_encoder"],{"max_seq_len":f.literal(15)},cls=("hi_module","LatentActionEncoder"))
    f.module([],{"_freeze_flags":f.literal({"encoder":True}),"_forward_hooks":f.literal({})},
             children={"encoder":enc,"low_predictor":low,"high_predictor":high,"latent_action_encoder":action},
             cls=("hi_jepa","HiJEPA"))
    return f


@pytest.mark.parametrize("value", [None,True,False,0,-100,2**100,"value",1e-12,-0.0,
    (1,2),[1,True,None],{1:"one","two":2},{"name"}])
def test_literal_builtin_values(value):
    f=Fixture(); i=f.literal(value)
    record=m.Literals(f.nodes).decode(i)
    out=m.literal_value(record)
    assert type(out) is type(value) and out==value
    assert record["origin"]["json_pointer"]==f"/ir/nodes/{i}"
    if type(value) is float:
        assert out.hex()==value.hex()


def test_nonfinite_float_preserved_without_nonfinite_JSON():
    f=Fixture(); i=f.literal(float("nan"))
    record=m.Literals(f.nodes).decode(i)
    json.dumps(record,allow_nan=False)
    assert math.isnan(m.literal_value(record))


def test_set_reducer_inert_and_foreign_refused():
    f=Fixture(); value=f.literal(["a"])
    args=f.node("tuple",items=[value])
    cls=f.node("global",module="__builtin__",name="set")
    i=f.node("reduce",callable=cls,args=args,states=[],executed=False)
    assert m.literal_value(m.Literals(f.nodes).decode(i))=={"a"}
    f.nodes[cls].update(module="os",name="system")
    record=m.Literals(f.nodes).decode(i)
    assert record["status"]=="opaque"
    with pytest.raises(ValueError): m.literal_value(record)


def test_alias_and_cycle_origin():
    f=Fixture(); index=f.node("list",items=[])
    f.nodes[index]["items"].append(index)
    record=m.Literals(f.nodes).decode(index)
    assert record["status"]=="opaque"
    assert "cyclic" in record["items"][0]["reason"]
    assert record["items"][0]["origin"]==record["origin"]


def test_duplicate_and_unhashable_keys_are_opaque():
    for case in ("duplicate","unhashable"):
        f=Fixture(); k=f.literal("x") if case=="duplicate" else f.literal([1])
        i=f.node("dict",entries=[[k,f.literal(1)],[k,f.literal(2)]])
        assert m.Literals(f.nodes).decode(i)["status"]=="opaque"


def test_unknown_object_layout_retained():
    f=Fixture(); i=f.obj({"a":f.literal(1)})
    f.nodes[i]["states"]=[]
    record=m.Literals(f.nodes).decode(i)
    assert record["status"]=="opaque" and record["origin"]["node"]==i


def test_bounded_decode():
    f=Fixture(); i=f.literal([1,2,3])
    with pytest.raises(ValueError,match="cap"):
        m.Literals(f.nodes,cap=1).decode(i)
    with pytest.raises(ValueError): m.Literals(f.nodes).decode(-1)


def test_config_runtime_registered_constructor_values_and_no_defaults():
    f=small_saved(); r=m.extract(f.saved(),sources())
    assert len(r["modules"])==5
    config=r["object_states"][0]
    native=m.object_literal_fields(config)
    assert native==dict(image_size=224,patch_size=14,id2label={0:"a",1:"b"},eps=1e-12,return_dict=True)
    root=next(x for x in r["modules"] if x["path"]==[])
    assert root["registered_fields"]["encoder"]["saved_in"]=="_modules"
    assert root["registered_fields"]["encoder"]["decoded"]["status"]=="module_reference"
    assert root["source_self_write_fields_absent_in_saved_state"]==["history"]
    assert root["source_class"]["constructor"][1]["default_source"]=="999"
    assert m.constructor_values(r,"low_predictor",["num_frames","input_dim"])==dict(num_frames=3,input_dim=192)
    assert m.constructor_values(r,"high_predictor",["num_frames"])==dict(num_frames=4)
    with pytest.raises(ValueError): m.constructor_values(r,"low_predictor",["heads"])
    assert r["defaults_applied"] is False and r["constructor_approved"] is False
    assert r["hooks"]["nonempty_or_opaque_saved_fields"]==[]
    la=next(x for x in r["modules"] if x["path"]==["latent_action_encoder"])
    assert la["source_self_write_fields_absent_in_saved_state"]==["input_dim","latent_dim"]


def test_hook_normalizer_concrete_fields_and_no_fitting():
    f=small_saved()
    root=next(x for x in f.modules if x["path"]==[])
    state=f.nodes[f.nodes[root["node"]]["states"][0]]
    state["entries"].append([f.literal("action_normalizer"),f.literal({"mean":[1,2],"scale":[3,4]})])
    hook=next(v for k,v in state["entries"] if f.nodes[k].get("value")=="_forward_hooks")
    f.nodes[hook]["entries"].append([f.literal("hook"),f.node("global",module="os",name="system")])
    r=m.extract(f.saved(),sources())
    risk=r["hooks"]["nonempty_or_opaque_saved_fields"][0]
    assert risk["field"]=="_forward_hooks" and risk["module_path"]==[]
    norm=r["normalizer"]["saved_named_field_candidates"][0]
    assert norm["field"]=="action_normalizer"
    assert m.literal_value(norm["decoded"])["mean"]==[1,2]


def test_AST_records_never_executes_and_config_interpolation_opaque():
    text='''
raise RuntimeError("MUST NOT RUN")
class X:
    def __init__(self,x=__import__("os").system("false")):
        self.width=x
    def __setstate__(self,state):
        self.width=state
def fit():
    scaler=preprocessing.StandardScaler()
    scaler.fit(secret_data)
'''
    r=m.source_ast(text,{"path":"ram.py"})
    x=r["classes"]["X"]
    assert x["constructor"][0]["default_source"]=='__import__("os").system("false")'
    assert x["special_methods"][0]["name"]=="__setstate__"
    assert len(r["normalizer_calls"])==2
    scalars=m.released_scalars("wm:\n  history: 3\n  action: ${wm.dim}\n  nested:\n    flag: false\nx: 2\n",{})
    assert scalars[0]["path"]==["wm","history"] and scalars[0]["value"]==3
    assert scalars[1]["value"] is None and scalars[1]["status"]=="opaque_source_expression"
    assert scalars[2]["path"]==["wm","nested","flag"] and scalars[2]["value"] is False
    assert scalars[3]["path"]==["x"]


def test_registered_field_collision_refused():
    f=small_saved(); root=next(x for x in f.modules if x["path"]==[])
    state=f.nodes[f.nodes[root["node"]]["states"][0]]
    state["entries"].append([f.literal("encoder"),f.literal(0)])
    with pytest.raises(ValueError,match="collision"): m.extract(f.saved(),sources())


def test_saved_report_exclusive_and_verify_RAM(tmp_path,monkeypatch,capsys):
    (tmp_path/"results").mkdir()
    monkeypatch.setattr(m,"ROOT",tmp_path)
    monkeypatch.setattr(m.data,"budget_bytes",lambda:0)
    monkeypatch.setattr(m.shutil,"disk_usage",lambda p:type("U",(),{"free":20*1024**3})())
    r=m.extract(small_saved().saved(),sources())
    path=tmp_path/"results"/(m.NAME+"_RAM.json")
    receipt=m.save(path,r)
    assert receipt["sha256"]==m.base.sha(path.read_bytes())
    with pytest.raises(FileExistsError):m.save(path,r)
    with pytest.raises(ValueError):m.save(Path("relative"),r)
    monkeypatch.setattr(m,"build",lambda:r)
    m.main(["verify","--input",str(path),"--sha256",receipt["sha256"]])
    assert "saved_attribute_data_verified" in capsys.readouterr().out
    r["defaults_applied"]=True
    with pytest.raises(ValueError):m.main(["verify","--input",str(path),"--sha256",receipt["sha256"]])


def test_cold_plan_no_foreign_pickle_arrays_or_production_extract():
    code='''
import builtins,pickle,sys
def forbidden(*a,**k):raise AssertionError("pickle forbidden")
pickle.load=pickle.loads=pickle.Unpickler=forbidden
original=builtins.__import__
def checked(name,*a,**k):
    if name.split(".")[0] in {"torch","numpy","transformers","hi_jepa","hi_module","module","sklearn"}:
        raise AssertionError("foreign import "+name)
    return original(name,*a,**k)
builtins.__import__=checked
def audit(event,args):
    if event=="open" and str(args[0]).endswith((".ckpt",".zip",".deflate",".npy")):
        raise AssertionError("storage forbidden")
    if event in {"socket.connect","subprocess.Popen"}:raise AssertionError(event)
sys.addaudithook(audit)
from experiments.spectral_reachability import agent_hilewm_runtime_attributes_20261028_v1 as m
m.extract=forbidden
m.main(["plan"])
'''
    proc=subprocess.run([sys.executable,"-B","-c",code],cwd=m.ROOT,capture_output=True,text=True,timeout=30)
    assert proc.returncode==0,proc.stderr
    assert json.loads(proc.stdout)["production_attributes_extracted"] is False


def test_cli_no_foreign_override():
    with pytest.raises(SystemExit):m.main(["extract","--output","/tmp/no","--pickle","/tmp/foreign"])


def predictor_layers(f,prefix="low_predictor"):
    f.module([prefix,"transformer","norm"],{"normalized_shape":f.literal((192,))})
    f.module([prefix,"transformer","output_proj"],cls=("torch.nn.modules.linear","Identity"))
    f.module([prefix,"dropout"],{"p":f.literal(0.0)})
    for i in range(2):
        p=[prefix,"transformer","layers",str(i)]
        f.module(p)
        f.module(p+["attn"],{"heads":f.literal(8),"dropout":f.literal(0.1)})
        f.module(p+["attn","to_qkv"],{"out_features":f.literal(1536)})
        f.module(p+["mlp","net","1"],{"out_features":f.literal(768)})


def replace_attr(f,path,name,value):
    module=next(x for x in f.modules if x["path"]==path)
    fields=f.nodes[f.nodes[module["node"]]["states"][0]]["entries"]
    pair=next(p for p in fields if f.nodes[p[0]].get("value")==name)
    pair[1]=f.literal(value)


def test_complete_predictor_constructor_values_without_defaults():
    f=small_saved(); predictor_layers(f)
    r=m.extract(f.saved(),sources())
    values=m.constructor_values(r,"low_predictor",["num_frames","input_dim","hidden_dim","output_dim",
        "heads","dim_head","mlp_dim","depth","dropout","emb_dropout"])
    assert values==dict(num_frames=3,input_dim=192,hidden_dim=192,output_dim=192,heads=8,
                        dim_head=64,mlp_dim=768,depth=2,dropout=0.1,emb_dropout=0.0)


@pytest.mark.parametrize("case",["heads","mlp_dim","dropout","qkv","gap"])
def test_layer_heterogeneity_not_replaced_with_first_layer(case):
    f=small_saved(); predictor_layers(f)
    p=["low_predictor","transformer","layers","1"]
    if case=="heads":replace_attr(f,p+["attn"],"heads",16)
    elif case=="mlp_dim":replace_attr(f,p+["mlp","net","1"],"out_features",999)
    elif case=="dropout":replace_attr(f,p+["attn"],"dropout",0.2)
    elif case=="qkv":
        for i in range(2):replace_attr(f,["low_predictor","transformer","layers",str(i),"attn","to_qkv"],"out_features",1537)
    else:
        next(x for x in f.modules if x["path"]==p)["path"][-1]="3"
    r=m.extract(f.saved(),sources())
    field={"qkv":"dim_head","gap":"depth"}.get(case,case)
    row=next(x for x in r["derived_constructor_fields"] if x["name"]=="low_predictor."+field)
    assert row["status"]=="unknown" and row["value"] is None
    with pytest.raises(ValueError):m.constructor_values(r,"low_predictor",[field])


def test_latent_encoder_checks_all_layers():
    f=small_saved()
    f.module(["latent_action_encoder","encoder"],{"num_layers":f.literal(2)})
    for i in range(2):
        p=["latent_action_encoder","encoder","layers",str(i)]
        f.module(p)
        f.module(p+["self_attn"],{"num_heads":f.literal(4)})
        f.module(p+["linear1"],{"out_features":f.literal(768)})
        f.module(p+["dropout"],{"p":f.literal(0.1)})
    r=m.extract(f.saved(),sources())
    assert m.constructor_values(r,"latent_action",["num_heads","mlp_dim","dropout"])==dict(num_heads=4,mlp_dim=768,dropout=0.1)
    replace_attr(f,["latent_action_encoder","encoder","layers","1","self_attn"],"num_heads",8)
    r=m.extract(f.saved(),sources())
    with pytest.raises(ValueError):m.constructor_values(r,"latent_action",["num_heads"])


def test_unknown_config_field_cannot_be_constructor_input():
    f=small_saved()
    config=next(n for n in f.nodes if n["kind"]=="newobj" and
        f.nodes[n["callable"]].get("name")=="ViTConfig")
    fields=f.nodes[config["states"][0]]["entries"]
    fields.append([f.literal("callable"),f.node("global",module="os",name="system")])
    r=m.extract(f.saved(),sources())
    with pytest.raises(ValueError):m.object_literal_fields(r["object_states"][0])


def test_duplicate_state_and_ambiguous_source_refused():
    f=small_saved()
    with pytest.raises(ValueError,match="ambiguous"):
        m.extract(f.saved(),sources()+sources())
    root=next(x for x in f.modules if x["path"]==[])
    fields=f.nodes[f.nodes[root["node"]]["states"][0]]["entries"]
    fields.append(fields[-1].copy())
    with pytest.raises(ValueError,match="unique"):
        m.extract(f.saved(),sources())


def test_embedder_mlp_dimensions_do_not_guess_norm_defaults():
    f=small_saved()
    f.module(["action_encoder","patch_embed"],{"in_channels":f.literal(10),"out_channels":f.literal(10)})
    f.module(["action_encoder","embed","0"],{"out_features":f.literal(768)})
    f.module(["action_encoder","embed","2"],{"out_features":f.literal(192)})
    f.module(["projector","net","0"],{"in_features":f.literal(192),"out_features":f.literal(768)})
    f.module(["projector","net","3"],{"out_features":f.literal(192)})
    f.module(["projector","net","1"],cls=("torch.nn.modules.batchnorm","BatchNorm1d"))
    f.module(["projector","net","2"],cls=("torch.nn.modules.activation","GELU"))
    r=m.extract(f.saved(),sources())
    assert m.constructor_values(r,"action_encoder",["input_dim","smoothed_dim","emb_dim","mlp_scale"])==dict(
        input_dim=10,smoothed_dim=10,emb_dim=192,mlp_scale=4)
    assert m.constructor_values(r,"projector",["norm_class"])["norm_class"]==["torch.nn.modules.batchnorm","BatchNorm1d"]
    assert r["defaults_applied"] is False
