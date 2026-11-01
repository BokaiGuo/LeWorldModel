"""Independent official vs cached HF DINOv2; prepare is strictly non-neural."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path, PurePosixPath
import sys
import tarfile

NAME = "p5_jepa_official_equivalence_20261027_v6"
COMMIT = "7764ea0f912e53c92e82eb78a2a1631e92725fc8"
ARCHIVE = Path("/tmp/lewm-dinov2-source-DSuJiP7Y/official.tar.gz")
ARCHIVE_SHA = "c27dcdaf50e9fb5bbdf2bb529da357716372e19c6afab17d5350f3f0094aed4b"
URL = f"https://codeload.github.com/facebookresearch/dinov2/tar.gz/{COMMIT}"
EPISODES = list(range(1, 9))


def sha(p):
    h = hashlib.sha256()
    with Path(p).open("rb") as f:
        for b in iter(lambda: f.read(1024**2), b""):
            h.update(b)
    return h.hexdigest()


def official_root(root):
    return root / "external" / f"dinov2_official_{COMMIT}"


def protocol():
    return dict(schema=NAME, episodes=EPISODES, frames=[0,5,10,15], input_size=224,
                official_config=dict(img_size=518, patch_size=14, init_values=1.,
                    block_chunks=0, num_register_tokens=0, interpolate_offset=.1,
                    interpolate_antialias=False, pretrained=False),
                dtype="float32", attention="official native SDPA math; HF native eager; xformers disabled",
                gates=dict(preprocess_max_abs=1e-6, feature_max_abs=1e-4, feature_rmse=1e-5),
                outputs=["x_norm_clstoken", "x_norm_patchtokens"],
                counts=dict(official_forward=8, HF_forward=8),
                selection="all fixed episodes, no replacement or outcome filtering",
                scope="cached weight paired implementation equivalence only, not official weight provenance, D2P, prediction or control benefit",
                interpolation="native behaviors unchanged: official offset scale_factor vs installed HF size; mismatch retained",
                P5_admission=False)


def safe_member(m):
    p = PurePosixPath(m.name)
    if p.is_absolute() or ".." in p.parts or not p.parts or p.parts[0] != f"dinov2-{COMMIT}":
        raise ValueError("archive traversal/root rejected")
    if not (m.isfile() or m.isdir()) or m.size > 8*1024**2:
        raise ValueError("archive special member/size rejected")
    return Path(*p.parts[1:])


def archive_audit(root):
    if ARCHIVE.stat().st_size > 10*1024**2 or sha(ARCHIVE) != ARCHIVE_SHA:
        raise ValueError("official archive SHA/size rejected")
    base = official_root(root).resolve()
    records = []; total = 0; seen = set()
    with tarfile.open(ARCHIVE, "r:gz") as tf:
        for index, m in enumerate(tf):
            if index >= 10000:
                raise ValueError("archive member bound")
            rel = safe_member(m)
            total += m.size
            if total > 80*1024**2 or str(rel) in seen:
                raise ValueError("archive total/duplicate bound")
            seen.add(str(rel))
            path = base / rel
            if path.is_symlink() or not path.resolve().is_relative_to(base):
                raise ValueError("extracted tree escape")
            if m.isfile():
                digest = hashlib.sha256(tf.extractfile(m).read()).hexdigest()
                if not path.is_file() or sha(path) != digest:
                    raise ValueError(f"extracted source differs: {rel}")
                records.append(dict(path=str(path), sha256=digest))
    if not any(Path(r["path"]).name == "LICENSE" for r in records):
        raise ValueError("license missing")
    # Extra Python modules could affect imports despite authentic archive files.
    expected = {r["path"] for r in records if r["path"].endswith(".py")}
    if {str(p) for p in base.rglob("*.py")} != expected:
        raise ValueError("unexpected official Python source")
    return records


def helpers(root):
    p = root / "experiments/spectral_reachability/p5_jepa_action_fidelity_20261027_v5.py"
    spec = importlib.util.spec_from_file_location("sealed_v5_for_v6", p)
    v5 = importlib.util.module_from_spec(spec); spec.loader.exec_module(v5)
    manifest = json.loads((root/f"results/{v5.NAME}/manifest_frozen.json").read_text())
    v4 = v5.import_v4(root)
    v5.verify_manifest(v4, root, manifest)
    return v5, v4


def mapping():
    # Each destination: ordered source keys, expected source shapes, destination shape.
    result = {}
    def add(dst, src, shapes, shape=None):
        result[dst] = dict(sources=src, source_shapes=shapes, shape=shape or shapes[0], dtype="F32")
    for dst, src, shape in [
        ("cls_token","embeddings.cls_token",[1,1,384]),
        ("mask_token","embeddings.mask_token",[1,384]),
        ("pos_embed","embeddings.position_embeddings",[1,1370,384]),
        ("patch_embed.proj.weight","embeddings.patch_embeddings.projection.weight",[384,3,14,14]),
        ("patch_embed.proj.bias","embeddings.patch_embeddings.projection.bias",[384]),
        ("norm.weight","layernorm.weight",[384]), ("norm.bias","layernorm.bias",[384])]:
        add(dst,[src],[shape])
    for i in range(12):
        o=f"blocks.{i}."; h=f"encoder.layer.{i}."
        for suffix, shape, merged in [("weight",[384,384],[1152,384]),("bias",[384],[1152])]:
            add(o+"attn.qkv."+suffix,
                [h+f"attention.attention.{q}.{suffix}" for q in ("query","key","value")],
                [shape]*3,merged)
        for dst, src, shape in [
            ("attn.proj.weight","attention.output.dense.weight",[384,384]),
            ("attn.proj.bias","attention.output.dense.bias",[384]),
            ("ls1.gamma","layer_scale1.lambda1",[384]),("ls2.gamma","layer_scale2.lambda1",[384]),
            *[(f"{n}.{s}",f"{n}.{s}",[384]) for n in ("norm1","norm2") for s in ("weight","bias")],
            ("mlp.fc1.weight","mlp.fc1.weight",[1536,384]),("mlp.fc1.bias","mlp.fc1.bias",[1536]),
            ("mlp.fc2.weight","mlp.fc2.weight",[384,1536]),("mlp.fc2.bias","mlp.fc2.bias",[384])]:
            add(o+dst,[h+src],[shape])
    return result


def header_audit(root):
    from safetensors import safe_open
    path=root/".cache/dinov2/dinov2-small/model.safetensors"
    with safe_open(path,framework="pt",device="cpu") as f:
        actual={k:dict(shape=f.get_slice(k).get_shape(),dtype=f.get_slice(k).get_dtype()) for k in f.keys()}
    expected={s:dict(shape=shape,dtype=r["dtype"]) for r in mapping().values()
              for s,shape in zip(r["sources"],r["source_shapes"])}
    compare_headers(actual,expected)
    return dict(source_header=actual, destination_mapping=mapping(), exact_source_coverage=True)


def compare_headers(actual, expected):
    missing=sorted(expected.keys()-actual.keys()); extra=sorted(actual.keys()-expected.keys())
    mismatched={k:dict(actual=actual[k],expected=expected[k]) for k in actual.keys() & expected.keys() if actual[k]!=expected[k]}
    if missing or extra or mismatched:
        raise ValueError(json.dumps(dict(missing=missing,unexpected=extra,mismatched=mismatched)))


def prepare(root):
    import transformers.models.dinov2.modeling_dinov2 as hf_source
    v5,v4=helpers(root)
    before=archive_audit(root)
    seals={r["path"]:r["sha256"] for r in v5.seals(v4,root)+before}
    paths=[Path(__file__).resolve(), root/"tests"/f"test_{NAME}.py",root.parent/"docs"/f"{NAME}.md",
           ARCHIVE, Path(hf_source.__file__),
           root/f"results/{v5.NAME}/manifest_frozen.json"]
    seals.update({str(p):sha(p) for p in paths})
    rows=[]
    for episode in EPISODES:
        rgb,length=v4.decode_video(v4.data_root(root)/f"obses/episode_{episode:03d}.mp4")
        rows.append(dict(episode=episode,frames=[0,5,10,15],length=length,shape=list(rgb.shape),rgb_sha256=v4.array_sha(rgb)))
    header=header_audit(root)
    if before!=archive_audit(root) or any(sha(p)!=h for p,h in seals.items()):
        raise ValueError("preparation drift")
    return dict(schema=NAME,status="prepared_non_neural",protocol=protocol(),rows=rows,
                seals=seals,weight_audit=header,official_origin=dict(commit=COMMIT,url=URL,
                    archive_sha256=ARCHIVE_SHA,license_sha256=sha(official_root(root)/"LICENSE")),
                cuda_executed=False, P5_admission=False)


def validate_cuda(root, manifest):
    # Reproduce all non-neural preparation before model construction.
    if prepare(root)!=manifest:
        raise ValueError("frozen manifest replay differs")
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required, no CPU neural fallback")
    os.environ.update(HF_HUB_OFFLINE="1",TRANSFORMERS_OFFLINE="1",XFORMERS_DISABLED="1")
    if any(k=="dinov2" or k.startswith("dinov2.") for k in sys.modules):
        raise ValueError("preimported dinov2 rejected; use fresh process")
    sys.path.insert(0,str(official_root(root)))
    from dinov2.hub.backbones import dinov2_vits14
    from transformers import Dinov2Config, Dinov2Model
    from safetensors.torch import load_file
    from torch.nn.attention import sdpa_kernel, SDPBackend
    v5,v4=helpers(root)
    sys.path.insert(0,str(v4.native_root(root)))
    from app.plan_common.datasets.transforms import make_transforms
    torch.manual_seed(0); torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    cache=root/".cache/dinov2/dinov2-small"
    cfg=Dinov2Config.from_pretrained(str(cache),local_files_only=True)
    cfg._attn_implementation="eager"
    with torch.device("cuda"):
        official=dinov2_vits14(**protocol()["official_config"])
        hf=Dinov2Model(cfg)
    weights=load_file(str(cache/"model.safetensors"),device="cuda")
    compare_headers({k:dict(shape=list(t.shape),dtype="F32" if t.dtype==torch.float32 else str(t.dtype)) for k,t in hf.state_dict().items()},manifest["weight_audit"]["source_header"])
    mapped={k:torch.cat([weights[s] for s in r["sources"]],dim=0) if len(r["sources"])>1 else weights[r["sources"][0]] for k,r in mapping().items()}
    compare_headers({k:list(t.shape) for k,t in official.state_dict().items()}, {k:list(t.shape) for k,t in mapped.items()})
    official.load_state_dict(mapped,strict=True); hf.load_state_dict(weights,strict=True)
    del mapped,weights
    models=dict(official=official,HF=hf)
    for model in models.values(): model.eval().requires_grad_(False)
    before={k:v4.model_receipt(m) for k,m in models.items()}
    imported={k:str(Path(m.__file__).resolve()) for k,m in sys.modules.items() if k.startswith("dinov2") and getattr(m,"__file__",None)}
    if not imported or any(not Path(p).is_relative_to(official_root(root)) for p in imported.values()):
        raise ValueError("official imported-source boundary rejected")
    transform=make_transforms(img_size=224,random_horizontal_flip=False,
        random_resize_aspect_ratio=(1.,1.),random_resize_scale=(1.,1.),reprob=0.,auto_augment=False,motion_shift=False)
    counts=dict(official_forward=0,HF_forward=0); rows=[]
    handles=[official.register_forward_hook(lambda *args: counts.__setitem__("official_forward",counts["official_forward"]+1)),
             hf.register_forward_hook(lambda *args: counts.__setitem__("HF_forward",counts["HF_forward"]+1))]
    with torch.inference_mode(), sdpa_kernel(SDPBackend.MATH):
        for saved in manifest["rows"]:
            rgb,_=v4.decode_video(v4.data_root(root)/f"obses/episode_{saved['episode']:03d}.mp4")
            if v4.array_sha(rgb)!=saved["rgb_sha256"]: raise ValueError("decoded RGB drift")
            pixels=torch.from_numpy(rgb.copy()).permute(0,3,1,2).to("cuda",torch.float32)/255.
            x=transform(pixels.clone())
            ref=torch.nn.functional.interpolate(pixels,size=(224,224),mode="bilinear",align_corners=False)
            ref=(ref-torch.tensor([.485,.456,.406],device="cuda")[None,:,None,None])/torch.tensor([.229,.224,.225],device="cuda")[None,:,None,None]
            prep=(x-ref).abs().max().item()
            if prep>protocol()["gates"]["preprocess_max_abs"]: raise ValueError("native preprocessing differs")
            # Official forward is_training returns its native feature dictionary.
            out=official(x,is_training=True); other=hf(pixel_values=x).last_hidden_state
            pairs={"x_norm_clstoken":(out["x_norm_clstoken"],other[:,0]),
                   "x_norm_patchtokens":(out["x_norm_patchtokens"],other[:,1:])}
            metrics={}
            for key,(a,b) in pairs.items():
                expected=(4,384) if key=="x_norm_clstoken" else (4,256,384)
                if tuple(a.shape)!=expected or a.shape!=b.shape or not torch.isfinite(a).all() or not torch.isfinite(b).all():
                    raise ValueError("output shape/finite gate rejected")
                d=a.double()-b.double()
                metrics[key]=dict(shape=list(a.shape),max_abs=d.abs().max().item(),rmse=d.square().mean().sqrt().item(),
                    per_frame_max_abs=d.reshape(4,-1).abs().max(dim=1).values.tolist())
            # No extra encoder forward: positional methods run on actual tokens.
            tokens=hf.embeddings.patch_embeddings(x)
            tokens=torch.cat((hf.embeddings.cls_token.expand(4,-1,-1),tokens),dim=1)
            pos=(official.interpolate_pos_encoding(tokens,224,224)-hf.embeddings.interpolate_pos_encoding(tokens,224,224)).double()
            rows.append(dict(episode=saved["episode"],preprocess_max_abs=prep,features=metrics,
                positional_max_abs=pos.abs().max().item(),positional_rmse=pos.square().mean().sqrt().item(),
                passed=all(m["max_abs"]<=1e-4 and m["rmse"]<=1e-5 for m in metrics.values())))
    for h in handles: h.remove()
    after={k:v4.model_receipt(m) for k,m in models.items()}
    source_unchanged=all(sha(p)==h for p,h in manifest["seals"].items())
    if before!=after or not source_unchanged or counts!=protocol()["counts"]:
        raise ValueError("freeze/source/count receipt rejected")
    return dict(schema=NAME,status="paired_equivalence_passed" if all(r["passed"] for r in rows) else "paired_equivalence_failed",
        rows=rows,counts=counts,receipts_before=before,receipts_after=after,source_hashes_unchanged=source_unchanged,
        imported_official_sources=imported,strict_loads=True,cuda_executed=True,protocol=protocol(),
        frozen_manifest_sha256=hashlib.sha256(json.dumps(manifest,sort_keys=True).encode()).hexdigest(),
        runtime=dict(torch=torch.__version__,cuda=torch.version.cuda),P5_admission=False)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",type=Path,default=Path(__file__).resolve().parents[2])
    parser.add_argument("--validate-cuda",action="store_true")
    parser.add_argument("--manifest",type=Path)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists(): raise ValueError("new outputs only")
    if args.validate_cuda:
        if args.manifest is None: parser.error("--manifest required")
        result=validate_cuda(args.root.resolve(),json.loads(args.manifest.read_text()))
    else: result=prepare(args.root.resolve())
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open("x") as f: json.dump(result,f,indent=2,sort_keys=True)
    print(result["status"])


if __name__=="__main__": main()
