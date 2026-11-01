"""Offline configuration/value provenance manifest, explicitly NOT training certification.

Stdlib plus safe YAML only. Never imports training, torch, model or env modules;
weights are streamed for byte SHA only; datasets are not opened or scanned.
"""
from __future__ import annotations
import argparse
import ast
import gzip
import hashlib
from io import BytesIO
import json
import math
import os
from pathlib import Path
import struct
import yaml

NAME='agent_p1_configuration_provenance_20261028_v1'
ROOT=Path(__file__).resolve().parents[2]
CHECKPOINT=Path('/home/gbk/.stable_worldmodel/checkpoints/pusht/lewm')
CONFIG_SHA='2564086e961e7b5c7c04dffc451091115b389a590645ff19653c64fd0bc16e09'
WEIGHTS_SHA='48938400ae3464c9680731287f583a9cb516f55a8ec64ea13a91be47fb15b607'
NORMALIZER_SHA='0a4eb4592951139e596b35becede17d175df7b48a2fde16679ab0aac7c4dce34'
TASK='results/p1_moving_overlay_recompile_cuda_20261027_v1_cuda_MAIN01/task_00.json.gz'
TASK_SHA='80f0f4ed14498509e596322b830f0f8d1bedc4d38f9d3b7d64b57da375f9a132'
STATS='results/pusht_d2p_archive_action_statistics_v1_20261026.json'
STATS_SHA='c816c535e08a946a6f7464c659b23e98022b0375359bf5a7add4edd6e018cdda'


def require(ok,message):
    if not ok:raise ValueError(message)
def canonical(x):return json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1024**2),b''):h.update(chunk)
    return h.hexdigest()
def decode(raw):
    def pairs(items):
        d={}
        for k,v in items:require(k not in d,'duplicate JSON key');d[k]=v
        return d
    result=json.loads(raw,object_pairs_hook=pairs);canonical(result);return result
def f32(values):
    require(isinstance(values,(list,tuple)) and len(values)==2 and
        all(type(x) in (int,float) and math.isfinite(x) for x in values),'finite literal raw2 statistics')
    raw=struct.pack('<2f',*values)
    return dict(literals=list(values),float32_values=list(struct.unpack('<2f',raw)),float32_le_hex=raw.hex())


def constants(path,mean_name,std_name):
    """Only AST literals in top-level array constructors; no eval/import/exec."""
    found={}
    for n in ast.parse(Path(path).read_text()).body:
        if not isinstance(n,ast.Assign):continue
        for target in n.targets:
            if not isinstance(target,ast.Name) or target.id not in (mean_name,std_name):continue
            require(target.id not in found and isinstance(n.value,ast.Call) and
                ast.unparse(n.value.func) in ('np.asarray','torch.tensor') and len(n.value.args)==1,
                'unique literal array constructor')
            try:values=ast.literal_eval(n.value.args[0])
            except (ValueError,TypeError) as e:raise ValueError('literal action statistics required') from e
            found[target.id]=dict(**f32(values),line=n.lineno)
    require(set(found)=={mean_name,std_name},'both action statistics present')
    require(all(x>0 for x in found[std_name]['float32_values']),'positive standard deviations')
    return dict(mean=found[mean_name],std=found[std_name])


def function_evidence(path,name):
    text=Path(path).read_text();tree=ast.parse(text)
    found=[n for n in ast.walk(tree) if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)) and n.name==name]
    require(len(found)==1,'unique function evidence '+name)
    n=found[0]
    return dict(path=str(path),line=n.lineno,end_line=n.end_lineno,source=ast.get_source_segment(text,n),executed=False)


def build():
    bindings={}
    def bind(path,expected=None):
        p=Path(path).absolute();require(p.resolve()==p and p.is_file() and not p.is_symlink(),'regular canonical evidence file')
        before=p.stat();h=sha(p);after=p.stat()
        require((before.st_size,before.st_mtime_ns)==(after.st_size,after.st_mtime_ns) and (expected is None or h==expected),'evidence byte SHA '+str(p))
        bindings[str(p)]=h;return dict(path=str(p),sha256=h,bytes=after.st_size)
    def small(path,expected=None):
        b=bind(path,expected);require(b['bytes']<=8*1024**2,'bounded metadata');return b,Path(path).read_bytes()
    config_record,raw=small(CHECKPOINT/'config.json',CONFIG_SHA);config=decode(raw)
    weights=bind(CHECKPOINT/'weights.pt',WEIGHTS_SHA)
    readme_record,readme=small(CHECKPOINT/'README.md')
    front=readme.decode().split('---')
    declared=yaml.safe_load(front[1]).get('datasets',[]) if len(front)>=3 else []
    cache={}
    for name in ('config.json','weights.pt','README.md'):
        record,raw=small(CHECKPOINT/'.cache/huggingface/download'/(name+'.metadata'))
        lines=raw.decode().splitlines();require(len(lines)==3,'original HF cache metadata three lines')
        cache[name]=dict(file=record,revision=lines[0],object_id=lines[1],download_timestamp=lines[2],
            meaning='local download metadata, not training dataset/run provenance')
    require(cache['weights.pt']['object_id']==WEIGHTS_SHA,'local weights LFS content identity')
    # Config is a Git blob object, not a SHA256 LFS object: independently verify it.
    raw=(CHECKPOINT/'config.json').read_bytes()
    require(hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()==cache['config.json']['object_id'],'config cached Git blob identity')
    source=ROOT/'experiments/run_pusht_counterfactual.py';bind(source,NORMALIZER_SHA)
    jepa=ROOT/'external/jepa-wms/app/plan_common/datasets/pusht_dset.py';bind(jepa)
    kernel=ROOT/'experiments/spectral_reachability/p3_goal_patch_consistent_compiler_20261027_v2.py';bind(kernel)
    values=dict(public=constants(source,'PUBLIC_ACTION_MEAN','PUBLIC_ACTION_STD'),
        jepa_loader=constants(jepa,'ACTION_MEAN','ACTION_STD'),P1_compiler=constants(kernel,'ACTION_MEAN','ACTION_STD'))
    for key,path in (('public',source),('jepa_loader',jepa),('P1_compiler',kernel)):
        values[key]['source_file']=dict(path=str(path),sha256=bindings[str(path)])
    task_record,compressed=small(ROOT/TASK,TASK_SHA)
    with gzip.GzipFile(fileobj=BytesIO(compressed)) as stream:expanded=stream.read(8*1024**2+1)
    require(len(expanded)<=8*1024**2,'bounded sealed task JSON');task=decode(expanded)
    norm=task['task_contract']['normalization']
    require(norm['source_sha256']==NORMALIZER_SHA and norm['training_provenance_verified'] is False and
        task['task']['model_sha256']==WEIGHTS_SHA and task['task']['model_config_sha256']==CONFIG_SHA,'sealed normalization and checkpoint value binding')
    values['sealed_task']={k:f32(norm[k]) for k in ('mean','std')}
    values['sealed_task']['source_file']=task_record
    reference=values['public']
    equal=all(all(v[k]['float32_le_hex']==reference[k]['float32_le_hex'] for k in ('mean','std')) for v in values.values())
    require(equal,'all current P1 public/stat value bindings agree; no silent replacement')
    snapshots={}
    for rel in ('train.py','utils.py','config/train/lewm.yaml','config/train/data/pusht.yaml','config/train/model/lewm.yaml',
        'config/eval/pusht.yaml','.venv/lib/python3.12/site-packages/stable_worldmodel/wm/utils.py'):
        record,raw=small(ROOT/rel)
        snapshots[rel]=dict(file=record,content=yaml.safe_load(raw) if rel.endswith('.yaml') else None)
    # Capture actual function text rather than executing a training pipeline.
    funcs=[function_evidence(ROOT/'train.py','run'),function_evidence(ROOT/'utils.py','get_column_normalizer'),
        function_evidence(ROOT/'utils.py','__call__'),
        function_evidence(ROOT/'.venv/lib/python3.12/site-packages/stable_worldmodel/wm/utils.py','save_pretrained')]
    stats_record,raw=small(ROOT/STATS,STATS_SHA);stats=decode(raw)
    require(stats['checkpoint_config']['sha256']==CONFIG_SHA and stats['statistics']['hash_scope']=='action_field_only_not_full_archive','sealed archive statistics scope')
    missing={
        'checkpoint_to_training_run':'No authenticated training run ID/receipt binding this weights SHA to a run.',
        'training_code_revision':'Current repository/source file hashes are snapshots, not checkpoint historical code identity.',
        'resolved_training_config':'No run-resolved training config authenticated against this checkpoint.',
        'training_dataset_immutable_revision':'README dataset name/download revision does not identify training dataset content.',
        'training_dataset_action_content_identity':'Saved local H5 action hash is not linked to the checkpoint training dataset.',
        'training_episode_split_and_filter_identity':'No checkpoint-linked episode list, split, NaN/padding/subset policy.',
        'training_action_units_and_conversion_identity':'Current raw-unit conventions do not certify historical dataset conversion.',
        'training_normalizer_export':'No checkpoint-linked exported fitted mean/std/dtype/ddof/statistics scope.',
        'normalizer_to_checkpoint_binding':'Matching public literal values alone do not bind a training scaler to weights.',
        'h5_to_lance_conversion_identity':'Current training YAML names Lance; local H5 stats do not certify identical training content.'}
    own=[Path(__file__).resolve(),ROOT/'tests'/f'test_{NAME}.py',ROOT/'innerdocs'/f'{NAME}.md']
    for path in own:bind(path)
    # Final full file recheck, including weight bytes; no unpickling/tensor hashing.
    for path,h in bindings.items():require(sha(path)==h,'evidence unchanged during manifest')
    result=dict(schema=NAME+'.manifest',status='value_bound_training_lineage_unestablished',bindings=bindings,
        checkpoint=dict(config=config_record,weights=weights,config_content=config,README=readme_record,
            declared_dataset_names=declared,cache_metadata=dict(objects=cache,same_download_revision=len({x['revision'] for x in cache.values()})==1),
            weight_identity='streamed file bytes SHA256, NOT deserialized tensor/state_dict identity',
            available_top_level_files=sorted(p.name for p in CHECKPOINT.iterdir() if p.is_file()),
            absent_config_training_fields=[k for k in ('training_run','training_config','dataset','dataset_revision','normalizer','action_mean','action_std') if k not in config]),
        value_binding=dict(all_float32_bits_equal=equal,sources=values,
            comparison='explicit literal -> IEEE float32 bits; not certification of JEPA execution dtype or training provenance'),
        sealed_normalization=dict(task=task_record,normalization=norm),
        current_training_snapshot=dict(files=snapshots,function_evidence=funcs,historical_checkpoint_training_identity=False,
            dataset_name=snapshots['config/train/data/pusht.yaml']['content']['dataset']['name'],
            hydra_resolved=False,interpretation='current code/config snapshot only; YAML interpolations not evaluated; functions not executed'),
        archive_statistics=dict(report=stats_record,source=stats['source'],statistics=stats['statistics'],
            hash_scope=stats['statistics']['hash_scope'],recomputed_in_this_tool=False,
            checkpoint_training_dataset_identity_verified=False),
        training_identity=dict(verified=False,status='not_established_from_bounded_existing_evidence',missing_fields=missing,
            scope='fixed checkpoint directory, named current source/configs, sealed task and archive statistics; no binary introspection or all-files absence claim'),
        operations=dict(network=0,GPU=0,physics=0,training=0,normalizer_fit=0,weight_deserialization=0,dataset_array_reads=0),
        normalizer_changed=False,model_changed=False,gate_changed=False,P1_admission=False)
    canonical(result);return result


def validate_saved(saved,expected):
    require(canonical(saved)==canonical(expected),'entire provenance manifest exact recomputation, no identity promotion')


def write(path,value):
    p=Path(path).absolute();require(p.resolve()==p and p.parent==ROOT/'results' and p.name.startswith(NAME+'_') and p.suffix=='.json','own new manifest only')
    raw=canonical(value);require(len(raw)<=8*1024**2,'bounded manifest')
    with p.open('xb') as f:require(f.write(raw)==len(raw),'complete manifest');f.flush();os.fsync(f.fileno())
    return dict(path=str(p),sha256=sha(p))


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('command',choices=('manifest','verify'))
    parser.add_argument('--output');parser.add_argument('--manifest');parser.add_argument('--manifest-sha')
    args=parser.parse_args();value=build()
    if args.command=='verify':
        require(args.manifest and args.manifest_sha and sha(args.manifest)==args.manifest_sha,'external saved manifest SHA')
        require(Path(args.manifest).stat().st_size<=8*1024**2,'bounded saved manifest')
        validate_saved(decode(Path(args.manifest).read_bytes()),value)
        value=dict(schema=NAME+'.verify',status='saved_manifest_verified_training_lineage_still_unestablished',
            manifest_sha256=args.manifest_sha,training_identity_verified=False)
    print(canonical(write(args.output,value) if args.output else value).decode())


if __name__=='__main__':main()
