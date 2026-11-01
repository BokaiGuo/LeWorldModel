"""Presealed loaded-file census; point-in-time evidence, not import tracing."""
import base64
import hashlib
from pathlib import Path
import sys
from experiments.spectral_reachability import agent_e3_transport_publication_20261029_v1 as pub

need=pub.base.need
ROOT=Path(__file__).resolve().parents[2]
MAX=8*1024**2
CHUNK=6000


def snapshot(bindings, modules=None, *, root=ROOT):
    """All loaded file-backed modules in this project tree, including venv.

    Standard-library and external paths are explicitly outside this census;
    transitive libraries loaded then unloaded between snapshots are not proved.
    """
    need(type(bindings) is dict,'presealed source bindings')
    modules=sys.modules if modules is None else modules;entries=[];checked={}
    for name,mod in sorted(list(modules.items())):
        if type(name) is not str or mod is None:continue
        filename=getattr(mod,'__file__',None)
        if not isinstance(filename,str):continue
        path=Path(filename)
        if not path.is_absolute():continue
        resolved=path.resolve()
        if root not in resolved.parents:continue
        need(path==resolved and resolved.is_file() and not resolved.is_symlink(),'canonical loaded module')
        namepath=str(resolved);need(namepath in bindings,'loaded module not presealed: '+namepath)
        if namepath not in checked:
            h=hashlib.sha256()
            with resolved.open('rb') as f:
                for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
            need(h.hexdigest()==bindings[namepath],'loaded module bytes changed')
            checked[namepath]=h.hexdigest()
        entries.append(dict(module=name,path=namepath,sha256=checked[namepath]))
    need(entries,'nonempty loaded module census')
    return entries


def record(store,stage,entries):
    need(stage in ('before','after') and type(entries) is list,'module census stage')
    data=pub.base.raw(entries);need(0<len(data)<=MAX,'bounded module census')
    sha=hashlib.sha256(data).hexdigest();parts=[data[i:i+CHUNK] for i in range(0,len(data),CHUNK)]
    for i,part in enumerate(parts):
        store.event('physical','record','module_census_chunk',store.index,
                    dict(stage=stage,index=i,total=len(parts),sha256=sha,data=base64.b64encode(part).decode('ascii')))
    store.event('physical','record','module_census_'+stage,store.index,
                dict(sha256=sha,total=len(parts),bytes=len(data)))


def saved(events,bindings):
    """Validate the two saved censuses and current pinned bytes, no imports."""
    results={}
    for stage in ('before','after'):
        end=[e for e in events if e['origin']=='physical' and e['resource']=='module_census_'+stage]
        chunks=[e for e in events if e['origin']=='physical' and e['resource']=='module_census_chunk'
                and e['detail'].get('stage')==stage]
        need(len(end)==1 and chunks and end[0]['kind']=='record','one complete module census')
        d=end[0]['detail'];need(set(d)=={'sha256','total','bytes'} and type(d['total']) is int
             and d['total']==len(chunks) and type(d['bytes']) is int and 0<d['bytes']<=MAX,'bounded census')
        pieces=[]
        for i,e in enumerate(chunks):
            c=e['detail'];need(e['kind']=='record' and e['index']<end[0]['index']
                and set(c)=={'stage','index','total','sha256','data'} and type(c['index']) is int
                and c['index']==i and c['total']==d['total'] and c['sha256']==d['sha256'],'census chunk order')
            b=base64.b64decode(c['data'],validate=True)
            need(len(b)==CHUNK if i<len(chunks)-1 else 0<len(b)<=CHUNK,'census chunk bytes');pieces.append(b)
        blob=b''.join(pieces);need(len(blob)==d['bytes'] and hashlib.sha256(blob).hexdigest()==d['sha256'],'census SHA')
        entries=pub.base.decode(blob);need(type(entries) is list and bool(entries),'nonempty census')
        names=[];paths={}
        for x in entries:
            need(type(x) is dict and set(x)=={'module','path','sha256'} and type(x['module']) is str
                 and x['module'] and type(x['path']) is str and x['path'] in bindings
                 and x['sha256']==bindings[x['path']], 'presealed saved module identity')
            names.append(x['module']);paths[x['path']]=x['sha256']
        need(names==sorted(set(names)),'ordered unique module names')
        for path,sha in paths.items():
            p=Path(path);need(p.is_absolute() and p==p.resolve() and p.is_file() and not p.is_symlink(),'canonical source')
            h=hashlib.sha256()
            with p.open('rb') as f:
                for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
            need(h.hexdigest()==sha,'saved module source changed')
        results[stage]=entries
    need({x['module']:x['path'] for x in results['before']}.items() <=
         {x['module']:x['path'] for x in results['after']}.items(),'no disappearing or replaced module')
    return dict(before=len(results['before']),after=len(results['after']),scope='project_tree_loaded_snapshots',
                transient_modules_proved=False,external_native_dependencies_proved=False,scientific_admission=False)
