"""Preseal environment imports and synthetic render transport, without model or physics."""
import argparse
import io
import sys
from pathlib import Path

from experiments.spectral_reachability import agent_e3_startup_closure_20261029_v1 as old
from experiments.spectral_reachability import agent_e3_native_capture_transport_20261029_v1 as capture

host = old.host
launch = old.launch
need = old.need
NAME = 'agent_e3_render_closure_20261029_v1'


def probe():
    """Import native factory dependencies and exercise PNG capture using fake RGB."""
    import gymnasium as gym
    import stable_worldmodel
    from stable_worldmodel.envs.pusht.env import PushT
    import inspect
    import numpy as np
    from PIL import Image
    native = launch.reset_identity.native_meter.native
    need(Path(inspect.getsourcefile(PushT)).resolve() == native.kernel.moving.ENV and
         native.sha(native.kernel.moving.ENV) == native.kernel.moving.ENV_SHA, 'frozen environment source')
    spec = gym.spec('swm/PushT-v1')
    need(spec.entry_point == 'stable_worldmodel.envs.pusht.env:PushT' and
         spec.max_episode_steps is None and spec.additional_wrappers == (), 'native registry identity')
    # Pillow lazily registers all image plugins on Image.open; initialize explicitly.
    Image.init()
    class FakeStore:
        def __init__(self): self.output=Path('/fixture'); self.frames=0; self.index=0; self.data={}
        def check(self,*args): pass
        def write(self,name,data,category): self.data[name]=data
        def event(self,*args): self.index+=1
    store=FakeStore(); bridge=capture.CaptureStore(store)
    bridge.frame('render.png',np.zeros((224,224,3),dtype=np.uint8))
    with Image.open(io.BytesIO(store.data['render.png'])) as image:
        need(image.size == (224,224) and image.mode == 'RGB', 'fixture png round trip')
        image.load()
    del stable_worldmodel
    return old.inventory()


def collect(source, source_sha, output):
    host.authority(source,source_sha)
    entries=probe()
    need(launch.host.sha(source)==source_sha,'unchanged input authority')
    value=dict(schema=NAME+'.closure',source=dict(path=source,sha256=source_sha),
               python=str(Path(sys.executable).resolve()),entries=entries,
               scope='native_imports_and_fake_rgb_png',model_executed=False,physics_executed=False)
    sha=old.write_new(output,value)
    return dict(path=output,sha256=sha,modules=len(entries),unique_paths=len({x['path'] for x in entries}),scientific_admission=False)


def read_closure(path,sha,source,source_sha):
    value=launch.host.read(path,sha)
    need(type(value) is dict and set(value)=={'schema','source','python','entries','scope','model_executed','physics_executed'}
         and value['schema']==NAME+'.closure' and value['source']==dict(path=source,sha256=source_sha)
         and value['python']==str(Path(sys.executable).resolve()) and value['scope']=='native_imports_and_fake_rgb_png'
         and value['model_executed'] is False and value['physics_executed'] is False,'render closure schema')
    entries=value['entries'];need(type(entries) is list and bool(entries),'render entries')
    names=[];pins={}
    for e in entries:
        need(type(e) is dict and set(e)=={'module','path','sha256'} and type(e['module']) is str and bool(e['module'])
             and type(e['path']) is str and type(e['sha256']) is str and len(e['sha256'])==64
             and all(x in '0123456789abcdef' for x in e['sha256']),'entry schema')
        p=Path(e['path']);need(p.is_absolute() and p==p.resolve() and old.ROOT in p.parents and p.is_file() and not p.is_symlink(),'canonical module')
        need(pins.setdefault(str(p),e['sha256'])==e['sha256'],'consistent file hash');names.append(e['module'])
    need(names==sorted(set(names)),'unique ordered module names')
    for p,sha in pins.items():need(old.file_hash(p)==sha,'render source drift: '+p)
    return value,pins


def build(source,source_sha,closure,closure_sha):
    binding=host.authority(source,source_sha);_,pins=read_closure(closure,closure_sha,source,source_sha)
    value=launch.host.read(source,source_sha);value=dict(value,additional_sources=dict(value['additional_sources']))
    pins[str(Path(__file__).resolve())]=old.file_hash(Path(__file__).resolve())
    for p,sha in pins.items():
        prior=binding['prepared']['bindings_sha256'].get(p,value['additional_sources'].get(p))
        need(prior is None or prior==sha,'source pin collision')
        if p not in binding['prepared']['bindings_sha256']:value['additional_sources'][p]=sha
    need(launch.host.sha(source)==source_sha and launch.host.sha(closure)==closure_sha,'stable inputs')
    return value


def verify(source,source_sha,closure,closure_sha,authority,authority_sha):
    need(launch.host.read(authority,authority_sha)==build(source,source_sha,closure,closure_sha),'authority reconstruction')
    host.authority(authority,authority_sha)
    return dict(verified=True,scientific_admission=False)


def prepare(source,source_sha,closure,closure_sha,output):
    sha=old.write_new(output,build(source,source_sha,closure,closure_sha))
    verify(source,source_sha,closure,closure_sha,output,sha)
    return dict(path=output,sha256=sha,scientific_admission=False)


def precheck(source,source_sha,closure,closure_sha,authority,authority_sha):
    verify(source,source_sha,closure,closure_sha,authority,authority_sha)
    sealed,pins=read_closure(closure,closure_sha,source,source_sha)
    known={x['module']:(x['path'],x['sha256']) for x in sealed['entries']}
    actual=probe()
    for e in actual:
        need(pins.get(e['path'])==e['sha256'] and known.get(e['module'])==(e['path'],e['sha256']),
             'new or changed render dependency: '+e['path'])
    verify(source,source_sha,closure,closure_sha,authority,authority_sha)
    return dict(verified=True,modules=len(actual),model_executed=False,physics_executed=False,scientific_admission=False)


def main():
    p=argparse.ArgumentParser(description=__doc__);subs=p.add_subparsers(dest='command',required=True)
    for name in ('collect','prepare','verify','precheck'):
        x=subs.add_parser(name)
        for arg in ('source','source-sha256'):x.add_argument('--'+arg,required=True)
        if name in ('collect','prepare'):x.add_argument('--output',required=True)
        if name!='collect':
            for arg in ('closure','closure-sha256'):x.add_argument('--'+arg,required=True)
        if name in ('verify','precheck'):
            for arg in ('authority','authority-sha256'):x.add_argument('--'+arg,required=True)
    a=p.parse_args()
    if a.command=='collect':r=collect(a.source,a.source_sha256,a.output)
    elif a.command=='prepare':r=prepare(a.source,a.source_sha256,a.closure,a.closure_sha256,a.output)
    elif a.command=='verify':r=verify(a.source,a.source_sha256,a.closure,a.closure_sha256,a.authority,a.authority_sha256)
    else:r=precheck(a.source,a.source_sha256,a.closure,a.closure_sha256,a.authority,a.authority_sha256)
    print(launch.host.raw(r).decode())


if __name__=='__main__':main()
