"""Freeze all importable venv site-package files, without importing them."""
import argparse
import hashlib
import os
from pathlib import Path
import sys
import sysconfig

from experiments.spectral_reachability import agent_e3_render_closure_20261029_v1 as render

old=render.old
host=render.host
launch=render.launch
need=old.need
NAME='agent_e3_installed_closure_20261029_v1'
EXTENSIONS=('.py','.pyc','.so')
MAX_FILES=100000
FLOOR=1024**3


def site_root():
    pure=Path(sysconfig.get_path('purelib'));plat=Path(sysconfig.get_path('platlib'))
    need(pure==plat and pure.is_absolute() and pure==pure.resolve() and pure.is_dir()
         and old.ROOT in pure.parents, 'one canonical project venv site-packages root')
    return pure


def selected(name):return name.endswith(EXTENSIONS)


def enumerate_walk(root):
    files=[]
    for directory,dirs,names in os.walk(root,followlinks=False):
        for name in dirs:
            need(not (Path(directory)/name).is_symlink(), 'symlink directory in installed scope')
        for name in names:
            if not selected(name):continue
            p=Path(directory)/name
            need(not p.is_symlink() and p.is_file() and p==p.resolve(), 'canonical regular installed file')
            files.append(str(p))
            need(len(files)<=MAX_FILES,'installed file count bound')
    return sorted(files)


def enumerate_independent(root):
    files=[];pending=[root]
    while pending:
        directory=pending.pop()
        with os.scandir(directory) as entries:
            for item in entries:
                if item.is_symlink():
                    need(not item.is_dir(follow_symlinks=True) and not selected(item.name),'symlink in installed scope')
                elif item.is_dir(follow_symlinks=False):pending.append(Path(item.path))
                elif selected(item.name):
                    need(item.is_file(follow_symlinks=False),'regular installed file')
                    files.append(item.path);need(len(files)<=MAX_FILES,'installed file count bound')
    return sorted(files)


def hashes(paths):return {path:old.file_hash(path) for path in paths}


def disk_check(output):
    p=Path(output);s=os.statvfs(p.parent)
    need(s.f_bavail*s.f_frsize >= FLOOR+2*launch.host.META,'metadata output disk floor')


def collect(source,source_sha,output):
    host.authority(source,source_sha);root=site_root();disk_check(output)
    paths=enumerate_walk(root);need(paths,'nonempty installed scope')
    pins=hashes(paths)
    need(enumerate_walk(root)==paths,'installed set stable during collection')
    value=dict(schema=NAME+'.closure',source=dict(path=source,sha256=source_sha),root=str(root),
               python=str(Path(sys.executable).resolve()),extensions=list(EXTENSIONS),files=pins,
               scope='all_regular_importable_files_under_site_packages',model_executed=False,physics_executed=False)
    need(launch.host.sha(source)==source_sha,'unchanged source authority')
    sha=old.write_new(output,value)
    return dict(path=output,sha256=sha,files=len(pins),scientific_admission=False)


def verify_closure(path,sha,source,source_sha):
    sha_path=path;value=launch.host.read(path,sha);root=site_root()
    need(type(value) is dict and set(value)=={'schema','source','root','python','extensions','files','scope','model_executed','physics_executed'}
         and value['schema']==NAME+'.closure' and value['source']==dict(path=source,sha256=source_sha)
         and value['root']==str(root) and value['python']==str(Path(sys.executable).resolve())
         and value['extensions']==list(EXTENSIONS) and value['scope']=='all_regular_importable_files_under_site_packages'
         and value['model_executed'] is False and value['physics_executed'] is False,'installed closure schema')
    pins=value['files'];need(type(pins) is dict and 0<len(pins)<=MAX_FILES,'bounded file map')
    current=enumerate_independent(root)
    need(current==sorted(pins),'installed file set changed')
    for path in current:
        digest=pins[path]
        need(type(digest) is str and len(digest)==64 and all(x in '0123456789abcdef' for x in digest), 'sha schema')
        need(old.file_hash(path)==digest,'installed file drift: '+path)
    need(enumerate_independent(root)==current,'installed set stable during verification')
    need(launch.host.sha(source)==source_sha and launch.host.sha(sha_path)==sha,'stable source and closure')
    return pins


def build(source,source_sha,closure,closure_sha):
    binding=host.authority(source,source_sha);pins=verify_closure(closure,closure_sha,source,source_sha)
    value=launch.host.read(source,source_sha);value=dict(value,additional_sources=dict(value['additional_sources']))
    additions=dict(pins);p=str(Path(__file__).resolve());additions[p]=old.file_hash(p)
    for p,sha in additions.items():
        prior=binding['prepared']['bindings_sha256'].get(p,value['additional_sources'].get(p))
        need(prior is None or prior==sha,'source pin collision: '+p)
        if p not in binding['prepared']['bindings_sha256']:value['additional_sources'][p]=sha
    need(launch.host.sha(closure)==closure_sha and launch.host.sha(source)==source_sha,'stable inputs')
    return value


def prepare(source,source_sha,closure,closure_sha,output):
    disk_check(output);value=build(source,source_sha,closure,closure_sha);sha=old.write_new(output,value)
    return dict(path=output,sha256=sha,files=len(value['additional_sources']),scientific_admission=False)


def verify(source,source_sha,closure,closure_sha,authority,authority_sha):
    """Independent set scan, hash verification, and full authority reconstruction."""
    binding=host.authority(source,source_sha);pins=verify_closure(closure,closure_sha,source,source_sha)
    prior=launch.host.read(source,source_sha);actual=launch.host.read(authority,authority_sha)
    expected=dict(prior,additional_sources=dict(prior['additional_sources']))
    pins=dict(pins);p=str(Path(__file__).resolve());pins[p]=old.file_hash(p)
    for p,sha in pins.items():
        a=binding['prepared']['bindings_sha256'].get(p);b=expected['additional_sources'].get(p)
        need((a is None or a==sha) and (b is None or b==sha),'independent source collision')
        if a is None:expected['additional_sources'][p]=sha
    need(actual==expected,'exact independently reconstructed authority')
    host.authority(authority,authority_sha)
    need(launch.host.sha(closure)==closure_sha and launch.host.sha(authority)==authority_sha,'stable artifacts')
    return dict(verified=True,files=len(pins)-1,scientific_admission=False,model_executed=False,physics_executed=False)


def main():
    p=argparse.ArgumentParser(description=__doc__);subs=p.add_subparsers(dest='command',required=True)
    for name in ('collect','prepare','verify'):
        x=subs.add_parser(name)
        for arg in ('source','source-sha256'):x.add_argument('--'+arg,required=True)
        if name in ('collect','prepare'):x.add_argument('--output',required=True)
        if name!='collect':
            for arg in ('closure','closure-sha256'):x.add_argument('--'+arg,required=True)
        if name=='verify':
            for arg in ('authority','authority-sha256'):x.add_argument('--'+arg,required=True)
    a=p.parse_args()
    if a.command=='collect':r=collect(a.source,a.source_sha256,a.output)
    elif a.command=='prepare':r=prepare(a.source,a.source_sha256,a.closure,a.closure_sha256,a.output)
    else:r=verify(a.source,a.source_sha256,a.closure,a.closure_sha256,a.authority,a.authority_sha256)
    print(launch.host.raw(r).decode())


if __name__=='__main__':main()
