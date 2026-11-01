"""Preseal bounded static project source roots without importing their modules."""
import argparse
import hashlib
import os
from pathlib import Path
import sys

from experiments.spectral_reachability import agent_e3_installed_closure_20261029_v1 as installed

host=installed.host
launch=installed.launch
old=installed.old
need=installed.need
ROOT=old.ROOT
NAME='agent_e3_project_closure_20261029_v1'
SOURCE_SHA='c306e309352d35f3ce086aacb5bee781eca28e909f885aa22c615cf48c2599c7'
DIRS=('experiments','scripts','tests','external')
SUFFIXES=('.py','.so')
SKIP=('__pycache__','.git','.pytest_cache','.mypy_cache','.ruff_cache')
MAX_FILES=100000


def selected(name): return name.endswith(SUFFIXES)


def walk(root):
    paths=[]
    for p in root.iterdir():
        if selected(p.name):
            need(not p.is_symlink() and p.is_file() and p==p.resolve(),'canonical root source')
            paths.append(str(p))
    for name in DIRS:
        base=root/name
        need(base.is_dir() and not base.is_symlink() and base==base.resolve(),'canonical source root')
        for directory,dirs,files in os.walk(base,followlinks=False):
            dirs[:]=[d for d in dirs if d not in SKIP]
            for d in dirs: need(not (Path(directory)/d).is_symlink(),'source directory symlink')
            for name in files:
                if not selected(name): continue
                p=Path(directory)/name
                need(not p.is_symlink() and p.is_file() and p==p.resolve(),'canonical source file')
                paths.append(str(p));need(len(paths)<=MAX_FILES,'source count bound')
    need(len(paths)==len(set(paths)) and bool(paths),'nonempty unique sources')
    return sorted(paths)


def scan(root):
    paths=[]
    with os.scandir(root) as entries:
        for item in entries:
            if selected(item.name):
                need(not item.is_symlink() and item.is_file(follow_symlinks=False),'canonical root source')
                paths.append(item.path)
    pending=[]
    for name in DIRS:
        p=root/name;need(p.is_dir() and not p.is_symlink() and p==p.resolve(),'canonical source root');pending.append(p)
    while pending:
        directory=pending.pop()
        with os.scandir(directory) as entries:
            for item in entries:
                if item.name in SKIP:continue
                if item.is_symlink():
                    need(not item.is_dir(follow_symlinks=True) and not selected(item.name),'source symlink')
                elif item.is_dir(follow_symlinks=False):pending.append(Path(item.path))
                elif selected(item.name):
                    need(item.is_file(follow_symlinks=False),'regular source');paths.append(item.path)
                    need(len(paths)<=MAX_FILES,'source count bound')
    need(len(paths)==len(set(paths)) and bool(paths),'nonempty unique sources')
    return sorted(paths)


def hash_files(paths): return {p:old.file_hash(p) for p in paths}


def stable(root):
    paths=walk(root); pins=hash_files(paths)
    need(scan(root)==paths, 'project source set changed during scan')
    need(hash_files(paths)==pins, 'project source bytes changed during scan')
    need(walk(root)==paths,'project source set changed after hashing')
    return pins


def collect(source,source_sha,output):
    need(source_sha==SOURCE_SHA,'fixed installed authority SHA')
    host.authority(source,source_sha);installed.disk_check(output)
    pins=stable(ROOT)
    need(str(ROOT/'experiments/research_runtime/__init__.py') in pins,'MAIN04 missing file included')
    value=dict(schema=NAME+'.closure',source=dict(path=source,sha256=source_sha),root=str(ROOT),
        python=str(Path(sys.executable).resolve()),directories=list(DIRS),suffixes=list(SUFFIXES),
        excluded=list(SKIP),files=pins,scope='root files and four named static source trees',
        concurrent_mutations='not supported; double scans are detection not a filesystem snapshot',
        model_executed=False,physics_executed=False)
    need(launch.host.sha(source)==source_sha,'source authority stable')
    sha=old.write_new(output,value)
    return dict(path=output,sha256=sha,files=len(pins),scientific_admission=False)


def verify_closure(path,sha,source,source_sha):
    need(source_sha==SOURCE_SHA,'fixed installed authority SHA')
    value=launch.host.read(path,sha)
    need(set(value)=={'schema','source','root','python','directories','suffixes','excluded','files','scope',
                       'concurrent_mutations','model_executed','physics_executed'} and
         value['schema']==NAME+'.closure' and value['source']==dict(path=source,sha256=source_sha) and
         value['root']==str(ROOT) and value['python']==str(Path(sys.executable).resolve()) and
         value['directories']==list(DIRS) and value['suffixes']==list(SUFFIXES) and value['excluded']==list(SKIP) and
         value['scope']=='root files and four named static source trees' and
         value['concurrent_mutations']=='not supported; double scans are detection not a filesystem snapshot' and
         value['model_executed'] is False and value['physics_executed'] is False,'closure schema')
    pins=value['files']; need(type(pins) is dict and 0<len(pins)<=MAX_FILES,'bounded source pins')
    paths=scan(ROOT);need(paths==sorted(pins),'project file set changed')
    for p in paths:
        h=pins[p];need(type(h) is str and len(h)==64 and all(c in '0123456789abcdef' for c in h),'hash schema')
        need(old.file_hash(p)==h,'project source drift: '+p)
    need(walk(ROOT)==paths and hash_files(paths)==pins and scan(ROOT)==paths,'project source changed during verify')
    need(launch.host.sha(source)==source_sha and launch.host.sha(path)==sha,'stable closure/source')
    return pins


def reconstruct(binding,prior,pins):
    value=dict(prior,additional_sources=dict(prior['additional_sources']))
    base=binding['prepared']['bindings_sha256']
    # Check both maps, including when the same path appears in both.
    for p,h in pins.items():
        a=base.get(p);b=value['additional_sources'].get(p)
        need((a is None or a==h) and (b is None or b==h),'source pin collision: '+p)
        if a is None:value['additional_sources'][p]=h
    merged=dict(base);launch.host.source.merge(merged,value['additional_sources'])
    return value


def build(source,source_sha,closure,closure_sha):
    binding=host.authority(source,source_sha)
    pins=verify_closure(closure,closure_sha,source,source_sha)
    value=reconstruct(binding,launch.host.read(source,source_sha),pins)
    need(launch.host.sha(source)==source_sha and launch.host.sha(closure)==closure_sha,'stable inputs')
    return value


def prepare(source,source_sha,closure,closure_sha,output):
    installed.disk_check(output);value=build(source,source_sha,closure,closure_sha)
    digest=old.write_new(output,value)
    return dict(path=output,sha256=digest,scientific_admission=False,full_upstream_strict_rebuilt=False)


def verify(source,source_sha,closure,closure_sha,authority,authority_sha):
    binding=host.authority(source,source_sha)
    pins=verify_closure(closure,closure_sha,source,source_sha)
    prior=launch.host.read(source,source_sha);actual=launch.host.read(authority,authority_sha)
    expected=reconstruct(binding,prior,pins)
    need(actual==expected,'whole authority mismatch')
    host.authority(authority,authority_sha)
    need(launch.host.sha(source)==source_sha and launch.host.sha(closure)==closure_sha and
         launch.host.sha(authority)==authority_sha,'stable final inputs')
    return dict(verified=True,files=len(pins),scientific_admission=False,model_executed=False,physics_executed=False)


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
