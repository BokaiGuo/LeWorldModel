"""Fixed saved algebraic negative controls; never controller inputs or forecasts."""
from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
import shutil
import time

import numpy as np
from experiments.spectral_reachability import agent_p2_six_target_error_decomposition_20261029_v1 as base

NAME='agent_p2_six_target_algebraic_controls_20261029_v1'
ROOT=Path(__file__).resolve().parents[2]
PROTOCOL='innerdocs/agent_p2_six_target_local_intervention_protocol_20261029_v1.md'
PROTOCOL_SHA='c9c03fe6df2188be7865ad55d8aae9fd6b2665c56e7bb57329639a20329a712c'
BASE_SHA='6aacaf5fb77546c6987f8d3f3de9318716c66953742ab944d4cdba8f50cf7b9f'
SIGNS=(-1,0,1)
AMPLITUDES=(-.5,0.,.5,1.)
DELTA=2.**-8
CAP,FLOOR,WALL=16*1024**2,200*1024**2,120
need,sha,encode,equal,digest,read=base.need,base.sha,base.encode,base.equal,base.digest,base.read


def closure():
    need(sha(Path(base.__file__))==BASE_SHA,'frozen local reader/arithmetic code')
    need(sha(ROOT/PROTOCOL)==PROTOCOL_SHA,'approved section2 protocol')
    out=base.closure()
    out[str(ROOT/PROTOCOL)]=PROTOCOL_SHA
    for p in (Path(__file__).resolve(),ROOT/'tests'/('test_'+NAME+'.py'),ROOT/'innerdocs'/(NAME+'.md')):
        out[str(p)]=sha(p)
    return out


def vec64(x):
    need(type(x) is np.ndarray and x.dtype==np.float64 and x.shape==(192,) and np.isfinite(x).all(),'finite f64 derived vector192')
    return x


def mean64(x):
    vec64(x);v=math.fsum(float(a) for a in x)/192
    need(math.isfinite(v),'finite f64 reduction');return v


def mse64(x,y):
    delta=vec64(x)-vec64(y);return mean64(delta*delta)


def record64(x):
    vec64(x)
    return dict(values=x.tolist(),dtype='float64',shape=[192],sha256=digest(x.tobytes()),byteorder=np.dtype(np.float64).byteorder)


def geometry(x,aux,g):
    if x is None:return None
    own=mse64(x,aux);gap=mse64(aux,g);cross=2*mean64((x-aux)*(aux-g));q=mse64(x,g)
    total=math.fsum((own,gap,cross))
    return dict(tracking_MSE=own,target_goal_geometry=gap,signed_cross_term=cross,
        original_task_MSE=q,reconstructed_task_MSE=total,identity_roundoff_residual=q-total)


def residual(p,z,r):
    if z is None:return None
    e=p-z;energy=mean64(e*e);cross=2*mean64(e*(z-r));q=mse64(p,r);actual=mse64(z,r)
    total=math.fsum((actual,energy,cross))
    return dict(energy=energy,signed_cross_term=cross,derived_MSE=q,actual_MSE=actual,
        direct_difference=q-actual,reconstructed_MSE=total,identity_roundoff_residual=q-total)


def controls(pred,actual,target,goal):
    # Only original inputs pass f32 validation; NO f32 conversion of derived vectors.
    p,t,g=(base.vector(x) for x in (pred,target,goal))
    z=None if actual is None else base.vector(actual)
    refs={'task_goal':g,'tracking_target':t}
    original={k:residual(p,z,v) for k,v in refs.items()}
    geometries=[]
    for s in SIGNS:
        aux=t.copy();aux[0]=aux[0]+s*DELTA
        geometries.append(dict(sign=s,auxiliary_target=record64(aux),
            model=geometry(p,aux,g),actual=geometry(z,aux,g),
            unchanged_task_residual=original['task_goal']))
    amplitudes=[]
    for a in AMPLITUDES:
        if z is None:
            amplitudes.append(dict(amplitude=a,derived_prediction=None,by_reference=None,
                original_prediction_distance=None,reason='actual_encoding_missing'))
            continue
        derived=z+a*(p-z);by_ref={}
        for name,r in refs.items():
            v=residual(derived,z,r);v0=original[name]
            expected_energy=(a*a)*v0['energy'];expected_cross=a*v0['signed_cross_term']
            expected_difference=math.fsum((expected_energy,expected_cross))
            by_ref[name]=dict(**v,expected_scaled_energy=expected_energy,expected_scaled_cross_term=expected_cross,
                expected_scaled_difference=expected_difference,energy_scaling_roundoff=v['energy']-expected_energy,
                cross_scaling_roundoff=v['signed_cross_term']-expected_cross,
                difference_scaling_roundoff=v['direct_difference']-expected_difference)
        amplitudes.append(dict(amplitude=a,derived_prediction=record64(derived),by_reference=by_ref,
            original_prediction_distance=mse64(derived,p),reason=None))
    return dict(original_residual=original,geometry_controls=geometries,residual_controls=amplitudes,
        original_target_goal_geometry=mse64(t,g),actual_missing=z is None)


def panel(p,r,c,latents):
    original,_=base.original_rows(p,r,latents)
    need(equal(original,c['rows']),'independently recomputed original24 MAIN scalar/identity rows')
    rows=[]
    for spec,z,scalar in zip(p['rows'],latents,original,strict=True):
        pred,target,goal=(base.f32(spec[k]) for k in ('predicted192','target192','goal192'))
        rows.append(dict(index=spec['index'],slot=spec['slot'],root=spec['root'],arm=spec['arm'],cp_public_step=spec['cp_public_step'],
            reason=spec['reason'],source_time=spec['source_time'],target_origin=spec['target_origin'],
            source=spec['pixel_source'],prediction_source=spec['prediction_pointer'],goal_source=spec['goal_source'],
            raw15_sha256=spec['raw15_sha256'],original_z_sha256=spec['original_z_sha256'],original_a_sha256=spec['original_a_sha256'],
            inputs=dict(prediction_sha256=digest(pred.tobytes()),actual_sha256=None if z is None else digest(z.tobytes()),
                target_sha256=digest(target.tobytes()),goal_sha256=digest(goal.tobytes()),dtype='float32'),
            original_scalars=scalar,**controls(pred,z,target,goal)))
    return dict(denominator=dict(slots=6,positions=24,geometry_controls=72,residual_controls=96),rows=rows,
        missing_actual=sum(v is None for v in latents),counterfactual_only=True,controller_inputs=False,
        winner=None,threshold=None,scientific_admission=False,P1_admission=False,P2_mechanism=False,
        known_roots=2,fresh=False,prior_physical_time=None,equal_generation_budget=False)


def build(refs=None,clock=time.monotonic):
    started=clock()
    def check():need(clock()-started<=WALL,'saved controls whole120 wall')
    refs=base.actual_refs() if refs is None else refs
    need(equal(refs,base.actual_refs()),'fixed actual four authorities')
    docs={}
    for k in ('prepare','report','completion','receipt'):
        check();docs[k]=read(refs[k]['path'],refs[k]['sha256'])
    p,r,d,c=(docs[k] for k in ('prepare','report','completion','receipt'))
    base.documents(p,r,d,c,refs)
    need(len(p['rows'])==len(r['rows'])==24,'all24 roster, no partial fallback')
    folder=Path(refs['report']['path']).parent;sealpath=folder/'native_seal.json'
    seal=read(sealpath,r['native_seal_sha256'])
    need(r['artifact_sha256'].get('native_seal.json')==r['native_seal_sha256'] and
        equal(seal,{k:r[k] for k in ('rows','modules','counts','errors')}),'original native seal exact')
    sources={v['path']:v['sha256'] for v in refs.values()};sources[str(sealpath)]=r['native_seal_sha256'];sources.update(closure())
    base.source_join(p,r,check);latents=[]
    for row in r['rows']:
        check();desc=row['latent'];latents.append(None if desc is None else base.load_latent(folder,desc,r['artifact_sha256']))
        if desc is not None:sources[str(folder/desc['file'])]=desc['sha256']
    result=panel(p,r,c,latents)
    for path,h in sources.items():check();need(sha(path)==h,'all local source/input bytes unchanged')
    check()
    return dict(schema=NAME+'.analysis',authorities=refs,local_sources=sources,**result,
        authentication='MAIN RGB strict imported; independent local input/scalar arithmetic only',
        upstream_authentication_imported=True,independent_fullstrict=False,independent_gradient=False,
        approved_protocol_sha256=PROTOCOL_SHA,implemented_section=2,action_interventions=False,
        new_calls=dict(encoder=0,predict=0,AD=0,physics=0),FLOPs=None,
        arithmetic='original f32 -> f64; derived vectors remain f64; fsum/192; signed terms and roundoff unmodified')


def write(path,value):
    need(path is not None,'output path required');p=Path(path).absolute();raw=encode(value)
    need(p.parent.resolve()==ROOT/'results' and p.name.startswith(NAME+'_') and not p.is_symlink() and len(raw)<=CAP and
        shutil.disk_usage(p.parent).free>=FLOOR+len(raw),'exclusive own16MiB metadata / free200MiB')
    with p.open('xb') as f:need(f.write(raw)==len(raw),'complete output write');f.flush();os.fsync(f.fileno())
    fd=os.open(p.parent,os.O_RDONLY)
    try:os.fsync(fd)
    finally:os.close(fd)
    return dict(path=str(p),sha256=sha(p))


def verify(path,h):
    v=read(path,h);need(equal(v,build(v['authorities'])),'entire local control output reconstructed')
    return dict(verified=True,independent_fullstrict=False,new_neural=0)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('command',choices=('build','verify'))
    parser.add_argument('--output',type=Path);parser.add_argument('--summary',type=Path);parser.add_argument('--summary-sha256')
    a=parser.parse_args(argv);start=time.monotonic()
    result=write(a.output,build()) if a.command=='build' else verify(a.summary,a.summary_sha256)
    need(time.monotonic()-start<=WALL,'inclusive read/arithmetic/fsync120 wall; failed artifacts retained')
    print(encode(result).decode())


if __name__=='__main__':main()
