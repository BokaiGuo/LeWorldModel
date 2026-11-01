"""Frame13-only saved geometry permutation diagnostic; no env/Body/Space/model import.

Permutation memberships are conditional on reconstructed sincos coefficients.
Historical cached transform equivalence is UNKNOWN, never promoted by a match.
"""
from __future__ import annotations

import argparse
import ctypes
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import platform
import struct
import subprocess
import sys

import numpy as np
import shapely
from shapely.geometry import MultiPolygon, Polygon

NAME = 'agent_p5_saved_geometry_order_20261028_v1'
ROOT = Path(__file__).resolve().parents[2]
FAILED = ROOT/'results/agent_p5_replay_target_renderer_20261028_v1_render_MAIN01/report.json'
FAILED_SHA = '1b02ed4a2ca136b4dc7f8549073ff030620df8587d654eb2a07ca86036cad0c6'
PREP = ROOT/'results/agent_p5_replay_target_renderer_20261028_v1_prepare_MAIN01.json'
PREP_SHA = 'c2ecf01381cf339bf8f7e89fa428f4f62ac46075b10458ee7cfc8deac661941d'
ENV = ROOT/'external/jepa-wms/evals/simu_env_planning/envs/pusht_env/pusht_env.py'
ENV_SHA = '6ba34790b6c2ce017a76322c2f1c205ef97d4d3b08a620f844a57075c608a723'
CHIP = ROOT/'.jepa-physics-site/pymunk/_chipmunk.abi3.so'
CHIP_SHA = '1224ec1d02916a7001d58e5a7c558f79a7941d3b6132cdfbe58b24665daf3bce'
SYMBOLS = ('cpBodyLocalToWorld','cpBodyGetPosition','cpBodyGetAngle','cpBodyUpdatePosition','cpBodySetAngle')
ORDERS = ((0,1),(1,0))
CAP = 2*1024**2


def require(ok, message):
    if not ok: raise ValueError(message)


def canonical(value):
    return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()


def sha(path):
    p = Path(path)
    require(p.is_file() and not any(q.is_symlink() for q in (p,*p.parents)), 'regular nonsymlink input')
    h = hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024**2),b''): h.update(b)
    return h.hexdigest()


def read(path, expected):
    require(sha(path)==expected and Path(path).stat().st_size<=16*1024**2, 'external SHA/read cap')
    def pairs(items):
        out = {}
        for k,v in items:
            require(k not in out,'duplicate JSON key'); out[k]=v
        return out
    return json.loads(Path(path).read_bytes(),object_pairs_hook=pairs,
                      parse_constant=lambda s:require(False,'nonfinite JSON'))


def bits(value):
    require(isinstance(value,(float,int,np.floating)) and math.isfinite(value),'finite scalar')
    return struct.pack('>d',float(value)).hex()


def exact(a,b):
    return bits(a)==bits(b)


def finite_vector(value, n):
    require(isinstance(value,list) and len(value)==n,'exact vector length')
    require(all(isinstance(v,(float,int)) and not isinstance(v,bool) and math.isfinite(v) for v in value),'finite vector')
    return [float(v) for v in value]


def extract(snapshot):
    polys = [s for s in snapshot['shapes'] if s['body']=='block']
    require(len(polys)==2 and all(s['type']=='Poly' and s['radius']==0. for s in polys),'two unrounded native T polygons')
    vertices = [[finite_vector(v,2) for v in s['vertices']] for s in polys]
    require(all(len(v)==4 for v in vertices),'exact two quadrilaterals')
    block = snapshot['bodies']['block']
    position = finite_vector(block['position'],2)
    require(all(x>0 for x in position),'positive translations: GetPosition zero-product signed-zero ambiguity excluded')
    angle = float(block['angle']); require(math.isfinite(angle),'finite angle')
    goal = snapshot['env']['goal_pose']
    require(goal['dtype']=='<f8' and goal['shape']==[3],'goal float64 pose')
    goal = finite_vector(goal['values'],3)
    require(all(x>0 for x in goal[:2]),'positive original goal translations')
    threshold = snapshot['env']['success_threshold']
    require(isinstance(threshold,float) and math.isfinite(threshold) and threshold>0,'original native threshold')
    return dict(vertices=vertices,block_position=position,block_angle=angle,goal=goal,
                center_of_gravity=finite_vector(block['center_of_gravity'],2),threshold=threshold)


def sincos_runtime():
    """Current-process libm sincos only; DOES NOT load the physics shared object."""
    process = ctypes.CDLL(None)
    fn = process.sincos
    fn.argtypes = (ctypes.c_double,ctypes.POINTER(ctypes.c_double),ctypes.POINTER(ctypes.c_double))
    fn.restype = None
    class DlInfo(ctypes.Structure):
        _fields_ = [('dli_fname',ctypes.c_char_p),('dli_fbase',ctypes.c_void_p),
                    ('dli_sname',ctypes.c_char_p),('dli_saddr',ctypes.c_void_p)]
    dladdr = process.dladdr
    dladdr.argtypes = (ctypes.c_void_p,ctypes.POINTER(DlInfo)); dladdr.restype = ctypes.c_int
    info = DlInfo()
    require(dladdr(ctypes.cast(fn,ctypes.c_void_p),ctypes.byref(info))!=0 and info.dli_fname,'resolved sincos library')
    path = Path(os.fsdecode(info.dli_fname)).resolve()
    def evaluate(angle):
        s,c = ctypes.c_double(),ctypes.c_double()
        fn(float(angle),ctypes.byref(s),ctypes.byref(c))
        require(math.isfinite(s.value) and math.isfinite(c.value),'finite native libm result')
        return s.value,c.value
    return evaluate,dict(path=str(path),sha256=sha(path),
        symbol=info.dli_sname.decode() if info.dli_sname else None,
        historic_loaded_bytes_attested=False)


def transform(vertices, position, angle, sincos):
    """Mirror pinned SSE2: (c*y+a*x)+tx, (d*y+b*x)+ty; no FMA/CoG re-subtraction.

    These coefficients are reconstructed, not extracted from historic cpBody.
    """
    s,c = sincos(angle)
    coefficients = [c,s,-s,c,*position]
    a,b,cc,d,tx,ty = coefficients
    out = []
    for x,y in vertices:
        # Separate Python float binary64 operations intentionally match mulpd/addpd order.
        px = cc*y; qx = a*x; sx = px+qx; wx = sx+tx
        py = d*y; qy = b*x; sy = py+qy; wy = sy+ty
        require(math.isfinite(wx) and math.isfinite(wy),'finite transformed vertex')
        out.append([wx,wy])
    return out,dict(values=coefficients,bits=[bits(v) for v in coefficients])


def permutations(snapshot, sincos):
    data = extract(snapshot)
    block,goal = [],[]; transforms = {}
    for vs in data['vertices']:
        b,bt = transform(vs,data['block_position'],data['block_angle'],sincos)
        g,gt = transform(vs,data['goal'][:2],data['goal'][2],sincos)
        block.append(b); goal.append(g)
        transforms = dict(block=bt,goal=gt)
    block_polys = [Polygon(v+[v[0]]) for v in block]
    goal_polys = [Polygon(v+[v[0]]) for v in goal]
    results = []
    history = snapshot['env']['coverage_arr']
    require(len(history)==13 and all(math.isfinite(v) for v in history),'only actual frame13 history')
    for gi,bi in itertools.product(ORDERS,ORDERS):
        gg = MultiPolygon([goal_polys[i] for i in gi])
        bg = MultiPolygon([block_polys[i] for i in bi])
        intersection = gg.intersection(bg)
        area = intersection.area; denominator = gg.area
        require(denominator>0,'positive native goal area')
        coverage = area/denominator
        reward = float(np.clip(coverage/data['threshold'],0,1))
        values = dict(coverage=coverage,reward=reward,final_coverage=coverage,
                      max_coverage=max([*history[:12],coverage]))
        results.append(dict(goal_order=list(gi),block_order=list(bi),
            native_input_validity=dict(goal=bool(gg.is_valid),block=bool(bg.is_valid)),
            intersection_area=area,goal_area=denominator,values=values,
            bits={k:bits(v) for k,v in values.items()},intersection_wkb_hex=intersection.wkb_hex))
    return dict(transforms=transforms,block_world_vertices=block,goal_world_vertices=goal,
                world_vertex_bits=dict(block=[[[bits(x) for x in v] for v in vs] for vs in block],
                    goal=[[[bits(x) for x in v] for v in vs] for vs in goal]),permutations=results)


def recorded(snapshot, transition):
    require(transition['kind']=='step' and transition['done'] is False,'original returned native step')
    c = snapshot['env']['coverage_arr'][-1]
    values = dict(coverage=c,reward=transition['reward'],final_coverage=transition['info']['final_coverage'],
                  max_coverage=transition['info']['max_coverage'])
    require(exact(c,values['final_coverage']),'snapshot/transition coverage identity')
    return dict(values=values,bits={k:bits(v) for k,v in values.items()})


def compare(results, observed):
    # Exact bit memberships only, never tolerance, rounding or nearest candidate selection.
    matches = [dict(goal_order=r['goal_order'],block_order=r['block_order']) for r in results
               if all(exact(r['values'][k],v) for k,v in observed['values'].items())]
    return dict(conditional_exact_matches=matches,conditional_exact_member=bool(matches),
        native_transform_certified=False,legal_native_permutation_member=None,
        reason='historic cached matrix/world vertices and resolved sincos bytes not saved; agreement cannot certify their reconstruction')


def build():
    require(not any(k=='torch' or k.startswith(('torch.','pymunk')) for k in sys.modules),'no model/physics imports')
    bindings = {}
    def bound(path, expected):
        bindings[str(path)] = expected
        return read(path,expected)
    report = bound(FAILED,FAILED_SHA); p = bound(PREP,PREP_SHA)
    require(report['prepare_sha256']==PREP_SHA and report['prepare_digest']==hashlib.sha256(canonical(p)).hexdigest(), 'failed execution -> exact prepare')
    require(report['status']=='partial_engineering_failure' and not report['all_baselines_passed'], 'v1 failure must stay failure')
    require(sha(ENV)==ENV_SHA and sha(CHIP)==CHIP_SHA,'exact native source/binary')
    bindings.update({str(ENV):ENV_SHA,str(CHIP):CHIP_SHA})
    old_report_path = Path(p['original_physical_report'])
    old_report = bound(old_report_path,p['bindings'][str(old_report_path)])
    require(old_report['rows'][0]['arm']=='xy' and old_report['rows'][0]['records'][13]['frame']==13, 'fixed original xy13 row')
    require(report['rows'][0]['state_id']=='xy' and report['rows'][0]['records'][13]['frame']==13,'fixed failed xy13 row')
    original = p['frames']['xy'][13]
    observed = report['rows'][0]['records'][13]['files']
    snapshots = {}; transitions = {}
    for label,records,base in (('original',original,None),('replay',observed,FAILED.parent)):
        for name in ('rgb.npy','state.npy','proprio.npy','full_public_state.json','transition.json'):
            rec = records[name]; path = Path(rec['path']) if base is None else base/rec['path']
            require(sha(path)==rec['sha256'] and path.stat().st_size==rec['bytes'],'actual frame13 artifact')
            bindings[str(path)] = rec['sha256']
            if name.endswith('.json'):
                value = read(path,rec['sha256'])
                (snapshots if name=='full_public_state.json' else transitions)[label] = value
    for name in ('rgb.npy','state.npy','proprio.npy'):
        require(Path(original[name]['path']).read_bytes() == (FAILED.parent/observed[name]['path']).read_bytes(),'saved RGB/state/proprio exact')
    old,new = snapshots['original'],snapshots['replay']
    stripped_old = json.loads(canonical(old)); stripped_new = json.loads(canonical(new))
    stripped_old['env']['coverage_arr'].pop(); stripped_new['env']['coverage_arr'].pop()
    require(stripped_old==stripped_new,'only current coverage may differ in full public snapshot')
    require(extract(old)==extract(new),'all geometric inputs exact')
    sincos,libm = sincos_runtime()
    bindings[libm['path']] = libm['sha256']
    # Read-only binary evidence, no loading of _chipmunk or construction of native objects.
    disassembly = {}
    require(platform.machine()=='x86_64' and sys.float_info.mant_dig==53,'pinned SSE2/binary64 host')
    for symbol in SYMBOLS:
        text = subprocess.check_output(['objdump','-d','--disassemble='+symbol,str(CHIP)],timeout=10,text=True)
        require('<'+symbol+'>:' in text,'exact exported binary symbol')
        disassembly[symbol] = text
    require('mulpd' in disassembly['cpBodyLocalToWorld'] and 'addpd' in disassembly['cpBodyLocalToWorld']
            and 'sincos@plt' in disassembly['cpBodyUpdatePosition'] and 'sincos@plt' in disassembly['cpBodySetAngle'], 'bound native arithmetic evidence')
    geometry = permutations(old,sincos)
    observations = {k:recorded(snapshots[k],transitions[k]) for k in ('original','replay')}
    memberships = {k:compare(geometry['permutations'],observations[k]) for k in observations}
    # Actual imported Shapely artifacts + GEOS bytes, without reading unrelated assets.
    probe = bound(FAILED.parent/'runtime_probe.json',report['artifacts']['runtime_probe.json']['sha256'])
    require(shapely.__version__==probe['versions']['shapely']['version'],'same declared Shapely version')
    for name,module in list(sys.modules.items()):
        filename = getattr(module,'__file__',None)
        if name.startswith('shapely') and filename and Path(filename).is_file():
            path = Path(filename).resolve(); h = sha(path); bindings[str(path)] = h
            historical = probe.get('imported_dependency_hashes',{}).get(str(path))
            require(historical is None or historical==h,'loaded Shapely file differs from replay probe')
    geos_dir = Path(shapely.__file__).resolve().parent.parent/'shapely.libs'
    require(geos_dir.is_dir(),'installed GEOS distribution assets')
    for path in sorted(geos_dir.glob('*.so*')): bindings[str(path)] = sha(path)
    for path in (Path(__file__).resolve(),ROOT/f'tests/test_{NAME}.py',ROOT/f'innerdocs/{NAME}.md'):
        bindings[str(path)] = sha(path)
    require(all(sha(f)==h for f,h in bindings.items()),'stable complete diagnostic inputs')
    return dict(schema=NAME+'.report',status='saved_frame13_order_diagnostic_transform_unknown',
        scope=dict(arm='xy',frame=13,states=1,permutations=4,no_other_frames=True),
        bindings=bindings,geometry=geometry,observations=observations,membership=memberships,
        transform_evidence=dict(binary_sha256=CHIP_SHA,disassembly=disassembly,current_sincos=libm,
            formula='(c*y+a*x)+tx ; (d*y+b*x)+ty; a=d=cos(angle), b=sin(angle), c=-sin(angle)',
            position='public GetPosition is transform(0,0), not CoG world point; no extra CoG subtraction',
            native_cached_matrix_saved=False,native_world_vertices_saved=False,
            exact_historical_transform_equivalence=None,
            caveat='SSE arithmetic mirrored under reconstructed current coefficients; no captured native matrix/vertices or historical libm resolution to independently certify those coefficients'),
        shapely=dict(version=shapely.__version__,geos_version=shapely.geos_version_string),
        tolerances=None,renderer_v1_release=False,scientific_admission=False,
        physics_calls=0,neural_calls=0,body_or_space_constructed=False,source_modified=False)


def save(path, value):
    path = Path(path).absolute(); data = canonical(value)+b'\n'
    require(len(data)<=CAP and path.parent==ROOT/'results' and path.name.startswith(NAME+'_')
            and not any(p.is_symlink() for p in (path,*path.parents)), 'exclusive own small results file')
    with path.open('xb') as f: f.write(data); f.flush(); os.fsync(f.fileno())
    fd = os.open(path.parent,os.O_RDONLY|os.O_DIRECTORY)
    try: os.fsync(fd)
    finally: os.close(fd)


def verify(path, expected):
    r = read(path,expected)
    require(r==build(),'complete deterministic diagnostic replay')
    return dict(schema=NAME+'.receipt',status='saved_diagnostic_verified_not_native_certification',
                report_sha256=expected,legal_native_permutation_member=None,renderer_v1_release=False,
                scientific_admission=False,physics_calls=0,neural_calls=0)


def main():
    parser = argparse.ArgumentParser(); sub = parser.add_subparsers(dest='command',required=True)
    p = sub.add_parser('analyze'); p.add_argument('--output',required=True)
    p = sub.add_parser('verify'); p.add_argument('--report',required=True); p.add_argument('--sha',required=True)
    x = parser.parse_args()
    if x.command=='analyze':
        r = build(); save(x.output,r)
        print(canonical(dict(report=str(Path(x.output).absolute()),sha256=sha(x.output),status=r['status'],membership=r['membership'])).decode())
    else: print(canonical(verify(x.report,x.sha)).decode())


if __name__=='__main__': main()
