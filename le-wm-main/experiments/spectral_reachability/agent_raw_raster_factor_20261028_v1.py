"""Fixed80 saved raster-input factorization; no simulation, model or selector.

AST extracts sealed geometry constructors and drawing functions. The only native
operations are detached Pymunk geometry (Body/Poly/Circle), never Space/env/step.
Snapshot does NOT seal debug contact caches; reconstructed shape inputs are not
misrepresented as an authenticated full renderer callback stream.
"""
from __future__ import annotations

import argparse
import ast
from collections import namedtuple
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import shutil
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
NAME = 'agent_raw_raster_factor_20261028_v1'
AUTH = 'MAIN_RAW_RASTER_DETACHED_GEOMETRY_ONLY'
SITE = ROOT/'.venv/lib/python3.12/site-packages'
UTILS = SITE/'stable_worldmodel/envs/utils.py'
ENV = SITE/'stable_worldmodel/envs/pusht/env.py'
UTILS_SHA = '2e5d98a5c13227cba6f5574bf72b6eab149ee5b3783df0481b8d240f34047f56'
ENV_SHA = 'c6d2104adc368ff1c352edc0e35ac74da9f07fda3a748ddbef0ba6b7a2d1daa2'
PINS = {
    'physical_prepare': ('agent_observed_raw_coordinate_physical_20261028_v1_prepare_MAIN01.json',
                         '47ee177e40cb1247a73e7d08bfe5f31ce5d50884a09a5e0bce909027a1b63bd1'),
    'physical_report': ('agent_observed_raw_coordinate_physical_20261028_v1_MAIN01/report.json',
                        'c4610c7f495137cbda7d70b147c74670db6f85339561da611ce5bee152d7bb10'),
    'physical_receipt': ('agent_observed_raw_coordinate_physical_20261028_v1_consume_MAIN01.json',
                         '8df105747cf6835056d41ca1b1fd80f7f5b4170f7f527ec538213addc4c9d0bb'),
    'observability': ('main_raw_observability_audit_20261028_v1_MAIN01.json',
                      '1411a87e95d36ae84971b27e2bbaaed848fbc5c277e9675d591d374125614d3f'),
}
CAP = 32*1024**2
FLOOR = 200*1024**2
ORDER = ((0,'off'), (0,'on'), (1,'off'), (1,'on'))
LABELS = ('same_draw_inputs_sameRGB', 'changed_draw_inputs_sameRGB', 'changedRGB',
          'unknown_draw_inputs_sameRGB', 'unknown_draw_inputs_changedRGB')
UNSEALED = ['Chipmunk transformed-shape/debug iteration cache',
            'collision-point/constraint debug callbacks and their order',
            'original pre-resize 512x512 surface (only224 PNG saved)']


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    p=Path(path)
    require(p.is_file() and not p.is_symlink(), 'regular source/input file')
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024**2), b''): h.update(b)
    return h.hexdigest()


def literal(x):
    if isinstance(x,np.ndarray): return literal(x.tolist())
    if isinstance(x,np.generic): return x.item()
    if isinstance(x,(tuple,list)): return [literal(v) for v in x]
    if isinstance(x,dict): return {k:literal(v) for k,v in x.items()}
    require(x is None or type(x) in (int,float,str,bool),'literal recording only')
    return x


def raw(x):
    return json.dumps(literal(x),sort_keys=True,separators=(',',':'),allow_nan=False).encode()


def read(path, expected, cap=128*1024**2):
    require(Path(path).stat().st_size<=cap and sha(path)==expected,'pinned complete input SHA/cap')
    result=json.loads(Path(path).read_bytes())
    require(sha(path)==expected,'input bytes unchanged after read')
    return result


def schedule():
    return [dict(slot=s,root_slot=ORDER[s][0],condition=ORDER[s][1],h=h,coord=c,
                 minus_index=41*s+1+20*hi+2*c,plus_index=41*s+2+20*hi+2*c)
            for s in range(4) for hi,h in enumerate((.01,.001)) for c in range(10)]


class Color(namedtuple('ColorBase','r g b a',defaults=[255])):
    __slots__=()
    def as_int(self): return tuple(round(v) for v in self)


def color(*args):
    x=args if len(args)!=1 else args[0]
    if isinstance(x,str):
        require(x=='LightGray','only source wall named color supported')
        x=(211,211,211,255)
    return Color(*x)


def extract(text, names, class_name=None):
    """Extract whole original function bodies, not a hand-transcribed renderer."""
    tree=ast.parse(text)
    nodes=tree.body if class_name is None else next(
        n for n in tree.body if isinstance(n,ast.ClassDef) and n.name==class_name).body
    selected=[deepcopy(n) for n in nodes if isinstance(n,ast.FunctionDef) and n.name in names]
    require({n.name for n in selected}==set(names),'exact source functions present')
    return selected


def compile_nodes(nodes, namespace):
    # Future annotations allow original Vec2d/pygame annotations without imports.
    module=ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0),*nodes],type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), '<sealed raster AST>', 'exec'),namespace)


class Recorder:
    """pygame.draw sink, not a rasterizer; full integer arguments are retained."""
    def __init__(self): self.commands=[]
    def circle(self,surface,color,center,radius,width=0):
        self.commands.append(literal(dict(op='circle',color=color,center=center,radius=radius,width=width)))
    def polygon(self,surface,color,points,width=0):
        self.commands.append(literal(dict(op='polygon',color=color,points=points,width=width)))
    def lines(self,surface,color,closed,points,width=1):
        self.commands.append(literal(dict(op='lines',color=color,closed=closed,points=points,width=width)))
    def aalines(self,surface,color,closed,points):
        self.commands.append(literal(dict(op='aalines',color=color,closed=closed,points=points)))


def draw_api(text):
    require(hashlib.sha256(text.encode()).hexdigest()==UTILS_SHA,'sealed renderer source before extraction')
    sink=Recorder()
    ns=dict(np=np,pygame=SimpleNamespace(draw=sink),SpaceDebugColor=Color,positive_y_is_up=False)
    compile_nodes(extract(text,('to_pygame','light_color'))+extract(text,
        ('draw_circle','draw_polygon','draw_fat_segment','draw_segment','draw_dot'),'DrawOptions'),ns)
    cls=type('ExtractedDrawOptions',(),{k:ns[k] for k in
        ('draw_circle','draw_polygon','draw_fat_segment','draw_segment','draw_dot')})
    obj=cls();obj.surface=SimpleNamespace(get_height=lambda:512)
    def result(points, radius, rgb, kind):
        sink.commands=[];c=color(rgb)
        if kind=='circle':obj.draw_circle(tuple(points),0,radius,c,c)
        elif kind=='polygon':obj.draw_polygon([tuple(v) for v in points],radius,c,c)
        else:obj.draw_fat_segment(tuple(points[0]),tuple(points[1]),radius,c,c)
        return dict(kind=kind,world=literal(points),radius=radius,
            integer_points=literal(ns['to_pygame'](points,obj.surface) if kind=='circle'
                else [ns['to_pygame'](v,obj.surface) for v in points]),commands=deepcopy(sink.commands))
    return SimpleNamespace(circle=lambda p,r,c:result(p,r,c,'circle'),
        polygon=lambda p,r,c:result(p,r,c,'polygon'),segment=lambda p,r,c:result(p,r,c,'segment'))


def shape_names(text):
    require(hashlib.sha256(text.encode()).hexdigest()==ENV_SHA,'sealed geometry source')
    init=extract(text,('__init__',),'PushT')[0]
    matches=[n for n in ast.walk(init) if isinstance(n,ast.Assign) and any(
        isinstance(t,ast.Attribute) and t.attr=='shapes' for t in n.targets)]
    require(len(matches)==1,'literal actual shape dispatch table')
    return ast.literal_eval(matches[0].value)


def geometry_api(text,pm):
    """Detached native geometry only. No gym/env constructor, Space or step."""
    names=shape_names(text)
    methods=('add_circle','add_tee','add_small_tee','add_plus','add_L','add_Z',
             'add_square','add_I','add_shape','_get_goal_pose_body')
    ns=dict(pymunk=pm,pygame=SimpleNamespace(Color=color),np=np)
    compile_nodes(extract(text,methods,'PushT'),ns)
    cls=type('DetachedSourceGeometry',(),{n:ns[n] for n in methods})
    obj=cls()
    # Original constructors call space.add; this records ownership only. It is
    # intentionally NOT pymunk.Space, so no solver/cache/physics is constructed.
    added=[];obj.space=SimpleNamespace(add=lambda *items:added.extend(items))
    def body(spec):
        i=spec['shape']
        if type(i) is not int or not 0<=i<len(names):
            return None,dict(status='unknown',reason='shape index absent from sealed source dispatch',shape_index=i)
        added.clear()
        b=obj.add_shape(names[i],position=spec['start_position'],angle=spec['angle'],
                        scale=spec['scale'],color=spec['color'])
        shapes=[x for x in added if isinstance(x,(pm.Circle,pm.Poly))]
        require(shapes and all(x.body is b for x in shapes),'source constructor body/shape ownership')
        return (b,shapes),dict(status='known',shape_index=i,shape_name=names[i],requested_scale=spec['scale'])
    return SimpleNamespace(body=body,goal_body=obj._get_goal_pose_body,pm=pm)


def check_snapshot(s):
    a=np.asarray(s['state'],dtype='<f8')
    require(a.shape==(7,) and np.isfinite(a).all() and a.tobytes().hex()==s['state_float64_bits'],
            'full native7 float64 snapshot bits')
    for name in ('agent','block'):
        b=s['bodies'][name]
        x=np.asarray([*b['position'],*b['velocity'],b['unwrapped_angle'],b['angular_velocity'],
                      *b['force'],b['torque']],dtype='<f8')
        require(x.shape==(9,) and np.isfinite(x).all() and x.tobytes().hex()==b['float64_bits']
            and type(b['is_sleeping']) is bool,'full body float64 snapshot bits, no wrapped-angle substitution')
    require(a[:2].tobytes()==np.asarray(s['bodies']['agent']['position'],dtype='<f8').tobytes()
        and a[2:4].tobytes()==np.asarray(s['bodies']['block']['position'],dtype='<f8').tobytes(),
        'native state/body positions agree exactly')


def reconstruct(renderer,snapshot,goal_pose,condition,draw,geometry):
    check_snapshot(snapshot)
    require(condition in ('off','on') and renderer['window_size']==512 and renderer['render_size']==224
        and renderer['render_action'] is False,'original512->224 no action marker renderer')
    v=renderer['variations'];out=dict(status='known',bodies={},scene_constants=dict(
        renderer=deepcopy(renderer),goal_pose=deepcopy(goal_pose),with_target=condition=='on',
        render_goal=bool(v['rendering']['render_goal']) and condition=='on'))
    goal_commands=[]
    for name in ('agent','block'):
        made,meta=geometry.body(v[name])
        if made is None:return dict(status='unknown',reason=meta['reason'],geometry=meta)
        original,shapes=made;b=snapshot['bodies'][name]
        # Body.position exports world(local0). A detached zero-CoG pose body
        # restores that exported origin directly, avoiding a second CoG shift.
        pose=geometry.pm.Body(body_type=geometry.pm.Body.KINEMATIC)
        pose.angle=b['unwrapped_angle'];pose.position=b['position']
        require(np.asarray(pose.local_to_world((0,0)),dtype='<f8').tobytes()==
            np.asarray(b['position'],dtype='<f8').tobytes(),'exact exported local0 pose restoration')
        records=[]
        for sh in shapes:
            if isinstance(sh,geometry.pm.Circle):
                local=list(sh.offset);world=list(pose.local_to_world(sh.offset))
                rec=draw.circle(world,float(sh.radius),v[name]['color'])
                rec['local']=local
            elif isinstance(sh,geometry.pm.Poly):
                local=[list(x) for x in sh.get_vertices()]
                world=[list(pose.local_to_world(x)) for x in local]
                rec=draw.polygon(world,float(sh.radius),v[name]['color']);rec['local']=local
            else:return dict(status='unknown',reason='unhandled actual shape class')
            records.append(rec)
            if name=='block' and out['scene_constants']['render_goal']:
                gb=geometry.goal_body(np.asarray(goal_pose,float))
                if isinstance(sh,geometry.pm.Circle):
                    p=list(gb.local_to_world(sh.offset))
                    goal_commands.append(dict(op='circle',color=v['goal']['color'],
                        center=[round(x) for x in p],radius=int(sh.radius),width=0))
                else:
                    ps=[[round(x) for x in gb.local_to_world(a)] for a in local]
                    goal_commands.append(dict(op='polygon',color=v['goal']['color'],points=ps+[ps[0]],width=0))
        out['bodies'][name]=dict(**meta,source_center_of_gravity=list(original.center_of_gravity),
            exported_position=b['position'],unwrapped_angle=b['unwrapped_angle'],shapes=records)
    out['scene_constants']['goal_commands']=goal_commands
    return out


def commands(t,name):
    return [c for s in t['bodies'][name]['shapes'] for c in s['commands']]


def factor(a,b,same_rgb):
    require(type(same_rgb) is bool,'exact PNG pixel identity')
    known=a['status']==b['status']=='known'
    r=dict(classification=('unknown_draw_inputs_sameRGB' if same_rgb else 'unknown_draw_inputs_changedRGB'),
        RGB_identical=same_rgb,integer_draw_inputs_changed=None,float_geometry_changed=None,
        rounding_erased_shape_change=None,same_draw_inputs_changedRGB=False,
        full_scene_sufficient_cause_certified=False,input_scope='reconstructed_agent_block_shape_primitives_plus_scene_constants',
        unsealed_inputs=UNSEALED)
    if not known:
        r['unknown_reasons']=[x.get('reason') for x in (a,b) if x['status']!='known'];return r
    same_constants=raw(a['scene_constants'])==raw(b['scene_constants'])
    require(same_constants,'same fixed renderer/goal/condition within pair')
    float_changed=False;parts={}
    for name in ('agent','block'):
        aa=a['bodies'][name]['shapes'];bb=b['bodies'][name]['shapes']
        require(len(aa)==len(bb),'same true geometry per fixed root')
        f=raw([(x['world'],x['radius']) for x in aa])!=raw([(x['world'],x['radius']) for x in bb])
        c=raw(commands(a,name))!=raw(commands(b,name));float_changed|=f
        parts[name]=dict(float_geometry_changed=f,integer_commands_changed=c,
            integer_vertex_or_center_changes=sum(x!=y for sa,sb in zip(aa,bb,strict=True)
                for x,y in zip((sa['integer_points'] if sa['kind']!='circle' else [sa['integer_points']]),
                               (sb['integer_points'] if sb['kind']!='circle' else [sb['integer_points']]),strict=True)))
    changed=any(p['integer_commands_changed'] for p in parts.values())
    r.update(classification=('changedRGB' if not same_rgb else
        'changed_draw_inputs_sameRGB' if changed else 'same_draw_inputs_sameRGB'),
        integer_draw_inputs_changed=changed,float_geometry_changed=float_changed,
        rounding_erased_shape_change=bool(float_changed and not changed),
        same_draw_inputs_changedRGB=bool(not changed and not same_rgb),components=parts,
        residual_interpretation=('changed shape draw inputs with unchanged saved RGB; occlusion/raster/resize not isolated'
            if changed and same_rgb else 'shape rounding only; full scene/cache causality not certified'))
    return r


def validate_rows(p,r,c,a):
    require(len(p['plans'])==len(p['jobs'])==len(r['rows'])==len(c['rows'])==164
        and len(a['rows'])==80 and r['rows']==c['rows'] and r['errors']==c['errors']==[]
        and c['physical_report_sha256']==PINS['physical_report'][1] and c['saved_PNG_verified']==10824
        and r['execution_binding']['prepared']==str(ROOT/'results'/PINS['physical_prepare'][0])
        and r['execution_binding']['prepare_sha256']==PINS['physical_prepare'][1],
        'fixed complete original164 native report/strict receipt/prepare')
    for i,(j,plan,row) in enumerate(zip(p['jobs'],p['plans'],r['rows'],strict=True)):
        s,local=divmod(i,41)
        require(j['index']==plan['index']==row['index']==i and j['slot']==s and j['local_index']==local
            and (j['root_slot'],j['condition'])==ORDER[s] and row['plan']==plan
            and row['status']=='horizon_completed' and row['own_returned']==10
            and row['first_error'] is None and row['secondary_errors']==[]
            and row['condition']==j['condition'] and row['root_slot']==j['root_slot']
            and row['task_id']==j['task_id']==plan['task_id'] and plan['commands']==j['full10']
            and plan['model_task']==j['task'],'whole original roster/task/raw10, no outcome selection')
        cp=[x for x in row['checkpoints'] if x['step']==10]
        require(len(cp)==1 and cp[0]['snapshot']['state']==cp[0]['state']==row['final_state'],
            'all164 genuine exact step10 snapshot/state')
        check_snapshot(cp[0]['snapshot'])
    for want,got in zip(schedule(),a['rows'],strict=True):
        require(all(got[k]==v for k,v in want.items()),'all80 fixed original two-h pair order')


def png(path,frame):
    from PIL import Image
    require(Path(frame['png']).name==frame['png'] and sha(path)==frame['file_sha256'],'original PNG file seal')
    with Image.open(path) as im:
        require(im.format=='PNG' and im.mode=='RGB' and im.size==(224,224),'original native224 RGB PNG')
        x=np.asarray(im).copy()
    require(hashlib.sha256(x.tobytes()).hexdigest()==frame['pixel_sha256'] and sha(path)==frame['file_sha256'],
            'original PNG pixel SHA and stable bytes')
    return x


def terminal_events(path,r,p):
    require(sha(path)==r['journal_sha256'],'original entire journal SHA')
    slots=[[] for _ in range(164)]
    with Path(path).open('rb') as f:
        for line in f:
            require(len(line)<=2*1024**2,'bounded native event line')
            e=json.loads(line)
            if e['kind']!='fresh_observation':continue
            t=e['receipt']
            if (t['phase'],t['step'])!=('plan',10):continue
            i=e['branch'];require(type(i) is int and 0<=i<164,'terminal actual branch index')
            row=r['rows'][i];root=p['source_roots'][row['root_slot']]
            require(e['pre']==e['post'] and e['pre']['snapshot']==t['snapshot']
                and t['snapshot']['state']==t['state']==row['final_state']
                and e['pre']['renderer']==root['renderer'] and e['pre']['goal_pose']==root['goal_pose']
                and e['pre']['goal_state']==root['goal'] and e['pre']['goal_rgb_sha256']==root['goal_pixels'][row['condition']]
                and e['selection_digest']==row['selection_digest'],'actual full renderer/reset variations/goal/time/state identity')
            slots[i].append(e)
    for i,events in enumerate(slots):
        row=r['rows'][i]
        require([e['receipt']['overlay_index'] for e in events]==[0,1,2]
            and [e['receipt']['condition'] for e in events]==['off','on','off'], 'all164 exact terminal off/on/off chronology')
        k=1 if row['condition']=='on' else 0
        selected=[t for t in row['trajectory'] if (t['phase'],t['step'])==('plan',10)]
        require(len(selected)==1 and events[k]['receipt']==selected[0]
            and {k:v for k,v in events[k]['receipt'].items() if k not in ('phase','condition','overlay_index')}
                ==next(x for x in row['checkpoints'] if x['step']==10), 'selected real cp10 joins complete journal triplet')
    require(sha(path)==r['journal_sha256'],'journal stable after stream')
    return slots


def bindings():
    own=[Path(__file__).resolve(),ROOT/'tests'/('test_'+NAME+'.py'),ROOT/'innerdocs'/(NAME+'.md')]
    natives=[SITE/'pymunk'/s for s in ('body.py','shapes.py','vec2d.py','space_debug_draw_options.py',
        '_callbacks.py','pygame_util.py','_chipmunk_cffi.py','__init__.py')]
    binaries=sorted((SITE/'pymunk').glob('_chipmunk*.so'))
    require(len(binaries)==1,'single actual installed geometry binary')
    return {str(x):sha(x) for x in [*own,ENV,UTILS,*natives,*binaries]}


def build(authorization):
    require(authorization==AUTH,'MAIN detached geometry authorization before native import')
    fixed={k:read(ROOT/'results'/n,h) for k,(n,h) in PINS.items()}
    p,r,c,a=(fixed[k] for k in ('physical_prepare','physical_report','physical_receipt','observability'))
    validate_rows(p,r,c,a)
    require(sha(UTILS)==UTILS_SHA==p['bindings'][str(UTILS)] and sha(ENV)==ENV_SHA==p['bindings'][str(ENV)],
            'original renderer/geometry installed source authority')
    sealed=bindings()
    # Geometry operations only. No imports of gym/stable_worldmodel/pygame/torch.
    import pymunk as pm
    require(Path(pm.__file__).resolve()==SITE/'pymunk/__init__.py','actual bound installed detached geometry API')
    geometry=geometry_api(ENV.read_text(),pm);draw=draw_api(UTILS.read_text())
    output=ROOT/'results'/Path(PINS['physical_report'][0]).parent
    journal=output/'journal.jsonl';events=terminal_events(journal,r,p)
    tokens=[];pixel_refs=[]
    for i,row in enumerate(r['rows']):
        frames=events[i];images=[png(output/e['receipt']['png'],e['receipt']) for e in frames]
        require(np.array_equal(images[0],images[2]),'actual terminal off/on/off pixel closure')
        k=1 if row['condition']=='on' else 0;frame=frames[k]['receipt'];pre=frames[k]['pre']
        token=reconstruct(pre['renderer'],frame['snapshot'],pre['goal_pose'],row['condition'],draw,geometry)
        tokens.append(dict(index=i,slot=i//41,local_index=i%41,root_slot=row['root_slot'],condition=row['condition'],
            task_id=row['task_id'],selection_digest=row['selection_digest'],producer_job_sha256=p['jobs'][i]['job_sha256'],
            raw_float32_sha256=row['plan']['raw_float32_sha256'],snapshot_sha256=hashlib.sha256(raw(frame['snapshot'])).hexdigest(),
            terminal=deepcopy(frame),draw_inputs=token))
        pixel_refs.append(images[k])
    rows=[]
    for s,old in zip(schedule(),a['rows'],strict=True):
        i,j=s['minus_index'],s['plus_index'];same=bool(np.array_equal(pixel_refs[i],pixel_refs[j]))
        require(same==(old['label']=='unobservable_RGB'),'independent actual PNG identity versus original80 partition')
        rows.append(dict(**s,original_label=old['label'],**factor(tokens[i]['draw_inputs'],tokens[j]['draw_inputs'],same)))
    summary={label:sum(x['classification']==label for x in rows) for label in LABELS}
    require(sum(summary.values())==80,'all80 including unknown, no exclusion')
    for path,h in sealed.items():require(sha(path)==h,'final exact source closure')
    for _,(n,h) in PINS.items():require(sha(ROOT/'results'/n)==h,'final fixed input closure')
    return dict(schema=NAME+'.analysis',bindings={**sealed,**{str(ROOT/'results'/n):h for n,h in PINS.values()},
        str(journal):r['journal_sha256']},fixed_jobs=164,fixed_pairs=80,known_roots=2,fresh_roots=0,
        summary=summary,rows=rows,jobs=tokens,new_physics_calls=0,new_neural_calls=0,new_raster_calls=0,
        geometry_operations='detached original AST Body/Poly/Circle geometry only; no Space/env/step',
        P1_admission=False,gate_changed=False,training=False,correction_deployed=False,
        scope='Exact reconstructed agent/block shape primitive factorization; original transformed-shape cache, '
              'debug contact callbacks/order and512surface are not sealed. same_draw_inputs is SHAPE-SCOPED, '
              'not full-scene certification. Changed-input sameRGB cannot separate occlusion/raster/resize.')


def write_new(path,result):
    p=Path(path).absolute();b=raw(result)+b'\n'
    require(p.resolve()==p and p.parent==ROOT/'results' and p.name.startswith(NAME+'_')
        and not p.exists() and not p.is_symlink() and len(b)<=CAP
        and shutil.disk_usage(p.parent).free>=FLOOR+len(b),'new own bounded output only')
    with p.open('xb') as f:f.write(b);f.flush();os.fsync(f.fileno())
    return dict(output=str(p),sha256=sha(p),bytes=len(b),summary=result['summary'])


def main():
    ap=argparse.ArgumentParser(description=__doc__);sub=ap.add_subparsers(dest='command',required=True)
    q=sub.add_parser('analyze');q.add_argument('--output',required=True);q.add_argument('--authorization',required=True)
    q=sub.add_parser('verify');q.add_argument('--analysis',required=True);q.add_argument('--analysis-sha',required=True)
    q.add_argument('--authorization',required=True);args=ap.parse_args()
    if args.command=='analyze':print(json.dumps(write_new(args.output,build(args.authorization)),sort_keys=True))
    else:
        old=read(args.analysis,args.analysis_sha,CAP);now=build(args.authorization)
        require(old==now,'complete fixed80 saved reconstruction equality')
        print(json.dumps(dict(verified=True,analysis_sha256=args.analysis_sha,summary=now['summary']),sort_keys=True))


if __name__=='__main__':main()
