"""Independent saved scalar arithmetic; no producer import, NN or fullstrict."""
import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import time

import numpy as np

NAME='agent_p5_temporal_independent_scalars_20261029_v1'
OWNER='agent_p5_temporal_history_factorial_20261029_v1'
ROOT=Path(__file__).resolve().parents[2]
N=164
CELLS=('T_AA','T_AR','T_RA','T_RR')
ARMS=CELLS[1:]
EARLY='agent_p5_early_history_factorial_20261029_v1'
REPORT_SHA='f046c617ba2eeb6fcbd79799238e8ccbd00ca7ecd6b5414f2021844550672747'
RECEIPT_SHA='86149cf64a0e5c423ee99f23f98dde9ce06ccf8ced837f70c65648e3518e51ea'
PREPARE_SHA='7fac018fdb7b84362c174f84345c0d0b10c0045e34bf675f8a912eb988d52bef'
EARLY_REPORT_SHA='4fc86a7fdc69fbe95003e417ee38e3a4d430360a8eb5bd0c89f73ee845926d0f'
EARLY_RECEIPT_SHA='d9461f7a21fa13454eca1f4371488431cf833a08ee758990c2dcb999b342600e'
ORDER=((0,'off'),(0,'on'),(1,'off'),(1,'on'))
SIGNS=('positive','negative','zero','null')
CAP,FLOOR,WALL=32*1024**2,200*1024**2,900


def require(ok,message):
    if not ok:raise ValueError(message)


def raw(v):return (json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n').encode()


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024**2),b''):h.update(b)
    return h.hexdigest()


def checked(path,expected,used):
    p=Path(path).absolute()
    require(type(expected) is str and len(expected)==64 and all(c in '0123456789abcdef' for c in expected),'explicit SHA256')
    require(p.resolve()==p and p.is_file() and not any(x.is_symlink() for x in (p,*p.parents)),'canonical nonsymlink regular source')
    require(str(p) not in used or used[str(p)]==expected,'no conflicting file identity')
    if str(p) not in used:require(sha(p)==expected,'actual SHA256')
    used[str(p)]=expected;return p


def unique(items):
    d={}
    for k,v in items:require(k not in d,'duplicate JSON field');d[k]=v
    return d


def read(path,expected,used):
    p=checked(path,expected,used);require(p.stat().st_size<=CAP,'bounded32MiB metadata')
    v=json.loads(p.read_bytes(),object_pairs_hook=unique,parse_constant=lambda x:(_ for _ in ()).throw(ValueError('nonfinite JSON')))
    raw(v);return v


def schedule():
    return [dict(slot=s,root_slot=r,condition=c,h=h,coord=k,minus_index=41*s+1+20*j+2*k,plus_index=41*s+2+20*j+2*k)
        for s,(r,c) in enumerate(ORDER) for j,h in enumerate((.01,.001)) for k in range(10)]


def array(directory,rec,shape,inventory,used):
    require(type(rec) is dict and Path(rec['file']).name==rec['file'] and rec['dtype']=='float32' and rec['shape']==list(shape) and
        rec['nbytes']==4*int(np.prod(shape)) and inventory.get(rec['file'])=={k:rec[k] for k in ('sha256','file_bytes')},'full finite native descriptor/role/inventory')
    p=checked(Path(directory)/rec['file'],rec['sha256'],used)
    require(p.stat().st_size==rec['file_bytes']<=4*1024**2,'bounded exact NPY bytes')
    b=io.BytesIO(p.read_bytes());a=np.load(b,allow_pickle=False)
    require(b.tell()==p.stat().st_size and a.dtype==np.float32 and a.shape==shape and a.nbytes==rec['nbytes'] and np.isfinite(a).all(),'one complete NPY/no trailer')
    return a


def effects(v):
    require(set(v)==set(CELLS),'all four cells')
    a,b,c,d=(np.asarray(v[k],np.float64) for k in CELLS)
    require(a.shape==b.shape==c.shape==d.shape and all(np.isfinite(x).all() for x in (a,b,c,d)),'common finite support')
    return {'slot5':((c-a)+(d-b))/2,'slot10':((b-a)+(d-c))/2,'interaction':d-c-b+a,'joint':d-a}


def row_scalars(cells,label):
    require(label.shape==(1,256,384) and all(x.shape==label.shape for x in cells.values()),'final20 label/support')
    values={c:float(np.mean(np.square(x.astype(np.float64)-label.astype(np.float64)),dtype=np.float64)) for c,x in cells.items()}
    return dict(MSE=values,MSE_effects={k:float(x) for k,x in effects(values).items()},
        feature_effect_L2={k:float(np.linalg.norm(x)) for k,x in effects(cells).items()})


def sign(x):return 'null' if x is None else 'positive' if x>0 else 'negative' if x<0 else 'zero'


def pair_scalars(spec,minus,plus):
    a,la=minus;b,lb=plus;observed=(lb.astype(np.float64)-la.astype(np.float64)).ravel(order='C')
    on=float(np.linalg.norm(observed));values={};responses={}
    for c in CELLS:
        delta=b[c].astype(np.float64)-a[c].astype(np.float64);flat=delta.ravel(order='C');pn=float(np.linalg.norm(flat));dot=float(np.dot(flat,observed))
        cos=None if pn==0 or on==0 else dot/(pn*on)
        values[c]=dict(predicted_L2=pn,observed_L2=on,dot=dot,cosine=cos,predicted_zero=pn==0,observed_zero=on==0,
            response_error_L2=float(np.linalg.norm(flat-observed)),cosine_sign=sign(cos),
            central_predicted_L2=float(np.linalg.norm(flat/(2*spec['h']))),central_observed_L2=float(np.linalg.norm(observed/(2*spec['h']))))
        responses[c]=delta
    return dict(spec,missing=[],cells=values,response_effect_L2={k:float(np.linalg.norm(x)) for k,x in effects(responses).items()})


def summarize(rows,pairs):
    output=[]
    for root,condition in [(None,None),*ORDER]:
        keep=lambda x:root is None or (x['root_slot'],x['condition'])==(root,condition)
        rs=[r for r in rows if keep(r)];ps=[p for p in pairs if keep(p)]
        mean=lambda xs:None if not xs else float(np.mean(xs,dtype=np.float64))
        output.append(dict(root_slot=root,condition=condition,branches=len(rs),pairs=len(ps),
            mean_MSE={c:mean([r['score']['MSE'][c] for r in rs]) for c in CELLS},
            mean_MSE_effects={k:mean([r['score']['MSE_effects'][k] for r in rs]) for k in ('slot5','slot10','interaction','joint')},
            responses={c:dict(observed_zero=sum(p['cells'][c]['observed_zero'] for p in ps),predicted_zero=sum(p['cells'][c]['predicted_zero'] for p in ps),
                signs={s:sum(p['cells'][c]['cosine_sign']==s for p in ps) for s in SIGNS}) for c in CELLS},
            sign_transitions={a+'__to__'+b:{s+'__to__'+t:sum(p['cells'][a]['cosine_sign']==s and p['cells'][b]['cosine_sign']==t for p in ps)
                for s in SIGNS for t in SIGNS} for j,a in enumerate(CELLS) for b in CELLS[j+1:]}))
    return output


def authenticate(r,c,report_sha):
    u=dict(api=3*N,batch_samples=3*N,frame_samples=9*N)
    require(r['schema']==OWNER+'.report' and c['schema']==OWNER+'.consume' and c['report_sha256']==report_sha and
        r['status']=='all164_temporal_history_492_recorded' and c['status']=='all164_temporal_two_corners_RR_bridge_and80_pairs_verified' and
        r['errors']==r['source_errors']==[] and len(r['rows'])==N and len(r['calls'])==3*N and len(r['pairs'])==len(schedule()) and
        c['branches']==N and c['pairs']==len(schedule()) and c['predictor']==u and c['RR_all_native_IO_bitwise_verified'] is True and c['inherited_AA'] is True,
        'explicit complete prior MAIN strict is mandatory; no partial fallback')
    require(r['counts']==dict(attempted=u,returned=u,validated=u) and r['protocol']['branches']==N and r['protocol']['cells']==list(CELLS) and
        r['protocol']['budget_mode']=='predeclared_328_plus164_RR_492' and r['protocol']['frames']==[5,10,15] and r['protocol']['target']==20 and
        r['protocol']['pairs']==schedule() and r['protocol']['predictor']==u and r['model_before']==r['model_after_predictions']==r['model_after'],
        'complete four-cell protocol/counters/unchanged loaded state')
    require(all(x['P1_admission'] is x['P5_admission'] is False for x in (r,c)) and c['prior_fullstrict_rebuilt'] is False,
        'closed science gates; inherit exact receipt not rerun strict')


def analyze(report,report_sha,receipt,receipt_sha):
    started=time.monotonic();used={};r=read(report,report_sha,used);c=read(receipt,receipt_sha,used)
    require(report_sha==REPORT_SHA and receipt_sha==RECEIPT_SHA,'pinned actual temporal report/receipt')
    authenticate(r,c,report_sha)  # no array access until COMPLETE external receipt
    require(r['execution_binding']['prepare_sha256']==PREPARE_SHA,'pinned actual temporal prepare')
    prep=read(r['execution_binding']['prepared'],PREPARE_SHA,used)
    require(prep['schema']==OWNER+'.prepare' and prep['status']=='prepared_no_neural_execution' and
        prep['protocol']==r['protocol'] and len(prep['requests'])==N and prep['inherited']==r['inherited'],'actual prepare identity/requests')
    ed=ROOT/'results'/(EARLY+'_cuda_MAIN01')
    er=read(ed/'report.json',EARLY_REPORT_SHA,used)
    ec=read(ROOT/'results'/(EARLY+'_consume_MAIN01.json'),EARLY_RECEIPT_SHA,used)
    require(er['schema']==EARLY+'.report' and ec['schema']==EARLY+'.consume' and
        er['status']=='all164_early_history_factorial_recorded' and
        ec['status']=='all164_early_history_four_NEW_cells_and80_pairs_verified' and
        ec['report_sha256']==EARLY_REPORT_SHA and ec['branches']==N and ec['pairs']==len(schedule()) and
        len(er['rows'])==N and len(er['calls'])==4*N and er['errors']==er['source_errors']==[] and
        er['model_before']==er['model_after_predictions']==er['model_after']==r['model_before'] and
        all(x['P1_admission'] is x['P5_admission'] is False for x in (er,ec)) and
        r['inherited']['report_sha256']==EARLY_REPORT_SHA and r['inherited']['receipt_sha256']==EARLY_RECEIPT_SHA,
        'complete pinned inherited AA authority, same loaded model; not rebuilding prior strict')
    d=Path(report).absolute().parent;inv=r['artifacts'];rows=[];pairs=[];slots=[];seen=set()
    for start in range(0,N,41):
        cache={}
        for i in range(start,min(start+41,N)):
            require(time.monotonic()-started<WALL,'bounded saved900s no fallback/subselection')
            row=r['rows'][i];root,cond=ORDER[i//41]
            require(row['index']==i and (row['root_slot'],row['condition'])==(root,cond) and set(row['predictions'])==set(ARMS) and
                row['RR_verified']==dict(inputs=[True]*3,full3_outputs=[True]*3),'all original164/four-cell/RR slots')
            old=er['rows'][i];q=prep['requests'][i]
            keys=('index','root_slot','condition','task_id','selection_digest')
            require(all(row[k]==old[k]==q[k] for k in keys),'temporal/old/prepare exact branch identity')
            cellarrays={};slotrefs={}
            for j,cell in enumerate(CELLS):
                directory,inventory=(ed,er['artifacts']) if j==0 else (d,inv)
                call=er['calls'][4*i] if j==0 else r['calls'][3*i+j-1]
                tree=old['predictions']['AA'] if j==0 else row['predictions'][cell]
                if j==0:
                    refs={key:dict(directory=str(ed),array=tree['$tuple'][slot]['$array'],
                        report_json_pointer=f'/calls/{4*i}/outputs/$tuple/{slot}/$array') for key,slot in (('visual',0),('proprio',2))}
                    require(row['inherited_T_AA']==q['AA_outputs']==refs,'AA original call pointer/directory/descriptor, no guessed lineage')
                require(call['index']==(4*i if j==0 else 3*i+j-1) and call['T']==3 and call['status']=='returned' and
                    call['units']==dict(api=1,batch_samples=1,frame_samples=3) and tree==call['outputs'] and
                    set(tree)=={'$tuple'} and len(tree['$tuple'])==3 and tree['$tuple'][1] is None,
                    'all656 native full3 output roles and order, never select cells')
                records=[tree['$tuple'][k]['$array'] for k in (0,2)]
                decoded=[]
                for rec,shape in zip(records,((1,3,256,384),(1,3,256,16)),strict=True):
                    require(str(directory/rec['file']) not in seen,'unique actual native output arrays, no aliasing cells')
                    seen.add(str(directory/rec['file']));decoded.append(array(directory,rec,shape,inventory,used))
                cellarrays[cell]=decoded[0][:,-1].copy()
                slotrefs[cell]=dict(directory=str(directory),visual=records[0],proprio=records[1],all3slots_finite=True,scored_slot=2)
            rec=row['label'];require(str(d/rec['file']) not in seen,'separate actual label buffer');seen.add(str(d/rec['file']))
            batch=array(d,rec,(2,256,384),inv,used)
            oldbatch=array(ed,old['label'],(2,256,384),er['artifacts'],used)
            require(batch.tobytes()==oldbatch.tobytes() and q['label']==dict(directory=str(ed),array=old['label'],
                report_json_pointer=f'/rows/{i}/label'),'entire common H2 batch label and source pointer exact')
            label=batch[1:2]
            score=row_scalars(cellarrays,label);require(score==row['score'],'independent exact fourcell MSE/effects/feature norm')
            rows.append({k:row[k] for k in ('index','root_slot','condition','task_id','selection_digest')}|dict(score=score))
            slots.append(dict(index=i,outputs=slotrefs,label=rec));cache[i]=(cellarrays,label)
        for spec in schedule():
            if start<=spec['minus_index']<min(start+41,N):
                p=pair_scalars(spec,cache[spec['minus_index']],cache[spec['plus_index']]);pairs.append(p)
    require(pairs==r['pairs'],'all80 exact cosine/response/central differences independent of producer math')
    summary=summarize(rows,pairs);require(summary==r['summaries'],'all plus four strata, all sign transitions')
    for path,h in used.items():require(sha(path)==h,'all used saved evidence unchanged')
    for p in (Path(__file__).resolve(),ROOT/'tests'/('test_'+NAME+'.py'),ROOT/'innerdocs'/(NAME+'.md')):checked(p,sha(p),used)
    require(time.monotonic()-started<WALL,'saved wall including final hashes')
    return dict(schema=NAME+'.analysis',status='all164_temporal_fourcell_and80_pair_independent_scalars_verified',
        arguments=dict(report=str(Path(report).absolute()),report_sha=report_sha,receipt=str(Path(receipt).absolute()),receipt_sha=receipt_sha),
        rows=rows,pairs=pairs,summaries=summary,all_slot_evidence=slots,bindings=used,
        verification_boundary=dict(prior_complete_strict_trusted=True,prior_fullstrict_rebuilt=False,
            native_input_causal_revalidated=False,journal_chronology_revalidated=False,all_full3_outputs_decoded=True,
            scalar_arithmetic_independently_recomputed=True,labels_common_H2_as_certified_by_prior_strict=True,common_full_label_batch_bytechecked=True,
            inherited_AA_full3_pointer_and_bytes_checked=True,full_native_input_arrays_redecoded=False),
        branches=N,pair_count=len(pairs),resources=dict(temporal_actual_counts=r['counts'],temporal_wall_seconds=r['elapsed_seconds'],
            inherited_early_counts=er['counts'],inherited_early_wall_seconds=er['elapsed_seconds'],
            inherited_cost_sources=r['inherited'],costs_not_summed_as_independent_jobs=True,
            new_encoder=0,new_predictor=0,new_AD=0,new_physics=0,new_training=0,FLOPs=None),P1_admission=False,P5_admission=False)


def write_new(output,value):
    p=Path(output).absolute();body=raw(value)
    require(p.parent==ROOT/'results' and p.name.startswith(NAME+'_') and not p.exists() and not any(x.is_symlink() for x in (p,*p.parents)) and
        len(body)<=CAP and shutil.disk_usage(p.parent).free>=FLOOR+len(body),'exclusive own32MiB metadata/floor')
    with p.open('xb') as f:require(f.write(body)==len(body),'full durable write');f.flush();os.fsync(f.fileno())
    fd=os.open(p.parent,os.O_RDONLY|os.O_DIRECTORY)
    try:os.fsync(fd)
    finally:os.close(fd)
    return dict(status=value['status'],path=str(p),sha256=sha(p))


def verify(analysis,expected):
    v=read(analysis,expected,{})
    require(v==analyze(**v['arguments']),'entire deterministic independent scalar analysis reproduced')
    return dict(status='independent_factorial_scalar_analysis_reproduced',analysis_sha256=expected,new_neural_calls=0,new_physics_calls=0)


def main():
    require(N==164,'fixed production denominator164')
    p=argparse.ArgumentParser(description=__doc__);s=p.add_subparsers(dest='command',required=True)
    a=s.add_parser('analyze')
    a.add_argument('--report',default=str(ROOT/'results'/(OWNER+'_cuda_MAIN01')/'report.json'))
    a.add_argument('--report-sha',default=REPORT_SHA)
    a.add_argument('--receipt',default=str(ROOT/'results'/(OWNER+'_consume_MAIN01.json')))
    a.add_argument('--receipt-sha',default=RECEIPT_SHA)
    a.add_argument('--output',required=True)
    a=s.add_parser('verify');a.add_argument('--analysis',required=True);a.add_argument('--analysis-sha',required=True)
    q=p.parse_args()
    if q.command=='analyze':
        args=vars(q).copy();args.pop('command');output=args.pop('output');value=write_new(output,analyze(**args))
    else:value=verify(q.analysis,q.analysis_sha)
    print(json.dumps(value,sort_keys=True,allow_nan=False))


if __name__=='__main__':main()
