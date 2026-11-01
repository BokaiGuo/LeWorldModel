"""Independent original-NPY Q arithmetic on the fixed physical 2x2 panel."""
import argparse
import gzip
import hashlib
import io
import json
from pathlib import Path

import numpy as np

ROOT=Path(__file__).resolve().parents[2]
NAME='main_anchor_guard_latent_factorial_20261028_v1'
PINS={
    'physical':('main_anchor_guard_physical_factorial_20261028_v1_MAIN01.json','cb1827b8b1e65805766b1e53ffca2f7045a9b9d05ddd547914bae3d93accfa30'),
    'cells':('agent_p2_safeguard_factorial_cuda_20261028_v1_consume_MAIN01.json','8a59a189feaedf3b215b7aa9245c3b027228e2a51cfef32a47df286c664e5d89'),
    'control':('agent_p2_contextual_physical_rgb_encode_20261028_v1_cuda_MAIN01/report.json','b57fa03560696c3c0b75fe842ec8263d6ec3961341d6736a4f6e782d1f0e92ea'),
    'incumbent':('agent_p2_midfeedback_terminal_rgb_20261028_v2_cuda_MAIN01/report.json','9a63b5a3db263a3f9accdc8d54faa31b40787c362ca52e5e29ca793f785c7941'),
    'incumbent_strict':('agent_p2_midfeedback_terminal_rgb_20261028_v2_consume_MAIN01.json','a66d4187cdbf714509a8bc931d12ef6c51410603da8c4445c7884bbde2e143e0'),
}
AXES=('joint_xy','object_xy','angle','velocity','native_distance')


def require(ok,msg):
    if not ok:raise ValueError(msg)


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path,expected):
    p=Path(path);require(p.is_file() and not p.is_symlink() and sha(p)==expected,'external actual input SHA')
    return json.loads(p.read_bytes())


def latent(folder,entry):
    require(Path(entry['file']).name==entry['file'],'flat original NPY')
    p=Path(folder)/entry['file'];require(sha(p)==entry['sha256'],'NPY gzip file SHA')
    with gzip.open(p,'rb') as f:data=f.read(8193)
    require(len(data)<=8192,'bounded native192 NPY')
    stream=io.BytesIO(data);z=np.load(stream,allow_pickle=False)
    require(stream.tell()==len(data) and z.shape==(192,) and z.dtype==np.dtype('float32')
        and entry['shape']==[192] and entry['dtype']=='float32' and np.isfinite(z).all(),'full native192 array ABI')
    return z


def q(z,g):
    z,g=np.asarray(z),np.asarray(g)
    require(z.shape==g.shape==(192,) and np.isfinite(z).all() and np.isfinite(g).all(),'finite whole192')
    return float(np.mean((z.astype(np.float64)-g.astype(np.float64))**2))


def contrast(left,right):
    require(left['branch']==right['branch'],'paired fixed branch')
    p=right['model_Q']-left['model_Q'];a=right['actual_Q']-left['actual_Q']
    return dict(branch=left['branch'],predicted_delta=p,actual_delta=a,model_to_actual_reversal=p*a<0,
        actual_to_physical_reversal={k:a*(right['metrics'][k]-left['metrics'][k])<0 for k in AXES},
        physical_delta={k:right['metrics'][k]-left['metrics'][k] for k in AXES})


def build(output,report_sha,receipt,receipt_sha):
    bound={k:read(ROOT/'results'/p,h) for k,(p,h) in PINS.items()}
    folder=Path(output);zero=read(folder/'report.json',report_sha);strict=read(receipt,receipt_sha)
    require(strict['schema']=='agent_p2_safeguard_factorial_terminal_rgb_20261028_v1.consume'
        and strict['status']=='zero8_terminal_RGB_verified_NOT_science' and strict['report_sha256']==report_sha
        and strict['P1_admission'] is strict['P2_admission'] is False,'accepted actual zero terminal authority')
    reports={k:bound[k] for k in ('control','incumbent')};reports['zero']=zero
    folders={k:(ROOT/'results'/PINS[k][0]).parent for k in ('control','incumbent')};folders['zero']=folder
    require(all(len(r['rows'])==8 for r in reports.values()) and len(bound['cells']['cells'])==32
        and len(bound['physical']['cells'])==32,'fixed native and cell denominators')
    rows=[]
    for cell,physical in zip(bound['cells']['cells'],bound['physical']['cells'],strict=True):
        i=cell['branch'];source=physical['source'];r=reports[source]['rows'][i]
        require((physical['branch'],physical['anchor'],physical['guard'])==(i,cell['anchor'],cell['guard'])
            and all(r[k]==physical[k] for k in ('root_slot','condition','task_id'))
            and r['raw_float32_bits']==cell['full10_raw_bits']==physical['action_bits'],'same complete action/task source')
        entry=r['latent'] if source!='control' else next(x for x in r['frames'] if x['step']==10)['latent']
        z=latent(folders[source],entry);actual=q(z,r['goal192'])
        predicted=q(cell['new_suffix_context']['latent_history'][-1],r['goal192'])
        require(predicted==physical['model_Q']==cell['selection']['model_Q'],'independent whole192 predicted Q')
        old=bound['incumbent_strict']['rows'][i]
        expected=strict['rows'][i]['actual_Q'] if source=='zero' else old['new_suffix_actual_Q' if source=='incumbent' else 'old_fixed_actual_Q']
        require(actual==expected,'independent original NPY actual Q equals strict authority')
        rows.append(dict(physical,actual_Q=actual))
    contrasts={name:[] for name in ('anchor_no_guard','anchor_guard','guard_zero','guard_incumbent')}
    for i in range(8):
        z0,z1,a0,a1=rows[4*i:4*i+4]
        for name,left,right in (('anchor_no_guard',z0,a0),('anchor_guard',z1,a1),('guard_zero',z0,z1),('guard_incumbent',a0,a1)):
            contrasts[name].append(contrast(left,right))
    return dict(schema=NAME,cells=rows,contrasts=contrasts,
        counts={name:dict(model_to_actual_reversals=sum(r['model_to_actual_reversal'] for r in rr),
            actual_to_physical_reversals={k:sum(r['actual_to_physical_reversal'][k] for r in rr) for k in AXES})
            for name,rr in contrasts.items()},
        authority=dict(pins=PINS,zero_output=str(folder),zero_report_sha256=report_sha,zero_receipt=str(receipt),zero_receipt_sha256=receipt_sha),
        independent_NPY_arithmetic=True,full_upstream_rebuilt_here=False,new_neural_calls=0,new_physics=0,
        known_roots=2,fresh_roots=0,P1_admission=False,P2_admission=False,
        interpretation='All fixed32 cells; exact action-based reuse, descriptive known-root contrasts. Strict reversals exclude ties; not unique-cause proof.')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('zero-output','report-sha','receipt','receipt-sha','output'):p.add_argument('--'+key,required=True)
    a=p.parse_args();result=build(a.zero_output,a.report_sha,a.receipt,a.receipt_sha);path=Path(a.output).absolute()
    require(path.parent==ROOT/'results' and path.name.startswith(NAME+'_'),'exclusive own metadata')
    with path.open('xb') as f:f.write(json.dumps(result,sort_keys=True,allow_nan=False).encode()+b'\n')
    print(json.dumps(dict(output=str(path),sha256=sha(path),counts=result['counts'])))


if __name__=='__main__':main()
