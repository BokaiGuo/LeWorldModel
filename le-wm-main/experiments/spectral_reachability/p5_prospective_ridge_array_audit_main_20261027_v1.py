"""Independent saved-array arithmetic; never imports neural or simulator code."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np

NAME='p5_prospective_ridge_array_audit_main_20261027_v1'
ROOT=Path(__file__).resolve().parents[2]
DIM=98304
ATOL=1e-10
RTOL=1e-8
LAYOUT={
    'response_columns':((DIM,10),np.dtype('float32')),
    'goal_features':((1,256,384),np.dtype('float32')),
    'nominal_successor':((1,256,384),np.dtype('float32')),
    'ridge_successor':((1,256,384),np.dtype('float32')),
    'selected_replay':((1,256,384),np.dtype('float32')),
    'gram':((10,10),np.dtype('float64')),
    'rhs':((10,),np.dtype('float64')),
    'ridge_delta':((10,),np.dtype('float64')),
}
INPUT_LAYOUT={'normalized_history':((1,2,10),np.dtype('float32')),
    'normalized_nominal':((10,),np.dtype('float32')),
    'normalized_ridge':((10,),np.dtype('float32'))}


def require(ok,message):
    if not ok:raise ValueError(message)


def sha(path):
    path=Path(path)
    require(path.is_file() and not any(p.is_symlink() for p in (path,*path.parents)),'regular non-symlink file required')
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1024**2),b''):h.update(block)
    return h.hexdigest()


def strict_object(pairs):
    value={}
    for key,item in pairs:
        require(key not in value,'duplicate JSON key')
        value[key]=item
    return value


def read_json(path):
    require(Path(path).stat().st_size<4*1024**2,'bounded JSON required')
    return json.loads(Path(path).read_text(),object_pairs_hook=strict_object,
        parse_constant=lambda x:(_ for _ in ()).throw(ValueError('nonfinite JSON')))


def write(path,value):
    with Path(path).open('xb') as stream:
        stream.write((json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+'\n').encode())
        stream.flush();os.fsync(stream.fileno())


def prepare():
    files=[Path(__file__).resolve(),ROOT/'tests'/f'test_{NAME}.py',ROOT/'innerdocs'/f'{NAME}.md']
    return dict(schema=NAME,stage='source_prepared_no_result',seals={str(p):sha(p) for p in files},
        formulas=dict(gram='J.T@J/D',rhs='J.T@(goal-nominal)/D',delta='solve(gram+.01I,rhs)',
            selection='minimum nonlinear MSE, nominal-first exact tie',replay='bitwise selected successor'),
        arithmetic_tolerance=dict(atol=ATOL,rtol=RTOL,scope='saved float64 reduction consistency, not scientific efficacy'),
        neural_calls=0,physical_calls=0,P1_admission=False,P5_admission=False)


def load_array(path,expected_sha,shape,dtype,directory):
    path=Path(path);dtype=np.dtype(dtype)
    require(path.is_absolute() and path.parent==Path(directory),'artifact path outside receipt directory')
    require(sha(path)==expected_sha,'artifact SHA mismatch')
    expected_bytes=math.prod(shape)*dtype.itemsize
    require(expected_bytes<path.stat().st_size<=expected_bytes+1024,'bounded numeric artifact required')
    with path.open('rb') as stream:
        version=np.lib.format.read_magic(stream)
        require(version in ((1,0),(2,0)),'unsupported npy version')
        reader=np.lib.format.read_array_header_1_0 if version==(1,0) else np.lib.format.read_array_header_2_0
        actual_shape,fortran,actual_dtype=reader(stream,max_header_size=1024)
        require(actual_shape==shape and actual_dtype==dtype and not dtype.hasobject and not fortran,'numeric npy layout mismatch')
        require(stream.tell()+expected_bytes==path.stat().st_size,'trailing or truncated npy rejected')
    value=np.load(path,allow_pickle=False,mmap_mode='r',max_header_size=1024)
    require(np.isfinite(value).all(),'nonfinite array')
    return value


def close(a,b):
    a,b=np.asarray(a,np.float64),np.asarray(b,np.float64)
    return a.shape==b.shape and bool(np.isfinite(a).all() and np.isfinite(b).all() and np.allclose(a,b,atol=ATOL,rtol=RTOL))


def bitwise(a,b):
    a,b=np.ascontiguousarray(a),np.ascontiguousarray(b)
    return a.shape==b.shape and a.dtype==b.dtype and a.tobytes()==b.tobytes()


def arithmetic(arrays,candidates,selected_id):
    """No source-runner numerical helpers reused; accepts small numeric fixtures."""
    j=np.asarray(arrays['response_columns'],np.float64)
    goal=np.asarray(arrays['goal_features'],np.float64).reshape(-1)
    nominal=np.asarray(arrays['nominal_successor'],np.float64).reshape(-1)
    require(j.ndim==2 and j.shape[1]==10 and len(j)==len(goal)==len(nominal) and len(goal)>0,'ten aligned response columns required')
    require(all(np.isfinite(np.asarray(a)).all() for a in arrays.values()),'finite arrays required')
    gram=j.T@j/len(j);rhs=j.T@(goal-nominal)/len(j)
    delta=np.linalg.solve(gram+.01*np.eye(10),rhs)
    require([c['id'] for c in candidates]==['nominal','ridge'],'exact ordered two-candidate census')
    scores={key:float(np.mean((np.asarray(arrays[key+'_successor'],np.float64).reshape(-1)-goal)**2)) for key in ('nominal','ridge')}
    selected=min(('nominal','ridge'),key=lambda k:scores[k])
    checks=dict(gram=close(arrays['gram'],gram),rhs=close(arrays['rhs'],rhs),ridge_delta=close(arrays['ridge_delta'],delta),
        selected_argmin=selected_id==selected,
        selected_replay_exact=bitwise(arrays['selected_replay'],arrays[selected+'_successor']))
    for candidate in candidates:
        key=candidate['id'];checks[key+'_nonlinear_score']=close(candidate['independent_goal_mse'],scores[key])
        packet=np.asarray(candidate['env_packet'])
        checks[key+'_bounded_packet']=bool(packet.shape==(5,2) and packet.dtype.kind in 'fi' and np.isfinite(packet).all() and np.all(np.abs(packet)<=1))
        if key=='nominal':checks['native_zero_nominal']=np.array_equal(packet,np.zeros((5,2)))
    return dict(checks={k:bool(v) for k,v in checks.items()},all_checks=all(checks.values()),scores=scores,
        selected_id=selected,solve_residual_norm=float(np.linalg.norm((gram+.01*np.eye(10))@delta-rhs)),
        predicted_linear_residual_mse=float(np.mean((goal-nominal-j@delta)**2)),
        limitation='saved-array algebra only; does not certify JVP against finite differences, RGB semantics or real task benefit')


def coordinates(receipt,delta,saved_inputs=None):
    c=receipt['action_coordinates']
    mean=np.asarray(c['action_mean_float32'],np.float32)
    std=np.asarray(c['action_std_float32'],np.float32)
    require(mean.shape==std.shape==(2,) and np.isfinite(mean).all() and np.isfinite(std).all() and np.all(std>0),'finite native action statistics required')
    def normalize(u):
        u=np.asarray(u,np.float64)
        return (((100*u).astype(np.float32)/np.float32(100)-mean)/std)
    nominal=np.zeros((5,2),np.float64)
    ridge=np.asarray(delta,np.float64).reshape(5,2)*std.astype(np.float64)
    histories=np.asarray(c['history_native_u64'],np.float64)
    require(histories.shape==(2,5,2) and np.isfinite(histories).all(),'two finite native history packets required')
    bank={r['id']:np.asarray(r['env_packet'],np.float64) for r in receipt['candidates']}
    checks=dict(native_nominal_exact=np.array_equal(c['nominal_native_u64'],nominal),
        normalized_nominal_exact=np.array_equal(np.asarray(c['nominal_normalized_f32'],np.float32),normalize(nominal)),
        normalized_history_exact=np.array_equal(np.asarray(c['history_normalized_f32'],np.float32),normalize(histories)),
        ridge_centered_inverse_exact=np.array_equal(bank['ridge'],ridge) and np.array_equal(c['ridge_native_u64'],ridge),
        normalized_ridge_exact=np.array_equal(np.asarray(c['ridge_normalized_f32'],np.float32),normalize(ridge)),
        selected_packet_exact=np.array_equal(receipt['selected_env_packet'],bank[receipt['selected_id']]))
    if saved_inputs is not None:
        checks.update(saved_normalized_history_exact=bitwise(saved_inputs['normalized_history'],normalize(histories).reshape(1,2,10)),
            saved_normalized_nominal_exact=bitwise(saved_inputs['normalized_nominal'],normalize(nominal).reshape(10)),
            saved_normalized_ridge_exact=bitwise(saved_inputs['normalized_ridge'],normalize(ridge).reshape(10)))
    return {k:bool(v) for k,v in checks.items()}


def audit(prepared,receipt_path,receipt_sha):
    require(prepared==prepare(),'auditor source drift')
    receipt_path=Path(receipt_path).absolute()
    require(sha(receipt_path)==receipt_sha,'explicit compiler receipt SHA mismatch')
    receipt=read_json(receipt_path)
    require(receipt['schema']=='p5_prospective_control_bridge_20261027_v1.compiler' and
        receipt['status']=='selected_packet_replayed' and receipt['backend']=='host_cuda_only' and
        receipt['physical_calls']==0 and receipt['used_observed_frame15'] is False and
        receipt['partial_work_unknown'] is False and receipt['selected_replay_exact'] is True and
        receipt['finalization_errors']==[],'completed prospective CUDA receipt required')
    require(set(receipt['artifact_roles'])>=set(LAYOUT),'all eight original arrays required')
    bindings=receipt['artifact_bindings'];arrays={}
    for role,(shape,dtype) in LAYOUT.items():
        path=Path(receipt['artifact_roles'][role])
        require(str(path) in bindings,'array not bound in compiler receipt')
        arrays[role]=load_array(path,bindings[str(path)],shape,dtype,receipt_path.parent)
    saved_inputs={}
    for name,(shape,dtype) in INPUT_LAYOUT.items():
        record=receipt['arrays'][name];path=Path(record['path'])
        require(str(path) in bindings and record['sha256']==bindings[str(path)],'normalized input array not bound')
        saved_inputs[name]=load_array(path,bindings[str(path)],shape,dtype,receipt_path.parent)
    result=arithmetic(arrays,receipt['candidates'],receipt['selected_id'])
    result['checks'].update(coordinates(receipt,arrays['ridge_delta'],saved_inputs))
    result['all_checks']=all(result['checks'].values())
    require(all(sha(p)==h for p,h in bindings.items()),'compiler artifact drift during audit')
    require(sha(receipt_path)==receipt_sha and prepared==prepare(),'receipt/source drift during audit')
    return dict(schema=NAME,status='saved_prospective_ridge_consistent' if result['all_checks'] else 'saved_prospective_ridge_inconsistent',
        arithmetic=result,compiler_receipt_sha256=receipt_sha,source_preparation=prepared,
        artifact_bindings=bindings,neural_calls=0,physical_calls=0,P1_admission=False,P5_admission=False)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('prepare','audit'))
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--prepared',type=Path);parser.add_argument('--receipt',type=Path);parser.add_argument('--receipt-sha256')
    args=parser.parse_args()
    if args.command=='prepare':result=prepare()
    else:
        require(args.prepared is not None and args.receipt is not None and args.receipt_sha256 is not None,'explicit prepared/receipt/SHA required')
        result=audit(read_json(args.prepared),args.receipt,args.receipt_sha256)
    write(args.output,result)
    print(json.dumps(dict(status=result.get('status',result.get('stage')))))
    if result.get('arithmetic',{}).get('all_checks') is False:raise SystemExit(2)


if __name__=='__main__':main()
