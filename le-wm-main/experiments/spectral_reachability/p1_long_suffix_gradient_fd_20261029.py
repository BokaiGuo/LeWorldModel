"""Fixed four-context two-packet CUDA directional derivative audit."""
import argparse
import hashlib
import inspect
import json
import os
from pathlib import Path
import numpy as np
from experiments.spectral_reachability import p1_long_suffix_consume_20261029 as consumer

NAME = 'p1_long_suffix_gradient_fd_20261029'
RESULT_SHA = 'b178a90b90b63f7a04d9eeafa7c55f71697a1bd121c5050b4de01de3a2bb334b'
EPS = np.float32(.001)


def difference(plus, minus, gradient, plus_raw, minus_raw):
    p, m = np.asarray(plus, np.float64), np.asarray(minus, np.float64)
    g = np.asarray(gradient, np.float64)
    d = (np.asarray(plus_raw, np.float32).astype(np.float64)-np.asarray(minus_raw, np.float32).astype(np.float64))/2
    if p.shape != () or m.shape != () or g.shape != (2,5,2) or d.shape != g.shape or not all(np.isfinite(v).all() for v in (p,m,g,d)):
        raise ValueError('invalid finite difference inputs')
    finite = float((p-m)/2); predicted = float(np.sum(g*d))
    return dict(finite_half_difference=finite, linear_half_difference=predicted,
        relative_error=None if max(abs(finite),abs(predicted)) == 0 else abs(finite-predicted)/max(abs(finite),abs(predicted)))


def run(output):
    producer=consumer.producer; source,saved,host=producer.one.source,producer.one.saved,producer.one.host
    result=saved.read_verified(source.ROOT/'results/p1_long_suffix_host_20261029_MAIN01.json',RESULT_SHA)
    report=saved.read_verified(source.REPORT,source.REPORT_SHA)
    receipt=saved.read_verified(source.RECEIPT,source.RECEIPT_SHA)
    if receipt['report_sha256'] != source.REPORT_SHA: raise ValueError('receipt binding')
    prepared=saved.read_verified(report['prepared'],report['prepare_sha256'])
    saved.analyze(report,receipt,prepared); consumer.verify(result,report,prepared)
    output=Path(output)
    if not output.is_absolute() or output.parent != source.ROOT/'results' or output.resolve()!=output or not output.name.startswith(NAME+'_') or output.exists():
        raise ValueError('exclusive output')
    import torch
    from torch.nn.attention import SDPBackend,sdpa_kernel
    from experiments.checkpoint_compat import load_compatible_pretrained
    from experiments.gpu_preflight import require_cuda_allocation
    os.environ['HF_HUB_OFFLINE']=os.environ['TRANSFORMERS_OFFLINE']='1'
    device_info=require_cuda_allocation('cuda:0'); device=torch.device('cuda:0')
    torch.backends.cuda.matmul.allow_tf32=torch.backends.cudnn.allow_tf32=False
    calls=dict(predict=0,action_encoder=0,reverse=0); rows=[]
    with torch.autocast(device_type='cuda',enabled=False),sdpa_kernel(SDPBackend.MATH):
        model=load_compatible_pretrained(prepared['policy'],device=device).eval().requires_grad_(False)
        before=host.signal.model_hash(model)
        if before != report['model_before']: raise ValueError('model hash')
        for module in model.modules():
            if str(Path(inspect.getfile(type(module))).resolve()) not in prepared['bindings']: raise ValueError('unbound model source')
        def hook(name):
            def count(_module,_args): calls[name]+=1
            return count
        hooks=[model.predictor.register_forward_pre_hook(hook('predict')),model.action_encoder.register_forward_pre_hook(hook('action_encoder'))]
        try:
            for slot in range(4):
                arm_index=2*slot; candidate=report['rows'][arm_index]['result']['candidates'][0]
                saved_row=result['rows'][20*arm_index]; saved_eval=saved_row['evaluations'][0]
                task=prepared['requests'][slot]['task']
                z=torch.tensor(candidate['mid']['latent_history'],dtype=torch.float32,device=device)[None]
                h=torch.tensor(candidate['mid']['action_history'],dtype=torch.float32,device=device)[None]
                mean=torch.tensor(np.tile(task['action_mean'],5),dtype=torch.float32,device=device)
                std=torch.tensor(np.tile(task['action_std'],5),dtype=torch.float32,device=device)
                goal=torch.tensor(task['goal'],dtype=torch.float32,device=device)
                raw=np.asarray(saved_eval['raw'],np.float32)
                def evaluate(raw_value,grad):
                    action=torch.tensor(raw_value,dtype=torch.float32,device=device,requires_grad=grad)
                    tokens=[(action[k].reshape(10)-mean)/std for k in range(2)]
                    end=producer.advance(z,h,tokens,lambda states,actions:model.predict(states,model.action_encoder(actions)))[-1]
                    loss=torch.mean((end-goal)**2)
                    derivative=None
                    if grad:
                        derivative=torch.autograd.grad(loss,action)[0].detach().cpu().numpy(); calls['reverse']+=1
                    return end.detach().cpu().numpy(),derivative
                nominal,gradient=evaluate(raw,True)
                if not np.array_equal(nominal,np.asarray(saved_eval['endpoint'],np.float32)) or not np.array_equal(gradient,np.asarray(saved_eval['raw_gradient'],np.float32)):
                    raise ValueError('saved nominal/gradient replay mismatch')
                directions=[]
                for packet in range(2):
                    plus=raw.copy(); minus=raw.copy()
                    plus[packet]+=EPS; minus[packet]-=EPS
                    if np.max(np.abs(plus))>1 or np.max(np.abs(minus))>1: raise ValueError('perturbation exits raw box')
                    p,_=evaluate(plus,False); m,_=evaluate(minus,False)
                    g=np.asarray(task['goal'],np.float64)
                    lp=float(np.mean((p.astype(np.float64)-g)**2)); lm=float(np.mean((m.astype(np.float64)-g)**2))
                    directions.append(dict(packet=packet,plus_raw=plus.tolist(),minus_raw=minus.tolist(),plus_endpoint=p.tolist(),minus_endpoint=m.tolist(),plus_loss=lp,minus_loss=lm,**difference(lp,lm,gradient,plus,minus)))
                rows.append(dict(slot=slot,arm_index=arm_index,nominal=nominal.tolist(),gradient=gradient.tolist(),directions=directions))
        finally:
            for handle in hooks: handle.remove()
        torch.cuda.synchronize(device);after=host.signal.model_hash(model)
        if before!=after or calls!=dict(predict=40,action_encoder=40,reverse=4): raise ValueError('resource/hash mismatch')
    host.F.check_bindings(prepared['bindings'])
    value=dict(schema=NAME,source_result_sha256=RESULT_SHA,source_report_sha256=source.REPORT_SHA,
        code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),device=device_info,model_before=before,model_after=after,
        calls=calls,epsilon=float(EPS),rows=rows,scope='known roots two fixed directions; not physical gradient',P1_admission=False,training=False)
    return dict(path=str(output),sha256=saved.write(output,value),max_relative_error=max(x['relative_error'] for r in rows for x in r['directions'] if x['relative_error'] is not None))


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',required=True)
    print('Checking saved two-packet gradients against fixed directional differences on host CUDA.',flush=True)
    print(json.dumps(run(parser.parse_args().output)))


if __name__=='__main__': main()
