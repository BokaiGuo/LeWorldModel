"""Saved prefix parity only; both actual CUDA jobs remain incomplete.

No torch, weights, full upstream strict or scientific admission.
"""
import argparse
from pathlib import Path
from experiments.spectral_reachability import agent_hilewm_transition_cuda_probe_20261029_v1 as old

TAG='main_hilewm_gradient_prefix_parity_20261029_v1'
BASE=old.ROOT/'results'
SPECS=(('agent_hilewm_transition_cuda_probe_20261029_v1',
        '44ce2355c454f690016ad73d1565659881da796c354c1799044c7ed11512c02d'),
       ('agent_hilewm_transition_cuda_probe_20261029_v2',
        'a7b28cddf19c5a1d62b60811c936a53ad80bdfbf8b2f8d26586808dabb1c5bb8'))


def compare_prefix(a,b,la,lb):
    old.need(a['route']==b['route']=='reference' and len(a['events'])==len(b['events'])==10,
             'same fixed first reference route, nine forward events and VJP attempt')
    count=0
    def eq(x,y):
        nonlocal count
        old.equal(la(x),lb(y),'old/new forward byte identity');count+=1
    for key in ('z','a_hist','actions','prediction','scalar'):eq(a[key],b[key])
    old.need(len(a['contexts'])==len(b['contexts'])==3,'all three checkpoints')
    for aa,bb in zip(a['contexts'],b['contexts'],strict=True):
        for key in ('z','a'):eq(aa[key],bb[key])
    for i,(aa,bb) in enumerate(zip(a['events'][:9],b['events'][:9],strict=True)):
        old.need(aa['index']==bb['index']==i and aa['kind']==bb['kind']==old.KINDS[i%3]
                 and aa['returned'] is bb['returned'] is True,'same returned module chronology')
        old.need(len(aa['inputs'])==len(bb['inputs']),'same actual input arity')
        for x,y in zip(aa['inputs']+[aa['output']],bb['inputs']+[bb['output']],strict=True):eq(x,y)
    old.need(a['events'][9]['returned'] is False and b['events'][9]['returned'] is True,
             'old VJP failure and new actual VJP return')
    old.need(a['events'][0]['output_layout']['requires_grad'] is False and
             b['events'][0]['output_layout']['requires_grad'] is True,'actual first graph boundary repaired')
    gradient=lb(b['gradient'])
    old.need(gradient.shape==(1,1,3,10),'all thirty gradient coordinates returned')
    return dict(forward_arrays_compared=count,forward_bitwise=True,
        new_gradient_shape=list(gradient.shape),new_gradient_raw_sha256=old.digest(gradient.tobytes()),
        gradient_parity=False,whole_segmented_verified=False,finite_difference_verified=False)


def analyze():
    reports=[];arrays=[];bindings={str(Path(__file__).resolve()):old.sha(__file__)}
    for name,h in SPECS:
        d=BASE/(name+'_cuda_MAIN01');p=d/'report.json';r=old.read(p,h)
        old.need(r['completed'] is False and r['errors'] and
                 r['model_before']==r['model_after'] and r['sources_before']==r['sources_after'],
                 'actual incomplete reports with stable model and sources')
        aa={n:old.load_array(d,n,desc) for n,desc in r['arrays'].items()}
        old.need({x.name for x in d.iterdir()}==set(aa)|{'report.json'},'entire prefix artifact census')
        bindings[str(p)]=h
        for n,desc in r['arrays'].items():bindings[str(d/n)]=desc['file_sha256']
        reports.append(r);arrays.append(aa)
    old.need(reports[0]['model_before']==reports[1]['model_before'],'same original all425 tensor values')
    result=compare_prefix(reports[0]['rows'][0],reports[1]['rows'][0],arrays[0].__getitem__,arrays[1].__getitem__)
    return dict(schema=TAG+'.summary',bindings=bindings,result=result,
        scope='partial reference-forward parity and returned gradient; not completed probe or original-class parity',
        new_neural_calls=0,new_physics_calls=0,upstream_fullstrict_rebuilt=False,scientific_admission=False)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=('analyze','verify'))
    p.add_argument('--output',required=True);p.add_argument('--sha256');a=p.parse_args();v=analyze()
    out=Path(a.output)
    old.need(out.is_absolute() and out.parent==BASE and out.name.startswith(TAG+'_'),'exclusive MAIN result path')
    if a.command=='analyze':old.write(out,old.encode(v))
    else:old.need(old.read(out,a.sha256)==v,'entire deterministic saved reconstruction')
    print(old.encode(dict(path=str(out),sha256=old.sha(out),result=v['result'])).decode(),end='')


if __name__=='__main__':main()
