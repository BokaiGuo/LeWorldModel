from copy import deepcopy
import pytest
from experiments.spectral_reachability import agent_p1_midpoint_physical_seam_20261029_v1 as m
import test_agent_observed_suffix_physical_20261028_v1 as old


@pytest.fixture(scope='module')
def inputs():
    _,_,c,_,_,bp,br,_=old.fixture()
    args=dict(host_output='/RAM/output',host_report_sha='a'*64,host_prepared='/RAM/prepare',host_prepare_sha='b'*64,
              host_receipt='/RAM/receipt',host_receipt_sha='c'*64)
    rows=[];audits=[]
    for i,s in enumerate(c['rows']):
        oldplan=bp['plans'][i];cmd=deepcopy(oldplan['commands'])
        s['request']['incumbent']['commands']=deepcopy(cmd[5:])
        chosen=None if i==0 else 1
        selected=None if chosen is None else cmd
        numeric={'iterates':[{},dict(raw_bits=m.host.F.bits(cmd[5:]))]}
        result=dict(selected_iteration=chosen,selected_commands=selected,status='abstain' if chosen is None else 'model_candidate',
                    original=dict(numeric=numeric))
        rows.append(dict(index=i,status='sealed',result=result))
        audits.append(dict(selected_iteration=chosen,selected_commands=selected,midpoint_count=8))
    p=dict(schema=m.host.NAME+'.prepare',contract=m.host.contract(),production_ready=True,original_prepare=c)
    r=dict(schema=m.host.NAME+'.report',status='complete_model_only',errors=[],fixed_denominator=4,prepared=args['host_prepared'],
           prepare_sha256=args['host_prepare_sha'],rows=rows,P1_admission=False,freshness=False,physical_release=False)
    u=dict(schema=m.host.NAME+'.consume',status='verified_model_only',complete=True,report_sha256=args['host_report_sha'],
           fixed_denominator=4,known_roots=2,rows=deepcopy(rows),audits=audits,
           actual_new_compute={k:m.host.F.caps(104,32) for k in ('attempted','completed')},
           P1_admission=False,freshness=False,physical_release=False)
    return args,p,r,u,bp,br


def test_four_slots_abstention_and_full_raw(inputs):
    p=m.assemble(*deepcopy(inputs))
    assert len(p['plans'])==4 and p['plans'][0]['status']=='abstain' and p['plans'][0]['commands'] is None
    assert all(len(q['commands'])==10 for q in p['plans'][1:])
    assert not p['physical_release'] and not p['P1_admission']


@pytest.mark.parametrize('change',['partial','release','wrong_root','wrong_cp5','raw','audits','count'])
def test_fail_closed(inputs,change):
    args,p,r,u,bp,br=deepcopy(inputs)
    if change=='partial':u['complete']=False
    elif change=='release':u['physical_release']=True
    elif change=='wrong_root':br['rows'][1]['root_sha256']='0'*64
    elif change=='wrong_cp5':br['rows'][1]['checkpoints']=[x for x in br['rows'][1]['checkpoints'] if x['step']!=5]
    elif change=='raw':r['rows'][1]['result']['selected_commands'][6][0]+=.1;u['rows']=deepcopy(r['rows']);u['audits'][1]['selected_commands']=deepcopy(r['rows'][1]['result']['selected_commands'])
    elif change=='audits':u['audits'].pop()
    else:u['actual_new_compute']['completed']['predict_API']-=1
    with pytest.raises((ValueError,KeyError)):m.assemble(args,p,r,u,bp,br)
