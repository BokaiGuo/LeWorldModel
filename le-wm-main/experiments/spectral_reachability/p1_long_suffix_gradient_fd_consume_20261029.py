"""Rebuild fixed CUDA directional audit from saved vectors and source identities."""
import argparse
import json
import numpy as np
from experiments.spectral_reachability import p1_long_suffix_gradient_fd_20261029 as producer


def verify(value, report, prepared, long_result):
    source = producer.consumer.producer.one.source
    if (value['schema'] != producer.NAME or value['source_result_sha256'] != producer.RESULT_SHA
        or value['source_report_sha256'] != source.REPORT_SHA
        or value['epsilon'] != float(producer.EPS)
        or value['P1_admission'] is not False or value['training'] is not False
        or value['calls'] != dict(predict=40,action_encoder=40,reverse=4)):
        raise ValueError('source/call binding')
    if len(value['rows']) != 4 or value['model_before'] != value['model_after'] or value['model_before'] != report['model_before']:
        raise ValueError('roster/model mismatch')
    maximum=0.
    for slot,row in enumerate(value['rows']):
        index=slot*40; source=long_result['rows'][index]['evaluations'][0]
        if row['slot']!=slot or row['arm_index']!=slot*2 or not np.array_equal(row['nominal'],source['endpoint']) or not np.array_equal(row['gradient'],source['raw_gradient']):
            raise ValueError('nominal or gradient mismatch')
        raw=np.asarray(source['raw'],np.float32);grad=np.asarray(row['gradient'],np.float64)
        goal=np.asarray(prepared['requests'][slot]['task']['goal'],np.float64)
        if len(row['directions'])!=2: raise ValueError('direction denominator')
        for packet,d in enumerate(row['directions']):
            plus=raw.copy();minus=raw.copy();plus[packet]+=producer.EPS;minus[packet]-=producer.EPS
            if d['packet']!=packet or not np.array_equal(d['plus_raw'],plus) or not np.array_equal(d['minus_raw'],minus): raise ValueError('perturbation')
            pp=np.asarray(d['plus_endpoint'],np.float64);mm=np.asarray(d['minus_endpoint'],np.float64)
            if pp.shape!=(192,) or mm.shape!=(192,) or not np.isfinite(pp).all() or not np.isfinite(mm).all(): raise ValueError('endpoints')
            lp=float(np.mean((pp-goal)**2));lm=float(np.mean((mm-goal)**2))
            calculated=producer.difference(lp,lm,grad,plus,minus)
            for k,v in dict(plus_loss=lp,minus_loss=lm,**calculated).items():
                if v is None:
                    if d[k] is not None: raise ValueError('null mismatch')
                elif not np.isclose(v,d[k],rtol=1e-12,atol=1e-12): raise ValueError('scalar mismatch')
            if calculated['relative_error'] is not None: maximum=max(maximum,calculated['relative_error'])
    return dict(saved_arithmetic=True,directions=8,max_relative_error=maximum,gradient_recomputed=False,hardware_attested=False,physical_gradient=False,P1_admission=False)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--result',required=True);parser.add_argument('--sha256',required=True)
    a=parser.parse_args();consumer=producer.consumer; source,saved=consumer.producer.one.source,consumer.producer.one.saved
    value=saved.read_verified(a.result,a.sha256);report=saved.read_verified(source.REPORT,source.REPORT_SHA)
    receipt=saved.read_verified(source.RECEIPT,source.RECEIPT_SHA)
    if receipt['report_sha256']!=source.REPORT_SHA: raise ValueError('receipt')
    prepared=saved.read_verified(report['prepared'],report['prepare_sha256']);saved.analyze(report,receipt,prepared)
    long_result=saved.read_verified(source.ROOT/'results/p1_long_suffix_host_20261029_MAIN01.json',producer.RESULT_SHA)
    consumer.verify(long_result,report,prepared)
    print(json.dumps(verify(value,report,prepared,long_result)))


if __name__=='__main__':main()
