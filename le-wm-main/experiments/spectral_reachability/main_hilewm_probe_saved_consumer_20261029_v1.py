"""MAIN saved consumer: certify original manifest plus explicit loader metadata scope.

No sealed file changes, no neural replay. Loader code was manifest-bound;
loader test/doc SHA were captured at construction, not before/after manifest.
"""
import argparse
import math
from pathlib import Path
from experiments.spectral_reachability.agent_hilewm_transition_cuda_probe_20261029_v3 import (
    scoped,read,verify,need,TAG,protocol,load_array,CAP,validate_rows,sha,ROOT,encode,write)
NAME='main_hilewm_probe_saved_consumer_20261029_v1'
LOADER={
 str(ROOT/'experiments/spectral_reachability/agent_hilewm_trusted_loader_20261028_v1.py'):
 'df313393f1ead36a512dc549f8883a1da85dd3e7ffa057f690d39627a1e7b411',
 str(ROOT/'tests/test_agent_hilewm_trusted_loader_20261028_v1.py'):
 '5f217dff654d1609025f87a608f11814ba1be6057b350b9a36c8255ec57bfda7',
 str(ROOT/'innerdocs/agent_hilewm_trusted_loader_20261028_v1.md'):
 'bee0f03bb7813bb35f03ce71fe2ae2f64b64575fb5ae0d365cc0b01455002662'}
CODE=next(iter(LOADER))


def check_loader_sources(c,m):
    need(c['own_sha256']==LOADER,'all exact original loader source receipt entries')
    need(m['sources'].get(CODE)==LOADER[CODE],'executed loader code must be original manifest bound')
    for path,h in LOADER.items():
        need(sha(path)==h,'each actual loader source/metadata byte SHA')
        need(path not in m['sources'] or m['sources'][path]==h,'no contradictory original manifest entry')
    missing=sorted(set(LOADER)-set(m['sources']))
    need(missing==sorted(set(LOADER)-{CODE}),'only fixed nonexecuted test/doc absent from original manifest')
    return dict(executed_loader_manifest_bound=True,construction_source_sha256=LOADER,
        nonexecuted_metadata_not_in_original_before_after_manifest=missing,
        metadata_current_bytes_verified=True,metadata_before_after_manifest_verified=False)


def consume(output,rh):
    output=scoped(output);r=read(output/'report.json',rh);m=verify(r['manifest'],r['manifest_sha256'])
    need(r['schema']==TAG+'.report' and r['protocol']==protocol() and r['completed'] is True and r['errors']==[]
         and r['scientific_admission'] is False,'complete actual report')
    need(r['sources_before']==r['sources_after']==m['sources'] and
        r['model_before']==r['model_after']==m['expected_model_hashes']==r['construction']['tensor_value_hashes'],
        'all source/input/export receipts and weights before/after')
    need(set(r['runtime'])=={'torch','cuda','device','peak_allocated_bytes','peak_reserved_bytes'} and
         all(type(r['runtime'][k]) is int and r['runtime'][k]>=0 for k in ('peak_allocated_bytes','peak_reserved_bytes')),
         'actual runtime and GPU allocation scopes')
    c=r['construction'];need(c['export_report_sha256']==m['export']['export_report_sha256'] and
        c['export_receipt_sha256']==m['export']['export_receipt_sha256'] and c['all_parameters_frozen'] is True and c['forward_calls']==0,
        'original frozen loader construction receipt')
    metadata_scope=check_loader_sources(c,m)
    need(all(c[k]==v for k,v in m['construction_expected'].items()) and
         c['tensor_copy_attempts']==c['tensor_copy_returns']==425 and c['device']==r['runtime']['device'] and
         c['torch']==r['runtime']['torch'],
         'full loader counts/source/runtime closure')
    need(set(r['timings'])=={'initial_saved_verify','construct','model_hash_before','dispatch_all5','model_hash_after','final_saved_verify'}
         and all(e['attempted'] is True and e['returned'] is True and type(e['seconds']) in (int,float)
         and math.isfinite(e['seconds']) and e['seconds']>=0 for e in r['timings'].values()),'complete actual timing scope ledger')
    arrays={n:load_array(output,n,d) for n,d in r['arrays'].items()}
    need({p.name for p in output.iterdir()}==set(arrays)|{'report.json'} and
        sum(p.stat().st_size for p in output.iterdir())<=CAP,'complete bounded directory census')
    refs=set()
    def load(n):need(n in arrays,'actual array reference');refs.add(n);return arrays[n]
    result=validate_rows(m,r,load);need(result==r['diagnostics'] and refs==set(arrays),'full diagnostics and array census')
    need(sha(output/'report.json')==rh and verify(r['manifest'],r['manifest_sha256'])==m,'report/authority stable')
    return dict(schema=NAME+'.consume',loader_source_scope=metadata_scope,checker_source_sha256=sha(__file__),status='fixed_owned_low_probe_saved_verified_not_official_parity',
        report_sha256=rh,manifest_sha256=r['manifest_sha256'],diagnostics=result,resources=r['resources'],scientific_admission=False)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',required=True);p.add_argument('--report-sha256',required=True)
    p.add_argument('--receipt',required=True);a=p.parse_args()
    target=Path(a.receipt)
    need(target.is_absolute() and target.parent==ROOT/'results' and target.name.startswith(NAME+'_'),'own new receipt')
    result=consume(a.output,a.report_sha256)
    write(target,encode(result))
    print(encode(dict(receipt=str(target),sha256=sha(target),diagnostics=result['diagnostics'])).decode(),end='')


if __name__=='__main__':main()

