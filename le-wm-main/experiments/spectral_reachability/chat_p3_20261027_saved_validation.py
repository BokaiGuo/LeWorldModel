"""Actual saved-only validation after queue seal; no candidate scoring."""
import gzip
import json
import time
from pathlib import Path
from experiments.spectral_reachability import chat_p3_20261027_scheduler as s


def main():
    start = time.monotonic()
    seal = s.ROOT/'results/chat_p3_20261027_queue_seal.json'
    expected = '2f39483bceb87e638f74b2da170db8cf7ddbf6a1050df4137bbbcf9dde37ed78'
    if s.digest(seal) != expected or json.loads(seal.read_text()) != s.build():
        raise ValueError('external seal or saved closure drift')
    prepared = json.loads(seal.read_text())
    with gzip.open(s.OLD/'panel.json.gz', 'rt') as f:
        panel = json.load(f)
    mappings = {'U':'U_uniform', 'D_rank':'D_geometry', 'S_rank':'S_swapped', 'N':'N_no_signal_uniform'}
    consumed = {}
    old_winners = {}
    for arm, previous in mappings.items():
        rows = [r for r in panel['rows'] if r['arm'] == previous]
        candidates = [{'slot':r['source_slot'], 'index':c['index'], 'cost':c['cost']}
                      for r in rows for c in r['candidates']]
        consumed[arm] = s.consume_saved(prepared['plans'][arm]['jobs'], candidates)
        old_winners[arm] = [r['winner_index'] for r in sorted(rows,key=lambda r:r['source_slot'])]
    out = {'schema':'chat_p3_20261027.saved_validation.v1',
           'seal_sha256':expected, 'source_sha256':s.digest(Path(__file__)),
           'status':'separated_queues_missing_new_tail_scores_not_admitted',
           'old_weight_fraction':prepared['weights'][0]/sum(prepared['weights']),
           'old_allocations':prepared['old_allocations'], 'old_winners':old_winners,
           'new_allocations':{a:p['blocks'] for a,p in prepared['plans'].items()},
           'saved_queue_consumption':consumed, 'paired_gain':None,
           'new_resources':{'predict_API':0,'predictor_samples':0,'encoder':0,'AD':0,
                            'model_SVD':0,'physics':0,'wall_seconds':time.monotonic()-start},
           'shared_historical_predict_API':8,
           'historical_predictor_samples':'not independently inferred from API count; see native hook arrays',
           'historical_encoder':8,'historical_projector':8,'historical_anchor_predict_API':288,
           'historical_AD_directions':384,'historical_J':2,'historical_SVD':2,
           'new_full_run_expected_predict_API':520,
           'equal_prediction_budget_design_only':True,'equal_FLOPs':False,
           'P1_admission':False,'P3_admission':False,'freshness':False,
           'compute_advantage':False,'walltime_advantage':False}
    output=s.ROOT/'results/chat_p3_20261027_saved_validation.json'
    with output.open('xb') as f:
        f.write(s.canonical(out)+b'\n')
    print(json.dumps({'output':str(output),'sha256':s.digest(output),
                      'missing':{a:len(r.get('missing',[])) for a,r in consumed.items()}}))


if __name__ == '__main__':
    main()
