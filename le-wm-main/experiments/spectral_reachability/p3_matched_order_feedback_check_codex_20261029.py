"""Independent structural callback transcript check; never physical certification."""
from experiments.spectral_reachability import p3_matched_order_feedback_seam_codex_20261029 as seam


def check(result, events):
    seam.need(result['schema'] == seam.NAME+'.slot' and
              result['status'] == 'callback_feedback_complete_not_certified', 'complete callback slot only')
    seam.need(not result['errors'] and result['source_authenticated'] is False and
              result['native_execution_certified'] is False and result['scientific_pass'] is False, 'uncertified boundary')
    blocks, packets = result['blocks'], result['packets']
    seam.need(len(blocks) == 2 and len(packets) == 4 and
              sorted(b['horizon'] for b in blocks) == [1, 3], 'two matched blocks and four packets')
    seam.need(result['policy'] in ('fixed_order','signal_order'), 'policy')
    if result['policy'] == 'fixed_order':
        seam.need([b['horizon'] for b in blocks] == [1,3], 'fixed order')
    for stage, b in enumerate(blocks):
        seam.need(b['stage'] == stage and len(b['candidates']) == 4, 'block identity')
        ranked = min(b['candidates'], key=lambda c: (c['cost'], c['index']))
        seam.need(b['winner_index'] == ranked['index'] and b['actions'] == ranked['actions'], 'saved model winner')
        selected = [p for p in packets if p['stage'] == stage]
        seam.need(len(selected) == b['horizon'] and
                  [x for p in selected for x in p['raw']] == b['actions'], 'all selected raw actions')
    seam.need([(p['start'],p['end'],p['stopped']) for p in packets] ==
              [(i*5,(i+1)*5,False) for i in range(4)], 'contiguous returned packets')
    actual = [e for e in events if e['kind'] == 'actual_context']
    returns = [e for e in events if e['kind'] == 'packet_return']
    intents = [e for e in events if e['kind'] == 'packet_intent']
    compiled = [e for e in events if e['kind'] == 'block_compiled']
    seam.need(len(actual) == len(returns) == len(intents) == 4, 'actual context per packet')
    seam.need(len(compiled) == 2, 'two compile records')
    positions = {id(e): i for i,e in enumerate(events)}
    for stage,b in enumerate(blocks):
        e = compiled[stage]
        seam.need(e == dict(kind='block_compiled', stage=stage, context_sha256=b['context_sha256'],
                            horizon=b['horizon'], actions=b['actions']), 'compile event binding')
    seam.need(positions[id(compiled[0])] < positions[id(intents[0])], 'compile before first dispatch')
    for i, (a, r, intent, packet) in enumerate(zip(actual, returns, intents, packets, strict=True)):
        seam.need(positions[id(intent)] < positions[id(r)] < positions[id(a)] and
                  (i == 3 or positions[id(a)] < positions[id(intents[i+1])]) and
                  all(r[k] == packet[k] for k in ('stage','start','end','raw','stopped')) and
                  intent['stage'] == packet['stage'] and intent['start'] == packet['start'] and
                  intent['raw'] == packet['raw'] and a['end'] == packet['end'], 'ordered intent return observation')
    boundary = 5*blocks[0]['horizon']
    first_end = next(a for a in actual if a['end'] == boundary)
    seam.need(blocks[1]['context_sha256'] == first_end['context_sha256'] and
              positions[id(first_end)] < positions[id(compiled[1])] < positions[id(intents[blocks[0]['horizon']])],
              'second block uses actual joined context')
    seam.core.audit_ledger(result['resource_receipt'])
    seam.need(result['resource_receipt']['attempted'] == result['resource_receipt']['completed'] == seam.design.expected(), 'matched measured axes')
    return dict(status='callback_structure_only', physical_certified=False, scientific_pass=False)
