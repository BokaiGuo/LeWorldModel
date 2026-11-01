"""Streaming fullarray verifier for the new paired signal-before-compile ABI."""
from copy import deepcopy
from experiments.spectral_reachability import agent_p3_paired_quota_runtime_20261028_v1_gate as wiring

core,contract,paid,compiler=wiring.core,wiring.contract,wiring.paid,wiring.compiler
require,digest,canonical,sha=wiring.require,wiring.digest,wiring.canonical,wiring.sha

def replay(g,e):
    kind=e['kind']
    if kind=='pair_opened':g.open_pair()
    elif kind=='attempted':
        a=e['entry'];g.begin(a['ticket'],a['resource'],shape=a['shape'],directions=a['directions'])
    elif kind=='returned':g.returned(e['operation_id'])
    elif kind=='signal_sealed':g.signal_done(e['ticket'],e['decision'],e['signals'],e['signal_sha256'],e['resource_sha256'])
    elif kind=='pair_sealed':g.pair_seal()
    elif kind=='finished':g.finish(e['ticket'])
    elif kind=='pair_closed':g.close_pair()
    else:raise ValueError('failure or unknown gate event cannot certify successful panel')
    require(g.events[-1]==e,'full exact chained gate event reconstruction')

def signal(p,slot,row,trace):
    child=p['wiring']['children'][slot//2];mode=contract.MODES[slot%2]
    task=p['wiring']['tasks'][child['root_slot']]
    initial=wiring.initial_rows(p)[slot];core.keys(row,tuple(initial))
    require(row['child_id']==child['child_id'] and row['mode']==mode and row['status']=='paid_signal' and
        row['error'] is row['compiler'] is row['execution_quota'] is row['pair_seal_sha256'] is None,
        'complete independent signal snapshot BEFORE any compiler')
    a=dict(group=child['child_id']+':'+mode,status='returned',error=None,backend_kind='HOST_CUDA',
        signal=row['signal'],resource_receipt=row['signal_resources'])
    signals=paid.validate_trace(task,child,a,trace,1)
    require(row['signal_resources']['caps']==row['signal_resources']['attempted']==row['signal_resources']['completed']==contract.signal_caps(1),
        'exact independent8 signal, no reuse or double count')
    require(all(r['producer_sha256']==sha(paid.__file__) and r['binding_sha256']==digest(task['binding']) and
        r['parent_context_sha256']==child['context_sha256'] for r in row['signal']['responses']),
        'same child paid native ancestry and goal, no root/sibling response')
    schedule=[e for e in row['signal_resources']['events'] if e['resource']=='schedule']
    require(len(schedule)==1 and schedule[0]['scope']==dict(child_id=child['child_id'],group=mode,mode=mode,
        phase='schedule',parent_context_sha256=child['context_sha256']),'one original method schedule')
    require(row['decision']==core.choose(task,dict(context=child['context'],depth=0),signals,mode),
        'untouched original paid chooser')

def validate_stream(events,p,report):
    g=wiring.Gate(p['wiring']);stage='denominator';pair=0;expect=None;trace=[];trace_bytes=0
    maps={};signal_rows={};compiled_rows={};sealed_signals=set();finished=set();tail='0'*64
    census=dict(compiled=0,paired_paid_refusal=0,not_reached=0)
    for event,tail in events:
        kind=event['kind']
        if stage=='denominator':
            require(event==dict(kind='denominator',rows=wiring.initial_rows(p),fixed_denominator=48,child_denominator=24,
                sources_sha256=digest(p['sources'])),'full48 durable before all work');stage='before';continue
        if stage=='before':
            require(event==dict(kind='model_before',hashes=report['model_before']),'frozen model before work');stage='work';continue
        if kind=='work':
            core.keys(event,('kind','slot','payload'))
            require(stage=='work' and pair<24 and type(event['slot']) is int and event['slot']//2==pair,
                'exact same contiguous child pair trace')
            slot=event['slot'];e=event['payload'];c=p['wiring']['children'][pair]
            if e['kind']=='budget_gate':
                core.keys(e,('kind','payload'));entry=e['payload'];k=entry['kind']
                if expect is not None:
                    require(expect[0]=='gate_return' and k=='returned' and entry['operation_id']==expect[1],
                        'native return immediately followed by exact gate completion');expect=None
                else:require(k!='returned','no premature gate return')
                if k=='pair_opened':require(slot==2*pair and not trace,'pair open at first slot')
                elif k=='attempted':
                    require(slot==2*pair+g.turn and g.stage in ('signal','compile') and
                        (g.stage=='signal' and slot not in signal_rows or g.stage=='compile' and slot not in compiled_rows),
                        'actual entry in current unsealed method phase')
                elif k=='signal_sealed':
                    require(slot in signal_rows and slot not in sealed_signals and not trace,
                        'full validated signal_return BEFORE signal seal')
                    row=signal_rows[slot]
                    require(entry['decision']==row['decision'] and entry['signals']==row['signal']['scheduling_signal'] and
                        entry['signal_sha256']==digest(row['signal']) and entry['resource_sha256']==digest(row['signal_resources']),
                        'pairseal uses independently verified signal, not summary')
                    sealed_signals.add(slot);maps={}
                elif k=='pair_sealed':
                    require(sealed_signals=={2*pair,2*pair+1} and slot==2*pair+1 and not trace,
                        'both full signals before durable pairseal');maps={}
                elif k=='finished':
                    require(slot in compiled_rows and slot not in finished and not trace,'full arrays BEFORE finish')
                    finished.add(slot);maps={}
                elif k=='pair_closed':require(not trace,'closed complete phases')
                replay(g,entry)
                if k=='attempted':expect=('native_attempt',entry['entry'])
            else:
                require(g.stage in ('signal','compile') and slot==2*pair+g.turn and
                    (g.stage=='signal' and slot not in signal_rows or g.stage=='compile' and slot not in compiled_rows),
                    'no arrays outside current paid stage')
                if expect is not None:
                    require(expect[0]=='native_attempt' and e['kind']=='resource_attempt',
                        'predispatch directly before actual call')
                    a=expect[1];n=e['entry']
                    require(all(a[k]==n[k] for k in ('resource','shape','directions','delta')) and n['index'] not in maps,
                        'gate and actual native operation bijection')
                    maps[n['index']]=a['operation_id'];expect=None
                else:require(e['kind']!='resource_attempt','no ungated actual call')
                if e['kind']=='resource_return':
                    ident=e['entry']['index'];require(ident in maps and maps[ident] in g.pending,'actual unique attempted return')
                    expect=('gate_return',maps[ident])
                trace_bytes+=len(canonical(e));require(trace_bytes<=p['profile']['slot_trace_bytes'],'bounded phase full arrays')
                trace.append(e)
        elif kind in ('signal_return','compile_return'):
            core.keys(event,('kind','slot','row'));slot=event['slot']
            require(stage=='work' and pair<24 and type(slot) is int and slot==2*pair+g.turn and
                expect is None and not g.pending,'complete actual returns before phase snapshot')
            row=event['row'];child=p['wiring']['children'][pair];mode=contract.MODES[slot%2]
            if kind=='signal_return':
                require(g.stage=='signal' and slot not in signal_rows,'one independent signal return')
                signal(p,slot,row,trace);signal_rows[slot]=deepcopy(row)
            else:
                require(g.stage=='compile' and slot not in compiled_rows and slot in sealed_signals,
                    'compiler only after paired paid seal')
                seal=g.seals[-1];expected=deepcopy(signal_rows[slot])
                expected.update(status='compiled',execution_quota=seal['execution_quotas'][slot%2],pair_seal_sha256=digest(seal),compiler=row['compiler'])
                require(row==expected and seal['dispatch_allowed'],'unchanged original decision/signal and exact execution quota')
                branch=wiring.branch_for(child,mode,row)
                compiler.validate_branch(p['wiring']['tasks'][child['root_slot']],child,branch,row['compiler'],trace)
                compiled_rows[slot]=deepcopy(row)
            trace=[];trace_bytes=0
        elif kind=='pair_return':
            core.keys(event,('kind','pair','rows'))
            require(stage=='work' and type(event['pair']) is int and event['pair']==pair and g.cursor==pair+1 and
                g.stage=='closed' and not trace and expect is None,'one complete pair before advance')
            rows=report['session']['rows'][2*pair:2*pair+2]
            require(event['rows']==rows,'durable full pair matches complete report')
            child=p['wiring']['children'][pair]
            for j,row in enumerate(rows):
                slot=2*pair+j
                if child['status']=='not_reached':
                    require(row==wiring.initial_rows(p)[slot] and not signal_rows and not compiled_rows,'missing stays unknown without paid work')
                    census['not_reached']+=1
                elif g.seals[-1]['dispatch_allowed']:
                    require(slot in finished and row==compiled_rows[slot],'both full compiled rows');census['compiled']+=1
                else:
                    expected=deepcopy(signal_rows[slot]);seal=g.seals[-1]
                    expected.update(status='paired_paid_refusal',execution_quota=seal['execution_quotas'][j],pair_seal_sha256=digest(seal))
                    require(row==expected and not compiled_rows and not finished,'both paid refused; no candidate subset')
                    census['paired_paid_refusal']+=1
            pair+=1;signal_rows={};compiled_rows={};sealed_signals=set();finished=set();maps={}
        elif kind=='model_after':
            require(stage=='work' and pair==24 and event==dict(kind='model_after',hashes=report['model_after']),
                'all48 before final model guard');stage='sources'
        elif kind=='sources_final':
            require(stage=='sources' and event==dict(kind='sources_final',sha256=digest(p['sources'])),'final source closure');stage='done'
        else:raise ValueError('unknown/duplicate/out of order paired journal record')
    require(stage=='done' and tail==report['journal_tail'] and g.cursor==24 and not g.failed,
        'complete journal hash chain/24pairs/48slots')
    require(canonical(g.receipt())==canonical(report['session']['gate']),'entire reconstructed gate receipt')
    actual={m:dict(attempted=wiring.zero(),completed=wiring.zero()) for m in contract.MODES}
    for row in report['session']['rows']:
        for receipt in (row['signal_resources'],None if row['compiler'] is None else row['compiler']['resource_receipt']):
            if receipt is None:continue
            core.audit_ledger(receipt)
            for k in ('attempted','completed'):actual[row['mode']][k]=wiring.add(actual[row['mode']][k],receipt[k])
    require(actual==g.actual,'signal plus compiler actual equals gate mirror, never sum gate twice')
    require(all(actual['fixed'][k][axis]==actual['adaptive'][k][axis] for k in ('attempted','completed')
        for axis in ('predict_API','predict_samples')),'actual matched predict API/sample panel, NOT FLOPs')
    return census,actual
