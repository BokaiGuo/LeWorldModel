"""Reconstruct all saved CUDA winners and bind them to the exact cp15 source."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import numpy as np
from experiments.spectral_reachability import codex_p2_prefix_recompile_saved_20261030_v1 as saved
from experiments.spectral_reachability import agent_p2_full30_control15_feedback_physical_20261029_v1 as original

NAME='codex_p2_prefix_recompile_physical_binding_20261030_v1'
ROOT=saved.host.ROOT
RUN=ROOT/'results/codex_p2_prefix_recompile_host_20261030_v1_RUN_01'
PREP=ROOT/'results/codex_p2_prefix_recompile_host_20261030_v1_PREP_01.json'
RECEIPT=ROOT/'results/codex_p2_prefix_recompile_saved_20261030_v1_RECEIPT_01.json'
REPORT_SHA='cec7be120e93f51f9e4fdd69ec0d29ab4898b5bcf479249550fd26793a684d88'
SUP_SHA='092803d14da764b2049550c778a34c6f04f4c793249744149b08110fb267cec7'
PREP_SHA='531333a5316dfdf20c4bcb628c9e0894d8c58e8af4d24e3af6157993e29ffd84'
RECEIPT_SHA='d58d1938435d1be3251e0c0b280ab46a2f0834f3b3370c89b9d78455baf508ca'
need=saved.need


def assemble(base,physical,case,branches):
    need(type(base) is int and base in range(4) and physical['job_index']==case['base']==base
         and physical['observed_context']==case['context'] and len(physical['prefix_reference'])==106,
         'same fixed cp15 context and native reference')
    parent=physical['plans'][1];raw=np.asarray(parent['commands'],dtype='<f4')
    prefix=np.asarray(case['prefix_raw'],dtype='<f4')
    need(raw.shape==(30,2) and prefix.shape==(15,2) and raw[:15].tobytes()==prefix.tobytes()
         and hashlib.sha256(prefix.tobytes()).hexdigest()==case['prefix_float32_sha256']
         and parent['task_id']==case['task_id'] and parent['root_slot']==case['root'],
         'exact original native task and 15 raw steps')
    need(len(branches)==2,'fixed two arms')
    plans=[]
    for arm,branch in zip(('joint','prefix_chain'),branches,strict=True):
        need((branch['base'],branch['arm'],branch['physical'],branch['P2_admission'])==
             (base,arm,False,False),'fixed complete saved branch')
        winner=branch['winner']
        if arm=='joint':
            suffix=bytes.fromhex(winner['candidate']['raw_float32_hex'])
            index=winner['candidate']['index'];ids=winner['candidate']['program_ids']
            need(''.join(edge['raw_float32_hex'] for edge in winner['edges'])==suffix.hex(),
                 'joint winner literal packet edges')
        else:
            suffix=bytes.fromhex(winner['first_raw_float32_hex']+winner['second_raw_float32_hex'])
            index=winner['index']
            ids=[f"prefix_{winner['prefix_index']}",f"mode_{winner['mode']}_sign_{winner['sign']:+d}"]
        need(len(suffix)==80,'exact two saved five-step packets')
        a=np.frombuffer(suffix,dtype='<f4').reshape(10,2)
        need(np.isfinite(a).all() and np.all(np.abs(a)<=1),'native raw action domain')
        plans.append(dict(base=base,arm=arm,task_id=case['task_id'],root=case['root'],
            prefix_sha256=case['prefix_float32_sha256'],feedback_context_sha256=case['feedback_context_sha256'],
            winner_index=index,program_ids=deepcopy(ids),winner_suffix_float32_hex=suffix.hex(),
            winner_suffix_sha256=hashlib.sha256(suffix).hexdigest(),
            full25_raw_float32_hex=prefix.tobytes().hex()+suffix.hex(),horizon=25,
            source='actual saved CUDA winner; no reconstruction or padding'))
    return dict(schema=NAME+'.binding',base=base,plans=plans,physical_ready=False,
        prefix_reference=deepcopy(physical['prefix_reference']),observed_context=deepcopy(case['context']),
        original_bindings=deepcopy(physical['bindings']),cuda_report_sha256=REPORT_SHA,
        cuda_receipt_sha256=RECEIPT_SHA,scientific_pass=False)


def build(base):
    need(type(base) is int and base in range(4),'base before reads')
    need(saved.host.prep.original.previous.sha(RECEIPT)==RECEIPT_SHA,'actual saved receipt hash')
    receipt=saved.consume(RUN,REPORT_SHA,SUP_SHA,PREP,PREP_SHA)
    need(receipt==json.loads(RECEIPT.read_bytes()),'independent full CUDA saved reconstruction')
    prepared=saved.host.verify(PREP,PREP_SHA)
    panel=json.loads(Path(prepared['parent']['panel_path']).read_bytes())
    physical,_=original.material(base)
    branches=[json.loads((RUN/f'branch_{base*2+i}.json').read_bytes()) for i in range(2)]
    return assemble(base,physical,panel['cases'][base],branches)
