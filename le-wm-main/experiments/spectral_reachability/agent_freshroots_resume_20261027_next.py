"""Incremental saved dispatch receipts and bounded historical metadata delta.

No historical commands are executed, no models/environments imported. A
returned process is not a reset ledger; metadata scope is not global authority.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import gzip
from io import BytesIO
import json
from pathlib import Path
import re
import sys
import zlib

NAME = 'agent_freshroots_resume_20261027_next'
ROOT = Path(__file__).resolve().parents[2]
PREVIOUS = 'results/agent_freshroots_resume_20261027_coverage.json'
PREVIOUS_SHA = '00a3fffeaaa8943e851d129b941a7f07feaa1db387117beb98110e4ec95f5eeb'
PREVIOUS_CODE = 'experiments/spectral_reachability/agent_freshroots_resume_20261027_resolver.py'
PREVIOUS_CODE_SHA = '941434dc406605e9499c54e8fc7758ed0fc28f5ddc5c7ee698c50c3242a4406c'
INVENTORY = 'results/moving_authority_inventory_bridge_20261027_v1.inventory.json'
INVENTORY_SHA = '9fc0218a8c39b194abac363ca59bb7877323bdcf882e8aefd54a1c9effb8bd9a'
FILE_CAP = 128*1024**2
TOTAL_CAP = 1024*1024**2
OUTPUT_CAP = 8*1024**2
SEMANTIC_CAP = 64*1024**2
ENTRY_CAP = 4096
LINE_CAP = 8*1024**2
INFLATE_CAP = 16*1024**2
INFLATE_TOTAL_CAP = 256*1024**2


def need(ok, message):
    if not ok:
        raise ValueError(message)


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def encode(value):
    return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()


def parse(raw):
    def unique(items):
        result={}
        for key,value in items:
            need(key not in result,'duplicate JSON key')
            result[key]=value
        return result
    return json.loads(raw,object_pairs_hook=unique,
        parse_constant=lambda _:need(False,'nonfinite JSON'))


def pack_result(result):
    raw=encode(result);need(len(raw)<=SEMANTIC_CAP,'semantic result cap')
    if len(raw)<=OUTPUT_CAP:return raw
    return encode(dict(schema=NAME+'.lossless-envelope.v1',semantic_bytes=len(raw),
        semantic_sha256=digest(raw),gzip_base64=base64.b64encode(
            gzip.compress(raw,mtime=0)).decode()))


def unpack_result(raw):
    value=parse(raw)
    if value.get('schema')!=NAME+'.lossless-envelope.v1':return value
    need(set(value)=={'schema','semantic_bytes','semantic_sha256','gzip_base64'},'envelope fields')
    binary=base64.b64decode(value['gzip_base64'],validate=True)
    z=zlib.decompressobj(31);decoded=z.decompress(binary,SEMANTIC_CAP+1)
    need(z.eof and not z.unconsumed_tail and not z.unused_data and
         len(decoded)<=SEMANTIC_CAP,'single result gzip and semantic cap')
    need(len(decoded)==value['semantic_bytes'] and digest(decoded)==value['semantic_sha256'],
         'result semantic binding')
    return parse(decoded)


def literal(path):
    path=Path(path)
    need(path.is_absolute() and '..' not in path.parts and path.resolve()==path
         and path.is_relative_to(ROOT/'results') and path.is_file() and not path.is_symlink(),
         'literal regular results path')
    return path


def read(path,expected,cap=FILE_CAP):
    path=Path(path)
    need(path.stat().st_size<=cap,'file byte cap')
    raw=path.read_bytes()
    need(digest(raw)==expected,'external SHA mismatch: '+str(path))
    return raw


def previous():
    read(ROOT/PREVIOUS_CODE,PREVIOUS_CODE_SHA)
    from experiments.spectral_reachability import agent_freshroots_resume_20261027_resolver as old
    value=parse(read(ROOT/PREVIOUS,PREVIOUS_SHA))
    for binding in value['implementation_bindings']:
        read(binding['path'],binding['sha256'])
    need(value['scope']['historical_sources']==8 and value['scope']['inherited_blockers']==16,
         'previous 8/16 denominator')
    need(len(value['recovery']['dispatches'])==10,'previous dispatch denominator')
    return old,value


def eligible(path):
    return (path.suffix=='.json' and not path.name.startswith(NAME) and
            (path.name in ('report.json','sourcepack.json','plan.json','protocol.json') or
             'prepare' in path.name or 'request' in path.name))


def capture_scope():
    """Filename-only finite rule fixed before projecting any delta contents."""
    old,value=previous()
    inventory=parse(read(ROOT/INVENTORY,INVENTORY_SHA))
    prior={row['path'] for row in inventory['files']}
    paths=sorted(p for p in (ROOT/'results').rglob('*.json')
                 if eligible(p) and p.is_file() and str(p) not in prior)
    need(len(paths)<=ENTRY_CAP,'delta entry cap')
    files=[];used=0
    for p in paths:
        literal(p);size=p.stat().st_size;used+=size
        need(size<=FILE_CAP and used<=TOTAL_CAP,'delta metadata byte cap')
        h=hashlib.sha256()
        with p.open('rb') as stream:
            for block in iter(lambda:stream.read(1024*1024),b''):
                h.update(block)
        need(p.stat().st_size==size,'changed during scope capture')
        files.append(dict(path=str(p),sha256=h.hexdigest(),bytes=size))
    ranges=[]
    for binding in value['log_bindings']:
        path=Path(binding['path']);size=path.stat().st_size
        need(binding['bytes']<=size<=650*1024**2,'bounded original log append')
        full=hashlib.sha256();prefix=hashlib.sha256();tail=hashlib.sha256();offset=0
        with path.open('rb') as stream:
            while offset<size:
                block=stream.read(min(1024*1024,size-offset));need(block,'short log capture')
                full.update(block);boundary=max(0,min(len(block),binding['bytes']-offset))
                prefix.update(block[:boundary]);tail.update(block[boundary:]);offset+=len(block)
        need(prefix.hexdigest()==binding['sha256'],'original log prefix rewritten')
        ranges.append(dict(path=str(path),bytes=size,sha256=full.hexdigest(),
            previous_bytes=binding['bytes'],previous_prefix_sha256=prefix.hexdigest(),
            append_bytes=size-binding['bytes'],append_sha256=tail.hexdigest()))
    return dict(schema=NAME+'.scope.v1',previous_sha256=PREVIOUS_SHA,
        inventory_sha256=INVENTORY_SHA,files=files,
        log_ranges=ranges,
        selection='sorted literal results JSON report/sourcepack/plan/protocol or prepare/request filenames minus original 618 paths and own stem',
        hash_capture_only=True,contents_projected_before_scope=False,
        complete_global_history=False,neural_calls=0,physics_calls=0)


def operations(arguments,old):
    """Find literal tool invocations; unsupported control flow stays unassigned."""
    ops=[]
    # Parenthesis/string scanner, never eval JS or interpolate its literals.
    for match in re.finditer(r'\btools\.(exec_command|write_stdin|apply_patch)\s*\(',arguments):
        i=match.end();start=i;depth=1;quote=None;escape=False
        while i<len(arguments) and depth:
            ch=arguments[i]
            if quote:
                if escape:escape=False
                elif ch=='\\':escape=True
                elif ch==quote:quote=None
            elif ch in ('"',"'",'`'):quote=ch
            elif ch=='(':depth+=1
            elif ch==')':depth-=1
            i+=1
        need(depth==0,'truncated tool invocation')
        text=arguments[start:i-1];kind=match.group(1)
        op=dict(kind=kind)
        if kind=='exec_command':
            # Force literal-field extraction for JS {cmd:...}, which is not a
            # JSON object. The frozen helper's JSON-object error path returns [].
            commands=old.dispatch_commands('wrapper('+text+')')
            op['command']=commands[0] if len(commands)==1 else None
        if kind=='write_stdin':
            session=re.search(r'(?:"session_id"|\bsession_id)\s*:\s*(\d+)\b',text)
            op['session_id']=int(session.group(1)) if session else None
        ops.append(op)
    return ops


def texts(output):
    if isinstance(output,str):
        # New formats can wrap text blocks in a JSON string; only wrappers,
        # never arbitrary stdout, are parsed here.
        if output.startswith('[{"type":'):
            output=parse(output)
        else:
            return [output]
    need(isinstance(output,list),'known tool output wrapper')
    result=[]
    for block in output:
        if isinstance(block,dict) and block.get('type') in ('input_text','text'):
            text=block.get('text','')
            if not text.startswith(('Script completed','Script running with cell ID')):
                result.append(text)
    return result


def receipt(text,old):
    result=dict(kind='stdout_only',exit_code=None,session_id=None,
                exceptions=[],producer_traceback_frames=[])
    if text.lstrip().startswith('{"chunk_id"'):
        # Only transport metadata leaves; .output (potential labels) is skipped.
        p=old.project(text.encode(),[(k,) for k in ('chunk_id','exit_code','session_id')])
        result.update(kind='exec_transport',exit_code=p.get(('exit_code',)),
                      session_id=p.get(('session_id',)))
    # Failure diagnostics only. No scalar scientific outputs are projected.
    # Transport .output can contain escaped line breaks/quotes; normalization
    # is only for these anchored traceback/error lexemes, not JSON labels.
    normalized=text.replace('\\n','\n').replace('\\"','"')
    result['exceptions']=re.findall(r'(?m)^((?:[A-Za-z_]+Error|[A-Za-z_]+Exception):[^\n]{0,300})',normalized)
    for path,line,function in re.findall(r'File "([^"\n]+)", line (\d+), in ([^\n]+)',normalized):
        if path in (str(ROOT/old.P0),str(ROOT/old.P1)):
            result['producer_traceback_frames'].append(dict(path=path,line=int(line),function=function))
    if result['exceptions']:
        result['kind']='traceback_failure' if result['kind']=='stdout_only' else result['kind']
    return result


def join_records(records,prior,old):
    """Join by exact call_id, operation order and session_id, not proximity."""
    dispatches=prior['recovery']['dispatches']
    patches=prior['recovery']['edits']
    wanted={d['call_id'] for d in dispatches}|{d['call_id'] for d in patches}
    calls={};active={};results=[];patch_receipts=[];unassigned=[]
    for event in records:
        p=event['payload'];kind=p.get('type');cid=p.get('call_id')
        if kind in ('function_call','custom_tool_call'):
            arg=p.get('arguments',p.get('input',''))
            if not isinstance(arg,str):continue
            if cid not in wanted:
                polled={int(x) for x in re.findall(r'(?:"session_id"|\bsession_id)\s*:\s*(\d+)\b',arg)}
                if not any((event['log'],sid) in active for sid in polled):continue
            ops=operations(arg,old)
            if cid in wanted or any(op['kind']=='write_stdin' and
                (event['log'],op.get('session_id')) in active for op in ops):
                calls[cid]=dict(event=event,ops=ops)
        elif kind in ('function_call_output','custom_tool_call_output') and cid in calls:
            call=calls[cid];blocks=texts(p.get('output',''))
            if len(blocks)!=len(call['ops']):
                unassigned.append(dict(call_id=cid,line=event['line'],
                    reason='tool_operation_output_arity_mismatch',operations=len(call['ops']),blocks=len(blocks)))
                continue
            for index,(op,block) in enumerate(zip(call['ops'],blocks)):
                rec=receipt(block,old)
                ref=dict(log=event['log'],line=event['line'],call_id=cid,operation_index=index,
                         line_sha256=event['line_sha256'])
                if op['kind']=='apply_patch' and cid in wanted:
                    patch_receipts.append(dict(**ref,receipt_kind=rec['kind'],
                        empty_response=block=='{}',actual_source_bytes_attested=False))
                owners=[]
                if op['kind']=='exec_command' and op.get('command'):
                    matched={sid for sid,_,_ in old.launch_matches(op['command'])}
                    owners=[d for d in dispatches if d['call_id']==cid and d['source_id'] in matched]
                elif op['kind']=='write_stdin':
                    owner=active.get((event['log'],op.get('session_id')))
                    if owner:owners=[owner]
                need(len(owners)<=1,'ambiguous dispatch/session owner')
                for owner in owners:
                    if rec['session_id'] is not None:
                        active[(event['log'],rec['session_id'])]=owner
                    if op['kind']=='write_stdin' and rec['exit_code'] is not None:
                        active.pop((event['log'],op.get('session_id')),None)
                    results.append(dict(source_id=owner['source_id'],dispatch_call_id=owner['call_id'],
                        dispatch_line=owner['line'],operation=op['kind'],polled_session=op.get('session_id'),
                        **ref,**rec))
    return dict(dispatch_receipts=results,patch_receipts=patch_receipts,
                unassigned=unassigned,open_sessions=[dict(log=k[0],session_id=k[1],
                source_id=v['source_id']) for k,v in sorted(active.items())])


def log_records(prior,ranges=None):
    records=[]
    # Pinned original logs only; report/session results are projected below, not
    # treated as new candidate evidence. No command from the logs is executed.
    if ranges is None:ranges=prior['log_bindings']
    need([b['path'] for b in ranges]==[b['path'] for b in prior['log_bindings']],'fixed original log paths')
    for binding,original in zip(ranges,prior['log_bindings']):
        p=Path(binding['path']);need(original['bytes']<=binding['bytes']<=650*1024**2 and
            p.stat().st_size>=binding['bytes'],'bounded log range')
        h=hashlib.sha256();prefix=hashlib.sha256();offset=0
        with p.open('rb') as f:
            i=0
            while offset<binding['bytes']:
                line=f.readline(min(LINE_CAP+1,binding['bytes']-offset));i+=1
                need(len(line)<=LINE_CAP and line.endswith(b'\n'),'partial log')
                h.update(line)
                boundary=max(0,min(len(line),original['bytes']-offset))
                prefix.update(line[:boundary]);offset+=len(line)
                if not re.search(rb'"type"\s*:\s*"(?:function_call|custom_tool_call|function_call_output|custom_tool_call_output)"',line):
                    continue
                # Narrow to relevant original lines/call IDs and session polls.
                relevant=(any(d['call_id'].encode() in line for d in prior['recovery']['dispatches']) or
                    any(d['call_id'].encode() in line for d in prior['recovery']['edits']) or
                    b'write_stdin' in line)
                if not relevant and b'call_id' not in line:continue
                if not relevant:
                    # Transport outputs corresponding to a poll call are found
                    # using already selected call IDs, no unrelated body decode.
                    ids={e['payload'].get('call_id') for e in records if e['payload'].get('type') in ('function_call','custom_tool_call')}
                    if not any(k and k.encode() in line for k in ids):continue
                obj=parse(line);payload=obj.get('payload',{})
                if payload.get('type') not in ('function_call','custom_tool_call','function_call_output','custom_tool_call_output'):continue
                records.append(dict(log=str(p),line=i,timestamp=obj['timestamp'],payload=payload,
                                    line_sha256=digest(line)))
        need(h.hexdigest()==binding['sha256'],'captured log range SHA drift')
        need(prefix.hexdigest()==original['sha256'],'pinned original log prefix SHA drift')
    records.sort(key=lambda x:(x['timestamp'],x['log'],x['line']))
    return records


def metadata_patterns():
    contexts=[(),('protocol',),('plan',),('source',),('metadata',),('rows','*'),
              ('roots','*'),('slots','*'),('plan','slots','*'),('source','slots','*')]
    fields=('schema','environment','env_id','root_seed','reset_seed','calibration_root_seed',
            'root_count','root_oversample_factor','root_sha256','root_goal_sha256',
            'root_state_sha256','blind_sha256','blind_payload_sha256','history_pixels_sha256')
    pats=[c+(f,) for c in contexts for f in fields]
    for container in ('bindings','input_bindings','source_bindings','plans_bindings','prepared_bindings'):
        pats.extend([(container,'*'),(container,'*','path'),(container,'*','sha256')])
    for container in ('sourcepack','source','prepared','prepare','task','report','compiler_report','parent_report','execution_binding'):
        for field in ('path','sha256'):
            pats.append((container,field))
    for prefix in ('sourcepack','source_report','prepare','prepared','compiler_report','parent_report','upstream_report','task'):
        pats.extend([(prefix+'_path',),(prefix+'_sha256',)])
    for container in (('slots','*'),('plan','slots','*'),('source','slots','*')):
        pats.append(container+('reset_seed_span','*'))
    return pats


def delta_row(binding,leaves):
    ids=[];seeds=[];links=[];spans=[]
    hash_fields={'root_sha256','root_goal_sha256','root_state_sha256','blind_sha256',
                 'blind_payload_sha256','history_pixels_sha256'}
    for key,value in sorted(leaves.items(),key=lambda kv:str(kv[0])):
        pointer='/'+ '/'.join(map(str,key))
        if key[-1] in hash_fields and isinstance(value,str) and re.fullmatch('[0-9a-f]{64}',value):
            ids.append(dict(pointer=pointer,value=value))
        if key[-1] in ('root_seed','reset_seed','calibration_root_seed') and type(value) is int:
            seeds.append(dict(pointer=pointer,value=value))
        if key[-2:]==('reset_seed_span',0):
            end=leaves.get(key[:-1]+(1,))
            if type(value) is int and type(end) is int and 0<=value<end:
                spans.append(dict(pointer='/'+ '/'.join(map(str,key[:-1])),
                                  half_open=[value,end],usage='declared_not_observed'))
        if key[-1]=='path' and isinstance(value,str):
            h=leaves.get(key[:-1]+('sha256',))
            if isinstance(h,str) and re.fullmatch('[0-9a-f]{64}',h):
                links.append(dict(path=value,sha256=h))
        if len(key)==2 and key[0].endswith('bindings') and str(key[1]).startswith('/'):
            if isinstance(value,str) and re.fullmatch('[0-9a-f]{64}',value):
                links.append(dict(path=key[1],sha256=value))
        if len(key)==1 and key[0].endswith('_path') and isinstance(value,str):
            h=leaves.get((key[0][:-5]+'_sha256',))
            if isinstance(h,str) and re.fullmatch('[0-9a-f]{64}',h):links.append(dict(path=value,sha256=h))
    need(len(ids)+len(seeds)+len(links)<=65536,'projected row cap')
    return dict(**binding,schema=leaves.get(('schema',)),declared_identities=ids,
        declared_seeds=seeds,declared_reset_spans=spans,exact_links=links,
        physical_reset_coverage_certified=False,projection_status='verified_captured_bytes',
        metadata_role=('declares_identity_or_seed' if ids or seeds or spans else
                       'bound_reference_only' if links else 'no_origin_fields_in_whitelist'))


def project_delta(raw,old):
    pats=metadata_patterns()
    leaves=old.project(raw,pats+[('gzip_base64',)])
    compressed=leaves.pop(('gzip_base64',),None)
    if compressed is None:return leaves,0
    need(isinstance(compressed,str),'gzip envelope scalar')
    binary=base64.b64decode(compressed,validate=True)
    z=zlib.decompressobj(31);decoded=z.decompress(binary,INFLATE_CAP+1)
    need(z.eof and not z.unconsumed_tail and not z.unused_data and len(decoded)<=INFLATE_CAP,
         'bounded single-member prepare gzip, no trailing bytes')
    # The original captured envelope SHA binds these bytes. This is metadata
    # projection, not producer-specific prepare/seal verification or authority.
    inner=old.project(decoded,pats)
    # Keep outer schema and separate exact pointer provenance for inner fields.
    inner={('decoded_envelope',)+k:v for k,v in inner.items()}
    return dict(list(leaves.items())+list(inner.items())),len(decoded)


def analyze_delta(scope,old):
    need(scope['schema']==NAME+'.scope.v1' and scope['previous_sha256']==PREVIOUS_SHA and
         scope['inventory_sha256']==INVENTORY_SHA,'scope authority mismatch')
    need(scope['complete_global_history'] is False,'cannot self-approve global scope')
    files=scope['files'];need(len(files)<=ENTRY_CAP,'scope entry cap')
    paths=[x['path'] for x in files];need(paths==sorted(set(paths)),'fixed sorted unique scope')
    rows=[];used=0;inflated=0
    for i,binding in enumerate(files):
        path=literal(binding['path']);need(eligible(path),'scope filename rule')
        used+=binding['bytes'];need(used<=TOTAL_CAP,'scope total cap')
        try:
            raw=read(path,binding['sha256']);need(len(raw)==binding['bytes'],'captured size mismatch')
            leaves,size=project_delta(raw,old);inflated+=size
            need(inflated<=INFLATE_TOTAL_CAP,'aggregate inflate cap')
            rows.append(delta_row(binding,leaves))
        except (OSError,ValueError,UnicodeError,zlib.error) as exc:
            rows.append(dict(**binding,projection_status='not_verified',error=str(exc),
                declared_identities=[],declared_seeds=[],declared_reset_spans=[],exact_links=[],
                physical_reset_coverage_certified=False,metadata_role='unresolved_container_not_assumed_origin'))
        if (i+1)%100==0:print('metadata projected '+str(i+1)+'/'+str(len(files)),file=sys.stderr)
    # Join only literal path+SHA, never basename, row position, or guessed roots.
    indexed={(r['path'],r['sha256']):r for r in rows}
    graph_edges=0
    for row in rows:
        row['links_resolved_inside_delta']=sum((x['path'],x['sha256']) in indexed for x in row['exact_links'])
        graph_edges+=row['links_resolved_inside_delta']
    return dict(files=rows,captured_input_bytes=used,inflated_metadata_bytes=inflated,
        metadata_capture_complete=all(r['projection_status']=='verified_captured_bytes' for r in rows),
        exact_delta_edges=graph_edges,
        files_with_declared_identity=sum(bool(r['declared_identities']) for r in rows),
        files_with_declared_seed=sum(bool(r['declared_seeds']) for r in rows),
        missing_fields_are_not_origin_failures=True,global_scope_certified=False)


def summarize(prior,joined):
    rows=[];cleared=[]
    for source in prior['sources']:
        ds=[d for d in prior['recovery']['dispatches'] if d['source_id']==source['id']]
        status=[]
        for d in ds:
            rs=[r for r in joined['dispatch_receipts'] if r['dispatch_call_id']==d['call_id']]
            terminal=[r for r in rs if r['exit_code'] is not None]
            failures=[r for r in rs if r['exceptions']]
            state=('terminal_exit_recorded' if terminal else 'traceback_failure_recorded' if failures
                   else 'stdout_return_without_exit_code' if rs else 'no_matched_receipt')
            status.append(dict(dispatch_call_id=d['call_id'],dispatch_line=d['line'],state=state,
                               exit_code=terminal[-1]['exit_code'] if terminal else None))
            if terminal or failures:
                cleared.append(dict(source_id=source['id'],dispatch_call_id=d['call_id'],
                    request='whether_this_specific_dispatch_failed_or_terminated',resolved_as=state))
        rows.append(dict(source_id=source['id'],report_sha256=source['sha256'],dispatch_status=status,
            original_reset_bound_certified=False,original_inherited_refs_cleared=0,
            remaining_minimum=['run-bound producer/P0/four-factory/installed-env bytes OR exhaustive reset ledger',
                              'failed/unselected/rejected attempts plus restarts outside pinned logs',
                              'independent complete historical scope and subsequent deltas']))
    return dict(sources=rows,atomic_requests_cleared=cleared,
        inherited_reset_blockers_before=16,inherited_reset_blockers_after=16,
        certified_fresh_candidates=0,coverage_complete=False,source_run_authorized=False,P1_admitted=False)


def build(scope,scope_sha):
    old,prior=previous()
    # Re-verify original fixed candidate order; no outcomes drive delta selection.
    reg=parse(read(old.OUTER/old.PREREG,old.PREREG_SHA));old.fixed_candidates(reg)
    joined=join_records(log_records(prior,scope['log_ranges']),prior,old)
    delta=analyze_delta(scope,old)
    bindings=[]
    for name in ('experiments/spectral_reachability/'+NAME+'.py',
                 'tests/test_'+NAME+'.py','innerdocs/'+NAME+'.md'):
        path=ROOT/name;raw=path.read_bytes();bindings.append(dict(path=str(path),sha256=digest(raw)))
    return dict(schema=NAME+'.delta.v1',previous_sha256=PREVIOUS_SHA,scope_sha256=scope_sha,
        implementation_bindings=bindings,dispatch_evidence=joined,metadata_delta=delta,
        disposition=summarize(prior,joined),neural_calls=0,physics_calls=0,
        future_labels_used_for_selection=False,model_or_environment_imports=False)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=['capture-scope','resolve','verify'])
    p.add_argument('--scope',type=Path);p.add_argument('--scope-sha256')
    p.add_argument('--saved',type=Path);p.add_argument('--sha256')
    a=p.parse_args()
    if a.mode=='capture-scope':result=capture_scope()
    else:
        need(a.scope is not None and a.scope_sha256 is not None,'explicit scope path+SHA')
        scope=parse(read(a.scope,a.scope_sha256,OUTPUT_CAP))
        result=build(scope,a.scope_sha256)
        if a.mode=='verify':
            need(a.saved is not None and a.sha256 is not None,'explicit saved path+SHA')
            saved=unpack_result(read(a.saved,a.sha256,OUTPUT_CAP))
            need(encode(saved)==encode(result),'saved delta drift')
            result=dict(status='saved_delta_recomputed',sha256=a.sha256,P1_admitted=False)
    raw=pack_result(result);need(len(raw)<=OUTPUT_CAP,'output cap');print(raw.decode())


if __name__=='__main__':main()
