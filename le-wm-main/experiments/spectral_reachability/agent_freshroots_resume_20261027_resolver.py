"""Bounded, outcome-blind historical dispatch/source recovery; never a certificate.

Only JSONL tool-call arguments are decoded, never tool results or messages.
Recovered patch text is parsed/replayed as data, never imported or executed.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
from io import BytesIO
import json
from pathlib import Path
import re
import shlex
import subprocess

STEM = 'agent_freshroots_resume_20261027_'
ROOT = Path(__file__).resolve().parents[2]
OUTER = ROOT.parent
REASON = 'historical_reset_attempt_bound_not_attested'
AUTHORITY = 'results/root_exclusion_authority_resolution_20261027_v1_MAIN01.json'
AUTHORITY_SHA = '032ea722f53da7c93db90377d775b75f7ff7a40771a43fdfa38dfd7761e27d00'
PREREG = 'docs/chat_freshroots_20261027_preregistration.json'
PREREG_SHA = '79e1f45e79d692cae72376c91a4830c94b6ac1f53bb346eeaa085715e4197d7e'
PROJECTION_SHA = '41474b63621545c04dc9ece6e81fc8bf9dd1c712e5ced5fee992c85d25450c04'
LEXER_SHA = 'e6521ae535bdf26b6544868bf3c9018761332ee3b339936475ea54707c0db79c'
SCANNER_SHA = '13e5b65c149f56d7a0285d262f264dfe933b55a8366de0223e6724aedb3dacde'
LOG_NAMES = (
 '2026/10/23/rollout-2026-10-23T21-46-56-01a167e2-eca3-7f63-bb9b-93030b5d366c_01a16903-c194-71a3-972b-7a268203b373.jsonl',
 '2026/10/26/rollout-2026-10-26T12-10-35-01a167e2-eca3-7f63-bb9b-93030b5d366c_01a17667-2cec-7780-b530-fb3aa5acdd59.jsonl',
)
LOG_ROOT = Path('/home/gbk/.codex/sessions')
LOG_FILE_CAP = 650 * 1024**2
LOG_TOTAL_CAP = 1100 * 1024**2
LINE_CAP = 8 * 1024**2
META_CAP = 128 * 1024**2
OUTPUT_CAP = 512 * 1024
COMMIT = 'd26e4bff84f31138adf5d1bda809bccafd9be10a'
P0 = 'experiments/spectral_reachability/preflight_pusht_d2p_p0.py'
P1 = 'experiments/spectral_reachability/preflight_pusht_d2p_p1_reconfiguration.py'
SHELL = 'experiments/spectral_reachability/run_pusht_d2p_p0_cuda.sh'
CHAIN = (
 ('experiments/audit_pusht_reachability_horizon.py', '73871055f10b09aea15f03acf87b04e79e30d546668d44bb8160c7be3c439a67'),
 ('experiments/run_pusht_mode_aware_shared_candidate.py', '1b027be83024ea1d4b868c86f2c1340032596520256e5eb5d7a9be869ad203e6'),
 ('experiments/run_pusht_signed_torque_audit.py', 'f35fe78b70237fcb7f10cabc950b30c163318c5eb5121cbaff8a5081e5ad3493'),
 ('experiments/run_causal_visual_pusht_world_model.py', '261f609e91da9110f8ba5254033529527314d6bcf38e058c3a237c920d3c4190'),
)
TARGETS = (P0, P1, SHELL) + tuple(p for p, _ in CHAIN)
# Fixed before log recovery; neither order nor seeds depend on outcome fields.
SOURCES = (
 ('A','pusht_d2p_p0_packet_contract_v2_20261026_cuda','2f29a195f291d040a666cc3db6673f9be12c315c96162b126ddaef9bcdb5e135',202753000,4,'pusht_d2p_p0_preflight.v2'),
 ('B','pusht_d2p_p0_preflight_20261024_cuda','b1b0fbc29ceb050939ae1403c7644d0ca704b133e47a2411c079dbd626ee0eaf',202610241,4,'pusht_d2p_p0_preflight.v1'),
 ('C','pusht_d2p_p0_preflight_packet_layout_repair_20261024_cuda','c499ececab67135a2f4d4d698359e6f62b557be66f9132480ef95b14a0dce080',202710241,4,'pusht_d2p_p0_preflight.v1'),
 ('D','pusht_d2p_p1_outcome_admission_20261024_cuda','a9266fb3cd3e68ee197ec187e8b7916cf3ca8c11966778df3d4c92083b7d6b54',202710281,24,'pusht_d2p_p1_reconfiguration_preflight.v3'),
 ('E','pusht_d2p_p1_packet_contract_v4_calibration_20261026_cuda','a0c9c53bd0161946ec4466878de49795abdf6db102dbb97b174035ed4028a006',202753010,4,'pusht_d2p_p1_reconfiguration_preflight.v4'),
 ('F','pusht_d2p_p1_reconfiguration_replay_bound_20261024_cuda','4911ae7e945abfb042b332f71d7cda0163e14c9727ee8131ff7881bc67257a14',202710251,4,'pusht_d2p_p1_reconfiguration_preflight.v2'),
 ('G','pusht_d2p_p1_rollout_alignment_repair_20261024_cuda','1c5fdd414a5104b6487c2c6e5cb38d05943f902a9d7a695b891ae73202ef7abb',202710251,4,'pusht_d2p_p1_reconfiguration_preflight.v3'),
 ('H','pusht_d2p_p1_root_anchored_v5_calibration_20261026_cuda','99cbb6b6dc4653eb629427353f6f6d05022c219333f3f7362cf4ca43a3076bc2',202753010,4,'pusht_d2p_p1_reconfiguration_preflight.v5'),
)


def require(ok, msg):
    if not ok:
        raise ValueError(msg)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def canonical(obj):
    return json.dumps(obj, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def strict_json(raw):
    def pairs(items):
        d = {}
        for k, v in items:
            require(k not in d, 'duplicate JSON key')
            d[k] = v
        return d
    return json.loads(raw, object_pairs_hook=pairs,
                      parse_constant=lambda _: require(False, 'nonfinite JSON'))


def pinned(path, expected, cap=META_CAP):
    require(path.stat().st_size <= cap, 'input cap: '+str(path))
    raw = path.read_bytes()
    require(sha(raw) == expected, 'SHA mismatch: '+str(path))
    return raw


def project(raw, patterns):
    # Sealed lexer skips outcome subtrees, not a full json.loads(report).
    folder = ROOT / 'experiments/spectral_reachability'
    pinned(folder/'root_exclusion_authority_resolution_20261027_v1.py', PROJECTION_SHA)
    pinned(folder/'moving_root_exclusion_manifest_20261027_v1.py', LEXER_SHA)
    pinned(folder/'p1_conditional_source_metadata_20261027_v1.py', SCANNER_SHA)
    from experiments.spectral_reachability import root_exclusion_authority_resolution_20261027_v1 as p
    return p.project(BytesIO(raw), patterns).leaves


def fixed_candidates(reg):
    expected = [dict(slot=i, root_seed=203027101+i, reset_seed=203027201+i,
                     witness_sign=1 if i % 2 == 0 else -1) for i in range(4)]
    require(reg['candidate_order'] == expected, 'candidate order/seed mutation')
    require(reg['sampling_denominator'] == 4 and reg['attempt_per_slot'] == 1
            and reg['proposal_per_slot'] == 1 and reg['replacement_or_retry'] is False,
            'fixed denominator/no replacement')
    return expected


def strings(text):
    """Only decode complete double-quoted JSON literals; never eval JS/Python."""
    for m in re.finditer(r'"(?:[^"\\]|\\.)*"', text):
        try:
            yield json.loads(m.group())
        except (ValueError, UnicodeError):
            continue


def patches(arguments):
    if arguments.startswith('*** Begin Patch\n'):
        return [arguments]
    return [s for s in strings(arguments) if s.startswith('*** Begin Patch\n')]


def apply_section(old, lines, adding=False, choices=None):
    if adding:
        require(all(s.startswith('+') for s in lines), 'invalid add patch')
        return '\n'.join(s[1:] for s in lines)+'\n'
    require(old is not None, 'update without historical base')
    current = old.splitlines()
    cursor = 0
    chunks = []
    for s in lines:
        if s.startswith('@@'):
            chunks.append([s[2:].strip(), []])
        else:
            require(chunks and (s[:1] in (' ', '+', '-') or s == ''), 'unsupported patch line')
            chunks[-1][1].append(s)
    for anchor, body in chunks:
        if anchor:
            where = [i for i in range(cursor, len(current)) if current[i] == anchor]
            require(len(where) == 1, 'ambiguous/missing patch anchor')
            cursor = where[0]+1
        before = [s[1:] if s else '' for s in body if not s.startswith('+')]
        after = [s[1:] if s else '' for s in body if not s.startswith('-')]
        require(before, 'context-free insertion not accepted')
        matches = [i for i in range(cursor, len(current)-len(before)+1)
                   if current[i:i+len(before)] == before]
        require(matches, 'missing patch context')
        if len(matches) > 1 and choices is not None:
            choices.append(dict(matches=matches, selected=matches[0],
                                rule='first_exact_match_after_cursor_not_execution_attestation'))
        at = matches[0]
        current[at:at+len(before)] = after
        cursor = at+len(after)
    return '\n'.join(current)+'\n'


def sections(patch):
    require(patch.endswith('*** End Patch') or patch.endswith('*** End Patch\n'), 'partial patch')
    groups = []
    for line in patch.splitlines()[1:]:
        if line == '*** End Patch':
            break
        if line.startswith(('*** Add File: ', '*** Update File: ', '*** Delete File: ')):
            op, path = line[4:].split(' File: ', 1)
            path = path.removeprefix(str(ROOT)+'/').removeprefix('le-wm-main/')
            groups.append([op, path, []])
        elif groups:
            groups[-1][2].append(line)
    return groups


def ast_metadata(source):
    tree = ast.parse(source)
    schema = None
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'SCHEMA' for t in node.targets):
            schema = ast.literal_eval(node.value)
    sites = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and (
                isinstance(node.func, ast.Attribute) and node.func.attr == 'reset' or
                isinstance(node.func, ast.Name) and node.func.id == 'collect_nonterminal_boundary_contact_roots'):
            sites.append(dict(line=node.lineno, expression=ast.unparse(node)))
    return dict(schema=schema, reset_and_collector_sites=sites)


def dispatch_commands(arguments):
    """Extract literal cmd fields only. Reads/results/mentions aren't dispatches."""
    found = []
    if arguments.startswith('{'):
        try:
            obj = strict_json(arguments)
            if isinstance(obj, dict) and isinstance(obj.get('cmd'), str):
                found.append(obj['cmd'])
        except ValueError:
            return []
    for m in re.finditer(r'(?:"cmd"|\bcmd)\s*:\s*("(?:[^"\\]|\\.)*")', arguments):
        found.append(json.loads(m.group(1)))
    return list(dict.fromkeys(found))


def launch_matches(command, shell_source=None):
    # Shell program strings are NEVER executed. Restrict to an invocation token
    # and literal output operand in the same simple command segment.
    found = []
    for segment in re.split(r'&&|\|\||[;\n]', command):
        try:
            tokens = shlex.split(segment)
        except ValueError:
            continue
        if not tokens or not (Path(tokens[0]).name in ('env','systemd-run','bash') or
                              re.fullmatch(r'python(?:3(?:\.\d+)?)?',Path(tokens[0]).name)):
            continue
        producer = None
        module = 'experiments.spectral_reachability.'
        for path in (P0, P1):
            token = module+Path(path).stem
            if '-m' in tokens and tokens[tokens.index('-m')+1:tokens.index('-m')+2] == [token]:
                producer = path
        shell = next((t for t in tokens if t.endswith('/'+Path(SHELL).name)), None)
        if shell is not None and 'bash' in tokens[:tokens.index(shell)+1]:
            producer = P0
        if producer is None:
            continue
        output = None
        if '--output' in tokens:
            j = tokens.index('--output')
            if j+1 < len(tokens):
                output = tokens[j+1]
        elif shell is not None and tokens.index(shell)+1 < len(tokens):
            output = tokens[tokens.index(shell)+1]
        elif shell is not None and shell_source:
            # Only a historical replayed shell default, never today's file.
            default = re.search(r'OUTPUT="\$\{1:-(results/[^}]+)\}"', shell_source)
            if default:
                output = default.group(1)
        for sid, directory, *_ in SOURCES:
            if output in ('results/'+directory, str(ROOT/'results'/directory)):
                found.append((sid, producer, segment.strip()))
    return found


def replay(records):
    state, errors, edits, launches = {}, [], [], []
    for rec in records:
        for patch in patches(rec['arguments']):
            for op, path, body in sections(patch):
                if path not in TARGETS:
                    continue
                item = {k: rec[k] for k in ('log', 'line', 'timestamp', 'call_id')}
                item.update(path=path, patch_sha256=sha(patch.encode()), operation=op)
                try:
                    require(op in ('Add', 'Update'), 'delete invalidates lineage')
                    choices = []
                    value = apply_section(state.get(path), body, op == 'Add', choices)
                    meta = ast_metadata(value) if path.endswith('.py') else {}
                    state[path] = value
                    item.update(recovered_sha256=sha(value.encode()), patch_context_choices=choices, **meta)
                except (ValueError, SyntaxError) as exc:
                    state[path] = None
                    item['error'] = str(exc)
                    errors.append(item)
                edits.append(item)
        for command in dispatch_commands(rec['arguments']):
            for sid, producer, segment in launch_matches(command, state.get(SHELL)):
                code = state.get(producer)
                launches.append(dict(source_id=sid, log=rec['log'], line=rec['line'],
                    timestamp=rec['timestamp'], call_id=rec['call_id'], command=segment,
                    producer=producer, recovered_producer_sha256=sha(code.encode()) if code else None,
                    recovered_producer_metadata=ast_metadata(code) if code else None,
                    root_factory_replayed_paths=[p for p, _ in CHAIN if state.get(p)],
                    replayed_P0_dependency_sha256=(sha(state[P0].encode())
                        if producer == P1 and state.get(P0) else None),
                    shell_recovered_sha256=sha(state[SHELL].encode()) if state.get(SHELL) else None,
                    dispatch_intention_only=True, process_return_attested=False,
                    execution_source_attested=False))
    return dict(edits=edits, recovery_errors=errors, dispatches=launches)


def scan_logs(paths):
    records, bindings, total = [], [], 0
    needles = tuple(p.encode() for p in TARGETS)+tuple(d.encode() for _, d, *_ in SOURCES)
    for path in paths:
        initial = path.stat().st_size
        total += initial
        require(initial <= LOG_FILE_CAP and total <= LOG_TOTAL_CAP, 'bounded log bytes')
        h = hashlib.sha256()
        selected = 0
        with path.open('rb') as f:
            for i, line in enumerate(iter(lambda: f.readline(LINE_CAP+1), b''), 1):
                require(len(line) <= LINE_CAP and line.endswith(b'\n'), 'partial/oversize JSONL line')
                h.update(line)
                if not any(n in line for n in needles):
                    continue
                # Never decode a tool result/message, even if it mentions a run.
                # Type gate is lexical, then confirmed after JSON decoding.
                if not re.search(rb'"type"\s*:\s*"(?:function_call|custom_tool_call)"', line):
                    continue
                event = strict_json(line)
                payload = event.get('payload', {})
                if payload.get('type') not in ('function_call', 'custom_tool_call'):
                    continue
                name = payload.get('name', '')
                if name not in ('exec', 'exec_command', 'apply_patch', 'functions.exec',
                                'functions.exec_command', 'functions.apply_patch'):
                    continue
                arguments = payload.get('arguments', payload.get('input', ''))
                if not isinstance(arguments, str):
                    continue
                records.append(dict(log=str(path), line=i, timestamp=event['timestamp'],
                    call_id=payload.get('call_id'), arguments=arguments))
                selected += 1
        require(path.stat().st_size == initial, 'log changed during scan')
        bindings.append(dict(path=str(path), bytes=initial, sha256=h.hexdigest(),
                             selected_tool_calls=selected, authority='first_capture_not_execution_seal'))
    records.sort(key=lambda r: (r['timestamp'], r['log'], r['line']))
    return replay(records), bindings


def candidates(reg, rows):
    output = []
    for slot in fixed_candidates(reg):
        conflicts = []
        for row in rows:
            # This family envelope is NOT a run certificate; kept as conservative
            # rejection information, never as proof that all old seeds are known.
            lo, hi = row['conditional_family_span']
            if any(lo <= slot[k] <= hi for k in ('root_seed', 'reset_seed')):
                conflicts.append(row['id'])
            reserved = row.get('reserved_family_span')
            if reserved and any(reserved[0] <= slot[k] <= reserved[1]
                                for k in ('root_seed', 'reset_seed')):
                conflicts.append(row['id']+':reservation')
        output.append(dict(**slot, conditional_seed_conflicts=conflicts,
            seed_conflict_check='conditional_only', actual_root_identity=None,
            certifiable=False, dispatch_authorized=False,
            missing=['complete_historical_scope', 'original_run_reset_coverage',
                     'future_candidate_actual_state_pose_snapshot_RGB3_exclusion']))
    return output


def validate_protocol(p, sid, seed, count, schema):
    actual_seed = p.get(('protocol','root_seed'), p.get(('protocol','calibration_root_seed')))
    actual_count = p.get(('protocol','root_count'), p.get(('protocol','calibration_roots')))
    require(p[('schema',)] == schema and type(actual_seed) is int and actual_seed == seed
            and type(actual_count) is int and actual_count == count, 'historical protocol mismatch')
    if sid not in ('A', 'B', 'C'):
        require(p[('protocol','root_oversample_factor')] == 4, 'historical oversample mismatch')


def summarize_dispatches(rows, dispatches):
    for row in rows:
        ds = [d for d in dispatches if d['source_id'] == row['id']]
        row['recovered_dispatch_count'] = len(ds)
        row['schema_matched_recovered_versions'] = sorted({d['recovered_producer_sha256'] for d in ds
            if d['recovered_producer_metadata'] and d['recovered_producer_metadata']['schema'] ==
            next(s[5] for s in SOURCES if s[0] == row['id'])})
        row['resolved_facts'] = ['original_report_protocol_and_two_SHA_refs_verified']
        row['dispatch_evidence'] = [dict(log=d['log'],line=d['line'],call_id=d['call_id'],
            producer_sha256=d['recovered_producer_sha256']) for d in ds]
        row['coverage_request'] = dict(report_sha256=row['sha256'],
            at_least_dispatch_intents_to_cover=len(ds),
            include_every_rejected_unselected_failed_reset=True,
            missing_factory_paths=[p for p, _ in CHAIN],
            missing_installed_env_binding=True, missing_execution_source_binding=True,
            complete_scope_authority_required=True)
        if ds:
            row['resolved_facts'].append('historical_dispatch_command_recovered')
        if row['schema_matched_recovered_versions']:
            row['resolved_facts'].append('historical_patch_replayed_schema_matched_producer_candidate')


def build():
    # Freeze and verify order/rules BEFORE opening reports/logs; no new outcomes.
    reg = strict_json(pinned(OUTER/PREREG, PREREG_SHA))
    fixed_candidates(reg)
    pinned(OUTER/'docs/D2P_MOVING_FRESH_ROOT_PROTOCOL_20261027.md',reg['protocol_sha256'])
    authority = pinned(ROOT/AUTHORITY, AUTHORITY_SHA)
    refs = project(authority, [('history_file_requests','*','path'),
        ('history_file_requests','*','inherited_refs','*','original_reason'),
        ('history_file_requests','*','inherited_refs','*','parent'),
        ('history_file_requests','*','inherited_refs','*','historical_sha256')])
    inherited = {}
    for pointer, value in refs.items():
        if pointer[-1] == 'original_reason' and value == REASON:
            path = refs[pointer[:2]+('path',)]
            inherited.setdefault(path, []).append(dict(parent=refs.get(pointer[:-1]+('parent',)),
                sha256=refs.get(pointer[:-1]+('historical_sha256',))))
    require(len(inherited) == 8 and sum(map(len, inherited.values())) == 16, '8/16 scope mutation')
    rows = []
    for sid, directory, expected, seed, count, schema in SOURCES:
        path = ROOT/'results'/directory/'report.json'
        require(str(path) in inherited, 'source absent in original 16 references')
        require(len(inherited[str(path)]) == 2 and all(r['sha256'] == expected for r in inherited[str(path)]),
                'inherited source SHA/reference mismatch')
        raw = pinned(path, expected)
        p = project(raw, [('schema',), ('protocol','root_seed'), ('protocol','root_count'),
            ('protocol','root_oversample_factor'), ('protocol','calibration_root_seed'),
            ('protocol','calibration_roots'), ('protocol','outcome','root_seed'),
            ('protocol','outcome','root_count'), ('protocol','outcome','oversample_factor')])
        validate_protocol(p, sid, seed, count, schema)
        reserve_seed = p.get(('protocol','outcome','root_seed'))
        reserve_count = p.get(('protocol','outcome','root_count'))
        reserved = [reserve_seed,reserve_seed+8*reserve_count-1] if reserve_seed is not None else None
        rows.append(dict(id=sid, path=str(path), sha256=expected, references=inherited[str(path)],
            protocol_projection=[dict(pointer='/'+ '/'.join(k), value=v) for k, v in sorted(p.items())],
            conditional_family_span=[seed,seed+8*count-1], conditional_span_certified=False,
            reserved_family_span=reserved, reserved_span_not_evidence_of_dispatch=True,
            explicit_state_reset_seed=0, original_execution_reset_bound_certified=False,
            unresolved=['original_run_to_source_and_env_closure',
                        'reset_attempts_including_failed_prior_dispatches', 'global_historical_scope']))
    recovered, logs = scan_logs([LOG_ROOT/name for name in LOG_NAMES])
    chain = []
    for path, expected in CHAIN:
        raw = subprocess.run(['git','show',COMMIT+':le-wm-main/'+path],cwd=OUTER,
                             check=True,capture_output=True,timeout=15).stdout
        require(sha(raw) == expected, 'historical chain blob changed')
        chain.append(dict(path='git:'+COMMIT+':le-wm-main/'+path, sha256=expected,
            scope='historical_code_family_not_original_execution_binding'))
    summarize_dispatches(rows, recovered['dispatches'])
    implementations = []
    for name in ('experiments/spectral_reachability/'+STEM+'resolver.py',
                 'tests/test_'+STEM+'resolver.py', 'innerdocs/'+STEM+'resolver.md'):
        path = ROOT/name
        require(path.stat().st_size <= 512*1024, 'implementation cap')
        raw = path.read_bytes()
        implementations.append(dict(path=str(path),sha256=sha(raw),bytes=len(raw)))
    return dict(schema=STEM+'coverage.v1', status='metadata_resolved_with_explicit_gaps',
        scope=dict(historical_sources=8,inherited_blockers=16,inventory_not_coverage=True,
                   bounded_log_files=len(LOG_NAMES),new_outcomes_decoded=False),
        preregistration=dict(path=str(OUTER/PREREG),sha256=PREREG_SHA),
        authority=dict(path=str(ROOT/AUTHORITY),sha256=AUTHORITY_SHA),
        implementation_bindings=implementations,
        projection_bindings=[dict(path=str(ROOT/'experiments/spectral_reachability'/p),sha256=h)
            for p,h in [('root_exclusion_authority_resolution_20261027_v1.py',PROJECTION_SHA),
                        ('moving_root_exclusion_manifest_20261027_v1.py',LEXER_SHA),
                        ('p1_conditional_source_metadata_20261027_v1.py',SCANNER_SHA)]],
        log_bindings=logs, historical_chain=chain, recovery=recovered, sources=rows,
        candidates=candidates(reg,rows), certified_candidate_count=0,
        original_execution_bounds_certified=0, coverage_complete=False,
        source_run_authorized=False, P1_admitted=False, neural_calls=0,physics_calls=0,
        evidence_requests=[
            'A-H original execution source+factory+installed env binding, not postrun/later seals',
            'original reset attempt ledgers or independently attested exhaustive envelopes covering all launches',
            'complete historical scope including later delta inventory; 618-path cutoff is not global authority',
            'future fixed-slot actual root state7/wrapped-pose/snapshot/RGB3 identity, before witness dispatch'])


def verify_saved(saved, actual):
    require(canonical(strict_json(saved)) == canonical(actual), 'saved closure drift')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['resolve','verify'])
    parser.add_argument('--saved', type=Path)
    parser.add_argument('--sha256')
    args = parser.parse_args()
    result = canonical(build())
    require(len(result) <= OUTPUT_CAP, 'output cap')
    if args.mode == 'verify':
        require(args.saved is not None and args.sha256 is not None, 'explicit saved path+SHA')
        saved = pinned(args.saved, args.sha256, OUTPUT_CAP)
        verify_saved(saved, strict_json(result))
        print(json.dumps(dict(status='saved_metadata_recomputed',sha256=sha(saved),P1_admitted=False)))
    else:
        print(result.decode())


if __name__ == '__main__':
    main()
