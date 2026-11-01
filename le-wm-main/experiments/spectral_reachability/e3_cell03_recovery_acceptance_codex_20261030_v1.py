"""Read-only recovery proposal and predecessor audit; never launches an episode."""
import hashlib
import json
from pathlib import Path

from experiments.spectral_reachability import agent_e3_transport_publication_20261029_v1 as transport
from experiments.spectral_reachability import e3_cell03_exclusive_host_codex_20261030_v1 as parent

ROOT = Path('/home/gbk/code/LeWorldModel/le-wm-main')
PREVIOUS = ROOT / 'results/agent_e3_chunked_identity_host_20261029_v1_CELL03_MAIN01'
PREVIOUS_CLAIM = ROOT / 'results/e3_cell03_exclusive_host_codex_20261030_v1_MAIN01.claim'
EXPECTED = {
    'claim': '595a811ec66cfb7233c79c7e2f0f149f6e41cfda2194acc4f28e465e77901f66',
    'report': '3f346697508d2254f7b57bf655ae53fdef1e8fa5f621f8a6c3bf4aaed8e428cb',
    'completion': '42e119fbbd50075e725ee2261e96d194b914e0507e44b7082f26a46d5b3abdf2',
    'journal': 'b41052ec46284a00f9e63b08ef538d835a0ef35970495b6c7dae2f900a0b1fea',
}
ATTEMPT = 'CELL03_RECOVERY01_PROPOSED'
FAILURE_PATH = str(ROOT / 'experiments/spectral_reachability/e3_cell03_exclusive_host_codex_20261030_v1.py')


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha_bytes(data):
    return hashlib.sha256(data).hexdigest()


def file_sha(path):
    p = Path(path)
    require(p.is_absolute() and p == p.resolve() and p.is_file() and not p.is_symlink(),
            'canonical regular file required')
    with p.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def audit_predecessor():
    require(file_sha(PREVIOUS_CLAIM) == EXPECTED['claim'], 'predecessor claim drift')
    require(file_sha(PREVIOUS / 'events.jsonl') == EXPECTED['journal'], 'predecessor journal drift')
    receipt = transport.consume(str(PREVIOUS), EXPECTED['report'], EXPECTED['completion'])
    require(receipt['complete'] is False, 'predecessor not partial')
    report = json.loads((PREVIOUS / 'report.json').read_bytes())
    completion = json.loads((PREVIOUS / 'completion.json').read_bytes())
    events = [json.loads(line) for line in (PREVIOUS / 'events.jsonl').read_bytes().splitlines()]
    expected_resources = (['launcher_started', 'launcher_authority_verified',
        'chunked_identity_host_started', 'launcher_gpu_preflight'] +
        ['loader_event'] * 4 + ['loader_receipt'] +
        ['preflight_binding_chunk'] * 5 + ['preflight_binding_seal', 'launcher_failure'])
    require(report['status'] == 'partial' and completion['complete'] is False and
            report['pending'] == [] and [e['resource'] for e in events] == expected_resources and
            all(e['origin'] == 'physical' and e['kind'] == 'record' for e in events),
            'exact pre-episode failure sequence')
    detail = events[-1]['detail']
    require(detail.get('type') == 'ValueError' and
            detail.get('message') == 'loaded module not presealed: ' + FAILURE_PATH,
            'exact census failure')
    require(events[3]['resource'] == 'launcher_gpu_preflight' and
            events[8]['detail']['receipt']['device'] == 'cuda:0',
            'CUDA load receipt')
    return {'predecessor_transport_verified': True, 'failure_index': 15,
            'cuda_loader_recorded': True, 'physical_episode_recorded': False,
            'scientific_admission': False}


def proposal():
    """Generate an unapproved candidate, not a source seal or launch authority."""
    audit_predecessor()
    src = ROOT / 'experiments/spectral_reachability'
    sources = {
        'parent_wrapper': file_sha(src / 'e3_cell03_exclusive_host_codex_20261030_v1.py'),
        'parent_precheck': file_sha(src / 'e3_cell03_host_cpu_precheck_codex_20261030_v1.py'),
        'child_runner': file_sha(src / 'agent_e3_chunked_identity_host_20261029_v1.py'),
        'child_code_utf8': sha_bytes(parent.CHILD.encode('utf-8')),
    }
    require(file_sha(parent.AUTHORITY) == parent.AUTHORITY_SHA, 'authority drift')
    body = {'schema': 'e3_cell03_recovery_proposal_v1', 'attempt': ATTEMPT,
            'predecessor_sha256': dict(EXPECTED), 'authority_sha256': parent.AUTHORITY_SHA,
            'sources_sha256': sources,
            'proposed_claim': str(ROOT / f'results/e3_cell03_{ATTEMPT}.claim'),
            'proposed_output': str(ROOT / f'results/agent_e3_chunked_identity_host_20261029_v1_{ATTEMPT}'),
            'recovery_authorized': False, 'production_enabled': False,
            'scientific_admission': False}
    raw = json.dumps(body, sort_keys=True, separators=(',', ':')).encode()
    return {'proposal': body, 'proposal_sha256': sha_bytes(raw)}


if __name__ == '__main__':
    print(json.dumps(proposal(), sort_keys=True))
