"""Bounded official-source GRASP contract; no remote code import/execution."""
from __future__ import annotations

import argparse
import base64
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

NAME = 'p2_grasp_official_contract_20261027_v1'
ROOT = Path(__file__).resolve().parents[2]
COMMIT = '64f988aebf04ad9f674f9821bc6d2b9e22879105'
REPO = 'michael-psenka/grasp'
LIMIT = 256 * 1024
TOTAL_LIMIT = 2 * 1024 * 1024
API = 'https://api.github.com/repos/' + REPO + '/'
RAW = 'https://raw.githubusercontent.com/' + REPO + '/' + COMMIT + '/'
ADAPTERS = {
    'lewm': ('michael-psenka/le-wm-grasp', '94be3c42aa2f0867c6cb1e38bda498b293b6921e'),
    'jepa': ('michael-psenka/jepa-wms-grasp', '2d81ea8d84d1f7c457fb1baf00621cff142afbe3'),
}
PINS = {
    'LICENSE.source.txt': 'd54bc4cd9f49bd38a602c6aaff6ff3ba60062d026039e8844ebf4863ac9856c2',
    'README.md.source.txt': 'ff2ddde4914c375a92d0d0fc35bc7f443d0e4cb09b0eb8f93c434b2ee54da935',
    'author.source.txt': '55aac5c9d44a24b370eab8d81a1fad40b896d3043d9e88443853a504dccc6bdf',
    'commit.source.txt': 'f1699f1b478f06db33422ea6e0fd7d0f69c34153576f6e134b1976803f5a2817',
    'conf__env__pusht.yaml.source.txt': '68a8514bf4e1aa9de3e16a42887ea3a7786ad710efb51d8f8ed3f3edcf39bc5b',
    'conf__plan_grasp.yaml.source.txt': '73599439f95f4f36b7d8a9eadec3a7f78a7b31173c2ff91b7669d8c5b3be9fc5',
    'conf__planner__grasp.yaml.source.txt': 'a574a1b1a78ba0b09a3d2462dd0c6267be09784fc209662a8c9ac7692c3fe293',
    'grasp_pseudocode.py.source.txt': '6ae8616e8fb950c7d5bdef49f34e3df234b43cddfab8ea273a87493238932ad1',
    'jepa__README.md.source.txt': '2a486440eb1e408a6ad41ae2612d9ef80595b9ea41860cfda870f0a27f3c3bb7',
    'jepa__commit.source.txt': '566b10c7454679bbcc0590b1cb6ad75923fa76b3bfccda701e9af394ea51b170',
    'jepa__evals__simu_env_planning__planning__planning__planner.py.source.txt': '8bd881e1d69e014c15c0da17c24755fb547fc392e7c4216038a7f547743b5d13',
    'jepa__repo.source.txt': '8b4d034cfb32105b6bab2ad52a4a555a56a3e950e0c45a07b2da5fef66a3bcb6',
    'jepa__tree.source.txt': 'ad0033ea7cd73a796676e7eec9d813be5f5d059730ddfccf4cc3fbb7fc9105be',
    'lewm__README.md.source.txt': '82492bc972db9eb6896a54f3dc482f09e4b39f8720d5836a22bda4732ea9d635',
    'lewm__commit.source.txt': 'cd50b761547afad08fd09a229f73a08f3a44acea80158cbfcd979e0c7ceaef2d',
    'lewm__config__eval__solver__grasp.yaml.source.txt': 'e424acb6c4d5bbfdaf4ec24323ad3c5b3a6948ea13e4042ee243cf3a15d424f5',
    'lewm__config__eval__solver__grasp_cem_sync.yaml.source.txt': 'd55fff59b62ca99030dab77af9ff2c57ef4a98a7f4a29bec5a29e8ed3de702f8',
    'lewm__grasp_solver.py.source.txt': '53e7761019e3eed0d776df46aa6e0f8e84c6a084d9588802b2528b006be690df',
    'lewm__repo.source.txt': '328858cced12b94d901edb5bae7ff837cff2303d2dfe9cf28fe0d845a1242e4d',
    'lewm__tree.source.txt': 'fa98a31684ead1c3355e3d9d21c053ea46f53c0c0726939c1fb1db445ba4f100',
    'models__visual_world_model.py.source.txt': '6dbdf9ab51ac6d5c1e91f56cbde2857176a16e2f46a2043a3c33f05bacf5ba5c',
    'planning__base_planner.py.source.txt': '767fe5b966d049c637e9cdf3b690095ae1d107dfa63f2de55413446d05b635ba',
    'planning__grasp.py.source.txt': 'dc1598e77e2fc1bd0438faf1fe044943bea52a3e835175748e8e313d7ace30ac',
    'preprocessor.py.source.txt': '48609786dbb787a7e96854b6295bbf738e2f961a959a73330d6fb36e8a8de8f8',
    'repo.source.txt': 'f696f479c287d8406728c9ab7ca0171b2afb1cba468143aa9140143aded56cb2',
    'tree.source.txt': 'cb2b7173ffa5d865214c646d1f4ace9fca1fbe112e6f092a73e2f786e3940e5a',
}


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def encoded(value):
    return (json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False) + '\n').encode()


def bounded(stream, limit):
    raw = stream.read(limit + 1)
    if len(raw) > limit:
        raise ValueError('source exceeds byte limit')
    return raw


def check_url(url):
    p = urlsplit(url)
    if p.scheme != 'https' or p.username or p.password or p.port or p.fragment or '..' in p.path.split('/'):
        raise ValueError('unapproved source URL')
    if p.hostname == 'raw.githubusercontent.com':
        prefixes = ['/' + repo + '/' + commit + '/' for repo, commit in [(REPO, COMMIT), *ADAPTERS.values()]]
        if p.query or '%' in p.path or not any(p.path.startswith(prefix) for prefix in prefixes) or not p.path.endswith(('.py', '.yaml', '.md', '/LICENSE')):
            raise ValueError('only pinned source/text, no weights or mutable refs')
    else:
        allowed = ['https://api.github.com/users/michael-psenka']
        for repo, commit in [(REPO, COMMIT), *ADAPTERS.values()]:
            api = 'https://api.github.com/repos/' + repo + '/'
            allowed.extend([api.rstrip('/'), api + 'commits/' + commit, api + 'git/trees/' + commit + '?recursive=1'])
            if re.fullmatch(re.escape(api) + r'git/blobs/[0-9a-f]{40}', url):
                return True
        if url not in allowed:
            raise ValueError('unapproved API endpoint')
    return True


def fetch(destination, paths=(), api_source=False, adapter=None):
    destination.mkdir(exist_ok=True)
    sources = destination / 'sources'
    sources.mkdir(exist_ok=True)
    repo, revision = ADAPTERS[adapter] if adapter else (REPO, COMMIT)
    api = 'https://api.github.com/repos/' + repo + '/'
    raw_base = 'https://raw.githubusercontent.com/' + repo + '/' + revision + '/'
    prefix = adapter + '__' if adapter else ''
    requests = ({prefix+'repo': api.rstrip('/'), prefix+'commit': api + 'commits/' + revision,
                 prefix+'tree': api + 'git/trees/' + revision + '?recursive=1'}
                if not paths else {prefix+p.replace('/', '__'): raw_base + p for p in paths})
    if not paths and not adapter:
        requests['author'] = 'https://api.github.com/users/michael-psenka'
    blobs = {}
    if api_source:
        tree = json.loads((sources / (prefix+'tree.source.txt')).read_text())
        commit = json.loads((sources / (prefix+'commit.source.txt')).read_text())
        if tree.get('truncated') or commit['sha'] != revision or tree['sha'] not in (revision, commit['commit']['tree']['sha']):
            raise ValueError('complete commit-bound tree required')
        entries = {e['path']: e for e in tree['tree'] if e['type'] == 'blob'}
        for p in paths:
            check_url(raw_base + p)
            if entries[p]['size'] > LIMIT:
                raise ValueError('source too large before request')
            blobs[prefix+p.replace('/', '__')] = entries[p]['sha']
        requests = {key: api + 'git/blobs/' + digest for key, digest in blobs.items()}
    receipt_name = 'retrieval_' + prefix + ('api_' if api_source else '') + ('metadata' if not paths else sha(encoded(list(paths)))[:12]) + '.json'
    if (destination / receipt_name).exists():
        raise FileExistsError('exclusive acquisition receipt already exists')
    total = sum(p.stat().st_size for p in sources.iterdir())
    receipts = []
    for key, url in requests.items():
        check_url(url)
        target = sources / (key + '.source.txt')
        if target.exists():
            raise FileExistsError(str(target))
        receipt = dict(id=key, url=url, utc=datetime.now(timezone.utc).isoformat())
        try:
            limit = min(LIMIT, TOTAL_LIMIT-total)
            if limit <= 0:
                raise ValueError('total acquisition limit reached')
            with urlopen(Request(url, headers={'User-Agent': 'LeWM-GRASP-source-audit/1.0'}), timeout=25) as response:
                check_url(response.url)
                if int(response.headers.get('Content-Length', 0)) > limit:
                    raise ValueError('declared body exceeds limit; body not read')
                raw = bounded(response, limit)
                receipt.update(http_status=response.status, final_url=response.url)
            raw.decode('utf-8')
            if api_source:
                payload = json.loads(raw)
                if payload['encoding'] != 'base64' or payload['sha'] != blobs[key]:
                    raise ValueError('GitHub blob identity/encoding rejected')
                raw = base64.b64decode(payload['content'])
                git_sha = hashlib.sha1(b'blob ' + str(len(raw)).encode() + b'\0' + raw).hexdigest()
                if git_sha != blobs[key] or len(raw) != payload['size']:
                    raise ValueError('Git object hash mismatch')
                receipt['git_blob_sha1'] = git_sha
            with target.open('xb') as stream:
                stream.write(raw)
            total += len(raw)
            receipt.update(status='retrieved', file=str(target.relative_to(destination)), bytes=len(raw), sha256=sha(raw))
        except Exception as exc:
            receipt.update(status='unavailable', error=type(exc).__name__ + ': ' + str(exc))
        receipts.append(receipt)
        print(json.dumps(receipt), flush=True)
    with (destination / receipt_name).open('xb') as stream:
        stream.write(encoded(dict(requests=receipts, byte_limit=LIMIT, total_limit=TOTAL_LIMIT, total_source_bytes=total)))


def read_source(destination, path, adapter=None):
    key = (adapter+'__' if adapter else '') + path.replace('/', '__') + '.source.txt'
    raw = (destination/'sources'/key).read_bytes()
    if len(raw)>LIMIT or sha(raw)!=PINS[key]:
        raise ValueError('official source SHA/size rejected: '+key)
    return raw


def anchor(destination,path,needle,adapter=None):
    raw=read_source(destination,path,adapter)
    lines=raw.decode().splitlines()
    hits=[i for i,line in enumerate(lines) if needle in line]
    if len(hits)!=1:
        raise ValueError('unique source anchor required: '+needle)
    first=max(0,hits[0]-2);last=min(len(lines),hits[0]+3)
    excerpt='\n'.join(lines[first:last])
    repo,commit=ADAPTERS[adapter] if adapter else (REPO,COMMIT)
    return dict(repo=repo,commit=commit,path=path,source_sha256=sha(raw),
        start_line_1based=first+1,end_line_1based=last,excerpt=excerpt,excerpt_sha256=sha(excerpt.encode()),
        url=f'https://github.com/{repo}/blob/{commit}/{path}#L{first+1}-L{last}')


def build_contract(destination):
    from experiments.spectral_reachability import p2_grasp_reference_core_20261027_v1 as core
    # Verify every acquired body, and independently its immutable tree blob ID.
    all_sources={}
    for key in PINS:
        raw=(destination/'sources'/key).read_bytes()
        if sha(raw)!=PINS[key] or len(raw)>LIMIT:
            raise ValueError('official source SHA/size rejected: '+key)
        all_sources[key]=raw
    repos=[]
    for adapter in (None,'lewm','jepa'):
        prefix=adapter+'__' if adapter else ''
        repo,revision=ADAPTERS[adapter] if adapter else (REPO,COMMIT)
        metadata=json.loads(all_sources[prefix+'repo.source.txt'])
        commit=json.loads(all_sources[prefix+'commit.source.txt'])
        tree=json.loads(all_sources[prefix+'tree.source.txt'])
        if metadata['full_name']!=repo or commit['sha']!=revision or tree.get('truncated') or tree['sha'] not in (revision,commit['commit']['tree']['sha']):
            raise ValueError('immutable official repository/tree binding rejected')
        entries={e['path']:e for e in tree['tree'] if e['type']=='blob'}
        for key,raw in all_sources.items():
            if key.startswith(('lewm__','jepa__')) != bool(adapter) or (adapter and not key.startswith(prefix)):
                continue
            name=key[len(prefix):].removesuffix('.source.txt')
            if name in ('repo','author','commit','tree'):
                continue
            path=name.replace('__','/')
            digest=hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()
            if path not in entries or digest!=entries[path]['sha'] or len(raw)!=entries[path]['size']:
                raise ValueError('commit tree Git blob mismatch: '+path)
        repos.append(dict(repo=repo,commit=revision,commit_tree=commit['commit']['tree']['sha'],
            url='https://github.com/'+repo+'/tree/'+revision,metadata_sha256=sha(all_sources[prefix+'repo.source.txt'])))
    paper_path=ROOT/'results/p2_reported_baseline_table_20261027_v3/sources/grasp_html.html'
    paper_raw=paper_path.read_bytes()
    if sha(paper_raw)!='2a17a21d05d1792b0fa7b7bd9442be973af6ab37236263994a59a4eb4e6fa899' or b'https://michaelpsenka.io/grasp' not in paper_raw:
        raise ValueError('sealed primary paper project-link binding rejected')
    if json.loads(all_sources['author.source.txt'])['name']!='Michael Psenka':
        raise ValueError('author identity rejected')
    readme=read_source(destination,'README.md').decode()
    if any('https://github.com/'+repo not in readme for repo,_ in ADAPTERS.values()):
        raise ValueError('official adapter README link missing')
    mapping=[]
    def claim(name,description,specs):
        mapping.append(dict(mechanism=name,description=description,
            anchors=[anchor(destination,path,needle,adapter) for path,needle,adapter in specs]))
    claim('virtual_states', 'DINO full planner cylinder states differ from adapter linear interpolation; both optimize intermediate states, not just action restarts.', [
        ('planning/grasp.py','def _init_states_cylinder(',None),
        ('grasp_solver.py','s_virtual = s_0.unsqueeze(1) + t_vals *', 'lewm'),
        ('evals/simu_env_planning/planning/planning/planner.py','s = (s_0_flat.unsqueeze(1) + t_interp *','jepa')])
    claim('soft_dynamics_and_goal', 'Full planner channel-sum/other-dimension-mean and extra nonterminal goal shaping differ from adapter sum of per-edge MSE plus dense goal MSE.', [
        ('planning/grasp.py','dynamics_loss = gamma_dynamics*',None),
        ('grasp_solver.py','loss = loss_dyn + cur_goal_weight * loss_goal','lewm'),
        ('evals/simu_env_planning/planning/planning/planner.py','return sum(losses_dyn) + goal_weight * sum(losses_goal)','jepa')])
    claim('state_input_detach', 'Stop only state inputs during lifted phase; residual target virtual states and action-input gradients remain live. Sync is full gradient through the serial state chain.', [
        ('planning/grasp.py','z0_input = z0_virtual if self.enable_state_gradient else z0_virtual.detach()',None),
        ('grasp_solver.py','s_t = s_full[:, t].detach()','lewm'),
        ('evals/simu_env_planning/planning/planning/planner.py','s_t.detach(), a_t, B, vis_shape, z_init','jepa')])
    claim('noise', 'Actual base additive noise uses current_noise_scale, NOT the previously computed residual noise_scale. Adapters inject Gaussian into virtual states only. Official randn_like uses global RNG, not their stored generator for this path.', [
        ('planning/grasp.py','noise = torch.randn_like(z_state) * current_noise_scale',None),
        ('grasp_solver.py','s_virtual.data += cur_noise * torch.randn_like(s_virtual)','lewm'),
        ('evals/simu_env_planning/planning/planning/planner.py','s.data += cur_noise * torch.randn_like(s)','jepa')])
    claim('sync', 'Base triggers at k%interval==0 and k>0, accepts improvement and refreshes virtual states. Adapters trigger (k+1)%interval==0 and k>0, replace actions/reset joint Adam, no virtual refresh. Base final_loss and rolled states come from final pre-update forward, not verification of post-update actions.', [
        ('planning/grasp.py','z_state.data = gd_rolled_out_states[:, 1:-1].data',None),
        ('grasp_solver.py','if self.gd_interval > 0 and (k + 1) % self.gd_interval == 0 and k > 0:','lewm'),
        ('evals/simu_env_planning/planning/planning/planner.py','if self.gd_interval > 0 and (k + 1) % self.gd_interval == 0 and k > 0:','jepa')])
    claim('world_model_interfaces', 'DINO rollout_encoded packs visual/proprio/action; LeWM action_encoder plus predict on H1; native adapter unroll takes full visual patches plus last initial proprio in H1, not CLS LeWM.', [
        ('models/visual_world_model.py','def rollout_encoded(',None),
        ('grasp_solver.py','def _jepa_predict_step(', 'lewm'),
        ('evals/simu_env_planning/planning/planning/planner.py','proprio = z_init["proprio"][:, -1:].detach()','jepa')])
    native_tree=json.loads(all_sources['jepa__tree.source.txt'])
    native_entries={e['path']:e['sha'] for e in native_tree['tree'] if e['type']=='blob'}
    comparisons={}
    for relative in ('app/plan_common/models/AdaLN_vit.py','app/plan_common/models/prop_embedding.py','app/plan_common/datasets/pusht_dset.py'):
        path=ROOT/'external/jepa-wms'/relative
        raw=path.read_bytes()
        digest=hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()
        comparisons[relative]=dict(local_path=str(path),local_sha256=sha(raw),local_git_blob=digest,
            official_adapter_git_blob=native_entries[relative],exact_git_blob_match=digest==native_entries[relative])
    return dict(schema=NAME, status='official_source_contract_verified_not_runtime_reproduction', repos=repos,
        paper=dict(version='2602.00475v1',path=str(paper_path),sha256=sha(paper_raw),
                   project_url='https://michaelpsenka.io/grasp',identity_basis='paper project link + repository homepage + author name + README adapter links'),
        paper_results_code_commit='unknown',algorithm_mapping=mapping,proposed_core=core.protocol(),
        proposed_config=core.asdict(core.Config()),proposed_budget_per_root=core.budget(core.Config()),
        native_source_comparison=comparisons,same_as_existing_real_H3_contract=False,
        local_reproduction=False,neural_execution=False,P0_admission=False,P2_admission=False,
        implementation_scope='independent exact-form H1 adapter mechanism subset; no claim of full DINO planner or paper-run equivalence',
        remaining=[
            'Main must choose an explicit H1-reset baseline with other arms matched to that input context, OR design/review a separate history-preserving GRASP extension; never relabel H1 as real-H3.',
            'Bind root IDs, full encoded goals, action normalization/bounds and frozen weight identities before any CUDA use; this prepare binds zero roots and loads no weights.',
            'Core has no action box. Review unbounded normalized actions versus existing bounded raw packet domain; adding tanh/clamp is a new declared variant, not official equivalence.',
            'Native direct AdaLN ABI is implemented but not runtime-verified. Its predictor matches adapter blob; proprio/data source differences must remain in provenance.',
            'Review official solver hyperparameters and resource estimates before CUDA; no outcome tuning. Default full H3 optimization needs2703 transition samples/root, not147.',
            'Add actual predictor/action-encoder/backward/encoding/verification hook ledger and peak-memory validation to a separately reviewed CUDA runner; no full-pipeline equivalence.',
            'Measure independent fixed root/task/feedback paired trials only after source freeze; current prior roots remain calibration, and no physical success is inherited.',
            'No exact experiment-config/checkpoint/commit binding for published Table1/3/4 has been verified.'])


def prepare(destination,output):
    contract=build_contract(destination)
    if output.exists():
        raise FileExistsError('exclusive NEW prepare directory required')
    sources=list((destination/'sources').iterdir())+list(destination.glob('retrieval*.json'))
    sources += [Path(__file__),ROOT/'experiments/spectral_reachability/p2_grasp_reference_core_20261027_v1.py',
        ROOT/'tests/test_p2_grasp_official_contract_20261027_v1.py',ROOT/'tests/test_p2_grasp_reference_core_20261027_v1.py',
        ROOT/'docs'/f'{NAME}.md',Path(contract['paper']['path']),ROOT/'jepa.py',
        ROOT/'experiments/spectral_reachability/p5_jepa_native_action_response_20261027_v8.py',
        ROOT/'experiments/spectral_reachability/pusht_composition_cuda_replay_v2.py']
    sources += [Path(row['local_path']) for row in contract['native_source_comparison'].values()]
    sources += [ROOT/'external/jepa-wms/app/vjepa_wm/modelcustom/simu_env_planning/vit_enc_preds.py']
    seals={str(p.resolve()):sha(p.read_bytes()) for p in sources}
    output.mkdir(exist_ok=False)
    with (output/'contract.json').open('xb') as stream:stream.write(encoded(contract))
    receipt=dict(schema=NAME,status='prepared_source_contract_not_executed',
        contract_sha256=sha((output/'contract.json').read_bytes()),sources_sha256=seals,
        roots_bound=0,new_model_transitions=0,new_encodings=0,new_physical_steps=0,
        model_loaded=False,torch_imported_for_prepare=False,cuda_launch_authorized=False,
        P0_admission=False,P2_admission=False)
    with (output/'prepare.json').open('xb') as stream:stream.write(encoded(receipt))
    seals.update({str(output/'contract.json'):receipt['contract_sha256'],str(output/'prepare.json'):sha(encoded(receipt))})
    with (output/'SHA256SUMS').open('x') as stream:
        stream.write('\n'.join(f'{digest}  {path}' for path,digest in sorted(seals.items()))+'\n')
    if any(sha(Path(p).read_bytes())!=h for p,h in seals.items()):
        raise ValueError('prepare source changed during sealing')
    return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fetch', action='store_true')
    parser.add_argument('--paths', nargs='*', default=[])
    parser.add_argument('--api-source', action='store_true')
    parser.add_argument('--adapter', choices=ADAPTERS)
    parser.add_argument('--prepare', action='store_true')
    args = parser.parse_args(argv)
    if args.prepare:
        if args.fetch or args.paths or args.adapter or args.api_source:
            parser.error('prepare is offline and cannot combine with acquisition')
        destination=ROOT/'results'/NAME
        receipt=prepare(destination,destination/'prepare')
        print(json.dumps(dict(status=receipt['status'],prepare_path=str(destination/'prepare/prepare.json'),
                             prepare_sha256=sha(encoded(receipt)),contract_sha256=receipt['contract_sha256'])))
        return 0
    if not args.fetch:
        parser.error('use --fetch for bounded official source acquisition')
    fetch(ROOT / 'results' / NAME, args.paths, args.api_source, args.adapter)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
