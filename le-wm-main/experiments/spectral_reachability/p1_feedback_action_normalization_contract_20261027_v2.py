"""Checkpoint-bound action normalizer gate; offline preparation only.

Conditional fallback: identity/full-goal experiments already actually exist.
Never guess checkpoint statistics from a local dataset, a weight tensor, or
prediction outcomes. A publisher/training export must be independently trusted.
This is a new opt-in interface, NOT a modification of old sealed consumers.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from experiments.spectral_reachability import p1_feedback_goal_contract_audit_20261027_v1 as audit

TAG = 'p1_feedback_action_normalization_contract_20261027_v2'
REPO = audit.REPO
EXPORT_SCHEMA = 'checkpoint_action_normalizer_training_export.v1'
CHECKPOINT = Path('/home/gbk/.stable_worldmodel/checkpoints/pusht/lewm')
IDENTITY = REPO/'results/pusht_d2p_nuisance_suffix_bank4x16_v1_20261026_cuda'
UNITS = REPO/'results/pusht_d2p_live_action_units16_v2_20261026_cuda.json'
STATS = REPO/'results/pusht_d2p_archive_action_statistics_v1_20261026.json'
FEEDBACK = REPO/'results/pusht_geometry_feedback_response_20261027_v1_prepare_v1'
COMPILER = REPO/'results/pusht_feedback_suffix_compile_20261027_v1_prepare_v1'


def digest_string(value):
    audit.require(isinstance(value, str) and len(value) == 64 and
                  all(c in '0123456789abcdef' for c in value), 'explicit lowercase SHA256 required')
    return value


def validate_export(value, weights_sha256, config_sha256):
    """Validate content/identity, NOT independently authenticate publisher claims."""
    audit.require(value.get('schema') == EXPORT_SCHEMA, 'training-export schema required')
    checkpoint = value['checkpoint']
    audit.require(checkpoint == dict(weights_sha256=digest_string(weights_sha256),
                  config_sha256=digest_string(config_sha256)), 'normalizer belongs to another checkpoint')
    p = value['provenance']
    audit.require(p['kind'] == 'fitted_training_export' and isinstance(p['training_run_id'], str)
                  and p['training_run_id'].strip(), 'actual training-run provenance required')
    for key in ('dataset_action_bytes_sha256', 'training_config_sha256', 'normalizer_recipe_sha256'):
        digest_string(p[key])
    n = value['normalizer']
    audit.require(n['command_dim'] == 2 and n['frameskip'] == 5 and
                  n['packet_order'] == 'time_major_xy' and n['formula'] == '(raw/divisor-mean)/std',
                  'exact raw-to-model packet convention required')
    mean, std = audit.array(n['mean'], (2,)), audit.array(n['std'], (2,))
    audit.require(np.all(std > 0) and np.isfinite(n['raw_action_divisor']) and
                  n['raw_action_divisor'] > 0 and n['std_ddof'] in (0,1), 'invalid fitted scaler')
    return dict(mean=mean.tolist(), std=std.tolist(), divisor=float(n['raw_action_divisor']))


class BoundActionNormalizer:
    """No public-stat default. Caller must independently review the export hash.

    Loading bytes with a trusted hash is an explicit trust decision; passing a
    self-authored JSON is not independent evidence of historical training truth.
    """
    def __init__(self, path, *, trusted_export_sha256, weights, config):
        digest_string(trusted_export_sha256)
        audit.require(audit.sha(path) == trusted_export_sha256, 'untrusted/changed normalizer export')
        self.export = audit.read(path)
        self.stats = validate_export(self.export, audit.sha(weights), audit.sha(config))
        self.export_sha256 = trusted_export_sha256

    def normalize(self, raw):
        raw = np.asarray(raw)
        audit.require(raw.dtype.kind in 'fi' and raw.ndim >= 2 and raw.shape[-2:] == (5,2)
                      and np.isfinite(raw).all() and np.max(np.abs(raw)) <= 1,
                      'finite bounded raw five-command packets required')
        # Arithmetic matches the model's float32 packet boundary, not float64
        # replay followed by a cast. Never normalize an already normalized token.
        raw = raw.astype(np.float32)
        mean = np.tile(np.asarray(self.stats['mean'], np.float32),5)
        std = np.tile(np.asarray(self.stats['std'], np.float32),5)
        return (raw.reshape(*raw.shape[:-2],10)/np.float32(self.stats['divisor'])-mean)/std

    def normalize_tensor(self, raw):
        """Differentiable drop-in packet boundary, preserving dtype/device.

        No torch import or neural operation: the caller supplies a floating
        tensor. Use this at the existing transition's raw-to-token boundary,
        only AFTER a separately authorized complete-history rebind.
        """
        audit.require(hasattr(raw,'new_tensor') and hasattr(raw,'is_floating_point'), 'floating tensor required')
        audit.require(raw.is_floating_point() and raw.ndim>=2 and tuple(raw.shape[-2:])==(5,2),
                      'floating five-command tensor required')
        audit.require(bool(raw.isfinite().all()) and bool((raw.abs()<=1).all()), 'finite bounded raw tensor required')
        mean=raw.new_tensor(self.stats['mean']*5)
        std=raw.new_tensor(self.stats['std']*5)
        return (raw.reshape(*raw.shape[:-2],10)/self.stats['divisor']-mean)/std


def check_prior_identity(report):
    audit.require(report['status'] == 'completed' and all(v is True for v in report['gates'].values()),
                  'prior identity experiment incomplete')
    p = report['protocol']
    audit.require(p['modes'] == ['identity_gn4','nuisance_gn4'] and p['iterations'] == 4
                  and p['damping'] == .01 and report['actual_predict_samples'] == 3968
                  and report['physical_future_branches'] == 128, 'prior objective contrast contract changed')
    audit.require(len(report['rows']) == 64 and {(r['root_id'],r['proposal_index']) for r in report['rows']}
                  == {(r,c) for r in range(4) for c in range(16)}, 'prior full-bank census missing')
    audit.require(p['root_ids'][1] == audit.ROOT, 'known root absent from prior full-goal contrast')
    for row in report['rows']:
        audit.require(set(row['branches']) == set(p['modes']), 'paired identity/Q branches missing')
        for branch in row['branches'].values():
            audit.require(branch['forecast']['metric_same_for_spectrum_and_GN'], 'mixed objective not a clean comparator')
    return dict(status='completed', modes=p['modes'], actual_predict_samples=3968,
        physical_branches=128, includes_known_root1=True,
        not_exact_current_six_context_protocol=True,
        difference='prior constant2 observed-prefix GN4 damping .01; current full10 spectral bank damping .002',
        decision='Honor conditional request: do not repeat identity/full-goal intervention.')


def witnesses(feedback, plans, normalizer=None):
    """All six contexts and all 60 predetermined candidate packets; no selection."""
    audit.require([(p['role'],p['arm']) for p in plans] == audit.ORDER, 'six fixed context order required')
    mean,std = audit.public_action_stats(REPO/'experiments/run_pusht_counterfactual.py')
    cases = {c['role']:c for c in feedback['cases']}
    rows = []
    for plan in plans:
        role = plan['role']; case = cases[role]
        audit.require(plan['root_sha256'] == audit.ROOT and plan['root_index'] == 1, 'known root required')
        audit.close(plan['prefix'],case['prefix'],'original prefix differs',atol=0)
        between = audit.array(case['between_raw'], (2,5,2))
        raw = np.stack([between[1],audit.array(plan['prefix'],(5,2))]).astype(np.float32)
        baseline = (raw.reshape(2,10)-mean)/std
        audit.close(baseline,plan['context']['action_history'],'saved incoming history differs',atol=1e-6)
        candidates = plan['candidates']
        audit.require(len(candidates)==10 and [c['candidate_id'] for c in candidates]
                      == [f'suffix_{i}' for i in range(10)],'all candidates retained in fixed order')
        suffixes = []
        for c in candidates:
            commands = audit.array(c['commands'],(10,2))
            audit.close(commands[:5],raw[1],'candidate prefix differs',atol=0)
            audit.close(commands[5:],np.tanh(audit.array(c['suffix_params'],(10,))).reshape(5,2),
                        'suffix chart differs',atol=1e-7)
            suffixes.append(dict(candidate_id=c['candidate_id'], raw_suffix=commands[5:].tolist(),
                bound_model_token=None if normalizer is None else normalizer.normalize(commands[5:]).tolist()))
        rows.append(dict(role=role,arm=plan['arm'],raw_incoming_packets=raw.tolist(),
            original_public_history_exact=True, original_public_tokens=baseline.tolist(),
            bound_incoming_tokens=None if normalizer is None else normalizer.normalize(raw).tolist(),
            candidates=suffixes, latent_reencoding_performed=False,
            warning='Do not reuse imagined latents as if regenerated under a different normalizer.'))
    return rows


def prepare(output, *, metadata=None, trusted_sha256=None):
    from experiments.spectral_reachability.d2p_prerun_source_seal import create_seal, verify_seal
    output = Path(output).resolve()
    audit.require(not output.exists(), 'exclusive NEW prepare directory required')
    audit.require((metadata is None) == (trusted_sha256 is None), 'metadata and independently trusted SHA are paired')
    inputs = audit.Inputs()
    prior = inputs.load(IDENTITY/'report.json')
    duplicate = check_prior_identity(prior)
    for p,h in prior['protocol']['bindings'].items(): inputs.bind(p,h)
    for name,h in prior['protocol']['implementation_sha256'].items():
        inputs.bind(IDENTITY/'implementation'/f'{name}.py',h)
    for row in prior['rows']:
        for b in row['branches'].values(): inputs.bind(b['stage_path'],b['stage_sha256'])
    units = inputs.load(UNITS); stats = inputs.load(STATS)
    audit.require(units['status']=='completed' and all(v is True for v in units['gates'].values())
                  and units['actual_predict_samples']==64, 'prior action-units evidence incomplete')
    audit.require(stats['status']=='completed' and stats['checkpoint_training_statistics_certified'] is False,
                  'archive statistics not training authority')
    feedback = inputs.load(FEEDBACK/'protocol.json')
    compilation = inputs.load(COMPILER/'report.json')
    audit.require(compilation['status']=='plans_sealed', 'completed six-context plans required')
    # Reuse the already executed independent audit's full bindings without
    # recomputing its metric diagnostics or rerunning any inference/physics.
    prior_audit = REPO/'results/p1_feedback_goal_contract_audit_20261027_v1_offline_v1'
    audited = inputs.load(prior_audit/'report.json')
    inputs.bind(prior_audit/'protocol.json',audited['protocol_sha256'])
    verified = inputs.load(prior_audit/'input_verification.json')
    for p,h in verified['bindings'].items(): inputs.bind(p,h)
    weights = inputs.bind(CHECKPOINT/'weights.pt'); config = inputs.bind(CHECKPOINT/'config.json')
    config_value = inputs.load(config)
    # Bounded top-level inventory; do not search arbitrary user directories.
    inventory = sorted(p.name for p in CHECKPOINT.iterdir() if p.is_file())
    sources = [Path(__file__),REPO/f'tests/test_{TAG}.py',REPO/f'innerdocs/{TAG}.md',
        REPO/'experiments/spectral_reachability/d2p_prerun_source_seal.py',
        Path(audit.__file__),REPO/'utils.py',REPO/'train.py',REPO/'config/train/data/pusht.yaml',
        REPO/'.venv/lib/python3.12/site-packages/stable_worldmodel/wm/utils.py']
    for p in sources: inputs.bind(p)
    normalizer = None
    if metadata is not None:
        inputs.bind(metadata,trusted_sha256)
        normalizer = BoundActionNormalizer(metadata,trusted_export_sha256=trusted_sha256,weights=weights,config=config)
    rows = witnesses(feedback,compilation['plans'],normalizer)
    inputs.unchanged()
    output.mkdir(exist_ok=False)
    def write(name,value):
        with (output/name).open('x') as f:
            json.dump(value,f,sort_keys=True,indent=2,allow_nan=False); f.write('\n')
    protocol = dict(schema=TAG,bindings=inputs.bindings,root_sha256=audit.ROOT,root_index=1,
        new_predict_calls=0,new_encoder_calls=0,new_physical_steps=0,training=False,outcome_selection=False,
        trusted_export_sha256=trusted_sha256,preparation_only=True,P1_admission=False,
        scope='Checkpoint-bound preprocessing gate and raw packet witnesses, not a new neural comparator.')
    write('protocol.json',protocol)
    seal = create_seal(files=sources,args_json=json.dumps(dict(protocol_sha256=audit.sha(output/'protocol.json'))),
                       output_dir=output/'source_seal')
    write('packet_witnesses.json',rows)
    mean,std = audit.public_action_stats(REPO/'experiments/run_pusht_counterfactual.py')
    result = dict(schema=TAG,status='prepared_not_run',duplicate_check=duplicate,
        checkpoint_inventory=inventory,config_top_level_keys=sorted(config_value),
        missing_artifact='Publisher/training-run fitted-normalizer export bound to these weights/config; not reconstructed from local archive statistics.',
        authoritative_metadata_available=normalizer is not None,
        runtime_gate='caller_trusted_checkpoint_bound_export' if normalizer else 'blocked_missing_authoritative_export',
        trusted_hash_is_not_independent_training_proof=True,
        public_mean=mean[:2].tolist(),public_std=std[:2].tolist(),
        archived_mean=stats['statistics']['mean'],archived_sample_std=stats['statistics']['sample_std'],
        archival_statistics_scope='Prior sealed action-only receipt, not a fresh archive scan or checkpoint-training proof.',
        archived_std_over_public=(np.asarray(stats['statistics']['sample_std'])/std[:2]).tolist(),
        prior_live_vs_extra100_already_completed=True,
        actual_budget=dict(predict_calls=0,encoder_calls=0,physical_steps=0,jacobians=0,svd=0,training_steps=0),
        witness_census=dict(contexts=6,incoming_packets=12,candidate_suffix_packets=60),
        protocol_sha256=audit.sha(output/'protocol.json'),source_seal=seal,
        packet_witnesses_sha256=audit.sha(output/'packet_witnesses.json'),
        no_runtime_or_policy_integrated=True,no_stats_substituted=True,inputs_sources_unchanged=True,
        P1_admission=False,efficacy_verified=False)
    inputs.unchanged()
    verify_seal(output_dir=seal['output_dir'],expected_manifest_sha256=seal['manifest_sha256'])
    write('prepare_receipt.json',result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['prepare','validate-export'])
    parser.add_argument('--output',type=Path)
    parser.add_argument('--metadata',type=Path)
    parser.add_argument('--trusted-export-sha256')
    args = parser.parse_args()
    if args.command=='prepare':
        audit.require(args.output is not None,'NEW output required')
        print('Prepare only: prior identity contrast exists; gate unknown checkpoint normalizer without guessing or neural work.',flush=True)
        r = prepare(args.output,metadata=args.metadata,trusted_sha256=args.trusted_export_sha256)
        print(json.dumps({k:r[k] for k in ('status','runtime_gate','witness_census','actual_budget')},indent=2))
    else:
        audit.require(args.metadata is not None and args.trusted_export_sha256 is not None,'trusted training export required')
        bound = BoundActionNormalizer(args.metadata,trusted_export_sha256=args.trusted_export_sha256,
            weights=CHECKPOINT/'weights.pt',config=CHECKPOINT/'config.json')
        print(json.dumps(dict(status='caller_trusted_export_bound',stats=bound.stats,
                             independently_authenticated_training_truth=False),indent=2))


if __name__=='__main__':
    main()
