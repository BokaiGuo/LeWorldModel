"""Read-only, identity-closed actual history inventory; never render or encode."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np

from experiments.spectral_reachability import agent_p2_midpoint_history_lineage_20261028_v1 as lineage

NAME = 'agent_p2_midpoint_triplet_inventory_20261028_v1'
suffix, mid, gap = lineage.suffix, lineage.mid, lineage.gap
ROOT, ARMS = lineage.ROOT, lineage.ARMS
read, require, sha, digest = lineage.read, lineage.require, lineage.sha, lineage.digest
P4_RECEIPT = ROOT/'results/p4_feedback_live_cuda_20261027_v1_cuda_main01/receipt.json'
P4_SHA = 'e3c91195540b7465e2fcbf8431d7f182c06bfdff9da6ab160489ae6fe23bbb47'
LINEAGE_SOURCES = dict(zip((ROOT/f'experiments/spectral_reachability/{lineage.NAME}.py',
                           ROOT/f'tests/test_{lineage.NAME}.py', ROOT/f'innerdocs/{lineage.NAME}.md'),
                          ('ae5c2bd2cb1e96101a0ea880000be682e2b2bc4e57d055e14c1d59a1aac4ff56',
                           'e573f713b3f0421bebc530a77ab125edfd15a757d5d6ac498dd4b6c40a0ca037',
                           'a61536990186fb611b33329bdffc407b0ebc287799e35df2b97acf723741bd8f')))
META_CAP = 64*1024**2


def pixel_sha(rgb):
    require(rgb.dtype == np.uint8 and rgb.shape == (224, 224, 3), 'actual RGB224 uint8, no conversion')
    return hashlib.sha256(rgb.tobytes()).hexdigest()


class Inputs:
    def __init__(self): self.files = {}

    def bind(self, path, expected):
        path = mid.regular(path)
        require(sha(path) == expected, 'saved input/source SHA: '+str(path))
        require(str(path) not in self.files or self.files[str(path)] == expected, 'one input identity')
        self.files[str(path)] = expected
        return path

    def doc(self, path, expected): return read(self.bind(path, expected))

    def finish(self):
        for path, h in self.files.items(): require(sha(path) == h, 'end-of-read source/input drift')


def image(inputs, path, expected, pixel=None):
    path = inputs.bind(path, expected)
    rgb, h = mid.pixels(path, expected)
    if pixel is not None: require(h == pixel, 'actual pixel SHA')
    return dict(path=str(path), file_sha256=expected, pixel_sha256=h), rgb


def validate_history(reference, frames, rgbs, goal_rgb):
    """Chronological event witnesses AND exact whole-stack pixels, not filenames."""
    require(len(frames) == len(rgbs) == 3 and [f['warmup_step'] for f in frames] == [5, 10, 15], 'three actual warmup boundaries')
    require(len({f['observation_id'] for f in frames}) == 3, 'distinct real observation events, never copy/relabel one frame')
    require(len({f['path'] for f in frames}) == 3, 'three real saved frame references, not repeated PNG alias')
    for frame, rgb in zip(frames, rgbs, strict=True):
        require(frame['root_sha256'] == reference['root_sha256'] and frame['actual_observation'] is True and
                frame['pixel_sha256'] == pixel_sha(rgb), 'frame root/actual pixels identity')
    require(hashlib.sha256(np.stack(rgbs).tobytes()).hexdigest() == reference['history_rgb_sha256'] and
            pixel_sha(goal_rgb) == reference['goal_rgb_sha256'], 'original full history/goal exact identity')
    return dict(root_sha256=reference['root_sha256'], history_rgb_sha256=reference['history_rgb_sha256'],
                goal_rgb_sha256=reference['goal_rgb_sha256'], frames=frames,
                proof='distinct observed boundary records + exact stacked RGB bytes and goal pixels', state_at_warmup10=None)


def p4_histories(inputs, references):
    """Known terminal schema adapter. Source-derived references, then content joins."""
    receipt = inputs.doc(P4_RECEIPT, P4_SHA)
    require(receipt['schema'] == 'p4_feedback_live_cuda_20261027_v1.receipt' and
            receipt['status'] == 'two_live_20step_3round_arms_completed' and receipt['errors'] == [] and
            receipt['source_verified'] is True and receipt['journal_failed'] is False, 'completed sealed P4 source run')
    folder, files = P4_RECEIPT.parent, receipt['output_files_sha256']
    def artifact(name):
        require(Path(name).name == name and name in files, 'flat terminal-manifest membership')
        return inputs.bind(folder/name, files[name])
    binding = read(artifact('execution_binding.json'))
    require(binding['prepare_sha256'] == receipt['prepare_sha256'], 'P4 terminal/prepare binding')
    prepared = Path(binding['prepared'])
    prep = inputs.doc(prepared/'prepare.json', binding['prepare_sha256'])
    p = inputs.doc(prepared/'protocol.json', prep['protocol_sha256'])
    for seal in prep['source_seals']:
        mid.verify_seal(output_dir=seal['output_dir'], expected_manifest_sha256=seal['manifest_sha256'])
        inputs.bind(Path(seal['output_dir'])/'manifest.json', seal['manifest_sha256'])
    for path, h in p['sources_sha256'].items(): inputs.bind(path, h)
    r = receipt['root_sha256']
    require(r == p['root_sha256'] and r in references and p['references'][r] == references[r], 'P4 exact original root/history identity')
    require(receipt['physics_attempts'] == receipt['physics_returns'] == dict(
        branch_resets=2, root_resets=32, selected_steps=40, warmup_steps=30), 'P4 actual warmup accounting')
    # These addresses follow the sealed schema's start()/guarded() implementation;
    # an address alone never admits a frame. All role/stack identities below apply.
    histories = []
    for arm in ('imagined', 'actual'):
        reset = read(artifact(arm+'.reset_identity.json'))
        require(reset['root'] == r and reset['actual_warmup_steps'] == 15 and
                reset['identity'] == {k: references[r][k] for k in ('history_rgb_sha256','goal_rgb_sha256')}, 'same actually reset history')
        goal_ref = reset['goal_rgb']
        goal, goal_rgb = image(inputs, artifact(goal_ref['file']), goal_ref['file_sha256'], goal_ref['pixel_sha256'])
        frames, rgbs = [], []
        for packet in range(3):
            prefix = arm+f'.warmup{packet}'
            physics = read(artifact(prefix+'.physics.json'))
            require(physics['status'] == 'completed' and physics['attempted_commands'] == physics['completed_commands'] == 5 and
                    physics['guard']['root_id'] == r and physics['error'] is None, 'five actual zero steps, no padded endpoint')
            for step in range(1, 6):
                event_path = artifact(prefix+f'.step{step}.json')
                event = read(event_path)
                require(event['command'] == [0.,0.] and event['terminated'] is event['truncated'] is False, 'actual zero command chronology')
                ref = event['rgb']
                img, rgb = image(inputs, artifact(ref['file']), ref['file_sha256'], ref['pixel_sha256'])
                if step == 5:
                    require(pixel_sha(rgb) == physics['terminal_pixel_sha256'], 'packet terminal frame identity')
                    frames.append(dict(**img, warmup_step=5*(packet+1), actual_observation=True, root_sha256=r,
                                       observation_id=str(event_path), state=None))
                    rgbs.append(rgb)
        h = validate_history(references[r], frames, rgbs, goal_rgb)
        h.update(source_receipt_sha256=P4_SHA, source_run=str(folder), source_arm=arm, goal_frame=goal)
        histories.append(h)
    return histories


def discover(references, directory):
    """Discovery only. Unrecognized schemas never become executable evidence."""
    directory = Path(directory).absolute()
    needles = sorted({v[k] for v in references.values() for k in ('root_sha256', 'history_rgb_sha256')})
    command = ['rg','-l','--null','--no-ignore','--glob','*.json','--glob','*.jsonl',
               '--glob','!*fresh*','--glob','!*reserved*','--glob','!**/*fresh*/**','--glob','!**/*reserved*/**',
               '--max-filesize','64M','--fixed-strings']
    for n in needles: command += ['-e', n]
    process = subprocess.run(command+[str(directory)], capture_output=True, timeout=60)
    require(process.returncode in (0,1), 'complete rg discovery, no silent scan failure: '+process.stderr.decode(errors='replace')[:500])
    paths = sorted(Path(x.decode()).absolute() for x in process.stdout.split(b'\0') if x)
    matches, candidates, errors = {}, [], []
    schemas = {}
    for path in paths:
        # This version's output cannot become its own input or alter a later verify.
        if any(part.startswith(NAME) for part in path.relative_to(directory).parts): continue
        try:
            mid.regular(path)
            raw = path.read_bytes()
            require(len(raw) <= META_CAP, 'bounded discovered metadata')
            h = hashlib.sha256(raw).hexdigest()
            matches[str(path)] = h
            # Most matching files are J arrays/context copies, not pixel assets.
            # Only parse possible actual-image/index documents, avoiding numeric banks.
            if path.suffix == '.json' and any(marker in raw for marker in
                    (b'.png', b'.npz', b'"rgb_history"', b'"history_frames"', b'"frames"')):
                value = json.loads(raw)
                schema = value.get('schema') if isinstance(value, dict) else None
                schemas[str(schema)] = schemas.get(str(schema),0)+1
                candidates.append(dict(path=str(path), sha256=h, schema=schema,
                                       admission='discovery_only; no image is admitted by a root string or filename'))
        except (OSError, ValueError, UnicodeError) as exc:
            errors.append(dict(path=str(path), error=type(exc).__name__+': '+str(exc)))
    return dict(scope=str(directory), metadata_max_bytes=META_CAP, matched_metadata_sha256=matches,
                candidate_image_documents=candidates, candidate_schema_counts=schemas, errors=errors,
                ignored_files_included=True, symlinks_not_followed=True, reserved_and_fresh_paths_excluded=True,
                unmatched_or_orphan_PNG_NPY_NPZ_not_execution_evidence=True,
                limitations=['identity-string-indexed JSON/JSONL within64MiB; no universal disk-absence claim',
                             'PNG/array names and pixel similarity cannot supply missing root/history/chronology witnesses',
                             'unsupported formats stay discovery-only, not silently accepted or declared absent'])


def build_rows(jobs, observations, histories):
    """Preserve all8; valid identical pixels at distinct actual times are allowed."""
    require([(j['root_slot'],j['arm']) for j in jobs] == [(i,a) for i in range(4) for a in ARMS], 'canonical all8 inputs')
    gap.census(observations)
    index = gap.census(observations)
    rows = []
    for job in jobs:
        obs = index[job['root_slot'], job['arm']]
        require(obs['root_sha256'] == job['root_sha256'], 'root identity, not root index alone')
        candidates = [h for h in histories if h['root_sha256'] == job['root_sha256'] and
                      h['history_rgb_sha256'] == obs['history_rgb_sha256'] and h['goal_rgb_sha256'] == obs['goal_rgb_sha256']]
        # Stable evidence order only; no outcomes/scores. Retain all alternatives.
        candidates.sort(key=lambda h: (h['source_receipt_sha256'], h['source_arm']))
        older = candidates[0]['frames'][1] if candidates else None
        if candidates:
            require(all(c['frames'][1]['pixel_sha256'] == older['pixel_sha256'] for c in candidates), 'alternative history evidence agrees')
            require(candidates[0]['frames'][2]['pixel_sha256'] == obs['root_frame']['pixel_sha256'], 'warmup15 equals actual P2 root pixels')
        selected = [older, obs['root_frame'], obs['step5_frame']]
        missing = ['warmup10_actual_observation_and_exact_original_history_witness'] if older is None else []
        rows.append(dict(root_slot=job['root_slot'], arm=job['arm'], root_sha256=job['root_sha256'],
                         job_binding_sha256=job['job_binding_sha256'], original_goal_vector_sha256=digest(job['goal']),
                         original_metric_id=job['metric_id'], original_suffix_actions_sha256=digest(job['raw_suffix5']),
                         history_rgb_sha256=obs['history_rgb_sha256'], goal_rgb_sha256=obs['goal_rgb_sha256'],
                         ordered_roles=['warmup10','warmup15_planning_root','actual_step5'], frames=selected,
                         equivalent_history_evidence=[dict(source_run=c['source_run'],source_arm=c['source_arm'],
                                                           receipt_sha256=c['source_receipt_sha256']) for c in candidates],
                         status='exact_triplet_available' if not missing else 'missing_identity_closed_history',
                         missing=missing, ready_for_public_rollout=not missing, new_model_calls=0, new_physics_calls=0))
    return rows


def reconstruction(rows):
    gap.census(rows)
    missing = sorted({r['root_slot'] for r in rows if r['missing']})
    n = len(missing)
    return dict(status='MAIN_review_required_not_authorized_not_executed', missing_root_slots=missing,
                missing_branch_slots=[dict(root_slot=r['root_slot'],arm=r['arm'],missing=r['missing']) for r in rows if r['missing']],
                reconstruct_once_per_root_share_only_after_identity_proof=True,
                procedure=['use original fixed4 collector/count/seed/oversample, no root/outcome reselection',
                           'fresh reset for each missing root, original state/goal recipe and real3 zero packets of5 raw calls',
                           'persist distinct actual warmup5/10/15 PNG+pixels+state7+snapshots+commands immediately',
                           'require original full history stack SHA, goal pixels, original root7/state identity before reuse',
                           'do not execute selected prefix/suffix: original actualstep5 already saved for all8',
                           'on mismatch/stop/engineering error retain all8, no pad/relabel/retry or model fallback'],
                planned_public_calls=dict(constructor_attempts=1 if n else 0,
                                          original_collector_reset_attempts=32 if n else 0,
                                          missing_root_reset_attempts=n, zero_warmup_step_attempts=15*n,
                                          selected_action_step_attempts=0,
                                          returned_counts=None),
                helper_explicit_render_lower_bound=15*n,
                native_constructor_reset_space_step_setter_render_goalbind_attempt_return='must be observed separately; unknown until MAIN runs; no claim hooks precede factory return',
                capture=dict(boundary_RGB=3*n, original_goal_RGB=n, state_dimensions=7,
                             full_raw_and_step_state_snapshots=15*n),
                encoder=0, projector=0, predict=0, AD=0,
                wall_seconds=None, disk_bytes=None,
                pending=['MAIN-reviewed metered reconstruction adapter and presealed engineering output/wall/floor caps',
                         'capture original reset/collector/goal semantics, verify installed dependencies and source hashes',
                         'MAIN authorization if new physical reconstruction is needed'],
                downstream_only_after_all8=dict(public_predict=8,manual_predict=8,action_encoder=16,predictor=16,pred_proj=16,
                                               encoder=0,projector=0,AD=0,physics=0,
                                               representation='reuse original cached teacher full3 and genuine triplet; no re-encoding or stageA'),
                P1_admission=False,P2_admission=False,scientific_admission=False)


def inventory(scan=True):
    inputs = Inputs()
    for path, h in {**LINEAGE_SOURCES, **lineage.FROZEN}.items(): inputs.bind(path,h)
    report_path = ROOT/lineage.INPUTS['report'][0]
    inputs.bind(report_path,lineage.INPUTS['report'][1])
    launch = read(report_path.parent/'execution_binding.json')
    p = suffix.verify(launch['prepared'],launch['prepare_sha256'])
    for path,h in {**p['bindings'],**p['sources_sha256'],**p['model_bindings']}.items(): inputs.bind(path,h)
    for path in (report_path.parent/'execution_binding.json',Path(launch['prepared'])/'prepare.json',Path(launch['prepared'])/'protocol.json'):
        inputs.bind(path,sha(path))
    physical = inputs.doc(ROOT/gap.INPUTS['physical'][0],gap.INPUTS['physical'][1])
    pi = gap.census(physical['rows'])
    require(physical['physical_calls']['resets']['root_collection'] == dict(attempts=32,returns=32), 'actual original collector budget32')
    references, observations = {}, []
    for job in p['jobs']:
        b = pi[job['root_slot'],job['arm']]['physical']
        ident = b['context_identity']
        r = job['root_sha256']
        reference = dict(root_sha256=r,history_rgb_sha256=ident['history_rgb_sha256'],goal_rgb_sha256=ident['goal_rgb_sha256'])
        require(r not in references or references[r] == reference, 'same paired original history identity')
        references[r] = reference
        journal = inputs.bind(b['journal_path'],b['journal_sha256'])
        events = [json.loads(line) for line in journal.read_bytes().splitlines()]
        initial = [e for e in events if e['kind'] == 'initial']
        later = [e for e in events if e['kind'] == 'observation' and e['primitive_step'] == 5]
        require(len(initial) == len(later) == 1 and initial[0]['state'] == ident['state'] and
                later[0]['command'] == job['raw10'][4], 'original actual root7/step5 action events')
        frames = []
        for role,e in zip(('warmup15_planning_root','actual_step5'),(initial[0],later[0]),strict=True):
            frame,rgb = image(inputs,e['rgb_path'],e['rgb_file_sha256'])
            if role == 'actual_step5': require(frame['pixel_sha256'] == job['midpoint_pixel_sha256'], 'same actual z5 input pixels')
            frame.update(role=role,root_sha256=r,state=e['state'],actual_observation=True,
                         observation_id=str(journal)+'#'+role)
            frames.append(frame)
        observations.append(dict(root_slot=job['root_slot'],arm=job['arm'],root_sha256=r,
                                 history_rgb_sha256=ident['history_rgb_sha256'],goal_rgb_sha256=ident['goal_rgb_sha256'],
                                 root_frame=frames[0],step5_frame=frames[1]))
    histories = p4_histories(inputs,references)
    rows = build_rows(p['jobs'],observations,histories)
    discovery = discover(references,ROOT/'results') if scan else dict(status='not_requested_no_global_search_claim')
    errors = discovery.get('errors',[])
    for path,h in discovery.get('matched_metadata_sha256',{}).items(): inputs.bind(path,h)
    complete = all(r['ready_for_public_rollout'] for r in rows) and not errors
    result = dict(schema=NAME+'.inventory',status='all8_exact_triplets_available' if complete else 'blocked_missing_or_unclassified_history',
                  root_denominator=4,arm_denominator=8,rows=rows,accepted_history_stacks=histories,
                  exact_triplet_branches=sum(r['ready_for_public_rollout'] for r in rows),
                  stageB_execution_ready=complete,stageB_adapter_delivered=False,
                  blocking='full8 triplet identity closure not established; no partial-root CUDA adapter or stageA replay',
                  discovery=discovery,reconstruction=reconstruction(rows),consumed_sha256=inputs.files,
                  limitations=['supported actual frame schemas: original P2 physical journal and sealed P4 live warmup schema',
                               'other matching metadata is discovery-only, not blanket evidence that no other assets exist',
                               'warmup10 native state7 absent from P4; its observed RGB chronology and exact full-stack identity are checked',
                               'identical pixel values do not justify copying one observation into another time slot'],
                  source_files_sha256={str(path):sha(path) for path in (Path(__file__).resolve(),ROOT/f'tests/test_{NAME}.py',ROOT/f'innerdocs/{NAME}.md')},
                  new_neural_calls=0,new_physics_steps=0,P1_admission=False,P2_admission=False,scientific_admission=False)
    inputs.finish()
    return result


def verify(path, expected):
    require(sha(mid.regular(path)) == expected,'external saved inventory SHA')
    r = read(path)
    require(r['schema'] == NAME+'.inventory' and len(r['rows']) == 8 and r['arm_denominator'] == 8 and
            r['root_denominator'] == 4 and r['scientific_admission'] is False,'own full8 inventory schema')
    gap.census(r['rows'])
    for file,h in {**r['consumed_sha256'],**r['source_files_sha256']}.items():
        require(sha(mid.regular(file)) == h,'snapshot input/source drift')
    require(r['reconstruction'] == reconstruction(r['rows']), 'fixed missing/budget reconstruction')
    rebuilt = inventory(scan=False)
    for key in ('rows','accepted_history_stacks','exact_triplet_branches','stageB_adapter_delivered','reconstruction','source_files_sha256'):
        require(r[key] == rebuilt[key], 'exact saved evidence reconstruction: '+key)
    require(r['stageB_execution_ready'] is False or
            (all(row['ready_for_public_rollout'] for row in r['rows']) and not r['discovery'].get('errors')), 'no incomplete-history execution gate')
    return dict(status='saved_inventory_inputs_verified_not_execution_authority',inventory_sha256=expected,
                stageB_execution_ready=r['stageB_execution_ready'],new_neural_calls=0,new_physics_steps=0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command',required=True)
    p = sub.add_parser('inventory'); p.add_argument('--output',type=Path)
    p = sub.add_parser('verify'); p.add_argument('--inventory',type=Path,required=True); p.add_argument('--sha',required=True)
    a = parser.parse_args()
    if a.command == 'verify':
        result = verify(a.inventory,a.sha)
    else:
        result = inventory()
        if a.output is not None:
            mid.write(a.output,result)
            result = dict(output=str(a.output.absolute()),sha256=sha(a.output),status=result['status'])
    print(gap.encode(result).decode())


if __name__ == '__main__': main()
