"""Saved resolver compatibility: preserve shared-panel original 256 MiB report ABI."""
from pathlib import Path
import argparse
from experiments.spectral_reachability import agent_p3_same_action_history_preconditions_20261029_v1 as resolver

NAME = 'main_p3_history_preconditions_compat_20261029_v1'
PIN = '4c2c786573bdde712fbfbbfb49fe45c6155f7d3a4136d5ac98497937d0c15898'
REPORT_CAP = 256*1024**2
require = resolver.require


def input_cap(key):
    require(key in resolver.KEYS, 'one of the six fixed authority roles')
    return REPORT_CAP if key == 'planner_report' else resolver.META


def build(**arguments):
    require(set(arguments) == {k+s for k in resolver.KEYS for s in ('', '_sha')}, 'six exact authority roles')
    require(resolver.sha(Path(resolver.__file__).read_bytes()) == PIN, 'unchanged frozen resolver')
    bound, caps, documents = {}, {}, []
    for key in resolver.KEYS:
        path = Path(arguments[key])
        require(path.is_absolute() and not path.is_symlink(), 'explicit absolute authority')
        path = str(path.resolve()); h = arguments[key+'_sha']; cap = input_cap(key)
        documents.append(resolver.saved.decode(resolver.saved.checked_bytes(path, h, cap)))
        bound[path], caps[path] = h, cap
    p, r, receipt, pp, pr, pc = documents
    for key, rep, rc in (('encoder', r, receipt), ('planner', pr, pc)):
        require(Path(rep['prepared']).resolve() == Path(arguments[key+'_prepared']).resolve()
                and rep['prepare_sha256'] == arguments[key+'_prepared_sha']
                and rc['report_sha256'] == arguments[key+'_report_sha'], 'complete actual authority chain')
    require(pc['prepare_sha256'] == arguments['planner_prepared_sha']
            and Path(pc['prepared']).resolve() == Path(arguments['planner_prepared']).resolve(), 'planner receipt join')
    for path, h in resolver.PINS.items():
        resolver.saved.checked_bytes(path, h, resolver.META); bound[str(path)] = h
    for key in ('planner_prepared', 'planner_report', 'planner_receipt'):
        require(p['bindings'][str(Path(arguments[key]).resolve())] == arguments[key+'_sha'], 'encoder inherited planner closure')
    folder = Path(arguments['encoder_report']).resolve().parent
    def latent(d):
        a = resolver.saved.load_latent(folder, d, r['artifact_sha256'])
        bound[str(folder/d['file'])] = d['sha256']; return a
    planner_dir = Path(arguments['planner_report']).resolve().parent
    def journal(n, scope):
        path = planner_dir/(n['episode_id']+'_b05')/'events.jsonl'
        data, ref = resolver.read_selected_journal(path, n['native_journal_sha256'], n['native_journal_tail'], scope)
        bound[str(path)] = ref['sha256']; return data, ref
    result = resolver.resolve(p, r, receipt, pp, pr, pc, latent, journal)
    own = [Path(__file__).resolve(), resolver.ROOT/'tests'/('test_'+NAME+'.py'),
           resolver.ROOT/'innerdocs'/(NAME+'.md'), *resolver.own_files()]
    for f in own:
        bound[str(f)] = resolver.sha(f.read_bytes())
    for f, h in bound.items():
        resolver.saved.checked_bytes(f, h, caps.get(f, resolver.JOURNAL_CAP))
    return dict(schema=NAME+'.prepare', arguments=arguments, contract=resolver.contract(), bindings=bound,
        **result, status='all8_saved_preconditions_ready_NOT_execution' if result['blocked'] == 0 else 'blocked_saved_preconditions',
        compatibility=dict(unchanged_resolver_sha256=PIN, planner_report_cap=REPORT_CAP,
            other_input_caps_unchanged=True, R_IO_math_resource_checks_unchanged=True,
            original_shared_panel_report_reserve_bytes=REPORT_CAP),
        evidence_scope=dict(full_upstream_strict_rebuilt_here=False, new_neural_parity=False,
            inherits_MAIN_complete_encoder_and_planner_receipts=True),
        inherited_resources=dict(encoder=receipt['counts'], planner=pr['per_episode_neural']),
        resources_here=dict(predict=0, encoder=0, projector=0, AD=0, physics=0),
        P1_admission=False, P3_admission=False, C3=False, physical_release=False)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in resolver.KEYS:
        p.add_argument('--'+key.replace('_', '-'), required=True)
        p.add_argument('--'+key.replace('_', '-')+'-sha', required=True)
    p.add_argument('--output', required=True)
    args = vars(p.parse_args()); output = args.pop('output')
    value = build(**args)
    # Reuse exclusive durable writer; keep original own-stem requirement visible.
    require(Path(output).name.startswith(resolver.NAME+'_'), 'original resolver result namespace')
    print(resolver.canonical(resolver.write(output, value)).decode())


if __name__ == '__main__':
    main()
