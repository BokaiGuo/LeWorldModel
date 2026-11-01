"""Native renderer contract and fail-closed fixed-state factorial dispatch.

No environment is constructed, reset, restored or stepped by this module.
Current bound runtime has no certified agent-hide switch: full dispatch BLOCKED.
The supported target-color kernel requires an already-live exact native state.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import importlib
from pathlib import Path
import sys

import numpy as np
from experiments.spectral_reachability import agent_p5_terminal_goal_attribution_20261028_v5 as saved

TAG = 'agent_p5_renderer_factorial_20261028_v1'
ROOT = saved.ROOT
WORDS = saved.WORDS
STATES = (*WORDS, 'goal')
CONDITIONS = (('visible', 'visible'), ('visible', 'white'), ('hidden', 'visible'), ('hidden', 'white'))
GOAL = [256., 256., float(np.pi/4)]
AGENT_NUISANCE = [64., 448.]
ENV = 'external/jepa-wms/evals/simu_env_planning/envs/pusht_env/pusht_env.py'
PILOT = 'experiments/spectral_reachability/p5_jepa_factual_pilot_20261027_v1.py'
PIN = {
    ENV: '6ba34790b6c2ce017a76322c2f1c205ef97d4d3b08a620f844a57075c608a723',
    PILOT: '1f67254e90bd7e231c7433e0b8891e3955f24c3f8864eb69f3b49403b3116632',
    '.jepa-physics-site/pymunk/shapes.py': '7472185a62d242c37548ef1b6c9fba4073fef2d298a8cd6d8f4b2861b79cb9d2',
    '.jepa-physics-site/pymunk/space.py': 'e2f662bf1aaf39f30aabd800aac33ea81237f21ab4c18ab843e224e86967d8bf',
    '.jepa-physics-site/pymunk/space_debug_draw_options.py': '0693f0fb6460fba13e5cc94463108ee83bbd64ce01e115433eb974abe4a89c6a',
    '.jepa-physics-site/pymunk/_callbacks.py': '468e48bd61a46649d9b9983d873c9acb22edef5f9d08f65172ef5fd246d172b9',
}
OWNER_SHA = 'aece72c75efa24f8c5e2df12aff38a954df31307a0476edf28be2295b1320025'
PREP_SHA = '79c3c2f1aa1b8f1b9223e8d87f22986f5c3437144e10bbe78a6cc556093beafd'
REPORT_SHA = '9505f332fc6c1cbb670d9a77f46290f8d2822486b00d89006ea038359898bdc5'
RECEIPT_SHA = '33f3f24735131fd8b569c1d0a87068d1b3f4e26dab31618a4afb90005cf8c927'
GOAL_DIR = ROOT/'results/p5_native_goal_scene_20261027_v1_scene_main_v1'
GOAL_REPORT_SHA = 'eb3776d2d0e6851eec0e40b56c0a9bed3e33bdebdddca8d1fbe3d2ef866e76e2'
GENERATION_SHA = 'a1336e95a3d764d1693844e557951cc925e40751d70d010746d88faf8d5c2f4f'
require, sha, read_json, write_new = saved.require, saved.impl.sha, saved.impl.read_json, saved.impl.write_new
BLOCKERS = ['agent_hidden_native_switch_uncertified', 'saved_states_are_not_live_native_handles']
ZERO_CALLS = dict(env_construct=0, close=0, reset=0, set_state=0, control_step=0,
                  space_step=0, native_render=0, debug_draw=0, goal_body=0, body_construct=0)


def protocol():
    return dict(schema=TAG, words=list(WORDS), states=list(STATES), conditions=[list(c) for c in CONDITIONS],
                denominator=20, terminal_cells=16, goal_cells=4, object_goal_pose=GOAL,
                goal_agent_nuisance=AGENT_NUISANCE, goal_agent_is_coverage_target=False,
                supported_target='goal_color LightGreen/White, NOT changing with_target after setup',
                agent_hidden='uncertified; ignore_draw dynamic assignment is not a verified switch',
                state_source='already-live native states only; no saved snapshot hydration or physics replay',
                state_invariance='exact complete declared public state plus RNG/topology identities; no hidden cache serialization claim',
                rendering_cache='screen surface is replaced by native renderer; excluded from physical state',
                banned=['Space.step', 'reset', '_set_state', '_setup', 'constructor', 'pixel edits', 'custom DrawOptions', 'shape removal'],
                physics=0, neural=0, winner=False, scientific_admission=False, task_success=None,
                prospective=False, fresh=False, wall_seconds=600, output_cap_bytes=8*1024**2,
                free_floor_bytes=200*1024**2)


def source_contract():
    for p, h in PIN.items():
        require(sha(ROOT/p) == h, 'bound native render source: '+p)
    env = (ROOT/ENV).read_text()
    shape = (ROOT/'.jepa-physics-site/pymunk/shapes.py').read_text()
    space = (ROOT/'.jepa-physics-site/pymunk/space.py').read_text()
    options = (ROOT/'.jepa-physics-site/pymunk/space_debug_draw_options.py').read_text()
    callbacks = (ROOT/'.jepa-physics-site/pymunk/_callbacks.py').read_text()
    require('pygame.draw.polygon(canvas, self.goal_color, goal_points)' in env and
            'self.space.debug_draw(draw_options)' in env and 'self.goal_color = pygame.Color("White")' in env,
            'original target and shape render route')
    require('ignore_draw' not in shape+space+options+callbacks, 'uncertified switch contract changed; new version review required')
    return dict(target_color_supported=True, agent_hide_supported=False, agent_hide_status='unestablished',
                evidence='no ignore_draw consumer/property in bound Shape/Space/options/CFFI callbacks; examples are not API certification',
                with_target_runtime_toggle=False, native_set_state_advances_space=True)


def prepare(manifest, report, receipt):
    require(sha(saved.__file__) == OWNER_SHA, 'frozen v5 source')
    require(tuple(map(sha, (manifest, report, receipt))) == (PREP_SHA, REPORT_SHA, RECEIPT_SHA), 'actual saved authorities')
    m = read_json(manifest, PREP_SHA)
    require(saved.consume(m, Path(report), REPORT_SHA) == read_json(receipt, RECEIPT_SHA), 'exact v5 full strict return')
    contract = source_contract()
    bindings = {str((ROOT/p).resolve()): h for p, h in PIN.items()}
    bindings.update({str(Path(p).resolve()): sha(p) for p in (manifest, report, receipt, saved.__file__)})
    records = []
    for w in WORDS:
        t = m['authority']['terminals'][w]
        records.append(dict(state_id=w, native_path=t['native']['state_file'], native_sha=t['native']['state_sha'],
                            rgb=t['rgb'], original_target='white'))
    generation = read_json(GOAL_DIR/'generation.json', GENERATION_SHA)
    gr = read_json(GOAL_DIR/'report.json', GOAL_REPORT_SHA)
    require(generation['execution_report_sha256'] == GOAL_REPORT_SHA and generation['object_goal_pose'] == GOAL
            and generation['agent_nuisance'] == AGENT_NUISANCE and gr['errors'] == []
            and gr['status'] == 'native_goal_scene_recorded', 'independent native goal provenance')
    for name, rec in gr['assets']['record']['files'].items():
        path = GOAL_DIR/'frame_00'/name
        require(sha(path) == rec['sha256'] and path.stat().st_size == rec['bytes'], 'goal state member')
        bindings[str(path)] = rec['sha256']
    for path, h in ((GOAL_DIR/'generation.json', GENERATION_SHA), (GOAL_DIR/'report.json', GOAL_REPORT_SHA),
                    (GOAL_DIR/'geometry.json', generation['geometry_sha256']), (GOAL_DIR/'journal.jsonl', generation['journal_sha256'])):
        require(sha(path) == h, 'goal producer member')
        bindings[str(path)] = h
    goal_rgb = m['authority']['gm']['public_manifest']['inputs']['goal_rgb']
    goal_saved = gr['assets']['record']['files']['rgb.npy']
    require(generation['goal_rgb_sha256'] == goal_saved['sha256'] == goal_rgb['sha256']
            and Path(goal_rgb['path']).read_bytes() == (GOAL_DIR/'frame_00/rgb.npy').read_bytes(), 'exact public goal image')
    records.append(dict(state_id='goal', native_path=str(GOAL_DIR/'frame_00/full_public_state.json'),
        native_sha=gr['assets']['record']['files']['full_public_state.json']['sha256'],
        rgb=dict(path=str(GOAL_DIR/'frame_00/rgb.npy'), **goal_saved), original_target='visible'))
    for rec in records:
        state = read_json(rec['native_path'], rec['native_sha'])
        require(state['env']['goal_pose']['values'] == GOAL, 'unchanged object coverage target')
        if rec['state_id'] == 'goal':
            require(state['bodies']['agent']['position'] == AGENT_NUISANCE and
                    state['bodies']['block']['position'] == GOAL[:2] and
                    state['bodies']['block']['angle'] == GOAL[2], 'goal agent nuisance and exact object pose')
        require(sha(rec['rgb']['path']) == rec['rgb']['sha256'], 'sealed state RGB')
        bindings[rec['native_path']] = rec['native_sha']
        bindings[rec['rgb']['path']] = rec['rgb']['sha256']
    for p in (Path(__file__), ROOT/'tests'/('test_'+TAG+'.py'), ROOT/'innerdocs'/(TAG+'.md')):
        bindings[str(p.resolve())] = sha(p)
    return dict(schema=TAG, status='blocked_before_any_native_dispatch', protocol=protocol(),
                source_contract=contract, blockers=list(BLOCKERS), states=records,
                design=[dict(state_id=s, agent=a, target=t) for s in STATES for a, t in CONDITIONS],
                authority=dict(manifest=str(Path(manifest).resolve()), report=str(Path(report).resolve()),
                               receipt=str(Path(receipt).resolve())), bindings=bindings,
                physical_release=False, scientific_admission=False)


def verify(m):
    require(all(sha(p) == h for p, h in m['bindings'].items()), 'binding drift')
    require(m == prepare(**m['authority']), 'full fixed-state contract recomputation')


def blocked_result(m):
    return dict(schema=TAG+'.report', status='blocked_not_executed', manifest_digest=saved.impl.digest(m),
                rows=[dict(**c, status='not_run', reason=list(BLOCKERS)) for c in m['design']],
                attempted=dict(ZERO_CALLS), returned=dict(ZERO_CALLS), blockers=list(BLOCKERS),
                physical_executed=False, neural_executed=False, scientific_admission=False,
                task_success=None, assets=[])


def run(m, main_authorized=False):
    require(main_authorized, 'MAIN authorization required')
    verify(m)
    # Crucially no native import/constructor/provider invocation before rejection.
    return blocked_result(m)


def consume(m, report, report_sha):
    verify(m)
    r = read_json(report, report_sha)
    require(r == blocked_result(m), 'cannot promote blocked factorial to rendered evidence')
    return dict(schema=TAG+'.consume', status='blocked_dispatch_record_verified_NOT_rendered',
                report_sha256=report_sha, denominator=20, rendered=0,
                blockers=list(BLOCKERS), physical_release=False, scientific_admission=False)


class RenderGuard:
    """Observe native call/return, reject forbidden Python entry before its body."""
    def __init__(self, env):
        cls, space, body = type(env), type(env.space), type(env.block)
        methods = dict(native_render=cls._render_frame, debug_draw=space.debug_draw,
                       goal_body=cls._get_goal_pose_body, body_construct=body.__init__,
                       reset=cls.reset, set_state=cls._set_state, control_step=cls.step,
                       space_step=space.step, setup=cls._setup, env_construct=cls.__init__,
                       space_construct=space.__init__, seed=cls.seed, close=cls.close)
        require(all(hasattr(fn, '__code__') for fn in methods.values()), 'Python guard interfaces required')
        self.codes = {fn.__code__: name for name, fn in methods.items()}
        require(len(self.codes) == len(methods), 'unambiguous native profiler methods')
        self.attempted, self.returned, self.pending = Counter(), Counter(), {}
        self.errors = []

    def profile(self, frame, event, arg):
        name = self.codes.get(frame.f_code)
        if name is None:
            return
        if event == 'call':
            self.attempted[name] += 1
            if name not in ('native_render', 'debug_draw', 'goal_body', 'body_construct'):
                self.errors.append(name)
                raise ValueError('forbidden native entry: '+name)
            self.pending[id(frame)] = name
        elif event == 'return':
            self.returned[name] += 1
            self.pending.pop(id(frame), None)

    def __enter__(self):
        require(sys.getprofile() is None, 'existing profiler; do not replace')
        sys.setprofile(self.profile)
        return self

    def __exit__(self, *exc):
        sys.setprofile(None)

    def check(self):
        want = dict(native_render=1, debug_draw=1, goal_body=1, body_construct=1)
        require(not self.errors and not self.pending and dict(self.attempted) == dict(self.returned) == want,
                'exact one native render/goal body/debug draw/detached body return')


def _target_pair(env, expected, sealed_rgb, state_reader, color_factory, guard_factory):
    """RAM seam; never hydration. Public entry validates actual native provenance."""
    require(state_reader(env) == expected, 'already-live exact complete public source required')
    require(env.render_action is False and env.window is None and env.clock is None, 'native offscreen no action overlay')
    require(sealed_rgb.shape == (224, 224, 3) and sealed_rgb.dtype == np.uint8, 'sealed original RGB ABI')
    original = env.goal_color
    topology = (id(env.space), id(env.agent), id(env.block), tuple(sorted(map(id, env.space.shapes))),
                tuple(sorted(map(id, env.space.bodies))))
    def stable():
        require(state_reader(env) == expected, 'physical state/RNG/settings changed')
        require(topology == (id(env.space), id(env.agent), id(env.block), tuple(sorted(map(id, env.space.shapes))),
                tuple(sorted(map(id, env.space.bodies)))), 'live topology replaced')
    rows = []
    try:
        # Baseline before either new condition: exact original raw RGB, not an endpoint substitute.
        for target, color in (('baseline', original), ('visible', color_factory('LightGreen')), ('white', color_factory('White'))):
            stable()
            env.goal_color = color
            with guard_factory(env) as guard:
                image = env._render_frame('rgb_array')
            guard.check()
            require(image.shape == (224, 224, 3) and image.dtype == np.uint8, 'native returned RGB ABI')
            stable()
            if target == 'baseline':
                require(np.array_equal(image, sealed_rgb), 'original RGB byte gate')
            rows.append(dict(target=target, rgb=image.copy(), before=deepcopy(expected), after=state_reader(env),
                             attempted=dict(guard.attempted), returned=dict(guard.returned)))
    finally:
        env.goal_color = original
        require(env.goal_color is original, 'restore original rendering color object')
        stable()
    return rows


def render_live_target_pair(env, expected, sealed_rgb, main_authorized=False):
    """Supported 1x2 kernel ONLY, not a full factorial runner or asset admission.

    Requires MAIN-owned live env; caller retains ownership/cleanup. No goal/body
    setter, snapshot reconstruction, env construction, reset, step, neural import.
    """
    require(main_authorized, 'MAIN authorization required')
    source_contract()
    require(not any(n == 'torch' or n.startswith('torch.') for n in sys.modules), 'no neural-loaded render process')
    require(Path(type(env)._render_frame.__code__.co_filename).resolve() == ROOT/ENV, 'original native renderer required')
    for name in ('pymunk', 'pygame'):
        mod = sys.modules.get(name)
        require(mod is not None and Path(mod.__file__).resolve().is_relative_to(ROOT/'.jepa-physics-site'), 'bound overlay '+name)
    base = importlib.import_module('experiments.spectral_reachability.p5_jepa_factual_pilot_20261027_v1')
    require(sha(base.__file__) == PIN[PILOT], 'original full-public-state reader')
    return _target_pair(env, expected, sealed_rgb, base.public_state,
                        sys.modules['pygame'].Color, RenderGuard)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=('prepare', 'verify', 'run', 'consume'))
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--manifest-sha256')
    p.add_argument('--authority', nargs=3, type=Path, help='v5 prepare/report/receipt')
    p.add_argument('--report', type=Path)
    p.add_argument('--report-sha256')
    p.add_argument('--output', type=Path)
    p.add_argument('--main-authorized', action='store_true')
    a = p.parse_args(argv)
    if a.command == 'prepare':
        require(a.authority is not None, 'actual three authorities required')
        result = prepare(*a.authority)
        write_new(a.manifest, result)
    else:
        require(a.manifest_sha256 and sha(a.manifest) == a.manifest_sha256, 'actual prepared SHA required')
        m = read_json(a.manifest, a.manifest_sha256)
        if a.command == 'verify':
            verify(m)
            result = dict(status='blocked_contract_verified_NOT_dispatch_ready')
        elif a.command == 'run':
            require(a.output is not None, 'exclusive blocked report output required')
            result = run(m, a.main_authorized)
            write_new(a.output, result)
        else:
            require(a.report and a.report_sha256 and a.output, 'actual report SHA and exclusive output required')
            result = consume(m, a.report, a.report_sha256)
            write_new(a.output, result)
    print(result['status'])
    return 2 if result['status'] == 'blocked_not_executed' else 0


if __name__ == '__main__':
    raise SystemExit(main())
