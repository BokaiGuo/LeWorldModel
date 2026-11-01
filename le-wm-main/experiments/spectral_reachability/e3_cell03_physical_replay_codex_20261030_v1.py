"""Exploratory fixed-action native replay of the saved cell03 trajectory; no neural calls."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time
import traceback

import numpy as np
from PIL import Image
import io

from experiments.spectral_reachability import e3_cell03_cuda_parity_codex_20261030_v2 as parity
from experiments.spectral_reachability import e3_cell03_encoder_parity_inputs_codex_20261030_v1 as inputs
from experiments.spectral_reachability import agent_e3_feedback_setup_20261029_v2 as setup
from experiments.spectral_reachability import agent_e3_native_region_adapter_20261029_v1 as region
from experiments.spectral_reachability import p1_moving_live_feedback_physical_20261027_v1 as native

ROOT = inputs.ROOT
MANIFEST = ROOT / 'e3_cell03_physical_replay_codex_20261030_v1_manifest.json'
OUTPUT = ROOT / 'e3_cell03_physical_replay_codex_20261030_v1_RUN01'
CAP = 16 * 1024**2
WALL = 600
CAPS = dict(constructor=1, reset=2, set_state=4, goal_render=4, goal_setter=4,
            goal_bind=1, public_step=30, physics_step=1200, explicit_render=32, close=1)


def require(ok, message):
    if not ok:
        raise ValueError(message)


def raw(value):
    return (json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)+'\n').encode()


def sha(data):
    return hashlib.sha256(data).hexdigest()


def extract(events):
    def pick(name):
        return [x['detail'] for x in events if x['origin'] == 'physical' and x['resource'] == 'native_'+name]
    observations = [x['receipt'] for x in pick('fresh_observation')]
    keys = [(x['phase'], x['step']) for x in observations]
    require(keys == [('warmup', x) for x in range(11)] + [('plan', x) for x in range(21)], 'observation sequence')
    warm, controls = pick('warmup_return'), pick('control_return')
    intents = pick('authorized_dispatch')
    require(len(warm) == 10 and len(controls) == len(intents) == 20, 'fixed action denominator')
    require([x['step'] for x in warm] == list(range(1, 11)) and
            [x['step'] for x in controls] == list(range(1, 21)), 'action chronology')
    for i, (intent, result) in enumerate(zip(intents, controls), 1):
        require(intent['step'] == i and intent['packet_boundary'] == ((i-1)//5)*5 and
                intent['raw_bits'] == result['raw_bits'] and not result['terminated'] and
                not result['truncated'], 'control identity or stop')
    return dict(observations=observations, warm=warm, controls=controls)


def matches_observation(expected, actual):
    for k in ('phase', 'step', 'pixel_sha256', 'state', 'snapshot', 'goal', 'goal_rgb_sha256', 'goal_pose'):
        require(expected[k] == actual[k], 'observation mismatch: '+k)


def load_native():
    import gymnasium  # noqa: F401
    import stable_worldmodel  # noqa: F401
    from stable_worldmodel.envs.pusht.env import PushT  # noqa: F401


def material():
    # Inspect original transport and source before deriving any replay actions.
    inputs.inputs()
    load_native()
    path = inputs.OUTPUT/'events.jsonl'
    report = json.loads((inputs.OUTPUT/'report.json').read_bytes())
    data = path.read_bytes()
    require(sha(data) == report['journal_sha256'], 'source journal drift')
    trace = extract([json.loads(x) for x in data.splitlines()])
    return dict(schema='cell03_fixed_physical_replay_v1', source_report=inputs.REPORT_SHA,
                source_completion=inputs.COMPLETION_SHA, journal_sha256=sha(data),
                authority_sha256=inputs.AUTHORITY_SHA, trace=trace,
                sources=parity.loaded_project_sources(), python_sha256=parity.sha(Path(sys.executable).resolve()),
                caps=CAPS, wall_seconds=WALL, output_cap_bytes=CAP, output=str(OUTPUT),
                comparison='exact saved state, full body snapshot, pixel digest, action and native return',
                neural_calls=0, scientific_admission=False)


def verify(expected):
    data = MANIFEST.read_bytes()
    require(sha(data) == expected, 'manifest digest')
    doc = json.loads(data)
    require(doc == material() and sha(MANIFEST.read_bytes()) == expected, 'manifest/source drift')
    return doc


class Store:
    def __init__(self, path, trace):
        self.path = path
        self.index = 0
        self.trace = trace
        self.obs = self.warm = self.controls = 0
        self.used = 0

    def check(self, size=4096):
        require(self.used + size <= CAP-65536 and shutil.disk_usage(self.path).free >= 200*1024**2+size,
                'output/disk cap')

    def write(self, name, data, append=False, final=False):
        require(Path(name).name == name, 'flat output')
        self.used += len(data)
        require(self.used <= CAP-(0 if final else 65536), 'output cap')
        flags = os.O_WRONLY | os.O_CREAT | os.O_NOFOLLOW | (os.O_APPEND if append else os.O_EXCL)
        with os.fdopen(os.open(self.path/name, flags, 0o600), 'ab' if append else 'wb') as f:
            f.write(data); f.flush(); os.fsync(f.fileno())
        return sha(data)

    def emit(self, kind, **fields):
        # Record actual event before any comparison can fail.
        self.write('events.jsonl', raw(dict(index=self.index, kind=kind, detail=fields)), append=True)
        self.index += 1
        if kind == 'fresh_observation':
            require(self.obs < len(self.trace['observations']), 'extra observation')
            matches_observation(self.trace['observations'][self.obs], fields['receipt'])
            self.obs += 1
        elif kind == 'warmup_return':
            require(self.warm < 10 and fields == self.trace['warm'][self.warm], 'warmup return mismatch')
            self.warm += 1
        elif kind == 'control_return':
            require(self.controls < 20 and fields == self.trace['controls'][self.controls], 'control return mismatch')
            self.controls += 1

    def frame(self, name, image):
        stream = io.BytesIO()
        Image.fromarray(image).save(stream, format='PNG')
        return self.write(name, stream.getvalue())

    def arrays(self, name, **arrays):
        stream = io.BytesIO()
        np.savez(stream, **arrays)
        return self.write(name, stream.getvalue())


def run(expected):
    doc = verify(expected)
    os.mkdir(OUTPUT, 0o700)
    store = Store(OUTPUT, doc['trace'])
    meter = native.Meter(store)
    meter.caps = dict(CAPS)
    meter.counts = {k: dict(attempted=0, completed=0) for k in CAPS}
    row = dict(episode_id='cell03', PNG=0, checkpoints=[], warmup_returned=0,
               control_returned=0, executed_raw=[], status='running')
    env = None
    started = time.monotonic()
    failure = None
    try:
        store.emit('replay_started', manifest_sha256=expected)
        binding = inputs.host.authority(str(inputs.AUTHORITY), inputs.AUTHORITY_SHA)
        launch = inputs.host.launch
        actual = launch.episode.old.authority.load(*binding['args'])
        args = launch.episode.old.setup_authority.resolve(binding['resolver']['path'],
                     binding['resolver']['sha256'], 3, actual['body'])
        args.pop('source_bindings')
        commands = np.stack([np.frombuffer(bytes.fromhex(x['raw_bits']), dtype='<f4')
                             for x in doc['trace']['controls']])
        require(commands.shape == (20, 2) and np.isfinite(commands).all() and
                np.array_equal(commands[:5], args['prefix5']), 'fixed source actions')
        env = meter.call('constructor', lambda: native.native_factory(meter))
        state = setup.setup(env, **args, row=row, meter=meter, store=store, packet_ops=region)
        require(state is not None and store.controls == 5, 'setup stopped or incomplete')
        for start in (5, 10, 15):
            packet = region.packet(env, commands[start:start+5], start=start, row=row,
                                   goal=state['goal'], goal_rgb=state['goal_rgb'], meter=meter, store=store)
            require(packet['complete'] and not packet['stopped'], 'unexpected native stop')
        require((store.obs, store.warm, store.controls) == (32, 10, 20), 'complete replay denominator')
        current = parity.loaded_project_sources()
        require(all(doc['sources'].get(k) == v for k, v in current.items()), 'new or changed loaded project source')
        store.emit('replay_complete', observations=store.obs, warmup=store.warm, controls=store.controls)
    except BaseException as exc:
        failure = dict(type=type(exc).__name__, message=str(exc), traceback=traceback.format_exc())
        try:
            store.emit('replay_failure', error=failure)
        except BaseException:
            pass
    finally:
        if env is not None:
            try:
                meter.call('close', env.close)
                store.emit('cleanup_return')
            except BaseException as exc:
                if failure is None:
                    failure = dict(type=type(exc).__name__, message=str(exc), stage='cleanup')
        result = dict(status='failed' if failure else 'completed', failure=failure,
                      manifest_sha256=expected, elapsed_seconds=time.monotonic()-started,
                      observations=store.obs, warmup=store.warm, controls=store.controls,
                      counts=meter.counts, pending=meter.pending, neural_calls=0,
                      full_saved_strict=False, scientific_admission=False)
        store.write('report.json', raw(result), final=True)
    return 1 if failure else 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('prepare', 'run'))
    parser.add_argument('--sha256')
    args = parser.parse_args()
    if args.command == 'prepare':
        data = raw(material())
        parity.write_new(MANIFEST, data)
        parity.syncdir(ROOT)
        print(sha(data))
        return 0
    require(args.sha256 is not None, 'external manifest digest required')
    return run(args.sha256)


if __name__ == '__main__':
    sys.exit(main())
