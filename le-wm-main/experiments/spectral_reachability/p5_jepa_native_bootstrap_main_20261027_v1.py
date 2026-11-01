"""Main-only bounded isolated dependency installer. No physics or neural work."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from experiments.spectral_reachability import p5_jepa_native_physics_runtime_20261027_v1 as runtime


def execute():
    root = runtime.ROOT
    manifest_path = root / f'results/{runtime.NAME}/manifest_frozen.json'
    manifest = json.loads(manifest_path.read_text())
    plan = runtime.bootstrap(root, manifest, manifest_path)
    overlay, staging = runtime.paths(root)
    if overlay.exists() or staging.exists():
        raise FileExistsError('fresh overlay and staging required; retain prior attempts')
    overlay.mkdir()
    staging.mkdir()
    env = os.environ.copy()
    env.update(plan['environment'])
    receipt = dict(schema=Path(__file__).stem, manifest_sha256=runtime.sha(manifest_path),
                   source_sha256=runtime.sha(__file__), plan=plan, stages=[],
                   physics_calls=0, neural_calls=0, status='install_started')
    def save():
        (staging / 'installer_receipt.json').write_text(json.dumps(receipt, indent=2, allow_nan=False))
    save()
    try:
        for index, argv in enumerate((plan['stage1_argv'], plan['stage2_argv']), 1):
            stage = dict(index=index, argv=argv, status='started')
            receipt['stages'].append(stage)
            save()
            start = time.monotonic()
            with (staging / f'stage_{index}.log').open('xb') as log:
                process = subprocess.Popen(argv, env=env, cwd=root, stdout=log, stderr=subprocess.STDOUT)
                stage['pid'] = process.pid
                try:
                    while process.poll() is None:
                        runtime.budget([overlay, staging])
                        if time.monotonic() - start > 240:
                            raise TimeoutError('bounded installer stage exceeded 240 seconds')
                        time.sleep(.2)
                except BaseException:
                    process.terminate()
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
                    raise
            stage.update(returncode=process.returncode, wall_seconds=time.monotonic()-start,
                         new_bytes=runtime.budget([overlay, staging]))
            stage['status'] = 'installed' if process.returncode == 0 else 'failed'
            save()
            print(json.dumps(stage), flush=True)
            if process.returncode:
                raise RuntimeError(f'installer stage {index} failed; inspect retained log')
        runtime.verify(manifest, root)
        receipt.update(status='isolated_install_completed', new_bytes=runtime.budget([overlay, staging]))
    except BaseException as exc:
        receipt.update(status='isolated_install_failed', failure=f'{type(exc).__name__}: {exc}')
        save()
        raise
    save()
    print(json.dumps(dict(status=receipt['status'], new_bytes=receipt['new_bytes'])))


if __name__ == '__main__':
    execute()
