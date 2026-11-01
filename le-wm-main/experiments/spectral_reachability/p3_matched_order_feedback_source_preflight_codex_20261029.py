"""Read-only source reconstruction and exclusive matched-order preflight seal.

Does not authorize physical or neural execution; live guards remain mandatory.
"""
import argparse
import gzip
import json
import os
from pathlib import Path
import numpy as np
from experiments.spectral_reachability import p3_matched_order_host_codex_20261029 as host
from experiments.spectral_reachability import p3_matched_order_feedback_seam_codex_20261029 as seam
from experiments.spectral_reachability import p3_matched_order_feedback_encoder_codex_20261029 as encoder
from experiments.spectral_reachability import p3_matched_order_feedback_encoder_saved_codex_20261029 as encoder_saved
from experiments.spectral_reachability import agent_p3_c3_k17_physical_handoff_20261029_v1 as handoff
from experiments.spectral_reachability import agent_allmode_physical_20261028_v1 as physical

NAME = 'p3_matched_order_feedback_source_preflight_codex_20261029'
ROOT = host.ROOT
PREP = ROOT/'results/p3_matched_order_host_codex_20261029_prepare_MAIN03.json'
PREP_SHA = '85e393983b1437630bd74243afe17c6764aff9d8505c5dd9d44d0603bc0066a4'
REPORT = ROOT/'results/p3_matched_order_host_codex_20261029_run_MAIN01'
REPORT_SHA = '8b5d1f618143557bb8a31ef3b924681ed6b54d7dfd52681d5bd300bafd0dc560'
RECEIPT = ROOT/'results/p3_matched_order_host_codex_20261029_consume_MAIN01.json'
RECEIPT_SHA = '9da648ebf1c511bba53f06d73cfb4ce2e8fb075bb3ed6ff0f41af18177967bb0'
need = host.need


def build():
    p = host.verify_prepare(PREP, PREP_SHA)
    need(host.sha(RECEIPT) == RECEIPT_SHA, 'saved receipt hash')
    reconstructed = host.consume(REPORT, REPORT_SHA, PREP, PREP_SHA)
    need(reconstructed == json.loads(RECEIPT.read_bytes()), 'reconstructed original native IO receipt')
    source = physical.source_material(); roots = physical.root_metadata(source)
    need(len(roots) == len(p['tasks']) == 2, 'fixed known2')
    histories = []
    bindings = dict(source['bindings'])
    for i, task in enumerate(p['tasks']):
        need(task['binding']['root_sha256'] == roots[i]['root_sha256'], 'original physical root identity')
        path = ROOT/f'results/p1_constructive_moving_history_encode_cuda_20261027_v1_cuda_main01/task_{i:02}.json.gz'
        digest = host.sha(path)
        with gzip.open(path, 'rt', encoding='utf-8') as f:
            export = json.load(f)
        history = handoff.history_export(task, export, digest)
        raw = np.asarray(export['task']['root_raw_action_history'], dtype=np.float32)
        old = np.asarray(source['roots'][i]['packets']['off']['raw_action_history'], dtype=np.float32)
        need(raw.shape == old.shape == (10,2) and raw.tobytes() == old.tobytes(), 'physical source warm10 raw bits')
        histories.append(history); physical.merge(bindings, {str(path): digest})
    physical.merge(bindings, p['sources'])
    for path in (PREP, REPORT/'report.json', REPORT/'events.jsonl', REPORT/'binding.json', RECEIPT):
        physical.merge(bindings, {str(path): host.sha(path)})
    for module in (host,seam,encoder,encoder_saved,handoff,physical):
        path = Path(module.__file__).resolve(); physical.merge(bindings, {str(path): host.sha(path)})
    path = Path(__file__).resolve(); physical.merge(bindings, {str(path): host.sha(path)})
    for path,digest in bindings.items():
        need(Path(path).is_file() and not Path(path).is_symlink() and host.sha(path) == digest, 'source closure drift')
    return dict(schema=NAME+'.preflight', roots=roots, histories=histories,
                roster=[dict(slot=s,root=r,policy=policy) for s,r,policy in seam.ROSTER],
                report_sha256=REPORT_SHA,receipt_sha256=RECEIPT_SHA,prepare_sha256=PREP_SHA,
                source_report_sha256=physical.SOURCE_SHA, bindings=bindings,
                live_reset_verified=False, live_RGB_verified=False, physical_authorized=False,
                gpu_coordinated=False, scientific_pass=False,
                limitation='saved source reconstruction only; not a runtime or physical approval')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    args = parser.parse_args(); path = Path(args.output).absolute()
    need(path.parent == ROOT/'results' and path.name.startswith(NAME) and path.resolve() == path and
         not path.exists(), 'exclusive own preflight path')
    value = build(); body = host.core.canonical(value)
    need(len(body) <= 16*1024**2, 'preflight cap')
    with path.open('xb') as f:
        need(f.write(body) == len(body), 'complete preflight write'); f.flush(); os.fsync(f.fileno())
    host.signal.fsync_dir(path.parent)
    print(json.dumps(dict(path=str(path),sha256=host.sha(path))))


if __name__ == '__main__':
    main()
