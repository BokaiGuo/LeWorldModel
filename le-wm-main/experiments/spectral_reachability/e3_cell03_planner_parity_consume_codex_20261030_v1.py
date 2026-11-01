"""Independent saved numerical reconstruction of cell03 planner parity."""
import hashlib
import json
import os

import numpy as np
from experiments.spectral_reachability import e3_cell03_planner_parity_codex_20261030_v1 as p
from experiments.spectral_reachability import agent_e3_multinode_saved_20261029_v1 as saved

MANIFEST_SHA = 'a076dc51dbcd8eba9fd505b51943c76a4a62d85d843797109e713bad696cf215'
REPORT_SHA = '88985c768e8974a7e5e4ccbeff199966edb3006311dc3dda7dcbc8c4fdcf44cc'
OUTPUT = p.ROOT/'e3_cell03_planner_parity_consume_codex_20261030_v1.json'


def read(path, digest):
    data = path.read_bytes()
    p.need(hashlib.sha256(data).hexdigest() == digest, 'pinned SHA mismatch')
    return json.loads(data)


def consume():
    manifest = read(p.MANIFEST, MANIFEST_SHA)
    report = read(p.OUTPUT/'report.json', REPORT_SHA)
    p.need(report['status'] == 'completed' and report['manifest_sha256'] == MANIFEST_SHA, 'runner status')
    result = saved.consume([x['body'] for x in manifest['rows']], p.OUTPUT/'numeric',
                           report['numeric_journal_sha256'])
    p.need(len(result['nodes']) == 2 and len(report['rows']) == len(manifest['rows']) == 2,
           'two node denominator')
    rows = []
    for node, original, actual in zip(result['nodes'], manifest['rows'], report['rows'], strict=True):
        bits = np.asarray(node['result']['first5'], np.float32).tobytes().hex()
        p.need(node['boundary'] == original['boundary'] == actual['boundary'] and
               node['row_sha256'] == actual['row_sha256'] and
               node['result'] == original['original_result'] and
               bits == original['original_action_bits'] == actual['action_bits'] and
               actual['action_exact'] is True and actual['result_exact'] is True and
               node['axes']['attempted'] == node['axes']['completed'], 'node mismatch')
        rows.append(dict(boundary=node['boundary'], result_exact=True, action_exact=True,
                         row_sha256=node['row_sha256'], axes=node['axes']))
    return dict(manifest_sha256=MANIFEST_SHA, report_sha256=REPORT_SHA, rows=rows,
                source_sha256=p.base.sha(__file__), full_saved_strict=False,
                scientific_admission=False, scope='separate CUDA node recomputation, not live closed-loop parity')


if __name__ == '__main__':
    data = p.base.raw(consume())
    p.base.write_new(OUTPUT, data)
    p.base.syncdir(OUTPUT.parent)
    print(hashlib.sha256(data).hexdigest())
