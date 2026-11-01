"""Join externally pinned MAIN control5 receipt, report, prepare and row.

This checks a saved authority join; it does not repeat the producer's strict
consumer, certify the following episode, or run a model.
"""
from pathlib import Path
import numpy as np
from experiments.spectral_reachability import agent_e3_gd_host_cuda_20261029_v1 as host

need = host.need


def join_objects(prepared, report, completion, receipt, row, cell):
    need(type(cell) is int and 0 <= cell < 16, 'fixed cell')
    need(prepared['schema'] == host.NAME+'.prepare' and prepared['fixed_cells'] == 16
         and len(prepared['bodies']) == 16, 'fixed original preparation')
    body = prepared['bodies'][cell]
    host.numerical.validate_body(body)
    need(report['schema'] == host.NAME+'.report' and report['cell'] == cell and report['boundary'] == 5
         and report['body_sha256'] == host.digest(body) and report['errors'] == []
         and report['sources_verified'] is True, 'control5 report/body/source')
    need(completion['schema'] == host.NAME+'.completion' and completion['cell'] == cell
         and completion['complete'] is True and completion['boundary'] == 'after_report_fsync_hash'
         and type(completion['elapsed_seconds']) in (int,float)
         and np.isfinite(completion['elapsed_seconds']) and 0 <= completion['elapsed_seconds'] <= host.WALL,
         'complete bounded control5 publication')
    need(receipt['schema'] == host.NAME+'.consume' and receipt['cell'] == cell
         and receipt['status'] == 'control5_cell_native_GD_verified_NOT_admitted', 'MAIN strict receipt')
    need(receipt['report_sha256'] == completion['report_sha256']
         and receipt['axes'] == report['axes'] and receipt['plan'] == row['result'], 'receipt/report/row join')
    need(report['model_before'] == report['model_after'] == prepared['expected_model_hash']
         and report['model_before'] == host.numerical.resolver.MODEL_SHA, 'original frozen model')
    need(report['precision_before'] == report['precision_after'] == host.primitives.precision_expected(),
         'original precision')
    classes = report['classes_before']
    need(type(classes) is list and bool(classes) and classes == report['classes_after'], 'loaded classes stable')
    names = set()
    for item in classes:
        need(type(item) is dict and set(item) == {'name','class_module','class_qualname','path','sha256'}
             and type(item['name']) is str and item['name'] not in names
             and all(type(item[k]) is str and item[k] for k in ('class_module','class_qualname','path','sha256'))
             and prepared['bindings_sha256'].get(item['path']) == item['sha256'], 'class source binding')
        names.add(item['name'])
    for key in host.FLAGS:
        need(report[key] is False and receipt[key] is False, 'all scientific flags false')
    plan = receipt['plan']
    a = np.asarray(plan['first5'], dtype=np.float32)
    need(a.shape == (5,2) and np.isfinite(a).all() and (np.abs(a)<=1).all(), 'fixed first5 raw action')
    commands = np.asarray(plan['commands'], dtype=np.float32)
    need(commands.shape == (5*body['horizon_packets'],2) and commands[:5].tobytes() == a.tobytes(),
         'first5 exact selected literal commands')
    return dict(cell=cell, body=body, first5=a, full_upstream_rebuilt_here=False, scientific_admission=False)


def load(prepared_path, prepared_sha, report_path, report_sha, completion_path, completion_sha,
         receipt_path, receipt_sha, cell):
    prepared = host.read(prepared_path, prepared_sha)
    report = host.read(report_path, report_sha)
    completion = host.read(completion_path, completion_sha)
    receipt = host.read(receipt_path, receipt_sha)
    need(report['prepared'] == str(Path(prepared_path)) and report['prepare_sha256'] == prepared_sha
         and completion['report_sha256'] == report_sha and receipt['report_sha256'] == report_sha
         and receipt['completion_sha256'] == completion_sha, 'explicit external SHA and path joins')
    out = Path(report_path).parent
    need(Path(report_path).name == 'report.json' and Path(completion_path) == out/'completion.json',
         'original report and completion same directory')
    row = host.read(out/'row.json', report['artifacts']['row.json'])
    host.source.rehash(prepared['bindings_sha256'])
    return join_objects(prepared, report, completion, receipt, row, cell)
