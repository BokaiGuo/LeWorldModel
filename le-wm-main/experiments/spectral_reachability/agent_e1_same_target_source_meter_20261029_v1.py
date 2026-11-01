"""Saved-source join and actual-hook meter seam. No runtime dispatch entrypoint."""
from __future__ import annotations
import gzip
import hashlib
import io
import json
from pathlib import Path

from experiments.spectral_reachability import agent_e1_six_slot_common_prepare_20261029_v1 as prep
from experiments.spectral_reachability import agent_e1_same_target_candidate_intervention_20261029_v1 as intervention
from experiments.spectral_reachability import moving_prediction_matched_sampling_cuda_20261027_v1 as native

ROOT = Path(__file__).resolve().parents[2]
INTERVENTION_SHA = '31ad853db50dba2b47d88582d612a689ccf471406c9d06ed8ed67718b5878de9'
INTERVENTION_TEST_SHA = 'bd18f3ad931f06a53b3b6b8121ff2a6b942bcd0ba295c6f3bce7d00639611ec7'
PREP_SOURCE_SHA = 'aa9f05b6ecb8c0215161c776ca331b9eebdc57bdb2fc834b87e68f1c3c00f6db'
METER_SOURCE_SHA = '72195df2a7f1d10948c2bf5c494dd26e66400eaa2e269737332cbd876e599d40'
PHASE_CAPS = {'controller': 144, 'extra': 2, 'replay': 4}
RESOURCES = native.RESOURCES


def require(ok, message):
    if not ok:
        raise ValueError(message)


def inflate(data):
    with gzip.GzipFile(fileobj=io.BytesIO(data)) as f:
        out = f.read(prep.CAP + 1)
    require(len(out) <= prep.CAP, 'inflated JSON cap')
    return prep.parse(out)


def source_inputs(root=ROOT):
    """Authenticate pinned saved artifacts, inheriting the old full receipt.

    No old fullstrict, model loading, or native execution is performed.
    """
    for path, expected in ((root/'experiments/spectral_reachability/agent_e1_same_target_candidate_intervention_20261029_v1.py', INTERVENTION_SHA),
                           (root/'tests/test_agent_e1_same_target_candidate_intervention_20261029_v1.py', INTERVENTION_TEST_SHA),
                           (root/'experiments/spectral_reachability/agent_e1_six_slot_common_prepare_20261029_v1.py', PREP_SOURCE_SHA),
                           (root/'experiments/spectral_reachability/moving_prediction_matched_sampling_cuda_20261027_v1.py', METER_SOURCE_SHA)):
        prep.bytes_at(path, expected)
    names = ('controller_prepare', 'controller_report', 'controller_receipt')
    objects = {k: prep.parse(prep.bytes_at(root/prep.FILES[k][0], prep.FILES[k][1])) for k in names}
    cp, report, receipt = (objects[k] for k in names)
    require(report['prepare_sha256'] == prep.FILES['controller_prepare'][1] and report['errors'] == []
            and report['model_hash_before'] == report['model_hash_after'] == cp['expected_model_hash']
            and receipt['status'] == 'common_controller_calibration_verified'
            and receipt['rows'] == report['rows'] and len(receipt['plans']) == 4,
            'original prepare/report/full receipt join')
    rows = []
    for i in (0, 1):
        slot, compiled, selected = cp['slots'][i], cp['compiled'][i], receipt['plans'][i*2+1]
        require(slot['source_slot'] == compiled['source_slot'] == selected['source_slot'] == i
                and selected['arm'] == 'dynamics_compiled_witness'
                and selected['task_id'] == slot['task_id'] and selected['task_sha256'] == slot['task_sha256']
                and selected['blind_sha256'] == slot['blind_sha256'], 'root/task/plan identity')
        task = inflate(prep.bytes_at(slot['task_export'], slot['task_sha256']))
        require(task['source_slot'] == i and task['task_id'] == slot['task_id']
                and task['blind_sha256'] == slot['blind_sha256']
                and task['task']['task_contract_sha256'] == slot['task_contract_sha256'], 'task export binding')
        result = inflate(prep.bytes_at(compiled['result_path'], compiled['result_sha256']))
        require(result['source_slot'] == i and result['task_id'] == slot['task_id']
                and result['commands'] == compiled['raw10'] and result['selected_path'] == compiled['path']
                and result['whole_contexts'] == result['segmented_contexts']
                and result['whole_contexts'][0] == compiled['prefix_context']
                and compiled['target'] == compiled['prefix_context']['latent_history'][-1]
                == cp['targets'][i]['targets']['dynamics_compiled_witness'] == selected['target'],
                'compiled result/prefix/target/receipt exact join')
        raw = intervention.raw_actions(compiled['raw10'], 10)
        target = prep.vector_bytes(compiled['target'])
        rows.append(dict(root=i, task_id=slot['task_id'], task_sha256=slot['task_sha256'],
                         task_export=slot['task_export'], blind_sha256=slot['blind_sha256'],
                         result_path=compiled['result_path'], result_sha256=compiled['result_sha256'],
                         compiled_selection_sha256=compiled['selection_sha256'],
                         target_hex=target.hex(), target_sha256=hashlib.sha256(target).hexdigest(),
                         witness_raw5=raw[:5], witness_source_raw10_sha256=intervention.digest(raw)))
    return dict(rows=rows, parent_authorities={k: prep.FILES[k][1] for k in names},
                pinned_intervention_sha256=INTERVENTION_SHA,
                authority_boundary='inherited complete receipt; compiled result exact join; no fullstrict rerun',
                physical_reachability_authenticated=False, production_authorized=False)


def compare_banks(left, right):
    """Require literal canonical JSON byte equality, not approximate values."""
    for bank in (left, right):
        require(isinstance(bank, list) and len(bank) == 72, '72 bank')
        require(all(isinstance(row, dict) and row.get('index') == i for i, row in enumerate(bank)), 'bank index order')
    a, b = prep.canonical(left), prep.canonical(right)
    require(a == b, 'independent banks not byte-identical')
    return dict(canonical_bytes=len(a), sha256=hashlib.sha256(a).hexdigest(), exact=True)


class HookMeter(native.Meter):
    """Use existing actual module hooks with separate caps, no global patch."""
    def __init__(self, phase, sink, sync=lambda: None, check=lambda: None):
        require(phase in PHASE_CAPS, 'fixed phase')
        super().__init__(sink, sync, check)
        self.phase = phase
        self.limit = PHASE_CAPS[phase]

    def available(self):
        self.check()
        require(not self.fault and all(x['attempted'] < self.limit for x in self.counts().values()), 'phase predispatch cap')

    def begin(self, resource, inputs):
        self.check()
        require(not self.fault and resource in RESOURCES and self.counts()[resource]['attempted'] < self.limit,
                'resource predispatch cap')
        return super().begin(resource, inputs)


def phase_complete(meter):
    n = PHASE_CAPS[meter.phase]
    require(not meter.fault and meter.counts() == native.budget(n)
            and meter.predictor_samples() == {'attempted': n, 'completed': n}, 'actual hook phase census')
    return dict(phase=meter.phase, resources=meter.counts(), predictor_samples=meter.predictor_samples())
