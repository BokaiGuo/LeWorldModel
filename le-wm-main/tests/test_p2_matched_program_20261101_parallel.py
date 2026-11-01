from contextlib import contextmanager
from copy import deepcopy

import numpy as np
import pytest

from experiments.spectral_reachability import p2_matched_program_kernel_20261101_parallel as kernel
from experiments.spectral_reachability import p2_matched_program_saved_20261101_parallel as saved


@contextmanager
def model_scope(_):
    yield


class FixtureAdapter:
    """Accounting fixture only; never CUDA or scientific evidence."""

    def __init__(self, meter):
        self.meter = meter

    def packet(self, fn):
        def nested():
            for axis in ("action_encoder", "predict", "predictor", "pred_proj"):
                self.meter.call(axis, lambda: None)
            return fn()

        return self.meter.call("packet", nested)

    def root_jacobian(self, _context):
        def work():
            self.packet(lambda: None)
            self.packet(lambda: None)
            return np.vstack(
                (np.diag(np.arange(20, 0, -1)), np.zeros((172, 20)))
            )

        return self.meter.call("jacobian", work)

    def packet_jacobian(self, _context):
        def work():
            self.packet(lambda: None)
            return np.vstack(
                (np.diag(np.arange(10, 0, -1)), np.zeros((182, 10)))
            )

        return self.meter.call("jacobian", work)

    def full_jacobian_at(self, _context, _anchor):
        def work():
            self.packet(lambda: None)
            self.packet(lambda: None)
            return np.vstack(
                (np.diag(np.arange(20, 0, -1)), np.zeros((172, 20)))
            )

        return self.meter.call("jacobian", work)

    def step(self, context, packet):
        def work():
            result = deepcopy(context)
            result["latent_history"][-1][0] += float(np.sum(packet)) / 100
            return result

        return self.packet(work)


def case():
    return dict(
        base=0,
        context={"latent_history": [[0.0] * 192], "action_history": []},
        goal=[0.0] * 192,
    )


def run(arm):
    events = []
    branch = kernel.run_branch(
        case(), arm, FixtureAdapter, events.append, model_scope
    )
    return branch, events


def test_exact_measured_resource_vector_for_both_arms():
    rows = [run(arm)[0] for arm in kernel.ARMS]
    assert all(row["resource_counts"]["attempted"] == kernel.CAPS for row in rows)
    assert rows[0]["resource_counts"]["attempted"] == rows[1]["resource_counts"]["attempted"]
    assert all(len(row["candidates"]) == 16 for row in rows)
    assert kernel.contract()["measured_resource_match"] is True
    assert kernel.contract()["equal_total_compute"] is False


@pytest.mark.parametrize("arm", kernel.ARMS)
def test_saved_branch_recomputes_structure_and_resources(arm):
    branch, events = run(arm)
    checked = saved.validate_branch(case(), branch, events)
    assert checked["counts"]["attempted"] == kernel.CAPS
    assert checked["P2_admission"] is False


def test_composition_candidate_prefix_is_replayed_sixteen_times():
    branch, events = run("prefix_chain")
    scoped = [row["event"] for row in events]
    candidate_prefixes = [
        event for event in scoped if event["kind"] == "candidate_prefix_return"
    ]
    assert len(candidate_prefixes) == 16
    assert all(
        row["candidate_prefix_context_sha256"] == row["prefix_context_sha256"]
        for row in branch["candidates"]
    )


def test_saved_consumer_rejects_resealed_candidate_prefix_tamper():
    branch, events = run("prefix_chain")
    damaged = deepcopy(branch)
    damaged["candidates"][0]["candidate_prefix_context_sha256"] = "0" * 64
    altered = deepcopy(events)
    altered[-1]["event"]["result_sha256"] = kernel.old.digest(damaged)
    with pytest.raises(ValueError):
        saved.validate_branch(case(), damaged, altered)


def test_host_material_binds_parallel_sources():
    from experiments.spectral_reachability import p2_matched_program_host_20261101_parallel as host

    value = host.material()
    names = {row["path"].split("/")[-1] for row in value["own_sources"]}
    assert {
        "p2_matched_program_kernel_20261101_parallel.py",
        "p2_matched_program_host_20261101_parallel.py",
        "p2_matched_program_saved_20261101_parallel.py",
        "test_p2_matched_program_20261101_parallel.py",
    }.issubset(names)
    assert value["protocol"]["P2_admission"] is False
