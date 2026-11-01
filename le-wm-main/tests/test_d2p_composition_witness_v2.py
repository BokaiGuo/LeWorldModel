"""Synthetic contract checks only; no model or physical efficacy evidence."""
from copy import deepcopy
import pytest
from experiments.spectral_reachability import d2p_composition_witness_v2 as w


def fixture():
    sha = "a" * 64
    provenance = {k: sha for k in w.PROVENANCE_KEYS}
    domain = dict(name="raw", units="position", low=[-2.], high=[2.],
                  action_block=1, normalization_sha256=sha)
    root = dict(root_id="r1", root_rgb_sha256=sha, goal_rgb_sha256=sha,
                context={"layout": "states_with_between_state_actions_v1",
                         "context_length": 2, "latent_dim": 1, "action_dim": 1,
                         "latent_history": [[0.], [0.]], "action_history": [[0.]]})
    def replay(context, actions):
        history = context["latent_history"] + [[context["latent_history"][-1][0] + sum(a[0] for a in actions)]]
        return dict(context, context_length=len(history), latent_history=history,
                    action_history=context["action_history"] + actions)
    edges = []
    context = root["context"]
    for i in range(3):
        post = replay(context, [[.25]])
        edges.append(dict(program_id=str(i), kind="progress", root_id="r1",
                          provenance=deepcopy(provenance), action_domain=deepcopy(domain),
                          input_context=deepcopy(context), actions=[[.25]], output_context=post))
        context = post
    return dict(root=root, provenance=provenance, action_domain=domain, edges=edges,
                replay=replay, max_edges=3, max_actions=3)


def test_three_edges_exact_concatenation_and_non_admission():
    args = fixture()
    receipt = w.compose(**args)
    assert receipt["actions"] == [[.25]] * 3
    assert receipt["P2_admission"] is False
    assert receipt["physical_efficacy_verified"] is False
    assert len(receipt["transitions"]) == 3
    assert receipt == w.compose(**deepcopy(args))


@pytest.mark.parametrize("mutation", [
    lambda a: a["edges"][1].update(root_id="other"),
    lambda a: a["edges"][1]["provenance"].update(checkpoint_sha256="b"*64),
    lambda a: a["edges"][1]["action_domain"].update(units="velocity"),
    lambda a: a["edges"][1]["input_context"]["action_history"].append([0.]),
    lambda a: a["edges"][1]["output_context"]["latent_history"][-1].__setitem__(0, 1.5),
    lambda a: a["edges"][0].update(actions=[[9.]]),
    lambda a: a["edges"][0].update(actions=[[float("nan")]]),
    lambda a: a["edges"][0].update(actions=[]),
    lambda a: a["edges"][0].update(kind="latent_waypoint"),
    lambda a: a.update(max_actions=2),
    lambda a: a.update(max_edges=2),
    lambda a: a.update(edges=[]),
    lambda a: a["root"].update(root_rgb_sha256=""),
    lambda a: a["action_domain"].update(action_block=0),
    lambda a: a["edges"][1].update(program_id="0"),
    lambda a: a["edges"][0].update(actions=[[True]]),
    lambda a: a["edges"][0].update(actions=[[.25, .1]]),
    lambda a: a["provenance"].pop("metric_sha256"),
    lambda a: a["action_domain"].update(name="atanh_params"),
    lambda a: a["action_domain"].update(action_block=2),
    lambda a: a["root"]["context"].update(latent_history=["arbitrary waypoint"]),
    lambda a: a["root"]["context"].update(latent_history=[[True], [0.]]),
    lambda a: a["root"]["context"].update(latent_history=[[0.], [0., 1.]]),
    lambda a: a["root"]["context"].update(latent_history=[[{}], [0.]]),
    lambda a: a["root"]["context"].update(action_history=[]),
    lambda a: a["root"]["context"].update(action_history=[[float("nan")]]),
    lambda a: a["root"]["context"].update(action_history=[["0"]]),
    lambda a: a["root"]["context"].update(action_history=[[False]]),
    lambda a: a["root"]["context"].update(action_history=[[0., 1.]]),
    lambda a: a["root"]["context"].update(context_length=3),
    lambda a: a["root"]["context"].update(action_dim=True),
    lambda a: a["root"]["context"].pop("layout"),
])
def test_fail_closed(mutation):
    args = fixture()
    mutation(args)
    with pytest.raises(ValueError):
        w.compose(**args)


def test_inputs_not_mutated_and_receipt_not_aliased():
    args = fixture()
    original = deepcopy(args)
    receipt = w.compose(**args)
    receipt["actions"][0][0] = 99
    assert args == original


def test_replay_cannot_mutate_claim_to_make_it_pass():
    args = fixture()
    def malicious(context, actions):
        context.clear()
        return {}
    args["replay"] = malicious
    with pytest.raises(ValueError):
        w.compose(**args)


def test_receipt_hash_covers_content_and_action_order():
    receipt = w.compose(**fixture())
    sha = receipt.pop("receipt_sha256")
    assert sha == w.digest(receipt)
    receipt["actions"].reverse()
    receipt["actions"][0][0] = -.25
    assert sha != w.digest(receipt)


@pytest.mark.parametrize("field,value", [
    ("latent_history", ["waypoint"]), ("latent_history", [[True], [0.]]),
    ("latent_history", [[0.], [0., 1.]]), ("latent_history", [[{}], [0.]]),
    ("action_history", []), ("action_history", [["0"]]),
    ("action_history", [[False]]), ("action_history", [[float("nan")]]),
    ("action_history", [[0., 1.]]), ("context_length", 3),
    ("action_dim", True), ("layout", "unknown"),
])
def test_context_validation_itself_rejects_invalid_layout(field, value):
    context = fixture()["root"]["context"]
    context[field] = value
    with pytest.raises(ValueError):
        w._context(context)
