import torch

from experiments.spectral_reachability import preflight_pusht_d2p_p1_reconfiguration as p1


class _Model(torch.nn.Module):
    def action_encoder(self, actions):
        return actions

    def predict(self, latent, actions):
        # Preserve batch/history layout while creating an action-dependent
        # successor.  This is a layout contract test, not an efficacy proxy.
        effect = actions[..., :2].mean(dim=-1, keepdim=True)
        return latent + effect


def test_batched_live_action_packets_preserve_each_time_step():
    params = torch.tensor([[[0.0, 0.5], [-0.5, 1.0]], [[0.25, -0.25], [0.75, -0.75]]])

    packets = p1._batched_live_action_tokens(params, action_block=5)

    assert packets.shape == (2, 2, 10)
    expected = (torch.tanh(params) / 100.0 - torch.as_tensor(p1.p0.PUBLIC_ACTION_MEAN)) / torch.as_tensor(p1.p0.PUBLIC_ACTION_STD)
    assert torch.allclose(packets, expected.repeat(1, 1, 5))
    assert torch.allclose(packets[..., 0::2], expected[..., :1].expand(-1, -1, 5))
    assert torch.allclose(packets[..., 1::2], expected[..., 1:].expand(-1, -1, 5))


def test_advance_accepts_a_batched_prefix_and_keeps_causal_packet_shape():
    model = _Model()
    emb = torch.zeros((1, 3, 4))
    actions = torch.zeros((1, 3, 10))
    prefix = torch.zeros((2, 1, 2))

    successor, successor_actions = p1._advance(model, emb, actions, prefix)

    assert successor.shape == (2, 3, 4)
    assert successor_actions.shape == (2, 3, 10)


def test_advance_matches_rollout_action_timing_with_a_bootstrap_transition():
    model = _Model()
    emb = torch.zeros((1, 3, 4))
    actions = torch.zeros((1, 3, 10))
    prefix = torch.ones((1, 1, 2))

    successor, _ = p1._advance(model, emb, actions, prefix)

    # The mock adds the mean of the first action pair.  A rollout-consistent
    # prefix must produce the history-only bootstrap followed by the prefix
    # transition, rather than applying the prefix to the bootstrap itself.
    packet = p1._batched_live_action_tokens(prefix, action_block=5)
    expected_last = packet[..., :2].mean(dim=-1, keepdim=True).expand(-1, -1, 4)
    assert torch.allclose(successor[:, -1:], expected_last)


def test_frozen_projection_is_deterministic_and_root_shared():
    first = p1._frozen_projection(latent_dim=12, sketch=8, device=torch.device("cpu"), seed=7)
    second = p1._frozen_projection(latent_dim=12, sketch=8, device=torch.device("cpu"), seed=7)
    distinct = p1._frozen_projection(latent_dim=12, sketch=8, device=torch.device("cpu"), seed=8)

    assert first.shape == (8, 12)
    assert torch.allclose(first, second)
    assert not torch.allclose(first, distinct)


def test_p1_receipt_schema_requires_self_contained_replay_binding():
    assert p1.SCHEMA.endswith(".v3")
