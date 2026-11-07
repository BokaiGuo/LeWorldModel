import torch

from experiments.pc_rgb_action_clock_internal_20261107_codex import manual_two_packets


def test_manual_two_packets_uses_actions_in_order():
    class Model:
        def encode(self, inputs):
            return {"emb": inputs["pixels"]}

        def action_encoder(self, actions):
            return actions

        def predict(self, embeddings, actions):
            return embeddings + actions

    pixels = torch.tensor([[[[10.]]]])
    actions = torch.tensor([[[[1.], [2.]]]])
    result = manual_two_packets(Model(), pixels, actions)
    torch.testing.assert_close(result, torch.tensor([[[10.], [11.], [13.]]]))
