import numpy as np
import torch

from experiments import pc_pointmaze_input_contract_audit_20261106_codex as audit


def test_wrapper_raw_state_is_normalized_exactly_once(monkeypatch):
    monkeypatch.setattr(audit, "rgb_tensor", lambda image: torch.tensor(image))
    monkeypatch.setattr(torch.Tensor, "cuda", lambda self: self)
    original_tensor = torch.tensor

    def cpu_tensor(*args, **kwargs):
        kwargs.pop("device", None)
        return original_tensor(*args, **kwargs)

    monkeypatch.setattr(torch, "tensor", cpu_tensor)

    class Dataset:
        proprio_mean = torch.tensor([1., 2.])
        proprio_std = torch.tensor([2., 4.])

    class Model:
        def encode_proprio(self, x):
            return x * 2

        def encode(self, value):
            if isinstance(value, dict):
                prop = (value["proprio"] - Dataset.proprio_mean) / Dataset.proprio_std
                return {"visual": value["visual"], "proprio": self.encode_proprio(prop)}
            return value

    result = audit.audit_one(Model(), Dataset(), np.ones((2,)), np.array([3., 6.]))
    assert result["max_proprio_single_normalization_error"] == 0
    assert result["max_visual_modal_path_error"] == 0
    assert result["double_normalization_proprio_difference"] > 0
