"""Versioned image-shaped RGB boundary for the declared LeWM packet contract."""
import torch

from experiments.pc_input_permissions import RGBLeWM


class RGBImageLeWM(RGBLeWM):
    """Reject mislabelled vector inputs before the frozen model is called.

    This validates tensor structure, not pixel provenance or caller trust.
    """

    def __init__(self, model, audit, *, image_size=224, packet_width=10):
        if (type(image_size) is not int or image_size < 8 or
                type(packet_width) is not int or packet_width < 1):
            raise ValueError("explicit image and packet dimensions required")
        super().__init__(model, audit)
        self.image_size = image_size
        self.packet_width = packet_width

    def _image(self, image, *, rank):
        if (not isinstance(image, torch.Tensor) or image.ndim != rank
                or not torch.is_floating_point(image)
                or tuple(image.shape[-3:]) != (3, self.image_size, self.image_size)
                or any(size < 1 for size in image.shape)):
            raise ValueError("expected nonempty floating-point RGB image tensor")

    def encode(self, inputs):
        if set(inputs) != {"pixels"}:
            raise ValueError("RGB encoder accepts pixels only")
        self._image(inputs["pixels"], rank=5)
        return super().encode(inputs)

    def score(self, inputs, actions):
        if set(inputs) != {"pixels", "goal", "action"}:
            raise ValueError("RGB scoring requires pixels, goal and action history only")
        pixels, goal, history = inputs["pixels"], inputs["goal"], inputs["action"]
        self._image(pixels, rank=6)
        self._image(goal, rank=6)
        if (not isinstance(actions, torch.Tensor) or actions.ndim != 4
                or not torch.is_floating_point(actions)
                or any(size < 1 for size in actions.shape)
                or actions.shape[-1] != self.packet_width):
            raise ValueError("invalid candidate packet tensor")
        batch, _, horizon, _ = actions.shape
        if (pixels.shape[0] != batch or goal.shape[0] != batch
                or pixels.shape[1] != 1 or goal.shape[1] != 1
                or goal.shape[2] != 1 or pixels.shape[2] > horizon):
            raise ValueError("incompatible image batch, sample or horizon")
        if (not isinstance(history, torch.Tensor) or history.ndim != 4
                or not torch.is_floating_point(history)
                or history.shape[:2] != (batch, 1)
                or history.shape[-1] != self.packet_width
                or history.shape[2] not in (0, pixels.shape[2])):
            raise ValueError("invalid explicit action history")
        return super().score(inputs, actions)
