"""Single-observation LeWM scoring session; never cache action-dependent outputs."""
import torch

from experiments.pc_input_permissions import RGBLeWM, tensor_receipt


class RGBObservationScoringSession(RGBLeWM):
    def __init__(self, model, audit):
        super().__init__(model, audit)
        self._context_key = None
        self._emb = None
        self._goal_emb = None
        self.cache_hits = 0
        self.cache_misses = 0

    def score(self, inputs, actions):
        if torch.is_grad_enabled():
            raise ValueError("observation cache requires inference without gradients")
        if set(inputs) != {"pixels", "goal", "action"}:
            raise ValueError("RGB observations and legal history required")
        identity = {k: tensor_receipt(v) for k, v in inputs.items()}
        if self._context_key is not None and identity != self._context_key:
            raise ValueError("observation or history changed: create a new session")
        self._record("score", {**inputs, "candidate_actions": actions})
        context = dict(inputs)
        if self._context_key is not None:
            if actions.ndim != 4 or actions.shape[0] != self._emb.shape[0]:
                raise ValueError("incompatible candidate batch")
            context["emb"] = self._emb.expand(-1, actions.shape[1], -1, -1)
            context["goal_emb"] = self._goal_emb
            self.cache_hits += 1
        cost = self._model.get_cost(context, actions)
        if self._context_key is None:
            emb, goal = context["emb"], context["goal_emb"]
            if emb.ndim != 4 or emb.shape[1] != actions.shape[1] or goal.ndim != 3:
                raise ValueError("unexpected upstream cached embedding shape")
            self._emb = emb[:, :1].detach().clone()
            self._goal_emb = goal.detach().clone()
            self._context_key = identity
            self.cache_misses += 1
        return cost, context
