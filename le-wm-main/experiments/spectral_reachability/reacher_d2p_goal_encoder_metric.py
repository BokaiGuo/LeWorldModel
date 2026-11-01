"""Frozen goal-encoder input co-metric, with no physical task labels.

Q = .01 * (J_E J_E.T + .01 I)^-1, J_E evaluated at the processed goal RGB.
The normalized-pixel tangent metric is an explicit candidate, not a claim that
pixel distance equals joint distance. Same goal => same Q across actuator masks.
"""
from __future__ import annotations

import torch

RIDGE=.01
POLICY="goal-encoder-input-cometric-v1"


class GoalEncoderMetric:
    def __init__(self,gram):
        if (gram.ndim!=2 or gram.shape[0]!=gram.shape[1] or gram.shape[0]<1
                or gram.dtype!=torch.float64 or not torch.isfinite(gram).all()
                or not torch.allclose(gram,gram.T,rtol=1e-12,atol=1e-12)):
            raise ValueError("finite symmetric float64 full goal-encoder Gram required")
        values,vectors=torch.linalg.eigh(gram)
        if float(values.min()) < -1e-10*max(1.,float(values.abs().max())):
            raise ValueError("positive semidefinite goal-encoder Gram required")
        self.eigenvalues=values.clamp_min(0.)
        self.sqrt_matrix=(vectors*torch.sqrt(RIDGE/(self.eigenvalues+RIDGE))[None])@vectors.T
        self.rows=gram.shape[0]
        self.device=gram.device

    def apply(self,value):
        if (value.ndim not in (1,2) or value.shape[0]!=self.rows or value.dtype!=torch.float64
                or value.device!=self.device or not torch.isfinite(value).all()):
            raise ValueError("finite aligned float64 values on metric device required")
        return self.sqrt_matrix@value
