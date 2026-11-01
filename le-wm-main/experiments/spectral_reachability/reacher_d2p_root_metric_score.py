"""Certified model-only common-gain rescoring; original stage/tail rules stay.

Pure arithmetic accepts CPU only for numeric unit fixtures. Live callers must
require host CUDA explicitly. No physical state/response/outcome is an input.
"""
from __future__ import annotations

import torch

from experiments.spectral_reachability import bounded_response_qp as qp
from experiments.spectral_reachability import reacher_d2p_masked_programs as compiler

RIDGE=.01
POLICY="model-root-response-metric-v1"


class RootMetricTensor:
    def __init__(self,response):
        if (response.ndim!=2 or min(response.shape)<1 or response.dtype!=torch.float64
                or not torch.isfinite(response).all()):
            raise ValueError("complete finite float64 model-root response required")
        self.u,self.s,_=torch.linalg.svd(response.detach(),full_matrices=False)
        self.correction=(RIDGE/(self.s.square()+RIDGE)).sqrt()-1
        self.device=response.device
        self.rows=response.shape[0]

    def apply(self,value):
        if (value.ndim not in (1,2) or value.shape[0]!=self.rows or value.device!=self.device
                or value.dtype!=torch.float64 or not torch.isfinite(value).all()):
            raise ValueError("aligned finite float64 values on the same device required")
        projected=self.u.T@value
        adjustment=self.correction*projected if value.ndim==1 else self.correction[:,None]*projected
        return value+self.u@adjustment


def model_rows(bank):
    """Project only model-owned fields; ignore snapshots/observations entirely."""
    if len(bank)!=17 or [b["index"] for b in bank]!=list(range(17)):
        raise ValueError("complete original ordered17 model programs required")
    rows=[{"index":b["index"],"prefix_cost":b["prefix_cost"],
        "terminal_cost":b["variants"]["cem_then_gn4"]["terminal_cost"],
        "common_response_gain":b["common_response_gain"],"response":b["response"]} for b in bank]
    compiler.select(rows)  # Validate unchanged stage/terminal/certificate inputs.
    return rows


def rescore(rows,common_residual,*,device):
    original=compiler.select(rows)
    tensor=lambda value:torch.as_tensor(value,dtype=torch.float64,device=device)
    root=tensor(rows[0]["response"])
    metric=RootMetricTensor(root)
    residual=tensor(common_residual)
    if residual.shape!=root.shape[:1]: raise ValueError("complete common model residual required")
    weighted_residual=metric.apply(residual)
    scored=[]
    for row in rows:
        response=tensor(row["response"])
        if response.shape!=root.shape: raise ValueError("all original full response shapes must agree")
        gain=qp.certified_response_gain(metric.apply(response),weighted_residual,damping=RIDGE)
        scored.append({"index":row["index"],"prefix_cost":row["prefix_cost"],
            "terminal_cost":row["terminal_cost"],"common_response_gain":gain})
    return {"bank":scored,"decision":compiler.select(scored),"original_decision":original,
        "metric":{"policy":POLICY,"ridge":RIDGE,"root_response_shape":list(root.shape),
            "root_response_singular_values":metric.s.tolist(),"device":str(root.device),
            "physical_labels_used":False,"trained_modules":False},
        "certified_qps":17,"new_predict_samples":0,"new_vjp_rows":0}
