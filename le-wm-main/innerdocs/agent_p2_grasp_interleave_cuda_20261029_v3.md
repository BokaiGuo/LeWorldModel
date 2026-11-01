# GRASP fixed-input CUDA comparison v3

This source-first future production diagnostic records each side's actual CUDA `torch.linspace(0,1,4,device=device)[1:-1][None,:,None]` fraction, including dtype, shape, stride and exact bytes, in the side artifact bound by the report SHA. The consumer requires both descriptors equal and independently reconstructs each side's initial `v` using float32 subtract, multiply, add in order and exact byte comparison. No MAIN01-specific coefficient or tolerance is used. Source SHA binding establishes the declared operation; saved bytes alone cannot independently prove the GPU executed it. All other v2 source, model, event, resource and frozen-weight guards remain in force.

This is a fixed-input diagnostic, not an independent gradient recomputation, whole-trajectory parity, equal-FLOPs claim, or scientific admission. Original and HOST module IO strides are not recorded. Partial failures remain fail-closed and retained. v2 and its results remain untouched.

MAIN commands from repository root (replace the protocol and its SHA with the actual approved authority; output names must not exist):

```bash
OPENBLAS_NUM_THREADS=1 .venv/bin/python -B -m experiments.spectral_reachability.agent_p2_grasp_interleave_cuda_20261029_v3 prepare --protocol PROTOCOL --protocol-sha PROTOCOL_SHA --output results/agent_p2_grasp_interleave_cuda_20261029_v3_prepare_MAIN01.json
OPENBLAS_NUM_THREADS=1 .venv/bin/python -B -m experiments.spectral_reachability.agent_p2_grasp_interleave_cuda_20261029_v3 verify --prepared results/agent_p2_grasp_interleave_cuda_20261029_v3_prepare_MAIN01.json --prepare-sha PREPARE_SHA
OPENBLAS_NUM_THREADS=1 .venv/bin/python -B -m experiments.spectral_reachability.agent_p2_grasp_interleave_cuda_20261029_v3 run --prepared results/agent_p2_grasp_interleave_cuda_20261029_v3_prepare_MAIN01.json --prepare-sha PREPARE_SHA --output results/agent_p2_grasp_interleave_cuda_20261029_v3_cuda_MAIN01 --authorization MAIN_P2_GRASP_INTERLEAVE_V3_CUDA_ONLY
OPENBLAS_NUM_THREADS=1 .venv/bin/python -B -m experiments.spectral_reachability.agent_p2_grasp_interleave_cuda_20261029_v3 consume --output results/agent_p2_grasp_interleave_cuda_20261029_v3_cuda_MAIN01 --report-sha REPORT_SHA --receipt results/agent_p2_grasp_interleave_cuda_20261029_v3_consume_MAIN01.json
```

No production operation is authorized by this document. The installed CUDA arithmetic must actually pass exact saved-byte reconstruction; a different implementation or fused operation fails closed rather than relaxing tolerance.
