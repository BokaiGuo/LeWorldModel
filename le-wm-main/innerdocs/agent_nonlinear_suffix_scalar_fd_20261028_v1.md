# Scalar raw-gradient 独立有限差分（固定 v1）

这是新增 scalar 路径的独立数值实验准备，不是原 P1 gate、selector 或修复。仅新增本 stem 的 source/tests/本文；不改任何旧封存。使用 addy-test-driven-development：先有缺失模块的 RED（collection ImportError），再实现并跑纯 NumPy/native-schema RAM 测试。没有执行神经 CPU/GPU、encoder、物理、训练或生产 prepare/run。

## 原始 authority 与固定点

所有路径相对 `/home/gbk/code/LeWorldModel/le-wm-main`：

- 原 source：`experiments/spectral_reachability/agent_p1_nonlinear_suffix_library_20261028_v2.py`，SHA `0a3e475af9271a0769a3829b4e714fa4e2d6f00eaed38ec8c3ea5d7b9a163fbc`。
- 原 CUDA `results/agent_p1_nonlinear_suffix_library_20261028_v2_cuda_MAIN01/report.json`，SHA `f91c44779de94e4abb87f4250b0db9b12f9e8cbe4f980628c031cb78301882de`。
- 原 full strict `results/agent_p1_nonlinear_suffix_library_20261028_v2_consume_MAIN01.json`，SHA `337330eda5e8f191deec615422cae53e0c260364f63db0cd14d58218033ff019`。
- 原 prepare `results/agent_p1_nonlinear_suffix_library_20261028_v2_prepare_MAIN01.json`，SHA `3da0299d73a865621f60b7bbd804a0f757044455dad30e562986635c1093eb4d`。

prepare 重新执行真实 `src.consume`，整份 canonical 比较已封存 strict，而非只检查 complete/status；其内包含原 full strict/source/ledger/actual arrays 校验。继承原 bindings，加入原全部 artifact、report/strict/prepare 和本三件套 SHA，复核八个 installed model-class source。无需新认证框架。

每 panel 仅取 `slot.original_generation_winner.index` 对应 candidate 的 `iterates[0].commands`、`gradients[0].raw_gradient`、`result.mid`、原 goal/normalizer/task。绝不取 suffix 新最终 winner/迭代最佳点。再单独核 `f002/gradient_1` 的真实 raw leaf、返回 gradient、callback、输入 full3 context 与 scalar loss。固定 root/condition/task 继承原严格绑定。

本 worker 已执行只读 actual-array extraction（未做完整生产 prepare）：四个原 prefix index 为 `[22,22,23,23]`，每点60请求合法，八个 installed sources 相符；该进程没有导入 torch。完整重建仍须 MAIN 实际 prepare/verify，不能将这次局部读取冒充原 full strict 的新执行。

## 数值实验与完整分母

按 panel→h→coordinate→sign 排序，h 固定 `[.01,.001,.0001]`，coordinate 为 raw5 的 C-order 十维，sign 为 `[-1,+1]`。扰动计算是 `float32(float64(saved_raw_coordinate) + sign*h)`，其余 raw float32 bits 原样保留（含 signed zero）。不经 tanh、投影或 clamp；不使用 logit gradient/chain。

固定240有符号请求 / 120中央差分对。域外或 float32 折叠明确 missing；不丢分母、不改 h、不补点。所有 pair 保留。缺任一侧则该 pair 无 FD；十维缺任一对则整个 h 的向量 L2/Linf/relative/cosine 为 null，禁止在幸存子集上汇总。零梯度范数的 relative/cosine 为 null。没有 best-h 选择或通过阈值。

每次合法请求仅调用原 v2 `Backend.forward` 一次，使用原完整三 latent/两 history、原 mean/std 输入归一化、action encoder→predict→predictor→pred_proj 及原 full3 返回 ABI。只进行 forward；代码不调用 `Backend.gradient` 或任何 AD API、不 patch 全局。固定冻结模型、eval、CUDA float32、TF32/autocast 关闭、SDP MATH；无 encoder、新图像或物理输入。

直接在该次 forward 的 CUDA 终态 tensor 上计算原 scalar 路径 `((y-goal)**2).mean()`。保存 endpoint、goal、residual、square、loss 的原生 float32 blob，不以 NumPy Q 或原选择器 float64 Q 冒充 native Q。主值为 `(float64(Qplus32)-float64(Qminus32))/(2*h)`；另列 actual float32 spacing 及以该间距作分母的 secant。比较已存 **raw** scalar gradient，报告逐坐标 signed/absolute error 和全向量误差；不是 logit derivative。

saved consume 独立精确核 endpoint/goal/residual/square bits，并分列 float64 endpoint Q、NumPy float32 mean、native reduction 差。GPU reduction 不保证与 NumPy mean bitwise 相同；仅使用非经验的保守舍入界 `gamma_193*mean(saved_squares)+193*min_subnormal`，`gamma_193=193u/(1-193u), u=2^-24`，做算术完整性检查，不据此签发 AD 正确或放宽 FD/science gate。该界基于192项非负 float32 求和与除法，含保守次正规绝对误差项；非有限/溢出拒绝。保存后不能重放 GPU reduction bits，字段明确 `bitwise_GPU_mean_replayed=false`。

## 资源和中断

预先固定最多240 predict API/240 samples/720 context tokens；action_encoder/predictor 各240 calls/240 samples/720 tokens；pred_proj240 calls/720 rows。grad/VJP/raw_backward/jacobian/encoder/physics 全部无新增请求。合法请求才支付 forward，域外不填充；本次固定锚点局部核查预期恰240，但仍以 runtime attempted/completed 真实 ledger 为准。FLOPs、CUDA kernel 时间、底层 backward-engine 数保持 unknown/null，不以 API 数证明 FLOPs 等价。

输出 cap128MiB，final reserve16MiB，disk floor200MiB，whole-stage wall1800s（含 verify），CUDA allocated3GiB/reserved4GiB。模型加载前后比较原 state_dict 与 all_buffers；运行前后再完整 source/file SHA。HOST CUDA-only/offline；不下载，不训练。

每 panel 有原 native journal 和独立 probe journal，事件 hashchain/fsync。先持久化 probe intent，再 forward，原生 nested resource attempt/return、模块输入输出立即保存，接着保存 scalar arrays/return。每点60槽运行前全部初始化；失败不重试，finally 保存该点原账本/partial row，剩余 panel not_started 保留；最终 partial report 不签完整。硬 kill 无最终 report 时，已 fsync 的 native/probe journal 仍保留，但不允许签发 strict receipt。工程 cap 是边界 watchdog，不宣称可中断单个挂死 CUDA kernel。

完整 consume 检查准确四点顺序、全部 artifact census/SHA、原 full strict、原模型/source、所有 native 事件白名单和 chronology/module wiring、所有 probe intent/return/missing 序列、原 Q buffers、各 h 全分母。partial 仅保存资源处置、不生成有效 FD 科学结论。输出所有 admission/certification 均 false。

## MAIN 可执行命令

在 `/home/gbk/code/LeWorldModel/le-wm-main`；每个输出须不存在。prepare/consume 为 saved-only；只有 run 会加载模型并进行 CUDA240次上限 forward，worker 未运行。SHA 占位符取前一步 CLI 的实际返回，不手填假值。

```bash
.venv/bin/python -m pytest -q tests/test_agent_nonlinear_suffix_scalar_fd_20261028_v1.py
.venv/bin/python -m experiments.spectral_reachability.agent_nonlinear_suffix_scalar_fd_20261028_v1 prepare --output results/agent_nonlinear_suffix_scalar_fd_20261028_v1_prepare_MAIN01.json
.venv/bin/python -m experiments.spectral_reachability.agent_nonlinear_suffix_scalar_fd_20261028_v1 verify --prepared results/agent_nonlinear_suffix_scalar_fd_20261028_v1_prepare_MAIN01.json --prepare-sha PREPARE_SHA
.venv/bin/python -m experiments.spectral_reachability.agent_nonlinear_suffix_scalar_fd_20261028_v1 run --prepared results/agent_nonlinear_suffix_scalar_fd_20261028_v1_prepare_MAIN01.json --prepare-sha PREPARE_SHA --output results/agent_nonlinear_suffix_scalar_fd_20261028_v1_cuda_MAIN01 --authorization MAIN_FIXED240_SCALAR_FD_HOST_CUDA_ONLY
.venv/bin/python -m experiments.spectral_reachability.agent_nonlinear_suffix_scalar_fd_20261028_v1 consume --output results/agent_nonlinear_suffix_scalar_fd_20261028_v1_cuda_MAIN01 --report-sha ACTUAL_REPORT_SHA --receipt results/agent_nonlinear_suffix_scalar_fd_20261028_v1_consume_MAIN01.json
```

不支持替换 checkpoint/原report/receipt、改变四点/h/原 Q/precision、只跑好看的 subset、按误差选点或 h、CPU neural fallback、补 AD、拼接旧8点 matrix-AD FD、重试/续写已有 output、将 FD 相合解释为物理正确或 P1 通过。不支持的输入或来源/类别/模型变化 fail closed，精度折叠/域外则按固定分母 missing 处理。旧8点属另一 actual-midcontext AD矩阵路径，不能替代本次原 predicted-mid 的 scalar 验证。

最终本 stem 19 tests；与原 v2 的60 tests 联合 **79 passed in 8.88s**，均为 RAM/保存格式测试，不是 CUDA 结果。保留中间真实错误记录：额外 packet signed-zero mutation 曾 **1 failed/18 passed in 1.08s**，原因是复用的上游 NumPy 数值相等不区分正负零；只在本 consumer 增加 `audit_native_bits`，逐字节核 packet/full3/normalizer/token/module wiring/boundary，未修改上游。初始14项通过和 MAIN 的74项联合通过是修订前测试数，不能冒充最终版本。

三件套 SHA 由交付消息记录（避免本文自引用 SHA）。MAIN 后续生产运行结果与科学判断另记，不回写冻结本三件套。
