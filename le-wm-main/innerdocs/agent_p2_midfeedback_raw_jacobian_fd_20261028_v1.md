# 固定 raw Jacobian 中心差分诊断（冻结交接）

实现：`experiments/spectral_reachability/agent_p2_midfeedback_raw_jacobian_fd_20261028_v1.py`；RAM 测试：`tests/test_agent_p2_midfeedback_raw_jacobian_fd_20261028_v1.py`。

## 固定输入与问题

仅绑定实际 v2 compiler `results/agent_p2_contextual_midfeedback_recompile_20261028_v2_cuda_MAIN01/report.json`，SHA `363925ff3714c6c9a5fbd2ec6c59e86dbcb63fe685cde1945d292ec1a9196117`；其源 SHA `e459cdc5de507353873346d6f86bc223afed291982904c4c4c6ce9bdfb7eaeab`。复用原 `consume_compile` 全闭包验证，不读取物理 outcome 或 terminal 编码。提取每臂原 incumbent（不是新选中候选）的 raw5、actual-mid context、history10、goal192、normalizer、原 nominal 输出与 literal raw AD Jacobian `(192,10)`。模型、源码、准备文件及全数组 SHA 继承绑定。

固定 8 anchors（2 known roots / 4 科学槽），steps `[1e-2,1e-3,1e-4]`，time-major 10 raw coordinates、每维正负、192 outputs。共 **480 signed forward 请求、240 central pairs；不增加 anchor 重放**。原 nominal 仅使用封存数组。每个请求只改变一个 raw float32 坐标：`float32(float64(anchor)+sign*h)`，不经过 tanh、logits、逆变换、clip 或量化 tanh 导数。

合法性为 public raw cube `[-1,1]`。这是 raw 导数诊断，不是 joint-logit 球内的 planner 候选；两种 domain 不混称。非法或 float32 collapsed 请求不执行、保留 missing/null；合法另一侧仍执行，绝不夹紧。480 是固定请求分母及最大实际 forward 数，存在非法请求时实际少于480；任何缺列均使对应完整矩阵 summary=null，不拿子集替代。

## 输出、成本与边界

每步每坐标保存完整192维 FD、AD 与误差；主 FD 分母为 `2*nominal_h`，另报真实 float32 正负 spacing 的 secant、spacing 非对称。各步独立列 absolute L2/maxabs/RMSE、relative-to-AD、symmetric relative 和 cosine；零分母为 null，不加 epsilon。不选最佳步长、不设置通过阈值。保留相对封存 nominal 的两侧一阶残差与不等 spacing 二阶差分，不能单凭这些把偏差归咎曲率或认证 AD 正确。

真实 native float32 predictor chain、CUDA-only、batch1、TF32/autocast off、MATH SDP。最大 action_encoder/predict/pred_proj 各480；inclusive predict API 最大480，不把嵌套轴相加成 FLOPs。原 compiler 成本和 AD 单列继承；本阶段新 AD/encoder/physics/优化/选择均0，total_FLOPs=null。持久化完整模块输入输出、API entry/return、无 AD 检查和逐点 intent/return；strict consumer 重建全部事件、数组、计量和差分。未知事件或未终态失败拒绝消费，不重试补臂。

预算：whole-stage 900秒（计时先于 verify，模型/运行也在内）；output128MiB，其中 final reserve8MiB；disk floor200MiB。失败保存已用账和8分母；超时为操作边界检查，不承诺中断一个卡住的 native call。消费无模型调用。所有 P1/P2/physical release=false，无 AD correctness、物理收益或新根声明。原策略 modeled-Q 改善与 native harm 不参与此诊断选择。

## 测试与未执行

实际 RAM 单测命令：`.venv/bin/python -m pytest -q tests/test_agent_p2_midfeedback_raw_jacobian_fd_20261028_v1.py`，**29 passed in 3.58s**。测试为 NumPy/device-labelled doubles，覆盖真实 wrapper/hooks、raw spacing、缺侧/全矩阵 null、全192误差、事件和数组篡改拒绝、零范数、AD 禁止与授权门。开发首轮28通过1失败保留为 mock 接线失败记录，不是科学证据；最终按固定480合同移除了额外8次 anchor 重放并通过29项。worker 未执行 production prepare、CUDA 或 physics。真实模型 FD 数值、实际资源账与 strict receipt 均待 MAIN。

## MAIN CLI

从仓库根运行；占位 SHA 必须使用上一命令实际返回值，不预填未来值。

```bash
.venv/bin/python -m experiments.spectral_reachability.agent_p2_midfeedback_raw_jacobian_fd_20261028_v1 prepare --compile-output results/agent_p2_contextual_midfeedback_recompile_20261028_v2_cuda_MAIN01 --compile-sha 363925ff3714c6c9a5fbd2ec6c59e86dbcb63fe685cde1945d292ec1a9196117 --output results/agent_p2_midfeedback_raw_jacobian_fd_20261028_v1_prepare_MAIN01.json
.venv/bin/python -m experiments.spectral_reachability.agent_p2_midfeedback_raw_jacobian_fd_20261028_v1 verify --prepared results/agent_p2_midfeedback_raw_jacobian_fd_20261028_v1_prepare_MAIN01.json --prepare-sha <ACTUAL_PREPARE_SHA>
.venv/bin/python -m experiments.spectral_reachability.agent_p2_midfeedback_raw_jacobian_fd_20261028_v1 run --prepared results/agent_p2_midfeedback_raw_jacobian_fd_20261028_v1_prepare_MAIN01.json --prepare-sha <ACTUAL_PREPARE_SHA> --output results/agent_p2_midfeedback_raw_jacobian_fd_20261028_v1_cuda_MAIN01 --authorize-host-cuda
.venv/bin/python -m experiments.spectral_reachability.agent_p2_midfeedback_raw_jacobian_fd_20261028_v1 consume --output results/agent_p2_midfeedback_raw_jacobian_fd_20261028_v1_cuda_MAIN01 --report-sha <ACTUAL_REPORT_SHA> --receipt results/agent_p2_midfeedback_raw_jacobian_fd_20261028_v1_consume_MAIN01.json
```

三文件交付后完全冻结；任何后续修复使用新版本，不改已有封存源码或 MAIN 结果。
