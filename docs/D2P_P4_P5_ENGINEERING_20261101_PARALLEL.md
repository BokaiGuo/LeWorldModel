# D2P P4/P5 工程闭环与 CUDA 前置（2026-11-01，PARALLEL）

## 范围

本轮只新增 P4/P5 独占文件，不修改 P1/P2/P3，不覆盖历史 sealed artifact。
没有训练冻结模型，没有运行物理，没有以 CPU 替代神经 CUDA，也没有改变任何
stage admission。

## P4：真实反馈闭环 saved closure

新增统一 saved-only 验收器：

- `le-wm-main/experiments/spectral_reachability/p4_feedback_closure_20261101_parallel.py`
- `le-wm-main/tests/test_p4_feedback_closure_20261101_parallel.py`

它固定并复核四份既有证据：

1. 两臂、20 步、3 轮反馈的真实反馈 CUDA receipt；
2. 独立 saved live audit；
3. multiround outcome 算术审计；
4. terminal20 CUDA encoding receipt。

验收器执行前后重读并校验 SHA256，检查 actual/imagined 两臂、完成状态、冻结模型
哈希、saved audit gates 和 admission=false。它不重新调用模型或物理环境。

产物：

- `le-wm-main/results/p4_feedback_closure_20261101_parallel.json`
- SHA256:
  `9abd90da949b95aa97c4a12e5dd27cc589437fbef608db15733fa9802dbd4f81`
- 状态：`p4_engineering_saved_closure_verified`

证据等级仅为 saved-artifact engineering validation。它证明已有真实反馈闭环接线、
三轮反馈、terminal encoding 和独立 saved validation 可共同通过统一 fail-closed
入口；不证明任务收益、P4 科学通过或独立泛化。

## P5：跨模型合同与 CUDA preflight

新增三阶段入口：

- `le-wm-main/experiments/spectral_reachability/p5_crossmodel_preflight_20261101_parallel.py`
- `le-wm-main/tests/test_p5_crossmodel_preflight_20261101_parallel.py`

### Prepare

固定 164 分支/80 配对的 panel、JEPA ABI 审计和 LeWM RGB checkpoint。合同明确保留：

- panel 只有 2 个已知根，fresh root 为 0；
- LeWM 路径为 RGB-only；
- native JEPA H1 需要 RGB history 加 agent position/velocity；
- 因此 same-permission cross-model comparison 尚未成立。

产物：

- `le-wm-main/results/p5_crossmodel_prepare_20261101_parallel.json`
- SHA256:
  `273ca12ad560111a6da3d57ae2351215f210b714a3ef80b563e107062298b066`

### Host CUDA strict-load

在 `le-wm-main/results/e3_p2_host_cuda_cooperative.lock` 的非阻塞独占锁内运行。
加载固定 LeWM checkpoint，执行 `weights_only=True`、strict state-dict load、CUDA
迁移、eval/frozen 检查；不执行 neural forward、训练或物理。checkpoint 在运行前后
SHA256 一致。

运行环境：

- GPU: NVIDIA GeForce RTX 5060 Ti
- PyTorch: `2.12.0+cu130`
- CUDA runtime: `13.0`
- neural forward calls: 0
- training calls: 0
- physics calls: 0

产物：

- `le-wm-main/results/p5_crossmodel_cuda_20261101_parallel.json`
- SHA256:
  `1445fd3b3227c57e73cb845cf43df7dadec10b307d917edff63ab47b6b7be5b6`
- 状态：`p5_cuda_strict_load_preflight_passed`

没有对 native JEPA 做 live strict-load。原因不是绕过失败，而是其已固定接口需要
privileged proprio，当前没有经过验证的同权限 RGB-only native checkpoint；强行加载
不能构成公平跨模型前置。

### Saved verification

独立重读 prepare 与 CUDA receipt，复核锁、CUDA、checkpoint identity、strict load、
frozen/eval、零 forward/training/physics 计数及所有 admission=false 边界。

产物：

- `le-wm-main/results/p5_crossmodel_verify_20261101_parallel.json`
- SHA256:
  `0e2b610699ce2ca5ecaad8558759be8d1f8a3a6caebe167efd6f606b6737babb`
- 状态：`p5_crossmodel_cuda_preflight_saved_validation_passed`

CUDA 作业结束后，合作锁已用非阻塞 `flock` 再次取得，确认锁已释放。

## 测试

先建立缺失模块的失败测试，再实现至通过：

```text
.venv/bin/pytest -q \
  tests/test_p4_feedback_closure_20261101_parallel.py \
  tests/test_p5_crossmodel_preflight_20261101_parallel.py

13 passed in 0.53s
```

覆盖：

- sealed input SHA 和稳定性；
- P4 两臂、反馈、模型哈希及 saved gates；
- P4/P5 admission 永远为 false；
- P5 权限差异与 fresh-root 缺口不可消失；
- CUDA-only、合作锁、strict-load、frozen/eval；
- checkpoint 前后哈希一致；
- saved receipt 重建；
- 独占输出拒绝覆盖。

## 当前阻塞项

1. P4 仍是已知根工程证据，上游阶段 admission 为 false；没有独立确认或任务收益证明。
2. P5 没有经过认证的 RGB-only native JEPA checkpoint/interface，当前 JEPA 与 LeWM
   不是 same-permission 输入。
3. P5 panel 只有两个已知根，fresh-root certification 为 0。
4. 本轮 CUDA 只建立 LeWM checkpoint 的运行前置，不建立跨模型 efficacy、P5 admission
   或独立泛化。
