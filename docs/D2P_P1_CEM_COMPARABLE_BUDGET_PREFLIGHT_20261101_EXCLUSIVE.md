# P1 CEM 预算可比性：独占 CPU 预检

本文件及 `codex_p1_cem_budget_preflight_20261101_exclusive.py` 是新的
只读分析，不是预注册确认、模型运行、物理证据，也不变更 v2 的两个
已封存 prepare。两个 root 的 v2 prepare 均哈希绑定到相应
first-feedback prepare；相同的目标名称、root slot、反馈机会和
规划 horizon 可以从这两份合同核对，但不能由此替代逐运行的实际
根状态、目标 RGB/latent、模型和环境认证。

## 差异与不能直接推断的结论

同目标 first-feedback prepare 的四臂 `structured` 静态
`node_costs` 声明 action encode / predict / predict samples 各 174、
backward 240、VJP 240、Jacobian 1、SVD 1。此处是其协议中的
静态值，不是逐边界已测实际消耗。CEM v2 在 0/5/10/15 的合同
分别为 195/130/65/65 次 action encode、predict 和 predict
samples，backward、Jacobian、SVD 为零；实际达到的节点取决于
早停。两个方法具有相同反馈机会和 H3/H2/H1/H1 horizon，但
first-feedback 在部分边界携带上一计划尾部，CEM v2 每节点
重新初始化；搜索算法、候选选择、反向和成本覆盖亦不同。
因此当前既不是逐边界同前向预算，也不是相同计算成本。

H4 的 reverse/centered prepare 使用 H4/H3/H2/H1，以及其他目标；
即使数字能对齐，也不能直接作为这两个 cp10 目标的配对对照。
旧报告已打开且 root 已知，任何后续调参或配对只能标注探索性，
不能说是针对这些已见结果的盲态独立确认。

## 可执行的预算设计检查

以 structured 静态 174 个 predict **作为假设的每节点前向上限**，
四轮、四精英、保留一次完整 winner replay，CEM 的调用为
`(4 × 每轮候选数 + 1) × horizon`。CPU 预检器求最大整数候选数：

| 边界 | horizon | 每轮候选 | 含重放前向上限 | 余量 |
|---|---:|---:|---:|---:|
| 0 | 3 | 14 | 171 | 3 |
| 5 | 2 | 21 | 170 | 4 |
| 10 | 1 | 43 | 173 | 1 |
| 15 | 1 | 43 | 173 | 1 |

这是**拟议配置**，不是 v2 runner 的现有行为：现有控制器固定每轮
16 个候选，既未实现逐边界变长采样，也没有这个方案的新 prepare。
相同 cap 不意味着相同实际用量，更不意味着等 FLOPs 或等成本；
不得虚填未到达节点、预算余量或重放。

执行只读检查（需从 `le-wm-main` 运行）：

```bash
env CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 \
  .venv/bin/python -B -m experiments.spectral_reachability.codex_p1_cem_budget_preflight_20261101_exclusive --root 0
```

root1 使用 `--root 1`。预检校验两份既有 v2 prepare 指定 SHA、
相应 first-feedback prepare SHA 及静态字段，若漂移则失败；
输出明确标记 `proposal_implemented=false` 和
`ready_to_run_comparison=false`。计算不会加载模型、调用 GPU、
启动物理或写文件。

## 下一步实现与准入门槛

1. 对将真正比较的 structured 运行和每一边界，从各自预封存合同、
   实际节点账本及 hook 回执确认输入、实际 horizon、预算和执行；
   特别不能假定上述静态 174 在后续反馈节点也真实适用。根与目标
   应比较实际 state、目标 RGB/latent、模型/normalizer 和物理设置，
   各 arm 的后续观察允许因为各自动作而不同。
2. 若预先确认统一上限适合该设计，另建独占 variable-bank CEM
   controller/host/prepare，显式绑定每节点采样数、seed、精英数、
   真实 driver 合同、事件和最终报告；对成本计数、超预算、
   提前终止及错误处理做 CPU 负向测试。不得修改现有 v2 prepare
   或把该方案追记为旧 v2 结果。
3. 双臂事前固定同等的观测与早停机会及一次性源闭包；分别记录
   encoder/projector、action-encoder、predictor、反向 API、方向、
   Jacobian/SVD、物理步、同步 wall time、显存和磁盘，不加权相加。
   模块 hooks 和叶适配器账本须交叉验证覆盖；无覆盖证明则相应轴
   标记未知，不用空值补零。比较实际资源向量及性能—预算曲线，
   不能将节点封顶等同于实际等成本。
4. 根目标身份、两个 arm 均完成的可检查回执、预算未超限、明确的
   失败/中断分母和独立验证全部具备之前，保持 fail-closed；
   按各自实际早停分别报资源与结果。完整比较的真实执行仍需新的
   用户授权、既有合作锁及现场安全检查。
