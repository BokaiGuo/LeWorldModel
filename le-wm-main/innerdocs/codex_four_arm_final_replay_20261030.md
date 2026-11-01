# 四臂 raw 候选、优化和最终 replay：接口状态

本轮新代码不加载 checkpoint、不启动 GPU 或物理；没有更改主线程文件、封存代码、
P1 门或已有运行。主线程的
`docs/D2P_FOUR_ARM_RAW_ACTION_CUDA_PROBE_20261030.md` 报告：真实冻结 checkpoint
上 raw 后缀两步与直接参考输出最大差 0、模型 hash 前后一致；这是主线程已有结果，
不是本轮对新四臂入口的 CUDA 验证，且参考路径未计入账本。

## 新接口

`codex_four_arm_final_replay_20261030.replay_and_select` 接收每个候选的最终
raw `[N,T,5,2]`、单个共同 `[1,H,D]` state、已归一化 `[1,H-1,10]` 历史动作、
共同 goal 和二维 mean/std。每个新 raw packet 单独做 `(raw-mean)/std` 再展平；
不会对历史动作重复归一化，不把 normalized `[-1,1]` 当物理界。只接受 raw
`[-1,1]`。策略哈希固定候选 ID 顺序、数量、horizon/history、mean/std、末端
latent mean-square 度量及精确平局优先最前 ID。全部候选逐一 fresh replay、
计费并评分后才返回 winner；有失败或超额则不返回部分 winner。

`codex_four_arm_raw_candidates_20261030.run_arm` 是相同 raw 图表上的可执行
试验性四臂入口，结果只能当模型侧工程准备：

- `native`：零均值 Gaussian 初始化的新 CEM 变体，**不是**已封存的 native CEM 实现。
- `random`：随机生成一个 raw 初始中心。
- `immediate_progress`：对固定数量候选的首 packet 逐个执行模型预测并按同一
  goal 的即时 latent MSE 筛选，平局按顺序；后续 packet 不参与即时评分。
- `structured`：在 raw 零点计算完整末端 latent 对 raw 的 Jacobian，
  按 ridge 最小二乘得到 raw 投影中心。这是结构化局部线性 **proxy**，并不是
  既有 D2P spectral-bank 实现或已证明的 D2P 机制。

四臂使用完全相同、配置固定的 clipped raw Gaussian CEM 更新（固定 population、
iteration、elite、seed、std 与 floor）；最后一代所有样本另做一次付费最终 replay。
返回的是最后一代样本 winner，不是假称重新评分过的 CEM 均值。该 CEM 未被证明
是强优化器，也不替代真实性能比较所需的各臂优化；实际强 baseline、原 native
CEM 与 D2P 结构化生成仍需单独接入，不应把本试验变体冒充它们。

示意接线（`contract`、任务、配置与输出均须由主线程事前固定，不是可直接复制的生产命令）：

```python
leaf = LeafAdapter(model, driver, contract, contract_sha256)
result = run_arm(leaf, arm, states, normalized_history, goal, mean, std,
                 config, replay_policy, contract)
receipt = driver.finish()
# 原独立 consumer 验证 receipt 的外部 SHA、合同、arm、level 和 run ID。
```

`identity.optimizer_config` 必须是 CEM 配置 canonical SHA；
`phase_policy_sha256` 必须是本 replay policy canonical SHA。需为每个 arm/预算点
创建独立 driver；选择预定任务和根不得依赖 outcome。采用同一冻结模型、
原始 RGB/history/goal、normalizer、动作域、精度和相同反馈时点；每次闭环
重规划必须重新取得同口径观测并分别计费，不能给某臂额外反馈。

## 已计和未计

使用现有持久 driver，逐个记录外层 action encoder、predict、标量反向/VJP、
Jacobian 标记、ridge solve、逐候选 replay 预留及 selection evaluation。
即时筛选也单独记录；各优化代不丢掉失败或已付前缀。`replay_calls` 事件是
开始整个 replay **之前的预留标记**，不是独立证明 replay 完成：完成还须检查
后续全部模型事件和调用结果。消费器验证账本顺序/额度，但目前日志**不含**
候选 raw 字节、逐候选模型输入输出或分数，不能独立重建 winner，也不能证明没有
绕过 hook。需新增真实 saved trace 与独立消费后才能交物理 runner。

RNG、CPU/GPU 张量运算、CEM 更新、矩阵乘法、输入转换、调用内部模型子模块、
环境与编解码、wall、峰值显存和 FLOPs 未被完整计量或限制；不可将向量 cap
称作总成本公平。`structured` Jacobian marker、`replay` marker 均非完整工作
的嵌套包装，模型及 backward 的叶子事件各自分开入账。强优化器是否充分、
结构化方法是否 faithful、实际 normalizer/编码来源、训练独立性、source/precision
closure 和四臂预算覆盖仍须先行审验。整链的超限、失败和中断保留未完成分母。

## 实际模型及物理接线前置

1. 新入口尚未对真实 checkpoint 做 CUDA ABI 对照；主线程已验证的是 raw suffix，
   不是本 CEM、结构化 Jacobian 或本 replay。先用同一固定已开放任务运行小规模
   host CUDA ABI 对照，逐 packet 检查输入、原始动作、normalizer、预测与输出，
   外部留存源码/模型/精度 hash，不能把 smoke 结果当性能证据。
2. 接真实 D2P candidate 与忠实 native/其他强 optimizer，保留各自初始化、搜索、
   全候选与失败的实际计量；提前固定比较预算和支持域，不得根据结果调参。
3. 补独立可复算候选/score/选中动作保存与绑定、四臂全路径计费、实际 wall/
   FLOPs/物理成本及完整源码闭包；处理统计轴未知而非填零。
4. 物理前固定相同根、动作起点和反馈时点、每臂固定分母与失败策略，取得相应
   authority 和主线程明确生产授权；不得以这些替身测试放宽 P1 或宣称独立性。

## 测试

在 `le-wm-main` 执行以下 CPU 解析替身及既有回归：`59 passed in 1.99s`。
包括主线程 raw 后缀的末次优化反例（旧评分 4、新动作复评 0.16）、H3 历史及
normalized 超出 `[-1,1]`、候选完整计费、确定性平局、各臂计数、失败与 cap 阻断。

```bash
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' .venv/bin/python -B -m pytest -q -p no:cacheprovider \
 tests/test_codex_four_arm_raw_candidates_20261030.py \
 tests/test_codex_four_arm_final_replay_20261030.py \
 tests/test_codex_four_arm_raw_suffix_20261030_main.py \
 tests/test_codex_four_arm_leaf_suffix_20261030.py \
 tests/test_codex_four_arm_durable_20261030.py \
 tests/test_codex_four_arm_full_cost_20261030.py
```
