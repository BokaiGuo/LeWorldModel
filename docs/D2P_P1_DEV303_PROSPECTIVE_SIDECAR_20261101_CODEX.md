# P1 selector v3 后 dev303 prospective sidecar

日期：2026-11-01

## 本轮边界

本轮只冻结 dev303 开发 holdout 的协议和根来源审计。没有读取 dev303 outcome，
没有初始化 CUDA，没有构造环境或运行物理，也没有修改任何 P1 selector/runner。

冻结产物：

- `le-wm-main/results/p1_dev303_prospective_sidecar_20261101_codex_PREP_MAIN01.json`
- SHA256:
  `f8db73f4763b5c68d722225f61fc7c54a4a3c96c5f90f170dd0b5da196686800`
- 状态：`dev303_prospective_protocol_frozen_waiting_for_source`

## 已冻结合同

### 候选库

候选顺序固定为：

1. zero
2. spectral
3. GN
4. RGB nearest contact
5. RGB goal-support direct
6. RGB goal-support top route
7. RGB goal-support bottom route
8. fixed random control

每个 prefix 为 5 步。spectral/GN 必须在 dev303 物理执行前封存；四个接触候选只能
由 dev303 初始 RGB 和 goal RGB 确定。outcome 后不得增加、删除或改序候选。

### RGB-only 接触几何

- pusher RGB: `[65, 105, 225]`
- object RGB: `[119, 136, 153]`
- component minimum: 8 pixels
- gain: 2.0
- pusher radius: 6.5 pixels
- obstacle clearance: 10 pixels
- cumulative raw scale: 17 pixels
- contact threshold: 8 pixels
- simulator state input: prohibited

分割或几何失败时 selector abstain，并作为方法失败记录；不得换根。

### Selector v3

- contact-feasible prior 为硬过滤；
- 每个 prefix 使用两包 tail；
- CEM 为 5 iterations、64 samples、8 elites；
- 每条 tail 评价 nominal 和四个固定 `0.05` 扰动；
- robust score 为五个 terminal latent goal MSE 的 mean 加 range；
- 全局最小 robust score，iteration/candidate 稳定次序打破平局；
- outcome 后任何 selector 修改都必须改名为 v4，并使用另一个未触碰 holdout。

### 反馈、预算和物理 roster

四臂固定为 selected-v3、same-bank greedy、zero-prefix、random-control。
每臂仅允许一次真实反馈，位于 prefix 执行完的 cp5；输入仅为实际 RGB 和实际五步
动作历史。反馈缺失时该臂失败，不允许退回 open-loop。

根选择阶段预算固定为：

- 8 prefix candidates
- 88 action-encoder calls
- 88 predict calls
- 25,608 predict samples
- 1,800 秒 wall cap

每臂反馈重规划预算固定为：

- 1 RGB encode
- 10 action-encoder calls
- 10 predict calls
- 3,200 predict samples
- 1,800 秒 wall cap

每臂最多 10 步 zero warmup、5 步 prefix 和 10 步 tail。禁止用无效调用补齐预算；
实际调用数按类型报告。

### 指标

主性能指标固定为第 15 control step 前是否达到 native success。

机制指标固定为：selected-v3 在 cp5 不具有相对初始状态的 object-pose-cost 优势，
但随后改善 terminal object-pose cost 或达到 native success。object-pose cost 定义为：

```text
object_xy_error + 20 * wrapped_object_angle_error
```

其余指标分栏报告 object XY、wrapped angle、object-only success、official native
success、RGB task score 和模型 terminal latent goal MSE。单根 dev303 仅是描述性开发
证据，不做显著性或独立泛化主张。

### 失败与删失

- reset/source identity 不匹配：技术无效，不换根；
- 无可行候选或 selector abstain：方法失败；
- cp5 前 native success：保留性能成功，机制与反馈指标删失；
- cp5 后提前成功：保留成功并记录停止步；
- 无成功的 termination/truncation：失败；
- 首个动作前的纯工程失败：允许同根、新 run ID 重跑；
- 任一动作派发后的失败：主分析不重跑，保留技术失败；
- 缺失指标不插补；
- outcome 后禁止换根或追加根。

## dev303 根来源能证明到什么程度

固定 seed 为 `301000003`。

审计结果：

- 该 seed 已在 DEV01 冻结队列中以 confirmation 角色、
  `not_a_confirmation` 槽预先列名；
- DEV301 ledger 实际只尝试了 `301000001`；
- DEV302 ledger 实际只尝试了 `301000002`；
- 当前固定 DEV301/DEV302 source receipts 也只绑定上述两次执行；
- 没有发现 dev303 reset、source、task、run 或 outcome 命名空间产物。

因此可以证明：

> 在已固定的 DEV301/DEV302 queue ledgers 范围内，seed 301000003 没有被 reset 或进入
> 已记录的开发执行历史。

不能证明：

- 对开发者完全盲；
- 没有任何未记录的历史访问；
- checkpoint 训练外；
- 全局物理状态新鲜；
- 独立确认。

旧 confirmation 槽不能静默改成 development。下一前置是为同一个 seed
`301000003` 建立新的 development-authorized DEV03 queue/source receipt 并封存；
在此之前不得运行 selector、CUDA 或物理。

## 测试

```text
.venv/bin/pytest -q \
  tests/test_p1_dev303_prospective_sidecar_20261101_codex.py

8 passed
```

测试覆盖候选库、RGB-only 几何、selector、反馈次数、预算、指标、失败/删失规则、
outcome 命名空间拒绝、dev303 seed 已尝试拒绝、source SHA 漂移拒绝、admission 漂移
拒绝及独占输出。
