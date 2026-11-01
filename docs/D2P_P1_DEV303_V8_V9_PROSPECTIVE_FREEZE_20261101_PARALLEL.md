# P1 dev303 V8/V9 prospective freeze 清单

日期：2026-11-01  
状态：**draft freeze；执行未授权；dev303 outcome 未打开**

## 1. 范围与硬边界

本文件只完成三件事：

1. 对照旧 selector-v3 dev303 sidecar 与后续 V8/V9 方法；
2. 固定一个可供 dev303 development holdout 使用的新协议草案；
3. 列出来源、代码闭包、预算、执行和判据的硬前置。

本文件不读取、生成或运行任何 dev303 outcome，不授权 selector、CUDA 或物理执行，
也不把 fixture、测试或 dev302 开发结果提升为 P1 科学证据。

旧文件
`docs/D2P_P1_DEV303_PROSPECTIVE_SIDECAR_20261101_CODEX.md`
及其 prep 必须保持不可变。V8/V9 不是旧 selector-v3 的实现修复，而是实质方法变化；
不得覆盖旧 sidecar、沿用其协议哈希或把新结果写入旧命名空间。

## 2. 旧 sidecar 与 V8/V9 的方法差异

| 项目 | 旧 selector-v3 sidecar | V8/V9 | 结论 |
|---|---|---|---|
| 候选数 | 8 | 17 | 搜索空间已改变 |
| pusher 轨迹模型 | memoryless 17 px cumulative-raw 标定 | 一步记忆 RGB pusher dynamics | 几何合同已改变 |
| 接触中心阈值 | 8 px | 6.5 px | 可行集合已改变 |
| task-support alignment | 无 | 至少 0.5 | 新硬过滤条件 |
| 路径族 | direct/top/bottom 等基础候选 | 新增 delayed-support 及 8 个 endpoint-preserving variants | 候选生成已改变 |
| CEM 随机流 | prefix 间不同随机流 | 全 prefix common random numbers | 排序方差合同已改变 |
| root action-encoder calls | 88 | 187 | 预算已改变 |
| root predict calls | 88 | 187 | 预算已改变 |
| root predict samples | 25,608 | 54,417 | 预算已改变 |
| greedy | 最小 cp5 latent goal MSE | V9 强制 index 4 direct 为 task-greedy | baseline 定义已改变 |
| 物理时域 | 5-step prefix + cp5 feedback + 10-step tail | 当前 V9 只执行 5-step prefix | V9 尚不是完整 P1 闭环 |

dev302 V8 的保存报告中，robust selected 与 latent same-bank greedy 均为 index 16。
因此该根不能提供 selected 对 latent-greedy 的物理区分证据。

V9 的 index 4 规则使用了 dev302 已知物理接触发生在 step 4/5 的信息。它是
outcome-informed development rule。若在 dev303 outcome 之前完整冻结，它可以作为
dev303 的开发性 `task_greedy_direct` 对照；它不能被称为独立、盲或 confirmation
baseline。代码中的角色名 `same_bank_greedy` 在此含义不正确，新协议必须使用
`task_greedy_direct`。

## 3. 新协议的证据等级

新协议固定为：

```text
P1 dev303 V8/V9 development holdout
```

它不属于 independent confirmation，理由是：

- V8/V9 均在 dev301/dev302 结果之后形成；
- V9 task-greedy 明确使用了 dev302 物理接触知识；
- seed `301000003` 只能证明在已封存的 DEV301/DEV302 bounded ledgers 中未被尝试；
- developer blindness、全局历史新鲜性和 checkpoint 训练外身份尚未建立。

dev303 单根结果只能用于开发判别和方法冻结决策，不做显著性、泛化或 P1 最终通过
主张。后续独立确认必须使用另一个在 V8/V9 及反馈 runner 全部冻结后才物化的根。

## 4. dev303 来源冻结前置

固定 reset seed：

```text
301000003
```

任何 selector、CUDA 或物理执行前，必须新建并封存一个明确
`development-authorized` 的 DEV03 queue/source receipt。旧 DEV01 中
`role=confirmation, slot_id=not_a_confirmation` 的槽不得静默改写为 development。

DEV03 source closure 至少必须逐项记录：

- queue JSON、queue ledger 和 reset-attempt event；
- source receipt、reset receipt、warmup report；
- reset seed、初始 simulator state hash；
- `rgb_00.png`、`rgb_05.png`、`rgb_10.png` 和 `goal.png` 的 SHA256；
- 五步以前的实际动作历史及其 dtype、shape、byte-order 和 SHA256；
- task JSON/gzip 原始字节 SHA256；
- source materialization 时间、进程、run ID 和独占输出路径；
- `training_membership=unknown` 与 `historical_coverage=unknown`，除非另有可审计权威证据；
- `independent_confirmation=false`、`P1_admission=false`。

以下任一条件不满足时 fail closed，不得换根：

- seed 不等于 `301000003`；
- DEV03 receipt 已存在但 SHA 漂移；
- outcome/source 命名空间在 freeze 完成前已出现不可解释产物；
- reset identity、RGB、动作历史或 goal identity 不一致；
- source 需要 simulator state 才能生成部署候选；
- 任一输入为 symlink、非普通文件或缺少完整 SHA。

## 5. V8 候选与选择器冻结

候选库固定为 17 个、顺序不可改变：

1. `zero`
2. `spectral`
3. `gn`
4. `rgb_nearest_contact`
5. `rgb_goal_support_direct`
6. `rgb_goal_support_top_route`
7. `rgb_goal_support_bottom_route`
8. `random_control`
9. `rgb_delayed_task_support`
10. `rgb_delayed_support_x12`
11. `rgb_delayed_support_y12`
12. `rgb_delayed_support_x23`
13. `rgb_delayed_support_y23`
14. `rgb_delayed_x12_pos06`
15. `rgb_delayed_x12_neg06`
16. `rgb_delayed_x23_pos06`
17. `rgb_delayed_x23_neg04`

必须在 dev303 selector 执行前冻结并保存：

- 每个候选的 `float32` 五步动作完整数组及 byte-level SHA256；
- 候选标签、稳定 index 和生成来源；
- 一步记忆 pusher dynamics 的全部系数和积分初值；
- action→pixel 符号、坐标轴定义和图像原点；
- 6.5 px 接触阈值；
- 0.5 minimum task-support alignment；
- delayed-support 固定动作；
- 8 个 endpoint-preserving path variant 的扰动位置和幅度；
- bbox、边界距离、clearance 和 component-size 判据；
- RGB 颜色、连通域规则和分割失败行为；
- common-random-number seed `30220261102`；
- stable tie-breaking 次序。

部署 selector 禁止读取 simulator state、接触事件、reward、success、物体真值姿态或
任何 dev303 后果。分割、几何或可行性失败时必须 abstain 并记为方法失败，不得退回
旧 V3、人工挑候选或换根。

V8 root selector 预算固定为：

| 资源 | 上限 |
|---|---:|
| prefix candidates | 17 |
| tail CEM iterations per prefix | 5 |
| samples per iteration | 64 |
| elites | 8 |
| robust variants | 5 |
| tail packets | 2 |
| action-encoder calls | 187 |
| predict calls | 187 |
| predict samples | 54,417 |
| candidate evaluations | 17 |
| wall time | 1,800 s |

禁止无效调用补齐预算。必须同时报告上限与实际分类消耗。

## 6. 四臂 roster 与 baseline 命名

物理 roster 固定为：

1. `selected_v8`
2. `task_greedy_direct`
3. `zero_prefix`
4. `random_control`

其中：

- `selected_v8`：V8 等预算 robust nonlinear terminal score 的全局最优；
- `task_greedy_direct`：固定候选 `rgb_goal_support_direct`，即 stable index 4；
- `zero_prefix`：stable index 0；
- `random_control`：stable index 7。

latent same-bank greedy 只作为保存诊断项，不占用物理臂，也不得与
`task_greedy_direct` 混名。若 dev303 上 latent greedy 与 selected 相同，必须原样报告，
不能另挑第二名制造区分。

## 7. 反馈、尾部与物理 runner 的硬前置

当前 V9 runner 只执行五步 prefix，且
`post_prefix_outcomes_used_for_planning=false`。它不能授权完整 P1 实验。

运行 dev303 物理 outcome 前，必须先实现、测试并封存一个新的独占 runner，满足：

1. 四臂分别从同一 reset/warmup 身份开始；
2. 执行各自五步 prefix；
3. 在 cp5 捕获真实 RGB 和实际五步动作历史；
4. 使用相同冻结 encoder/model、同一 CEM 实现和同一资源约束独立规划十步 tail；
5. 对四臂分别真实执行各自 tail；
6. 另外执行预注册的共同十步 tail 或共同 tail family，用于机制干预；
7. 所有 neural calls、RGB encode、候选生成、重评分和 wall time 分类计费；
8. 任何 arm 缺失 cp5 反馈时该 arm 失败，不得回退 open-loop；
9. runner 在任何动作派发前验证协议 SHA、source SHA、动作 SHA、模型 SHA 和环境 SHA；
10. 使用 cooperative GPU lock，禁止 CPU 替代神经 CUDA。

每臂 cp5 独立后缀规划预算固定为：

| 资源 | 上限 |
|---|---:|
| RGB encode calls | 1 |
| CEM iterations | 5 |
| samples per iteration | 64 |
| elites | 8 |
| robust variants | 5 |
| tail packets | 2 |
| action-encoder calls | 10 |
| predict calls | 10 |
| predict samples | 3,200 |
| wall time | 1,800 s |

共同 tail 干预与各臂 own-tail 性能比较必须分开记账。共同 tail 只能证明前缀改变了
该动作的真实效果；own-tail 才能检验相同搜索条件下的后继状态是否更易规划。

## 8. P1 预注册判据

预定任务代价：

```text
object_pose_cost =
    object_xy_error + 20 * wrapped_object_angle_error
```

必须分栏报告 object XY、wrapped angle、pusher state、object-only success、official
native success、RGB task score 和模型 terminal latent goal MSE。含 pusher 或速度的指标
不得替代 object-pose mechanism 判据。

开发性机制合取要求同一个事前选择的 `selected_v8` 同时满足：

1. **无即时 object-pose 优势**：cp5 的 object-pose cost 不低于该臂 prefix 前的值；
2. **真实响应改变**：相同预注册 tail/family 在 selected cp5 后产生比
   `task_greedy_direct`、`zero_prefix` 或 `random_control` 更有利的任务响应；
3. **可规划收益**：四臂用各自真实 cp5 RGB、相同优化器和相同预算规划时，
   selected 的终态 object-pose cost 或 native success 更优；
4. **性能终点**：在固定 control step 15 以前达到 native success，或在未成功时保留
   完整终态代价；
5. **模型事前性**：selected 的身份在任何 dev303 物理后果前已经由保存协议唯一确定。

共同 tail 成功但 own-tail 不占优，只支持“动作效果改变”，不支持“可规划能力更强”。
own-tail 占优但 cp5 已有 object-pose 进展，只支持性能收益，不支持非即时重配置机制。
单一 dev303 根即使满足全部开发性合取，也仍需冻结方法后在新来源根上做独立确认。

## 9. 失败、删失与禁止事项

- reset/source mismatch：技术无效，不换根；
- 无可行候选或 selector abstain：方法失败；
- cp5 前 native success：保留性能成功，机制和反馈指标删失；
- cp5 后提前成功：保留成功并记录实际停止步；
- termination/truncation 且未成功：失败；
- 首个动作前的纯工程失败：允许同根、新 run ID 重跑；
- 任一动作派发后的失败：主分析不重跑，保留技术失败；
- 缺失指标不插补；
- 不得按 dev303 outcome 改阈值、候选、排序、tail family、指标或预算；
- 不得换根、追加根、挑成功后缀或把测试结果称为物理证据；
- 不得把 V9 的 dev302 outcome-informed baseline 称为盲 baseline；
- 不得放宽 P1 gate。

## 10. Transitive source SHA closure

当前审查确认 V8 prep 只直接绑定部分 V1/V2/V8 文件，没有完整绑定 V3–V7
transitive chain。新协议必须将下列代码全部纳入单一 closure；任一 SHA 漂移均需新版本名
和新 holdout，不能继续使用本 freeze：

| 文件 | 当前 SHA256 |
|---|---|
| `p1_dev302_mixed_prefix_bank_20261101_codex.py` | `c3784aa9306007e1394381567e5b3bbfaf049747ed9e1678f171bd6d9142f4c0` |
| `p1_dev302_mixed_prefix_bank_v2_20261101_codex.py` | `3309c45a56e9b6ac38cb6b5903d0781c0a8aa4aec9cf0e1122fe9ee7f4d6b068` |
| `p1_dev302_mixed_prefix_bank_v3_20261101_codex.py` | `bb08f452bb3d4664fa3ea7cba5617ecc1f98b4dabd52fcede27251a91ca0a4a8` |
| `p1_dev302_mixed_prefix_bank_v4_20261101_codex.py` | `07aa2d03813ecc2513d37d4c70169278451def3a4e81da91555891ba0cb79b4e` |
| `p1_dev302_mixed_prefix_bank_v5_20261101_codex.py` | `a48c92b1c6de42ad0dd36d1edebb6c1b05f66e1215b9bea7da26b2d495af3228` |
| `p1_dev302_mixed_prefix_bank_v6_20261101_codex.py` | `05f333b9733b63afecda4164643114473e531fb9c821ad9be820df0c2df75d70` |
| `p1_dev302_mixed_prefix_bank_v7_20261101_codex.py` | `b35165156e45cc20e1d52ea7c0c13b3cf1a9fa2be1f733c58c9ac13dcf9971f3` |
| `p1_dev302_mixed_prefix_bank_v8_20261101_codex.py` | `e73f26b5993802ba6aa7b12ee33e41ff3509848c7da89296417070762cfc04a1` |
| `p1_dev302_mixed_bank_cuda_20261101_codex.py` | `69d49d09f000936c99a95adb3ae16ef57e5bce3151d011fa97b5e4ce8366cb0e` |
| `p1_dev302_mixed_bank_cuda_v2_20261101_codex.py` | `34037067815479651cdbd76d398a7f490a874266351f6e266d88fafd7718081a` |
| `p1_dev302_mixed_bank_cuda_v3_20261101_codex.py` | `5e8b3b2562096625da05208293e7aa3e31088b7be382919fec6f0b7e5577e779` |
| `p1_dev302_mixed_bank_cuda_v4_20261101_codex.py` | `09ce7347ad21c4f20c5f141e745890f1a17e77022dc9f6ad4432e23046cd52b1` |
| `p1_dev302_mixed_bank_cuda_v5_20261101_codex.py` | `9b75220c72c78b57391fac29bbe7d9a0ba93d4851a7e04cc503d0b4f5c3da0e6` |
| `p1_dev302_mixed_bank_cuda_v6_20261101_codex.py` | `6d9f705a138c695c44b86d12f8501b2dc6d9abbe08f4ff0461b9cb5a0d78c295` |
| `p1_dev302_mixed_bank_cuda_v7_20261101_codex.py` | `0c24841a56434032c539120291ea955da13e85090b1ba61212d19abddff51321` |
| `p1_dev302_mixed_bank_cuda_v8_20261101_codex.py` | `f9c73ef587c8a6b7eea1d7a19f9ad73379037192cad808a6c9f2a3088b8c019d` |
| `p1_dev302_v9_task_greedy_prefix_physical_20261101_codex.py` | `bfd019e18a0bc697e83c08e80e311a4b571729e52e871accdeeccd72ecd42c29` |
| `p1_dev302_mixed_prefix_physical_v3_20261101_codex.py` | `68ff86616712e13c08a9b9f4467794320f8c8032fe58e0923de4f4af7c96e860` |
| `p1_dev302_mixed_prefix_physical_v4_20261101_codex.py` | `ed78fa1dac85be530890dfbf1135205ebe886427aea564e05495ede76035f23c` |
| `p1_unified_cp5_feedback_cem_20261101_codex.py` | `f0510356e73d01f020f302e70bd0001a88e46f2e2f326b41d7d98145a86c8b1f` |
| `codex_p1_rgb_task_score_20261030_v2.py` | `5641c8c0d68be8bca28a2067c5c62f15e9cd04f8e08181f79c4a4e595bede1d7` |

closure 还必须在新 prep 中直接绑定：

- cooperative GPU lock；
- RGB segmentation/metric 实现；
- source/reset materializer；
- task loader；
- action normalizer 及其统计文件；
- encoder、action encoder、predictor 的代码和 checkpoint shards；
- 完整 checkpoint index/config；
- `stable_worldmodel.envs.pusht.env.PushT` 源文件；
- 依赖的 physical metrics/capture/store 代码；
- 新 cp5 feedback + common-tail + own-tail runner；
- Python、PyTorch、CUDA、driver、GPU 型号和关键 package 版本。

仅通过 import 间接到达但未在 prep 中列出 SHA，不算 source closure 完成。

## 11. 执行授权清单

只有以下项目全部为真时，另一个独占授权文件才可以打开 dev303 执行：

- [ ] 新 DEV03 development queue/source receipt 已封存；
- [ ] dev303 outcome namespace 仍未出现；
- [ ] 17 个候选动作字节和全部几何常数已封存；
- [ ] V1–V8、V9、反馈 runner、物理 engine、环境、模型和 normalizer 完成 transitive SHA closure；
- [ ] 四臂名称与 stable indices 已冻结；
- [ ] root selector 187/187/54,417 预算已冻结；
- [ ] 每臂 cp5 feedback 10/10/3,200 预算已冻结；
- [ ] common-tail family 已在 outcome 前冻结；
- [ ] own-tail 优化器、初始化、停止规则和失败处理已冻结；
- [ ] action/RGB/history/goal identity 的 pre-dispatch 检查已实现并测试；
- [ ] cooperative lock 与 host CUDA preflight 通过；
- [ ] physical runner 能保存逐步动作、RGB、state-for-evaluation、接触、停止和资源账本；
- [ ] 测试只被标为工程证据，不被标为 P1 outcome；
- [ ] 授权文件明确 `development_only=true`、`independent_confirmation=false`、
      `P1_admission=false`。

在此之前，dev303 selector、CUDA、物理和 outcome consumer 均保持关闭。

## 12. 本轮只读验证

本轮使用以下测试验证已有实现的工程合同；没有运行 dev303 outcome：

```text
.venv/bin/pytest -q \
  tests/test_p1_dev303_prospective_sidecar_20261101_codex.py \
  tests/test_p1_dev302_mixed_bank_cuda_v8_20261101_codex.py \
  tests/test_p1_dev302_v9_task_greedy_prefix_physical_20261101_codex.py

10 passed
```

这些测试只证明代码路径满足各自局部合同，不证明 dev303 来源合格、反馈闭环完成、
物理机制成立或 P1 通过。
