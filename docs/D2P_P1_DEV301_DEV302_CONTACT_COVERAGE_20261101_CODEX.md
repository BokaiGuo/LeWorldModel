# P1 开发根推进：真实机制切片与接触覆盖失败

日期：2026-11-01

## 本轮结论

P1 尚未通过，但瓶颈已进一步收窄：

1. 在开发根 `301000001` 上，冻结模型生成的 spectral D2P 计划真实执行优于 GN 的物体位置结果，尽管模型终态分数反而偏好 GN。
2. 对该根做首动作共同尾部干预后，D2P 首动作本身不移动物体，并让 native distance 暂时变差；但它让同一九步尾部更早接触物体，并获得更低的物体位置误差。这是重配置机制的开发性证据。
3. 将相同干预规则事前冻结到未参与该规则设计的第二开发根 `301000002` 后，三种首动作加共同尾部均无任何接触，物体完全不动。机制未复制。
4. 第二根的逐步轨迹显示，模型动作把推杆带离物体接触区域。因此当前主要方法缺口不是 runner、动作重放或 CUDA 接线，而是 latent 评分与候选库没有保证任务相关的接触覆盖。

这些根仍属于 development；训练成员和全历史覆盖均为 unknown，不能称独立确认。

## 已排除的工程问题

- 新 queue envelope 的 `development_seed` 与旧 planner schema 不兼容，已通过只移除该元数据的 adapter 修复；任务数组与 task ID 不变。
- 两个开发根均完成真实 reset、十步零动作 warmup、RGB 0/5/10、同实例 goal RGB、精确 saved-only 验收。
- 两个开发根均在主机 RTX 5060 Ti 上完成冻结模型 CUDA 编码和 GN／spectral D2P 规划；模型 hash 前后不变。
- 物理 runner 对每臂使用独立 native PushT 实例，逐位核对计划动作与实际动作，保存 RGB、状态、接触和停止状态；无 padding、无 outcome 选动作。
- seed `301000002` 的 planner saved audit 首次因手工输入错误 report SHA 而 fail-closed；随后使用实际 SHA 在新文件中通过。拒绝产物保留，不覆盖。

## 开发根 301000001

模型分数：

| 方法 | 模型终态代价 |
|---|---:|
| GN | 1.393353 |
| spectral D2P | 1.685180 |

真实十步执行：

| 方法 | native success | 物体位置误差 | 角度误差 | 接触步 |
|---|---:|---:|---:|---|
| GN | 否 | 256.9297 | 0.4676 | 无 |
| spectral D2P | 否 | 184.5963 | 0.9193 | 2,3,4,5,6,10 |
| zero | 否 | 256.9297 | 0.4676 | 无 |

模型排序与真实物体位置排序相反，继续支持“latent 评价失配”而非单纯执行代码错误。

首动作＋共同九步 D2P 尾部：

| 首动作 | 首步物体位置变化 | 首步 native distance | 最终物体位置误差 | 首次接触 |
|---|---:|---:|---:|---:|
| D2P | 0 | 549.3517 | 184.5963 | 2 |
| zero | 0 | 484.9145 | 210.9010 | 3 |
| GN | 0 | 531.1793 | 208.5175 | 3 |

这表明 D2P 首动作在没有立即移动物体、且 native distance 暂时恶化的情况下，改变了同一后续动作的真实效果。但该诊断是在看到原执行后设计的，且没有成功，不能直接打开 P1。

## 开发 holdout 根 301000002

模型分数：

| 方法 | 模型终态代价 |
|---|---:|
| GN | 1.679085 |
| spectral D2P | 1.694463 |

事前冻结相同首动作规则后：

| 首动作 | 首步物体位置变化 | 最终物体位置误差 | 接触步 |
|---|---:|---:|---|
| D2P | 0 | 225.5194 | 无 |
| zero | 0 | 225.5194 | 无 |
| GN | 0 | 225.5194 | 无 |

因此单一首动作规则未复制。该根的失败是“候选／尾部没有覆盖接触区域”，不能解释为 P1 理论已否定，也不能用第一根的开发性成功替代。

## 下一方法修改

下一版本不应继续复用开放环尾部或只调 latent gate。应在冻结模型候选中加入 fail-closed 的 RGB 接触覆盖项：

1. 从当前真实 RGB 分别提取推杆与物体的 renderer-specific mask；分割失败即拒绝，不回退到模拟器状态。
2. 候选生成仍由冻结模型动力学结构提出；选择时增加“推杆是否接近物体有效作用区域”的独立项，不把它混入 native success。
3. 执行短前缀后刷新真实 RGB，再以相同预算重规划尾部。
4. 与 zero／GN／随机前缀共用同一反馈和后缀优化器；结构计算计入预算。
5. 在新的 development holdout 上冻结验证后，才能进入来源可审计的独立确认。

## 关键证据

- `results/p1_dev301_blind_planners_saved_20261101_codex_MAIN01.json`
- `results/p1_dev301_blind_plan_physical_saved_20261101_codex_MAIN01.json`
- `results/p1_dev301_first_action_common_tail_saved_20261101_codex_MAIN01.json`
- `results/p1_dev302_blind_planners_saved_20261101_codex_MAIN02.json`
- `results/p1_dev302_first_action_common_tail_saved_20261101_codex_MAIN01.json`

P1 admission 保持 `false`。未放宽 gate，未按物理 outcome 替换同一实验内的动作。

## 追加：dev302 RGB 接触覆盖诊断

主线程随后增加了一个明确标记为 outcome-informed 的开发诊断：

- 前五步仅使用每步真实 RGB，通过 renderer-specific 推杆／物体 mask 将推杆朝最近物体像素移动；
- 后五步复用同一个封存 D2P 尾部；
- 对照为零前缀加同尾部，以及原封存 D2P；
- 不使用模拟器状态生成动作，不新增神经调用，不改变 P1 gate。

第一次执行在首动作后 fail-closed：运动中的蓝色推杆精确色 mask 被抗锯齿分成两个紧邻分量，旧的“必须单连通”规则错误拒绝。失败产物保留。代码随后改为只接受最多两个分量、且联合包围盒不超过 20 像素的紧凑运动 sprite；远距离碎片仍拒绝。该项属于已确认的代码／观测协议问题，不是理论负结果。

修复后 saved-only 验收通过：

- prepared SHA：`615e45c2b0a7e9409ca22f50f2729351564e1f401ea6c81ea6ae542af71d3dc1`
- physical report SHA：`aacb4d296e0a9aacba9bff9e1803bc82d48345beaddc2035b1e72128fd74e0a0`
- saved receipt SHA：`80d1b73886ad8e0461be21f8991958022979b07d06bdcc758777b483f8c74f8f`

RGB 前缀将推杆到最近物体像素的距离从 `20.56` 降至 `8.67` 像素，但五步内仍未发生 native contact；随后封存旧尾部立即把推杆带离物体。三臂物体均未移动，最终物体位置误差均为 `225.5194`。

因此本轮进一步排除了“只缺一个朝物体移动的短前缀即可复用旧尾部”。当前最具体的实现缺口是：前缀后必须基于新真实 RGB 重新规划尾部，且候选／后缀要显式覆盖接触进入，而不能把失效的开放环尾部接回去。该结论仍是开发诊断，不构成独立机制确认或 P1 admission。

新增证据：

- `results/p1_dev302_rgb_contact_feedback_20261101_codex_RUN_MAIN01/`：保留的 fail-closed 首次执行
- `results/p1_dev302_rgb_contact_feedback_20261101_codex_RUN_MAIN02/`
- `results/p1_dev302_rgb_contact_feedback_saved_20261101_codex_MAIN01.json`
