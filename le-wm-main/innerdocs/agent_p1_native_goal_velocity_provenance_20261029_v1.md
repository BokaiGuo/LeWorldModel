# P1 native goal agent velocity：来源与语义核查

## 结论与边界

P1 的两组非零目标速度来自**构造轨迹末端的真实 agent velocity**：固定 witness 的第 10 个 public step 返回完整 state7，构造器把这整个 state 原样保存为新 goal7；随后原任务重放和 P1 物理执行将该完整 goal 绑定到环境。它不是把随机 reset 目标速度直接带入 P1，也不是把 block velocity 误记作 agent velocity。全 40 个 P1 物理槽的 goal7 均与对应根的构造末端记录相等。

按已封存实现，这是选择完整末端状态作为 goal 的协议后果，不是观察到的 reset/record 违约；但这些证据不能证明研究设计者曾独立考虑并有意选择**非零速度作为任务目标**，也不能据此裁决原任务指标是否符合本研究想测的物理目标。非零本身不构成 bug。下述均为后验 source-bound 核查，不改原 success、reward、P1 gate，不构成独立根或确认实验。

## 完整已用根／条件对照

40 槽固定为 2 known roots × 2 renderer 条件 off/on × 2 前缀角色 Z/S × 5 library；每个 root 的四个 condition/role 组各有 5 槽，四组 goal7 相同。直接逐行比较 physical evaluation 的 40 个 `physical_goal` 与源日志相应 construction `native_return` 第 10 步 state7，结果 **40/40 相等**；两条 construction `fresh_observation` step 20 的 snapshot agent velocity 也分别与这两维相等。此处的 step 20 是 10 个 warmup 后的绝对 public 步，`native_return` step 10 是 witness 段内编号。

| root | 初次 seeded reset goal velocity | 固定 witness 末端 / P1 goal agent velocity | 构造 seed；placement root seed |
|---|---|---|---|
| 0 | (0.0, 0.0) | (60.30532455444336, -181.7025604248047) | 202798201；202798101 |
| 1 | (0.0, 0.0) | (1.9956680536270142, 188.54302978515625) | 202798202；202798102 |

这不是独立样本量 40：同根的 20 槽共享同一个目标。sourcepack 记录两个 root 均 `reachable_native_replayed`、两次 replay native success，均执行 10 步；这是构造 witness 的可达性记录，不可归因于 P1 模型。

## 逐段来源

1. 安装的 PushT `env.py` 第 14–18、114–119、237–281 行：默认 variation 只采样 agent 起始位置、block 起始位置和角度；agent velocity 默认值为 `(0,0)`，不在默认 variation 列表。未提供 `goal_state` 时，reset 的 goal 是单独采样的位置/角度加当前 variation velocity；显式 `goal_state` 可覆盖。`spaces.py` 第 13–53 行说明 reset 初始化并只更新配置的 variation。实际两个 construction 的 seed reset 日志均记录原 goal 速度为零，见上表。
2. `p1_constructive_moving_history_source_20261027_v1.py` 第 190–225 行：首次 seed reset 之后，以固定 root seed 计算 placement，第二次 reset 显式传入原 goal 和请求 state；以固定 raw warmup10 建立根，再按固定 radial/tangent witness 生成命令（第 116–122 行），在 witness 执行前落盘 commitment。第 231–250 行把执行 witness 后 `obs['state']` 完整复制成 goal，保存 `goal`、`future_states` 与 goal RGB。日志两条第 10 次 witness 返回 state 与上表完全相同；对应事件 SHA 分别为 `5ac4578476edf7505b389182e605b5f85b2df7218b1a769fc5b8c90fcfd79a9c`、`dfa37a4df93b21f762acc532ebb29472a04e1a0be9809b7e702987ca9a00c110`。
3. 同一源文件第 229–240、293–307 行：重放沿原 reset/placement/warmup，显式绑定 construction goal7 和 goal RGB，校验回放 state/snapshot、goal 与 native reward/termination。sourcepack 保存两次重放结果。该构造是固定 witness 后验端点形成的目标；不是宣称从无限候选或新独立根无选择采样。
4. `p1_moving_target_overlay_source_20261027_v1.py` 第 267–282、305–345 行明确区分 warmup 的 native reset goal 与之后绑定的 constructed goal；绑定验证物理 snapshot 和 renderer/goal pose 不变。`agent_p1_teacher_cross_library_physical_20261028_v1.py` 第 326–345 行重新执行原 reset/placement 与 warm10、验证 root/history，然后绑定 root 的 goal7/goal RGB。故必须区分初次 reset goal 与任务期最终评估 goal。

## 速度、单位与指标

`env.py` 第 360–368 行的 state 顺序是 agent XY、block XY、block angle、agent velocity XY；`_set_state` 第 514–527 行尽管局部变量叫 `vel_block`，实际赋给 `self.agent.velocity`。source snapshot 直接保存 agent 和 block 两个 body 的 velocity，上述两末端对照核对的是 agent body。RGB 是位置/形状渲染，不是目标速度通道；记录的 `goal_proprio` 则显式包括 goal position 与 velocity，不能将二者混同。

速度单位是 Pymunk 世界坐标单位／仿真时间单位：坐标对应 512 宽世界画布，`dt=0.01`、control frequency 10 Hz，每个 public step 10 次物理小步（`env.py` 第 40–49、310–330 行）。因此可按代码约定称世界画布单位／仿真秒；没有证据可换算为物理米／秒，也不应把 224 像素渲染分辨率直接当速度单位。`_set_state` 本身还调用一次 `space.step(dt)`；其后记录的 observation 是该步后的状态，不能把 placement 输入值直接当 reset 后状态。

原 `eval_state`（第 347–355 行）success 只检查前四维合并位置距离及角度阈值，不含 velocity；`state_dist=np.linalg.norm(goal_state-cur_state)` 用全部七维原始差值，reward 为其负值。故 velocity 项是 **当前 agent 速度相对非零目标 agent 速度**的误差，不是速度大小，不是 block 速度；更静止未必更接近目标。距离混合位置、原始角度差及速度量纲；success 的角度规则也不能自动等同 reward 的原始角度分量。静态目标图不能直接编码目标速度，这说明图像目标和 native7 评价存在可解释性限制，但不能单独证明任务不可解或协议出错。

## 封存证据与未覆盖项

以下 SHA 为本次直接计算并与 P1 evaluation `bindings` 对照的文件（journal/sourcepack 亦在该闭包内）：

- installed `stable_worldmodel/envs/pusht/env.py`：`c6d2104adc368ff1c352edc0e35ac74da9f07fda3a748ddbef0ba6b7a2d1daa2`；`stable_worldmodel/spaces.py`：`73079b2a66dace087abc7c417691e3cc339887866915553e21d0936a362229eb`。
- `experiments/spectral_reachability/p1_constructive_moving_history_source_20261027_v1.py`：`9fb76ecfa054f878ab9778fc29c59bc708728b240dd2cbc3060b166e52622f55`。
- `experiments/spectral_reachability/p1_moving_target_overlay_source_20261027_v1.py`：`768a45d127ad01a0e3ee8b55a6c7c535c92134211af10914e3e99bb4674e472b`；`agent_p1_teacher_cross_library_physical_20261028_v1.py`：`5367287a375cf89e45af2675997aa682ec63464d24b2b01ab413c5863882d8a4`。
- `results/p1_constructive_moving_history_source_20261027_v1_main01/journal.jsonl`：`f30790272a2d2bd1eaa0d533e1767577c69c8608d98b4fb3a3705bd9b70490e9`；同目录 `sourcepack.json`：`c2d8d573f94884ffc081163f25f2bfa269f98ffe8155887704109219910ffec6`。
- `results/agent_p1_model_library_selector_evaluation_20261028_v1_MAIN01.json`：`d8b3bedd38e9f9a56d0156f2452d5a20f363acbdad42ea5e3ee4d91f9626ff67`。

本次直接核文件 SHA、提取日志并逐行对比 40 槽，没有重跑上游 hash-chain/full strict、环境、模型或物理；沿用已封存 sourcepack/evaluation 的既有闭合。未核实历史作者对 velocity 分量的研究意图，未对其他 seed、其他环境版本或真正独立根外推。若要进一步判断指标设计，应在独立预注册对照中保留原 goal7/原 gate，并将速度语义及任何另设目标明确作为独立诊断；不能根据本批表现回改原结果。
