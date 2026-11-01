# Constructive moving-history source v1

## 冻结目的与范围

仅实施固定真实 nonzero history 的最小 source 协议可行性检验。使用原两 seed/placement
recipe 替代实际 warmup 路径；**不保留旧 post-warmup root，不是旧 task，不是 fresh roots**。
标记 `reused_recipe_calibration_not_fresh_not_same_task`。任务身份包含本协议、新 root/goal
与 blind payload SHA。P1 admission 始终 false，原 gate/模型/阈值/封存资产不变。

MAIN 已报告 archive 32H1 消融：8F 原复现逐位相同；平均 MSE
F=.00389331539、S=.00430292774、A=.00449873789、SA=.00383777611。
本实现未自行重算该报告。人工静止 joint history 没有 systematically 恶化；不得预设本协议
修复 failure 或将 history 静止断言为根因。此 source 不训练/编码/预测/选阈值，不读取旧 planner
或 physical outcomes；不增加比较 suite，不保证非零 commands 一定使 RGB 变化。

使用 experiment-code 的预声明、持久化及 mock fault 测试模式；不执行神经/物理或生成结果图。
当前 goal API 残留已撤销 review objective（blocked），未虚报完成以创建新 goal。

## 来源固定与真实动作

严格绑定原 prepare 文件（只读 recipe，不读取其后 outcomes）：

`results/p1_constructive_reachable_source_20261027_v1_prepare01.json`

外部 SHA256 `98ffb806018a270f5ef27ffa76be929393a74355c213bc274bf158c72d80b086`。
解码并通过原 source verifier 验证其原有 bindings/exclusions；直接继承两个 slots 的 root_seed、
reset_seed、sign、原 recipe 和资源上限。不会刷新 exclusions 后误将复用 seed 说成 fresh，
不会调用旧 `episode()`（它硬编码 zero）。原 prepare/tests/doc/env/source closure 与新三文件
都进入新 prepare bindings；verify 完整重建协议，拒绝变更任一 warmup/预算/recipe 字段。

每次 episode：指定 reset_seed 取 original goal；原 `sample_state` 使用 root_seed、proposal0、
attempt0，保持 jitter16、angle[-pi,pi]、edge[.15,.85]、surface_offset15；placement reset seed0。
with_target=False、relative=True、action_scale100、dt.01、control_hz10、render224 均不变。

MAIN 已确认预声明 raw schedule，按世界坐标直接送 `env.step`，不是 logits、不是 tanh 后的 .1：

| public warmup steps | float32 raw command |
|---|---|
| 1..5 | [0.1, 0] |
| 6..10 | [0, 0.1] |

每次独立 episode 都真的执行十步，并检查 native `latest_action` 的 dtype/shape/float32 bits。
历史 RGB 是实测 0/5/10，raw history 来自实际命令回执，不是对旧 payload 改写。
命令表在 episode warmup 前持久化；每步保留真实 command bits、返回标志、state、两 body dynamics。
不额外乘除 action_scale、不归一化、不插值、不缩幅、不按失败重选 seed。

## 一次构造 + 两次独立重放

1. 构造 episode 执行上述真实 warmup，在新 root 使用原几何 witness：radial=(object-agent)/norm，
   tangent=90°radial；前五 logit=.5radial，后五=.25radial+sign*.5tangent；两 packet cap=.75sqrt5，
   float32 tanh 得十条 raw commands。实际 logits/commands 在未来动作前 commit。
2. 十步 witness 的实际终态 state7/RGB 成为新 goal。原生早终止/truncate、guard 失败保留拒绝，
   不补动作、不造 goal。证书保留完整实际 warmup、witness、状态，均为 offline privileged。
3. replay0/replay1 各从独立 reset/sample/placement + 真实相同 warmup 重建，不恢复 poststate。
   history RGB/state float64/raw float32、两 body position/velocity/unwrapped-angle/angular-velocity/
   force/torque/sleep 精确一致后，才用原 native `_set_goal_state` + `_goal` 绑定新 goal。
   绑定前后全 snapshot 不变；没有 set_state/physics。没有序列化 solver warm-start caches，
   因而不声称全部隐藏缓存相同。
4. witness 重放逐步核实际 raw、state、RGB pixel SHA、full-body dynamics、goal state/RGB、
   原 native reward=-distance 和 terminated=success。仅到首次 native success；不强求十步全执行。
   root 已 native success 拒绝平凡任务；两个 replay 都成功才发布该 slot 的 blind payload。

固定分母始终2。科学/guard 拒绝记录 `rejected_fixed_candidate`，继续下个固定 slot；工程失败
fail-fast，未开始者保持 not_started。0/2 也可是诚实完成的 source feasibility 检验，非 admission。

## 持久化与预算

复用旧 runner 的 Store/Ops 设计但在本新文件独立实现执行循环；不 monkey-patch 封存模块。
每个调用 dispatch 前记 attempted、正常返回后记 completed；每次新 reset space 安装真实 step hook。
对 failed public/reset 保留已经尝试/完成的内部子调用，不退款。先 fsync 新鲜 PNG 与状态/body
receipt，再做 fresh RGB/canvas guard；拒绝帧保留，不再沿用旧 guard-before-save 缺陷。
每一 event hash-chain + fsync，NPZ/PNG/commit/report exclusive；无 resume、覆盖、删除或下载。
首错保留；close/hook restore/final seal/census/journal 次错独立列出。最终写盘错误不承诺 report 存在。

| 两 slot 全阶段上限 | attempted cap |
|---|---:|
| reset | 12 |
| public_step | 120 |
| physics space.step | 1224 |
| explicit_render / PNG | 126 |
| reset 内 goal_render | 12 |
| set_state | 24 |
| goal_bind | 4 |
| model/encoder/projector/predict/AD/训练/GPU | 0 |

正常完成 census 检查 physics=2*reset+10*public，goal_render=reset，set_state=2*reset。
native replay early success 可低于上限，不能以 cap 当实际计数。factory/close 各一次；factory
在 metered hooks 安装前，只允许已固定原 native constructor（不推进 physics），不声称任意工厂精确计账。
wall 由 MAIN 固定1..600s，每 dispatch 前检查，不能抢占单次卡住调用；建议外层 timeout。
总输出8MiB，final reserve128KiB，启动需208MiB；每次写前保留200MiB。cap 包括 PNG、NPZ、
journal、sourcepack、execution_binding。prepare envelope<200KB、解压元数据<=2MiB；无大闭包拷贝。

## 盲 payload 防火墙与尚未接通的下游

`sN_blind_inputs.npz` 严格仅：`rgb_history` uint8[3,224,224,3]、
`raw_action_history` float32[10,2]、`goal_rgb` uint8[224,224,3]。
新 `load_blind(path, external_sha)` 拒绝 zero/错 dtype/其他 schedule 或 privileged extra fields。
loader 不等于全 sourcepack 证书验证器：MAIN/新 adapter 必须先确认完整 sourcepack、两 replay、
journal 与 slot 状态，再仅向 selector 传 payload+SHA。不得把 results 目录/证书/commit 传 planner。
隔离是文件/API 合同，不声称 OS sandbox。witness 用于离线 source 构造/可达验证，不是 planner 候选。

本轮只有此 source+RAM/mock tests+本文档；**以下 sealed 消费者均未接通，不能直接运行新 source**：

- `p1_constructive_blind_encode_cuda_20261027_v1` 的 load_blind/APPROVED_INPUTS 为旧 zero/schema。
  NEW encoder adapter 需编码真实三帧，保留实际两包 raw 并按原 mean/std 归一化（不再除100）。
- `p1_constructive_blind_planners_cuda_20261027_v1.interface` 显式要求历史 zero，NEW wrapper
  需 source/root/goal/normalizer 绑定；底层 full-H2 数学 primitive 可复用但不能更换历史。
- `p1_constructive_plan_physical_20261027_v1.branch` 和 common-prefix physical 硬编码 warmup zero；
  NEW physical adapter 必须重建相同 recipe，真实执行这10条 raw，核实际 incoming tokens/history/root。
- selectedRGB/observedSuffix/commonPrefix/multiradius 等旧 wrappers 的 SHA/schema/原 task bindings 不适用。

此处不提前复制这些适配器，也不伪造 encoded task_id。source 实际由 MAIN 审阅运行后，再独立
授权推进下游。没有 direct-stall/阈值/独立确认资格，所有科学 admission 保持 false。

## MAIN 命令（交付 agent 不 prepare、不运行物理）

cwd `/home/gbk/code/LeWorldModel/le-wm-main`，新 output 路径不得已存在：

```bash
.venv/bin/python -B -m unittest discover -s tests -p 'test_p1_constructive_moving_history_source_20261027_v1.py' -v
.venv/bin/python -B -m experiments.spectral_reachability.p1_constructive_moving_history_source_20261027_v1 prepare --wall-seconds 300 --output results/p1_constructive_moving_history_source_20261027_v1_prepare01.json
.venv/bin/python -B -m experiments.spectral_reachability.p1_constructive_moving_history_source_20261027_v1 verify --prepared results/p1_constructive_moving_history_source_20261027_v1_prepare01.json --prepare-sha PREPARE_FILE_SHA
timeout 350s .venv/bin/python -B -m experiments.spectral_reachability.p1_constructive_moving_history_source_20261027_v1 run --prepared results/p1_constructive_moving_history_source_20261027_v1_prepare01.json --prepare-sha PREPARE_FILE_SHA --authorize-physics --output results/p1_constructive_moving_history_source_20261027_v1_main01
```

tests 只用 RAM/mock，fake dynamics 不是 PushT 成功证据。源/测试/文档 SHA 在交付时报告；MAIN
prepare 再绑定完整 closure。真实 source 尚未运行，不声称非零 history 改善任何模型/物理结果。
