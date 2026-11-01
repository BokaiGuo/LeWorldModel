# P2 matched composition：实际搜索 runner 与交付边界

只新增 `p2_matched_composition_20261027.py`、对应测试和本文。不修改任何历史源码、
测试、报告或 seal；不训练、不学 readout、不选 outcome 根、不运行 GPU。
P1 仍未通过，P2 admission 永远 false。另已读取最新
`D2P_REPRESENTATION_RUNTIME_AUDIT_20261027.md`：64 程序 direct/正确官方
rollout maxerror=0，12 张真实图 projected embedding 最大差约 2.15e-6；
它支持当前 recurrence 接线，不证明 Q 能力与物体控制一致。

## 已核对的缺口与实现

已读取 P2 comparison contract、composition CUDA integration、kinodynamic search、
CUDA replay、budget guard、当前 PushT P1 `_advance` / `_suffix_terminal` 和
representation runtime audit 源码。

原 search 跨所有 visited depth 选 best，不能满足固定 H2；已有 64 程序重放
只是接口完整性，不是组合优势。新模块实际执行两个搜索 arm，而不是再次校验声明：

| arm | 默认 planning | 选择 | 独立验证 |
| --- | --- | --- | --- |
| composition | 16 个首段 + beam4 × 16 个后段 = 80 packets | 只选 64 个完整双段中的 finite best | 逐段 2 + 同 root 完整重放 2 |
| flat_search | 预先 seed 打乱 256 个有序双段，完整评估其中 40 个 = 80 packets | 只选 40 个完整双段中的 finite best | 同上 |

两 arm 的候选都恰为 10 raw env steps；不补零、不裁剪、不选择 depth1。
flat 不使用中间 metric 或 prefix cache；beam 的首段分数仅决定 frontier。
两者都使用同一 bank、root context、goal metric、normalization、冻结模型。
这是一个明确的 flat search baseline，不冒充官方 SRA、AP 或 CEM。
flat 顺序由显式非负 seed 决定，不可按结果挑 seed。

root-anchored recurrence 沿用现有正确 `CudaReplay`，新专属类明确标识其语义：
三帧 `[z(t-2),z(t-1),z(t)]` 与两段 between-state 动作；拼入新 packet 后，
恰有三段 action tokens，预测一次 successor。它与当前 P1 incoming history
append 后取末三段等价；旧 adapter 本身不需要修复。不可把完整 incoming history
未经转换直接塞入 between-state witness，也不能添加 bootstrap prediction。

选中后独立再生成 intermediate/terminal，再从 root 完整 replay，精确检查三个
terminal 一致，然后复用 witness v2 绑定 raw actions、全部上下文和 provenance。
composition 保存两条边；flat 保存一条完整动作边。`kind=progress` 只是 witness
类型声明，绝不表示物理进展已证明。

## 可复用入口与调用约束

Python cwd `/home/gbk/code/LeWorldModel/le-wm-main`，解释器 `.venv/bin/python`。
模块提供 `schedule()`、`run_cuda_pair(...)`、`run_fixture_pair(...)`。
后者仅用于非神经测试，任何结果保留 `fixture_only=true`。

主线程在自身已验证／预封存的 CUDA runner 中调用：

```python
from experiments.spectral_reachability.p2_matched_composition_20261027 import (
    RootAnchoredCudaReplay, run_cuda_pair,
)
report = run_cuda_pair(
    replay=RootAnchoredCudaReplay(frozen_eval_model, cuda_device),
    primitive_bank=frozen_raw_one_packet_bank,
    root=full_witness_root,
    provenance=full_six_sha_provenance,
    action_domain=raw_witness_action_domain,
    target_metric=saved_goal_metric,
    beam_width=4,
    flat_seed=predeclared_seed,
)
```

bank 必须每项 `[5,2]`、有界有限；root/context/provenance/domain 在模型调用前检查。
所有输入必须由主线程在 outcomes 前外部封存；返回 declaration SHA 仅绑定内容，
不证明时间或无泄漏。本模块不自动读取旧 outcome 报告、不挑根，也不自动启动模型。
`run_cuda_pair` 拒绝普通回调、CPU adapter 和子类替换；冻结、eval、CUDA 检查
继承现有 adapter。主线程负责 checkpoint/source seals、before/after 参数与 buffers、
输入完整性和运行产物独占写入。另提供下节带校验的校准 CLI，不绕过旧 seal。

## 主线程可直接运行的 CUDA 校准命令

```bash
cd /home/gbk/code/LeWorldModel/le-wm-main
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m experiments.spectral_reachability.p2_matched_composition_20261027 \
  --compilation results/pusht_d2p_metric_program_preflight4_v1_20261026_cuda/report.json \
  --policy /home/gbk/.stable_worldmodel/checkpoints/pusht/lewm \
  --output results/p2_matched_composition_20261027_cuda_v1
```

本任务没有执行该 CUDA 命令。`--help` 已验证。CLI 固定纳入原四个校准根，
不会选根或读取 physical outcome；沿用旧 Q **仅做接口/成本校准**。
它核验 compilation/checkpoint/config/旧 source seal/input bindings，在 CUDA 分配前
独占写入新 protocol 和源码内联快照，打印 protocol digest；结束再检查源/输入/
参数/buffers/eval。旧 seal 不通过即停止，不重封绕过。正常 4 根 ×168=672
predict calls/samples，输出 actual 模型入口计数、峰值 allocated/reserved 显存和耗时。
会写 `p2_matched_sources.json`、`p2_matched_protocol.json`、每根
`p2_matched_root_N.json`、`p2_matched_report.json`，均只用独占新文件模式。
若根内可恢复失败则保留失败账本并继续另一 arm/根；fatal 环境/源码校验错误可
抛异常并保留已有文件，不保证每种 fatal 路径都有最终报告。

现场磁盘约 2.5 GB、单 RTX5060Ti 16GB 由主线程调度。CLI 不下载、不复制
checkpoint、不保存图片或 tensor；输出预计小于 100 MB（保守估计，非实测）。
源码快照按每文件 1 MiB 限制，实际运行前冻结；后续修改本模块会使该次 current
source 校验失效，主线程应在最终交付版本稳定后运行。已完成输出目录不可复用。

## 预算及失败

每 packet 在 dispatch 前记 attempted work，保留异常类型、阶段、路径和耗时；
失败、metric 非有限、验证失败均保留，不能退款或用替补 winner 隐藏。
若 frontier 失败不足 beam、终段失败或验证不完整，返回 incomplete_or_failed_pair，
即使两边碰巧花了相同 samples 也不宣布匹配完成；不做无意义 padding 来凑成本。

CUDA 入口另包裹真实 `model.predict` 和 `action_encoder.forward`，在进入函数前
记录 actual calls/samples，包括抛异常的调用，按 arm/phase/path 绑定。
dispatch 与真实入口不一致或入口失败时，prediction_budget_matched=false。
所有 override 在异常路径也恢复。predict 内部未通过该属性的递归、compiled/缓存
调用不在该 instrumentation 覆盖范围，外部不能据此声称所有底层算力已测量。

当前预算边界是 **prediction matched only**：每根正常路径 2 × (80+4) = 168
predict samples/calls，168 action_encoder calls。不同搜索的 metric evaluations、
attempted expansions、失败成本和实际 wall time 全量列出，不强行声称相等。
在线 image encoder/backward/optimizer/physics 均未调用；上游 bank 构建和 image
encoding、FLOPs、峰值显存、wall-time cap 未测量/未实现，明确保留 unmeasured。
因此 `fair_total_budget_comparison=false`，输出 **不能直接通过旧严格 P2 contract**，
更不能把调度账本转成旧合同的完整预算或把未知 FLOPs 填零。

GPU 未运行，暂无实测耗时或峰值显存。主线程若安排单根 pilot，串行 batch1、
官方约 69 MiB 权重、168 次 predict；建议预留单卡 4 GiB 和 10 分钟作为保守
调度额度（不是实测值或代码强制 cap），先记录实际峰值/耗时再扩大根数。
不需要训练资源。四根 CLI 暂建议同样预留 10 分钟作 pilot 调度额度；不是承诺。
本次交付 API 与 CUDA CLI，没有自行申请/占用 GPU。

## 验证与剩余缺口

非神经回归命令：

```bash
cd /home/gbk/code/LeWorldModel/le-wm-main
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q -p no:cacheprovider \
  tests/test_p2_matched_composition_20261027.py \
  tests/test_d2p_kinodynamic_program_search.py \
  tests/test_d2p_composition_witness_v2.py \
  tests/test_pusht_composition_cuda_replay_v2.py
```

测试只证明控制流、长度、确定性、失败记账和非神经 witness 可复算。
2026-10-27 最终补充 CLI provenance 回归后，专属测试 21 项、上述联合回归
117 项通过。CLI 集成测试只读取真实封存的四根 compilation/输入绑定，模型加载、
CUDA allocation/显存查询、pair 执行均由 metadata fake 替换，绝非 CPU 神经推理。
覆盖六项 provenance、protocol/source/bank 摘要、incoming3→between-state2 转换、
坏 root/history/goal/metric 摘要及错误动作历史在 CUDA 前拒绝、独占输出拒绝覆盖。
源码镜像覆盖 API、adapter、witness、search、metric、loader、packet normalization、
本测试/本文档以及 config 全部非 torch 构造模块，并逐文件复算镜像内容 SHA。
torch/transformers 等第三方框架依赖的完整源码和二进制不在该镜像完备性声明内。
剩余：P1 实证；预先外部封存新根/goal/bank/seed 与真实 CUDA 集成；完整算力测量
与共同停止预算；强无组合 SRA；全部候选物理执行、非贪心机制与配对不确定性。
没有新物理效益、没有组合优势、没有研究闭环。
