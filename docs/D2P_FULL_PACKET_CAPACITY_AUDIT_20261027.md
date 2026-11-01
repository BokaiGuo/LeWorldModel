# 完整包内动作空间与评分归因：2026-10-27 实证

## 结论边界

已确认旧 D2P 局部编译器把每个 5×2 动作包限制为同一个二维命令重复五次。
新独立版本去掉此限制，在四个既有校准根上完成主机 CUDA 规划与全部80条计划的
物理尝试。此限制真实存在，但解除限制尚未解决 P1：八组模型选择仍全部等于
greedy，完整逐步动作相对重复动作的物体 pose 有两根改善、两根变差。
不能将模型响应秩增加称为真实可控性提升，也不能将这批结果称为理论不可行。

本轮没有训练、改变checkpoint、引入readout、按真实结果重新选择动作或开放新根。
官方原 CEM 接口本来支持完整时序动作，限制针对被审计的 D2P 编译路径，不能
扩大为“整个项目不支持完整动作”。

## 真实执行与保留的失败

规划：`le-wm-main/results/pusht_full_packet_compiler_20261027_cuda_v1/report.json`。
7项门全部通过，572个实际CUDA predict样本、88次向量化Jacobian反向调用，
对应16896个标量输出VJP方向；后两项不能冒充CUDA kernel或FLOPs。
80条候选独立模型重放均逐值一致；四根模型Q响应秩从2变为10。
常量/完整chart的链式法则最大误差不超过2.385e-7；完整chart有限差分相对误差
约2.55e-4至2.67e-4。

物理：`le-wm-main/results/pusht_full_packet_physical_20261027_cuda_v1/report.json`。
保留原`integrity_fail`与`encodings_complete=false`，不能改成全绿实验。
实际791步、80条分支；77条完整十步，3条因RGB guard分别止于6、6、9步。
80个第五步prefix均可用；77个第十步终态可用，共157次实际CUDA编码。
三个缺失终态均有中止收据，不是编码失败；禁止拿提前终点填补固定十步终态。
所选、greedy、zero三类比较分支本身均完成十步。

| 已知根 | constant所选pose | full所选pose | constant零首段pose | full零首段pose |
| --- | ---: | ---: | ---: | ---: |
| 0 | 59.02848 | 63.92373 | 46.01346 | 43.95703 |
| 1 | 120.02555 | 113.29189 | 74.54011 | 67.10963 |
| 2 | 93.05557 | 94.59967 | 57.31855 | 59.90009 |
| 3 | 99.35120 | 92.81412 | 27.31225 | 27.29609 |

全部越低越好。这是已用根的描述性证据，不是正式独立泛化比较。
full所选官方state distance还在四根全部更差，不能仅报pose中有利两根。
相同chart参数范数约束的是pre-tanh logits，不是相同raw能量；新共同后缀球约束
也不同于历史无界后缀，不能把本轮constant臂称为旧实验的精确复刻。

## 主线程新增评分层级定位

独立离线审计：
`le-wm-main/results/pusht_full_packet_objective_audit_20261027_main_v1/report.json`。
对80条已执行见证按root SHA/arm/candidate连接，分别比较：

1. 线性后缀近似与非线性模型终态Q；
2. 非线性模型Q与真实RGB编码Q；
3. 真实RGB编码Q与物体pose、官方state distance。

把原objective的线性末段代价替换成已保存非线性末段Q，同时保留原首段和ridge项，
八组诊断winner全部不变。因此本批“只把线性末段换成非线性末段”不足以修复选择。
此比较是事后定位，不是已部署新方法；没有真实结果重新选择。

模型Q与真实编码Q每组都有反转（3至9对）；真实编码Q与物体pose每组也有反转
（20至30对）。每组共有45个候选对，中止导致涉及终态的9对缺失，明确单列。
这些候选对高度依赖，不能作为独立样本统计显著性或跨根泛化。
由此可见，预测误差与任务度量不一致均仍是具体缺口，不能统一归为理论失败，
也没有证明唯一代码根因。

## 复现与验证

从`le-wm-main`运行以下命令；结果目录必须为全新路径。神经命令在主机CUDA执行，
不可在不可见GPU的环境退回CPU。

```bash
.venv/bin/python -m experiments.spectral_reachability.pusht_full_packet_compiler_20261027 --compilation results/pusht_d2p_metric_program_preflight4_v1_20261026_cuda/report.json --policy /home/gbk/.stable_worldmodel/checkpoints/pusht/lewm --output results/pusht_full_packet_compiler_20261027_cuda_v1
SDL_VIDEODRIVER=dummy .venv/bin/python -m experiments.spectral_reachability.pusht_full_packet_physical_20261027 --planning-report results/pusht_full_packet_compiler_20261027_cuda_v1/report.json --output results/pusht_full_packet_physical_20261027_cuda_v1 --encode-terminals
MPLCONFIGDIR=/tmp/lewm-matplotlib .venv/bin/python -m experiments.spectral_reachability.analyze_pusht_full_packet_20261027 --planning-path results/pusht_full_packet_compiler_20261027_cuda_v1/report.json --physical-path results/pusht_full_packet_physical_20261027_cuda_v1/report.json --output results/pusht_full_packet_analysis_20261027_main_v1
.venv/bin/python -m experiments.spectral_reachability.pusht_full_packet_objective_audit_20261027 --planning-path results/pusht_full_packet_compiler_20261027_cuda_v1/report.json --physical-path results/pusht_full_packet_physical_20261027_cuda_v1/report.json --output results/pusht_full_packet_objective_audit_20261027_main_v1
```

物理命令原退出码1已解释为缺失固定终态，保留原报告；其他三条实际完成退出0。
chart/编译/物理/原分析四文件联合118 tests passed；新评分定位另4 passed。
这只是非神经逻辑回归，不代替以上CUDA/物理证据。
分析Figure_1与Figure_2已逐张目视核查：标签可读、无裁切、蓝橙对比，未隐藏不利根。
全部运行来源封存，不得原位改写其源码、测试或报告；后续修复需新版本。
