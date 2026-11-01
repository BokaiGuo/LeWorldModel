# 四 observed-prefix CEM：主机物理与独立复算

这是固定两个已知根上的四个校准分支，不是新根确认、完整 CEM-MPC 或等总算力比较。

## 工程问题已修复并真实重跑

v1 的真实 prepare 因 540,309 字节超过 512 KiB 上限而拒绝。v1 保留，不把它计作控制负结果。v2 只改 NAME 与 prepare cap=1 MiB，不改动作、Q、成功阈值或输出上限。MAIN 源码 diff 审读、80 项测试通过。

实际 prepare SHA256：`48a02bbb4b77efb0264ac2ce275adddb2c5390164fbdd3c7a3b7173b08097219`。主机物理运行完成，报告路径 `le-wm-main/results/agent_observed_CEM_physical_20261028_v2_MAIN01/report.json`，SHA256 `dd6b082c20f1d853c7500e806e99165969214b4c7d3953770a4a38dd53427dd4`。独立 strict consume 退出 0，`saved_physical_verified_not_admitted`。

四分支全部重建原 seed/placement/warmup10/prefix5，核对每支原 17 个 paired RGB/state/body 帧及 goal 后，才执行已封存 CEM 后缀。零工程失败、无换根或重试。实际 79 public steps、806 physics steps、87 explicit renders；一支在后缀第 4 步原生成功立即停止，其他三支执行完整 5 步。没有补步、旧 endpoint 回填或神经 CPU 降级；本物理执行没有新神经调用。

## 独立数值结果

MAIN 新增 `analyze_observed_CEM_MAIN_20261028_v1.py`，10 项测试通过；调用完整 strict consumer 后按原 selection/goal/prefix join，独立复算 native7、agent+object joint XY 及原 native angle 成功条件。分析报告 `le-wm-main/results/analyze_observed_CEM_MAIN_20261028_v1_MAIN01.json`，SHA256 `045a09239fa03d62abb6a0451adacbf282dc90598cd2ae0387b65a51b2632e33`，内嵌完整严格验收结果。

| 根 | 原 prefix | 历史后缀 joint XY | CEM 后缀 joint XY | 变化（像素） | CEM 原生成功 |
| --- | --- | ---: | ---: | ---: | --- |
| 0 | line_search_GN | 89.03088 | 27.91103 | −61.11984 | 否 |
| 0 | full10_conditional_D2P | 56.84685 | 11.81156 | −45.03529 | 是 |
| 1 | line_search_GN | 213.11568 | 136.66273 | −76.45295 | 否 |
| 1 | full10_conditional_D2P | 56.01562 | 31.67499 | −24.34063 | 否 |

四项均改善，平均 joint XY 变化 −51.73718 像素；历史 0/4 原生成功，新 CEM 1/4。原生成功位置条件包含 agent+object，不能以 object-only 替换。state7 distance 含速度，成功条件不含速度，故成功分支 state7 distance=79.07236 不矛盾。

## 解释与未关闭要求

此结果说明普通采样后缀优化也是重要强对照，不能只把新的控制收益归于谱程序。CEM 与历史后缀调用数不同，没有等总算力证据。四分支不是四个独立根，也没有统计确认。

成功的 D2P prefix 在第一段已经改善 joint XY（−118.34251 像素）；另一 D2P prefix 也改善（−148.42584）。因此本次成功不能证明“暂时退步以改变未来控制能力”。P1/C2 仍未通过，不能由该结果开放组合机制或动态调度科学结论。

八 teacher anchor 的另一套 CEM 已完成主机 CUDA 520 predict 及 own-consume，持久回执实际写入 `le-wm-main/results/agent_CEM8_release_20261028_v1_MAIN01/receipt.json`，SHA256 `878fb418d92fc8c8ec3fc3bf4f2bd0d7f40f5a50322280bcb7f91fd923c6f609`，完整绑定 9,902 个输入。它尚不是本表的物理分母，不混合两套 cohort。
