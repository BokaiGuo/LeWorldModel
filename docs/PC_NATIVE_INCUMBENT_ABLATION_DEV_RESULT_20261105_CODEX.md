# 原生 CEM 已评分最优链选择：开发结果

按[开发协议](PC_NATIVE_INCUMBENT_ABLATION_DEV_PROTOCOL_20261105_CODEX.md)
运行；seed 43、44 是看到 seed 42 后追加的探索性扩展。
每次规划 300×10 评分；基线返回最后一轮分布均值，实验臂返回
当前规划调用内、所有已评分链中的最低预测代价链，不包含未经评分
的最终均值。

| seed | 原生均值成功 / 50 | 已评分最优成功 / 50 | 仅最优臂成功 | 仅均值臂成功 |
|---|---:|---:|---:|---:|
| 42 | 47 | 46 | 1 | 2 |
| 43 | 42 | 37 | 1 | 6 |
| 44 | 39 | 40 | 4 | 3 |

逐任务布尔在对应结果目录保存。关闭组件的 seed 42 适配器结果
与历史同配置原生 `eval_metrics.json` 逐字节一致；单元测试覆盖
同随机流输出、跨 batch 已评分最优、非有限代价拒绝及两次求解间
重置，共 4 项通过。它不构成跨设备或全部输入的等价证明。

`world.evaluate` 秒数分别为 seed 42：关闭 28.79、开启 29.58；
seed 43：历史原生 28.86、开启 30.65；seed 44：历史原生
30.44、开启 30.18。这些时间包含执行和视频保存，不包含全部
初始化，且不是同步 GPU 的完整成本。虽然候选评分次数相同，
选择开销和真实执行路径不同，不应称为完整等成本认证。

原始数据：[关闭 42](../le-wm-main/results/pc_native_incumbent_off_seed42_20261105_codex/eval_metrics.json)、
[开启 42](../le-wm-main/results/pc_native_incumbent_on_seed42_20261105_codex/eval_metrics.json)、
[开启 43](../le-wm-main/results/pc_native_incumbent_on_seed43_20261105_codex/eval_metrics.json)、
[开启 44](../le-wm-main/results/pc_native_incumbent_on_seed44_20261105_codex/eval_metrics.json)。

已评分链选择本身没有形成稳定收益；它不涉及响应生成，也不是
Planning Compiler 的性能实验。不同选择改变下一次物理观测，后续
规划候选不保证相同；没有实际候选分数与动作日志，因此不能把性能
差异单独归因于评分错误或特定实现问题。任务面板、normalizer、
checkpoint 来源边界沿用原开发实验，seed 同时改变任务与随机流，
不能当作独立确认。原 PC0–PC5 状态不改变。
