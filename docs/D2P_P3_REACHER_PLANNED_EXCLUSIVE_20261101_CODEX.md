# P3 Reacher 模型规划候选的单节点执行器

入口 `experiments.spectral_reachability.codex_p3_reacher_planned_exclusive_20261101`
是独立文件，不覆盖零动作探针或既有调度器。“独占”指专用代码和新建输出目录；
它不锁定 GPU 或阻止其他进程使用 GPU。

## 设计

沿用 P0 已有的根观测条件 CEM、latent 目标成本、动作边界及 time-major
归一化/反归一化：先优化 3 个 token、即 15 个物理动作，再将同一候选分别交给
固定 5 步和动态 5/10/15 步执行。CEM seed 为根 seed 加 2000；样本数、
迭代数、elite 数和动态阈值必须由操作者事先指定。两个臂独立重建根与规划，
相同种子不证明根状态或候选相同；比较前必须核对根、候选、checkpoint、
动作统计及来源，不相同则不能称为配对。GPU 数值非确定性也可能破坏一致性。

此设计比较的是一个候选上不同的执行长度，固定臂不会因动态臂额外计算自动
获得等量计算；每臂的最终时域也可能不同。单 chunk 不构成任务完成率或
闭环性能实验。候选可能为零或无法到达目标；仅接线不保证任务可规划。
模型的预测改善和 persistence 差异不是物理安全证明或真实不确定性。

## 来源和计量

输出目录独占创建。`protocol.json` 在加载模型和创建环境前记录参数、已指定
源码、checkpoint 全部文件及动作统计的 SHA-256。`runtime_manifest.json`
记录 Python、平台、包版本和当时已加载项目 Python 源码的哈希；这些源文件
和 protocol 中的文件在规划前、规划后及执行后复核。`receipt.json` 另记录
最终已加载项目源码哈希，但晚加载文件未预封存、第三方源码及运行环境未
完全封存，不能称为完整供应链证明。已有统计 receipt 的训练语义声明也不是
checkpoint 训练数据和 normalizer 全部来源的独立证明。

`planning.json` 记录 CEM 配置对应的执行结果、seed、同步后的规划耗时及
非零动作元素数；`action_binding.json` 在调度预测和候选物理步之前记录并
重建物理动作及归一化 token。`receipt.json` 分阶段记录 encoder、
action_encoder、predict 的尝试与返回、输入样本量，动态扩展独立 meter，
包括 warmup 的环境步及候选逐步 reward/终止标记、CUDA 峰值和失败信息。
规划失败时调用计数仍保留；异步异常和进程被强杀不保证完整记录。未测
FLOPs、能耗、CPU 峰值或所有模型内部算子，各计数轴不可直接相加。单步
reward 不是成功率，也不替代预先定义的任务指标。

## CPU 验证与未来操作

CPU 测试使用真实 CEM 和转换代码、人工目标函数与模型 fixture，不构成真实
模型规划或物理证据：

```bash
cd /home/gbk/code/LeWorldModel/le-wm-main
CUDA_VISIBLE_DEVICES='' .venv/bin/pytest -q tests/test_codex_p3_reacher_planned_exclusive_20261101.py tests/test_codex_p3_reacher_live_exclusive_20261101.py tests/test_codex_p3_live_chunk_adapter_20261101.py tests/test_codex_p3_chunk_extension_20261101.py
```

未来仅在获准使用 GPU、MuJoCo EGL 及现有资产后，以事先冻结的参数分别执行；
以下数值仅为接线示例，非已验证阈值或预算。输出目录必须不存在：

```bash
cd /home/gbk/code/LeWorldModel/le-wm-main
.venv/bin/python -m experiments.spectral_reachability.codex_p3_reacher_planned_exclusive_20261101 --policy /home/gbk/.stable_worldmodel/checkpoints/reacher/lewm --adapter-artifact results/reacher_d2p_p1_distribution_repair_v1_20261024_cuda/reacher_training_action_stats.json --output results/p3_planned_fixed_NEW --seed 20261101 --arm fixed --min-improvement 0.01 --max-disagreement 1.0 --cem-samples 8 --cem-steps 2 --cem-elite 2
.venv/bin/python -m experiments.spectral_reachability.codex_p3_reacher_planned_exclusive_20261101 --policy /home/gbk/.stable_worldmodel/checkpoints/reacher/lewm --adapter-artifact results/reacher_d2p_p1_distribution_repair_v1_20261024_cuda/reacher_training_action_stats.json --output results/p3_planned_dynamic_NEW --seed 20261101 --arm dynamic --min-improvement 0.01 --max-disagreement 1.0 --cem-samples 8 --cem-steps 2 --cem-elite 2
```

本次未运行真实 GPU 或物理环境，未开放 P3 gate，也未宣称方法成功。
