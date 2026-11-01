# P3 Reacher 在线接线探针

独占入口为 `experiments.spectral_reachability.codex_p3_reacher_live_exclusive_20261101`。
它复用现有 P0 Reacher 的真实环境、已执行历史、RGB 预处理和冻结模型；
只执行一次 chunk，不修改现有 runner。候选为预先指定的 15 步零物理动作，
基础和每次扩展均为 5 步，最多扩展两次。它不是完整闭环策略。

`protocol.json` 在模型推理、环境创建和历史 warmup 前独占创建，包含参数、
候选规则和指定本地源码、动作统计及 checkpoint 文件的 SHA-256；这些文件在
chunk 决策前重新检查。此清单不是 Python 环境或全部依赖的完整封存。
warmup 结束后，`action_binding.json` 在预测和候选物理动作执行前独占创建：
记录实际物理动作、上下界、均值、标准差、time-major 归一化 token 和参数；
执行和预测均从该文件重建，重算不一致则停止。warmup 本身是既有固定种子
协议的物理动作，不属于这个候选 chunk。

`receipt.json` 保存 adapter 预测的尝试、返回和事件、encoder 与 action encoder
模块的入口/返回及输入数量、包括 warmup 的环境 step、耗时、CUDA 峰值分配、
失败和可能的部分 step 计数。失败不视为成功；上限拒绝不重新尝试。
这些是可观察资源而非完整计算量：未测 FLOPs、能耗、CPU 峰值内存、
模型内部操作及异步错误下全部资源；模型加载与 warmup 包含在总耗时内，
不同计数轴不能相加，也不能据此声称等计算或公平。

CPU 合约测试，不占 GPU：

```bash
cd /home/gbk/code/LeWorldModel/le-wm-main
CUDA_VISIBLE_DEVICES='' pytest -q tests/test_codex_p3_reacher_live_exclusive_20261101.py tests/test_codex_p3_live_chunk_adapter_20261101.py tests/test_codex_p3_chunk_extension_20261101.py
```

未来仅在获准使用空闲 CUDA、MuJoCo EGL、现有 checkpoint 和合法动作统计之后，
由操作者事先选定阈值与种子，以全新输出目录分别启动。例如以下阈值只是
接线探针参数，不是已验证阈值：

```bash
cd /home/gbk/code/LeWorldModel/le-wm-main
python -m experiments.spectral_reachability.codex_p3_reacher_live_exclusive_20261101 \
  --policy /home/gbk/.stable_worldmodel/checkpoints/reacher/lewm \
  --adapter-artifact results/reacher_d2p_p1_distribution_repair_v1_20261024_cuda/reacher_training_action_stats.json \
  --output results/p3_live_probe_fixed_new --seed 20261101 --arm fixed \
  --min-improvement 0.01 --max-disagreement 1.0
python -m experiments.spectral_reachability.codex_p3_reacher_live_exclusive_20261101 \
  --policy /home/gbk/.stable_worldmodel/checkpoints/reacher/lewm \
  --adapter-artifact results/reacher_d2p_p1_distribution_repair_v1_20261024_cuda/reacher_training_action_stats.json \
  --output results/p3_live_probe_dynamic_new --seed 20261101 --arm dynamic \
  --min-improvement 0.01 --max-disagreement 1.0
```

先核对输出封存、来源、动作绑定、终止及失败账本；本探针不产生公平比较、
物理安全、P3 准入或效果成功结论。阈值仍需独立确定及冻结，不能依据这些
运行的物理结果回调。固定与动态各自重建历史；同一种子不自动证明根相同。

2026-11-01 主机首次运行发现原示例中的
`cross_environment_prediction.json` 不满足 `validate_training_semantic_receipt`
合同，在模型加载和物理动作前被拒绝；失败记录保存在
`results/p3_live_probe_fixed_20261101_MAIN01/receipt.json`。示例已改为项目
已有的 Reacher 训练语义动作统计 receipt；其声明并不单独证明 checkpoint
训练数据及 normalizer 的完整来源链。
