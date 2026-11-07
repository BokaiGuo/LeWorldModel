# RGB 规划输入的 checkpoint 参数绑定：工程核验

## 问题与修复

已有 `RGBLeWM` 验证显式轨道、输入键、冻结状态和文件哈希，但不核对传入模型参数与被审计 checkpoint 是否一致。反例测试证明：模型参数为 2、checkpoint 参数为 1 时，旧边界仍可执行。这是认证覆盖缺口；不是已发现历史研究加载了错误模型，也不说明历史负结果由此产生。

新增可选 `CheckpointBoundRGB`：构造时检查权重和配置文件均在审计依赖中，并将实际模型每个参数和 buffer 与 checkpoint 经现有兼容键映射后的张量逐项精确比较；`finish()` 再校验运行收据、同一模型对象、冻结状态、全部参数、依赖哈希及初末模型状态摘要。新接口保留旧 `RGBLeWM` 的 RGB 输入检查和动作梯度能力，不修改历史边界或历史结果。

## 实际模型验证

按[协议](PC_RGB_CHECKPOINT_BINDING_SMOKE_PROTOCOL_20261107_CODEX.md)，使用实际 TwoRoom LeWM checkpoint、已曝光 episode 9000 的 RGB 和合法 25 步零动作，在 GPU 各执行一次旧边界与新边界评分。

- 两种评分的最大绝对差为 **0**，预测 latent 最大差为 **0**。
- checkpoint 所有 state 张量在运行前后与模型精确匹配；模型摘要与既有开发记录一致。
- 额外开销为 **2 条模型评分链、10 个模型动作包步**；没有环境执行。
- 相关测试覆盖错误冻结模型、依赖缺失和变更、参数修改、模型对象替换与空运行，及既有权限测试；详见测试输出。

该功能目前是**显式使用的新边界**，不会自动约束尚未迁移的运行器；调用方必须在结果封存前执行 `finish()` 并保存其结果。它检查运行前后状态，不是防恶意 Python 沙箱，也无法证明运行中短暂修改后恢复未发生。它不认证官方来源、代码实现语义、normalizer 拟合集或 RGB 任务收益。用户已说明权重当初从官方渠道下载；本测试没有独立重验下载来源。

`code-debugging` 用于建立反例、针对性修复和真实 checkpoint 复核。此项是 PC0 工程证据的一部分，**不改判任何 PC gate**。

[严格边界](../le-wm-main/experiments/pc_checkpoint_bound_rgb_20261107_codex.py) · [真实 GPU 核验代码](../le-wm-main/experiments/pc_rgb_checkpoint_binding_smoke_20261107_codex.py) · [原始结果](../le-wm-main/results/pc_rgb_checkpoint_binding_smoke_20261107_codex.json) · [测试](../le-wm-main/tests/test_pc_checkpoint_bound_rgb_20261107_codex.py)
