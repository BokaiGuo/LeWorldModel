# 双轨模型调用回执：版本化修复与 GPU 核验

## 发现与修复

旧 `pc_input_permissions.py` 在调用模型之前向 `receipts` 添加记录。因此，若被调用模型随后抛错，调用者捕获异常后仍可能拿到该条记录；`manifest` 不能从记录本身识别这次失败。此外，旧包装器仅在构造时检查 eval 与参数 `requires_grad`，之后的模式或冻结状态变化不会被包装器检查。这里发现的是**证据留痕风险**，不是已证明先前实验的负结果由该缺陷引起。

旧实验将旧文件绑定到哈希，本次保留旧文件不变；新增 `pc_input_permissions_v2_20261107_codex.py`。新版在调用前留存输入摘要，调用成功且调用前后仍为 eval、参数冻结后才添加回执；新版审计要求绑定新旧两个边界源文件。`call_manifest` 将状态限定为 `model_calls_recorded`，并采用独立 schema，不可直接作为旧 `completed` 证据登记。

这不是进程隔离或安全沙箱，也不通过 eval/`requires_grad` 检测全部可能的权重原位修改。完整 episode 的实际完成、动作时钟、合法性和模型权重前后指纹仍须由正式运行器单独验证。已有运行器没有自动迁移到新版；不能将本次测试当作所有历史路径均已修复。

## 运行结果

按[协议](PC_INPUT_RECEIPT_V2_SMOKE_PROTOCOL_20261107_CODEX.md)，复用原固定数据与动作，冻结模型并在 GPU 比较包装器输出和同一模型的直接调用：

| 独立轨道 | 检查范围 | 与直接调用的最大绝对差 | 结果边界 |
| --- | --- | ---: | --- |
| RGB-only | PushT 三种固定动作 | 0 | 接口一致性，不是任务成功 |
| State-assisted | PointMaze 三动作×两相机 | 0 | 使用完整 proprio，不是 RGB-only |

两次运行前后模型指纹一致，分别记录 3 与 6 条成功调用回执。最初沙箱不能初始化 CUDA，授权沙箱外执行后完成。复用旧 state 冒烟入口时，旧脚本仅允许远程仓库名字，而当前本地 DINO 兼容代码已直接请求本地路径，导致入口拒绝加载；未改旧脚本，新建 state 冒烟入口使用当前本地路径并绑定相关源码。此路径一致性不等于官方上游实现已完成权威等价认证。

失败调用、后续参数解冻、模式变化、state-assisted unroll 失败、轨道隔离、旧 schema 拒绝提升、动作梯度和现有逐层结果的复核合计 **27 项测试通过**。旧 RGB 冒烟记录绑定的 v1 源文件哈希仍与当前文件一致；这只核对该依赖项，不替代全部历史工件验证。

[新版边界](../le-wm-main/experiments/pc_input_permissions_v2_20261107_codex.py) · [测试](../le-wm-main/tests/test_pc_input_permissions_v2_20261107_codex.py) · [RGB 结果](../le-wm-main/results/pc_input_receipt_v2_smoke_20261107_codex_primary_rgb.json) · [state-assisted 结果](../le-wm-main/results/pc_input_receipt_v2_state_smoke_20261107_codex.json)。

两轨结果均只是模型调用级证据，PC0–PC5 不因此改判；也没有证明行动后果描述具有独立增量。
