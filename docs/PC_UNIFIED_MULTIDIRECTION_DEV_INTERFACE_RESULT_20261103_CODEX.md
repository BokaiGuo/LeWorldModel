# 多方向开发版统一决策入口：接口验证

新增[统一入口](../le-wm-main/experiments/spectral_reachability/pc_unified_multidirection_dev_v1.py)：
首轮调用既有多方向后继编译器，后两轮调用其固定终点反馈编译器，
以统一的模型预测回调、动作域、token 计量、执行片段和 carry 对接。
该入口是开发版接口，不是已经通过科学验收的正式方法；历史 RGB
runner 并未改接，目标对齐与零空间版本也未接入。

根据[接口合同](PC_UNIFIED_MULTIDIRECTION_DEV_INTERFACE_20261103_CODEX.md)，
已逐批回放四个暴露 RGB 根的三次封存决策，共 12 次。每次实际请求
动作、float32 动作、最终选择、弃权和预测次数/token 均匹配；
RGB 后端本次没有重新执行。原决策未保存真实目标投影，本次使用
二维零向量占位：已核实旧 v1 候选提案只使用目标维度，而模型评分
来自原轨迹。故此回放不能验证目标来源或目标条件化。

另外在一个暴露精确 Wall 根调用统一入口，连续三次分别规划 15、
10、5 步、各派发 5 步，计量为 478、480、480 token，carry 长度为
10、5、0。独立脚本逐轮重算实际派发、物理状态和选中链评分。
这证明一次 E0 接口执行与时钟衔接，不是独立样本，也不声称跨后端
或 PC4 通过。

实现过程中发现并修复**新入口自身**的 dtype 错误：最初入口将
`float32` 预测提升到 `float64`，导致 SVD 候选与原轨迹产生约
`1.16e-8` 的动作差异；保留预测输出 dtype 后逐批严格一致。
此错误发生在本次新建入口的初版，并非已证实历史 RGB 代码的 bug。
接口边界、dtype、动作越界、预算和既有多方向反馈测试共 5 项通过。

尚存的接口边界：调用方须实际认证 RGB、动作历史、normalizer、
checkpoint 和来源；`context_id` 只是标识。当前源码哈希覆盖主要核心
模块，但**尚未完整覆盖所有间接依赖**，不能作为完整依赖认证。
当前验证输出也未保存此次精确执行的所有预测批次；不能用其替代
正式方法完整审计或 PC2–PC5 证据。`api-and-interface-design` 技能
用于明确这些责任及失败边界，没有改变历史方法或其 gate。

- [接口回放与精确执行](../le-wm-main/results/pc_unified_multidirection_interface_verify_20261103_codex.json)，SHA-256 `aef370143a2f80f9df4a0d3173fad36c79b64ba045a0708c6d51302d61a83081`
- [独立绑定和执行审计](../le-wm-main/results/pc_unified_multidirection_interface_verify_20261103_codex_audit.json)，SHA-256 `77f8124c6ce6365008ac9c3c50b556069be27e995e83d3654fe4f01712ecdb88`
