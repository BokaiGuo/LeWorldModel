# Zero-reference 后果向量：RGB 反馈开发对照

按[事前协议](PC_RGB_HOLD_RETENTION_FEEDBACK_PROTOCOL_20261107_CODEX.md)，在已曝光的 40 个 TwoRoom 任务上，把完整 zero-reference-relative 两维向量接入固定前缀保留。四臂每次规划均评分 16 条合法动作链、共享候选构造和最终选择规则；各臂真实反馈后的观测与规划次数可不同。

| 规则 | 成功 | 整面板链评分 | 对照 effect 的独胜／独负 |
| --- | ---: | ---: | ---: |
| 普通 lookahead | 27/40 | 2496 | effect 为 2／2 |
| 旧的行中心化向量 | 26/40 | 2480 | effect 为 1／0 |
| zero-reference 后果向量 | 27/40 | 2512 | — |
| 行内动作身份打乱 | 27/40 | 2544 | effect 为 1／1 |

本配置**没有达到**事前规定的开发出口（相对 lookahead 和中心化向量分别至少独胜 3、独负至多 1，且不低于打乱臂），不应直接扩大确认。它不能证明其他描述或规划结构无效。zero 是合法比较动作；本次新生成前缀并未逐一认证其静止语义，因此不将该参考自动称作所有状态的物理 hold。打乱也是行内动作身份打乱，不是跨 root 打乱。

同环境重放 **160 条轨迹、2893 步**，重建 **627 次**已保存评分和选择，实际距离最大差为零；核对 RGB 收据、动作指纹、时钟与查询次数。审计第一次因为将 JSON 分数按 float64 而非原 float32 计算后果而失败；修正审计精度后完整通过，原运行和评分未改。此次没有独立 GPU 重算全部评分。定向搜索与适配层测试 **8 项通过**。

这是训练成员和 normalizer 来源未认证的开发面板，不是独立统计确认。既有同面板 CEM300 为 40/40，但其完整成本及单轮预算与此不同，不能将其直接作为同成本对照。此处没有新未查询动作预测、真实机制干预或跨模型结果；PC0–PC5 不改判。

代码与结果：[搜索实现](../le-wm-main/experiments/pc_rgb_hold_retention_search_20261107_codex.py)、[闭环记录](../le-wm-main/results/pc_tworoom_rgb_hold_retention_feedback_20261107_codex.json)、[重放审计](../le-wm-main/results/pc_tworoom_rgb_hold_retention_feedback_audit_20261107_codex.json)。
