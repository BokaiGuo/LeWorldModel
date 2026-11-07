# TwoRoom 预定跨墙机制集：可解性和 RGB CEM 开发结果

按[事前协议](PC_TWOROOM_CROSSWALL_MECHANISM_DEV_PROTOCOL_20261107_CODEX.md)，40 个任务的起点与目标按固定采样规则位于墙的两侧，没有按模型结果或见证结果筛选。标准性能集未修改。

| 评价 | 25 步内成功 | 实际环境步 | 评分完整链 | 预测 packet 步 |
| --- | ---: | ---: | ---: | ---: |
| 评价侧几何见证 | 40/40 | 1,000 | 不适用 | 不适用 |
| RGB CEM16 | 14/40 | 932 | 3,104 | 9,488 |
| RGB CEM300 | 37/40 | 731 | 48,900 | 165,000 |

初态成功为 0/40。CEM300 相对 CEM16 独胜 25 根、独负 2 根；两者每次规划预算不同，不能把差异归因于 Planning Compiler，也不是同成本比较。计数覆盖完整反馈过程，尚未测同步 GPU 时间或完整系统墙钟。

同一环境的事后诊断中，使用真实状态与目标、每步直接朝目标移动的控制器 **0/40** 成功。它是特权的简单直接移动对照，不是最优贪心控制器或规划基线；结果只说明该具体直线路径控制不能解决此面板。评价侧几何见证同样使用物理位置与预设门位置，不能作为 RGB 规划器结果。

对全部见证和两臂的 **2,663 个**环境步做了同模拟器物理重放，核对任务、初态与目标图、实际派发、距离、终止和每轮候选数量；未重新计算 GPU 候选分数。动作范围、跨墙任务构造和见证阶段相关测试 **6 项通过**。Gymnasium 观测类型和动作转换警告仍存在。

该结果说明这组预定义机制开发任务有合法见证，且低预算基线未饱和；它不是官方标准集，也不能证明模型训练外、来源独立或描述／程序机制有效。任务已曝光，不应再当作独立确认集。下一步的任何方法对照须提供新假设和同权限、完整成本的强基线，不能仅凭跨墙成功升级 PC2。PC0–PC5 未改判。

[公开 RGB](../le-wm-main/results/pc_tworoom_crosswall_mechanism_public_20261107_codex.npz) · [评价状态与见证](../le-wm-main/results/pc_tworoom_crosswall_mechanism_truth_20261107_codex.npz) · [见证轨迹](../le-wm-main/results/pc_tworoom_crosswall_mechanism_reference_20261107_codex.json) · [CEM 反馈](../le-wm-main/results/pc_tworoom_crosswall_mechanism_cem_20261107_codex.json) · [物理重放和直接移动诊断](../le-wm-main/results/pc_tworoom_crosswall_mechanism_audit_20261107_codex.json)
