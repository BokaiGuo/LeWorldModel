# V5 E0 新网格机制开发结果

## 结论

在 18 个新开发根上，v5 事前选中的链满足 zero、greedy 两对照机制交集的为 **0/18**；加入 lookahead、random 后仍为 **0/18**。这是指定协议下的开发负结果，不是 PC2 正式通过或对全部任务、全部优化预算的理论否定。

最终链开放环终距均值 0.501111，中位数 0.500535，终距小于 0.05 的为 0/18。非零且非即时的选中前缀为 1/18。该网格仅相对于协议列出的两批旧面板排重，完整历史独立性未知。

## 设计和结果

先按[计划协议](PC_V5_E0_PROSPECTIVE_MECHANISM_DEV_PROTOCOL_20261104_CODEX.md)封存 18 根各 1920 token 的最终链，再进行共同尾族、三流每流 480 样本的各前缀后缀 CEM、两尺度有限差分及障碍分支稳定性检查。计划审计覆盖 13,266 条轨迹。

| 前缀 | 各根三流后缀优化均值再跨根平均终距 | selected 对该前缀胜/平/负 |
|---|---:|---:|
| selected | 0.501100 | — |
| zero | 0.500900 | 8/0/10 |
| greedy | 0.500841 | 4/6/8 |
| lookahead | 0.467236 | 5/3/10 |
| random | 0.501618 | 12/0/6 |

胜平负容差为 `1e-10`，重复流不是独立根。lookahead 是九候选额外筛选后的结果，其额外成本和选择偏差不支持将上表作为公平部署优势比较。有限 CEM 预算也不证明全局优化充分。

唯一非即时根 3 上，selected 使用共同 selected 尾相对自身零尾的变化为 -0.622766，zero 为 +0.893103、greedy 为 +0.884588；但三个 CEM 流平均终距 selected 0.502656，高于 zero 0.502411 和 greedy 0.500620，且 selected 在两个响应尺度的障碍分支均不稳定，因而不满足机制交集。响应计算所用名义尾最大裁剪量为 0.035188；局部结果不能无条件外推到未裁剪尾。

干预计算包括 lookahead 的 19,440 条完整链、重规划 129,870 条完整链、共同尾 540 条完整链和 6,048 次局部探针；探针组含重复基准。成本口径不等同于实际硬件时间。

## 实现异常与审计边界

首版干预在第一个根输出前失败：加入随机桥后 selected 的候选索引由 9 变为 10，记录比较错误地要求索引也一致；核对其余字段一致。保留首版 manifest，另立 [v2 适配协议](PC_V5_E0_PROSPECTIVE_INTERVENTION_V2_PROTOCOL_20261104_CODEX.md)，比较时仅规范化该索引。适配器是在计划封存之后编写的，因此不能称整条干预代码严格事前注册；这个诊断适配问题也不是已证实的 v5 planner bug。

18 根干预的确定性重算审计通过；另行编写谓词代码并重放保存动作的[审计结果](../le-wm-main/results/pc_v5_e0_prospective_mechanism_dev_20261104_codex_intervention_v2_predicate_audit.json)验证两个交集均为零。它仍使用同一精确动力学后端，依赖记录亦明确环境闭包不完整，不是独立模型复现。

原始产物：[计划](../le-wm-main/results/pc_v5_e0_prospective_mechanism_dev_20261104_codex_plans.json)、[计划审计](../le-wm-main/results/pc_v5_e0_prospective_mechanism_dev_20261104_codex_audit.json)、[干预索引](../le-wm-main/results/pc_v5_e0_prospective_mechanism_dev_20261104_codex_intervention_v2_result.json)、[干预重算审计](../le-wm-main/results/pc_v5_e0_prospective_mechanism_dev_20261104_codex_intervention_v2_audit.json)。每根对应的 JSON 和 NPZ 保存动作与评分。

## 事后特权可行性诊断

按另立的[诊断协议](PC_V5_E0_PRIVILEGED_FEASIBILITY_DIAG_PROTOCOL_20261104_CODEX.md)，两侧解析绕墙轨迹在 18 根上均有 15 步以内的合法见证，共 36 条。经与 v5 同样的 float32 动作转换、精确动力学和欧氏目标评分，36 条均达到 `<0.05`，最大终距 `1.49e-7`，且均优于对应已封存 selected 链。原动作、转换后动作和逐根分数保存于[诊断结果](../le-wm-main/results/pc_v5_e0_prospective_mechanism_dev_20261104_codex_privileged_feasibility_diag.json)。

这是特权解析解存在性和事后评分排序检查，不是编译器实际发现、内部注入或剪枝测试；使用同一动力学后端，不能据此排除内部接线、覆盖或优化问题。下一步仍需检查实际生成和保留路径，并将注入与正式性能分离。不得仅增加根数或把该面板调成阳性。PC3 强基线、PC4 同版本视觉闭环和 PC5 来源合格独立复现仍需各自证据。
