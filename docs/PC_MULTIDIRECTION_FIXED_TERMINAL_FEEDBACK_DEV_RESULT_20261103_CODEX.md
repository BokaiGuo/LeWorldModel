# 多方向编译器固定终点反馈开发结果

按[预先协议](PC_MULTIDIRECTION_FIXED_TERMINAL_FEEDBACK_DEV_PROTOCOL_20261103_CODEX.md)，
复用多方向编译器封存首决策，执行前五步后分别在剩余 10、5 步以多方向
响应候选重新规划；当前状态、carry、预测、精修及最终选择均在每次决策
重新计算。24 条暴露根的三段动作按固定 15 步终点执行。

| 指标 | 多方向反馈版本 | 同根封存 full-CEM 反馈 |
|---|---:|---:|
| 平均最终末距 | 0.46146 | 0.50199 |
| 成功 `<0.05` | 2/24 | 0/24 |
| 多方向逐根末距严格更低 | 10/24 | — |
| 多方向减对照平均末距 | -0.04053 | — |

成功根为 6、17；它们对应的首次 selected 前缀不满足先前非即时条件，
因此不能作为非即时机制成功。首次规划花费 478–480 token，新增两次
决策各花费 480 token，新增决策共发生 2 次 zero-prefix 弃权。原控制
首决策及反馈轨迹已封存；它是单配置 CEM，并非强搜索上界。结构分解、
其他计算及机器时间不能与 token 机械视为等价。

每个新增时域的实际评分完整链均进入 incumbent；逐根检查最终链在该决策
所有已评分完整链中取最低分（含零对照）。carry 重评分、float32 动作、
时钟、每批评分、动作约束、分段与整链回放均已审计。这个修复不回改旧
反馈版本，也不证明其他模型路径不存在 incumbent 问题。

这是真实运行的**精确状态反馈开发轨迹**，不是合法 RGB 输入或独立
确认。完整反馈执行的三个片段是分别在各次观测后选择的，不能称为首次
事前选定的同一条完整链；未对反馈后的每次最终前缀完成事前机制确认。
24 根已暴露，且没有强基线和独立统计，PC2、PC3、PC4、PC5 均不能由
本结果通过。按协议保留本单配置结果，不追加种子或调参。

[完整轨迹](../le-wm-main/results/pc_multidirection_fixed_terminal_feedback_dev_20261103_codex_plans.json)
SHA-256：`81b5a315e6d88de226a01875a04cde31c3b15edc0df5f8eb34f0f657a4652066`；
[逐批审计](../le-wm-main/results/pc_multidirection_fixed_terminal_feedback_dev_20261103_codex_audit.json)
SHA-256：`5a423bacc74ba981928aca4c91fe87059c6253956d48ec5df590c69c336f4367`。
以下测试重建两个新增决策的随机样本及全部选择，并重放首次决策和其
事后机制诊断，结果 **3 passed**：

```bash
.venv/bin/pytest -q -p no:cacheprovider \
 tests/test_pc_multidirection_fixed_terminal_feedback_dev_20261103_codex.py \
 tests/test_pc_multidirection_successor_exact_dev_20261103_codex.py \
 tests/test_pc_multidirection_successor_same_chain_diag_20261103_codex.py
```
