# P4 独立 saved-data 闭环归因审计

只新增同 stem 审计器、测试与本文档。独立实现不导入旧 runner/compiler、torch、gym、pymunk，不分派代理，不执行神经/物理/训练/下载；Pillow 仅解码已有 PNG，NumPy 仅做保存数组的算术。未发现本仓库或适用父目录 AGENTS.md；已完整读取指定并行进度文档。

## 输入定位与科学边界

扫描 `le-wm-main/results/p4*/protocol.json`，只接纳协议明确为每臂20步、边界5/10/15的已落盘 report。按报告 mtime（同时间按绝对路径）选最近两批，不按效果选批；inventory保留被排除的短轮次协议。也可用重复 `--report` 显式指定最多两个不同报告，生产审计仍核验协议预算。

本次只有一批符合条件：

`le-wm-main/results/p4_multiround_feedback_runtime_20261027_v1_prepare_v1/report.json`

imagined/actual是同一批的两臂，不是两批；两轮10步实验不能替代第二批20step3round。`located_runs=1, missing_runs=1, two_run_evidence_verified=false`，第二批明确未证实。已有outcomes/mechanism报告不是新一次真实执行，不重复计数。

P1未通过，所以无论保存算术是否通过，`P1_admission=false, P4_admission=false, efficacy_verified=false`。本工具不接受用户布尔值开启准入。

## 检查与输出

- 按稳定root SHA、arm、round/boundary关联，拒绝重复槽位、错根、缺轮、缺step、事件乱序与错误selection内容SHA。plan绑定原goal/Q/prefix/context；逐候选重算Q成本、稳定argmin、selected动作和独立replay保存见证。
- 重建实际RGB三帧历史和两个已执行raw packet，验证PNG文件/像素SHA、原始root末帧及step5 prefix图、RoyalBlue guard。按原float32常量验证动作归一化；验证observed latent重叠、predicted successor滚动、actual/imagined正确选用各自context，防止跨臂或预测latent偷换观测。
- 每个root/round并列报告预测Q、下一边界真实编码Q与误差，以及真实pose/XY/wrapped-angle/official-state-distance/position-error。复算投影误差平方项与目标交叉项的恒等式。不把latent/Q单位映射成物理pose因果效应。
- step20有真实state/RGB，但没有保存真实latent；两臂15→20的prediction_error及observed_Q_cost_change均为null，状态`unconfirmed_not_saved`，不补造终点latent。跨臂逐轮端点差保留自身历史已经分叉的说明，不解释为孤立干预效应。
- 逐method重建prefix、compile、bank、independent-replay分阶段预测收据，检查phase/context/horizon/完成标记；核对candidate journal、有限192×10 Jacobian及nominal保存数组。核对final census及实际事件计数，不把计划预算直接当完成量。

本批保存预算复算：80 predict samples，其中78 compiler相关；6 Jacobian、1152数学标量输出VJP方向、12 SVD；6编码/18帧、240处理action-context tokens；40 selected步与30 setup步分账。VJP/SVD由保存收据和维度推导，不冒充独立底层API hook复测，也不宣称总算力匹配。

证据读取每文件32MiB、累计128MiB（含复核），输出报告硬上限2MiB、独占新目录，禁止覆盖；所有读取文件在结束前复hash。报告记录工具/测试/本文档及实际输入SHA。未递归认证全部上游source seals、原始编码器数值或隐藏模拟器状态；这些仍是证据范围外，不能宣称全部实现已排除。

## 已验证结果

`results/p4_feedback_attribution_audit_20261027_v1_verify02/report.json` 为最终只读实证报告（相对 `le-wm-main`）。该批六个root/arm/round槽位的算术与接线核验通过，但两批完整证据未满足，P4不准入。

actual减imagined终态：pose cost +68.1709449028，object XY +76.0117107898，wrapped angle −0.3920382944，official state distance −60.7454099244，official position error +67.2337761041。指标方向相反，不能只引用含速度的state distance下降宣称物体位置或任务改善。只是一已用根描述性结果，不是唯一根因、独立泛化或理论否定。

## 运行与测试

在 `le-wm-main` 下：

```bash
.venv/bin/python -B -m pytest -q -p no:cacheprovider tests/test_p4_feedback_attribution_audit_20261027_v1.py
.venv/bin/python -B experiments/spectral_reachability/p4_feedback_attribution_audit_20261027_v1.py --output results/p4_feedback_attribution_audit_20261027_v1_verify02
```

测试为纯算术/显式二维mock：完整三轮、缺末端latent、错根/错轮、重复事件、执行动作篡改、prefix拒绝、历史滚动/观测替换、预算/失败收据、残差恒等式、角度周期与速度指标分离、非有限/重复JSON、缺批次、短实验排除、输入改动及禁止runtime导入。mock通过不替代真实科学证据。

首轮测试检出非边界state未逐条检查的缺口，已在本新版本中补齐全部20步state有限性校验；最终测试与保存数据重验结果见交付回复。早期check01/verify01为本次新审计的调试记录，未覆写；不替代最终verify02。
