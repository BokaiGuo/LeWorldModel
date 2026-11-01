# P3 horizon 激活与分叉 saved-only v1

仅消费 `agent_p3_shared_panel_summary_20261029_v1.summary` 独立 summary 的显式 ABI，外部实际文件 SHA 必需；不 import/call producer 或旧 summary，不重跑 full strict，不读运行中结果。只新增本三件套，使用 experiment-code 技能的独立实现/回归要求；不生成训练、图表或生产模板。原封存保持不变。

## 固定统计与可识别边界

固定 roots0/1 × fixedH1/adaptive × endpoints/midpoints 八臂，boundary5/10/15 全24节点、两面板全48槽、每根四对共八对。不是24个独立根，不计算 CI 或新阈值，不签 P1/P3/freshness。

未到达节点也输出两条 `uncomputed_H1/H3` 缺项，error_categories 总和始终48；零计数栏固定保留。配对raw计数分母20（其中前5条ABI不可核），state计数分母21（含cp0），各自same/different/missing三栏全保留。

|字段|口径|不可解释为|
|---|---|---|
|sealed/refused/not_reached|全24分母，未到达仍保留节点|拒绝/未到达被删后只看成功|
|rejection_reasons_nonexclusive|逐字段检查 program/extension 是否 progress、continue 是否 true、horizon 是否1/3；保留原 decision|泛用拒绝字符串等于执行重配置失败|
|chosen_H1/H3|sealed 选择计数|一定已 dispatch；可能在边界早停|
|dispatched_H1/H3|至少一条选定首5动作已真实返回|完整包完成或任务收益|
|same_action_H1|实际选定H1、完整返回首5、终点误差存在|H3同动作精度|
|descriptive_H3|所有H3面板，有值/缺失分开|后续反馈动作未变的同动作见证|
|unexecuted_or_incomplete_H1|未选H1或首包不完整|同动作已完成负证据|

支持谓词是 program_type=extension_type=progress、continue_expanding=true、horizon∈{1,3}。完整decision字段缺失或 sealed/拒绝与谓词矛盾 fail closed，不从状态猜拒绝原因。拒绝原因非互斥，不将其和强制等于拒绝数；保留完整原decision供MAIN审读。合法拒绝是当前支持域覆盖边界，不是已执行重配置无益。chosen但无返回单列，不能冒称激活。

每对逐step0..20独立比较 float64 state 的7维字节；逐step6..20从真实 chosen 面板 first5_raw 和 control_returned 重建已返回 raw，比较float32 signed bits。给每步same/different/null、首个观察到的 raw/state 分叉、全分母计数。**上游 summary 不提供原 prefix5 raw，step1..5显式null原因 `prefix5_not_in_summary_ABI`**，不拿候选动作冒充执行，也不声称证明全20动作相同。step0没有动作。未返回动作/缺状态保持null；无观察分叉不等于未观测部分也相同。state分叉可能在raw首次可核验分叉前出现，二者不强行因果归一。固定比较方向/顺序沿上游八对，不另筛配置。

误差项直接保留独立上游已算 error_decomposition，不调用producer数学，也不在此另造Q；本工具重点是类别/激活，而不是第二次数值认证。H3 b10/b15终点25/30必须missing，b5到20仅描述；选定H1实际返回不足5步也不是完整同动作终点。没有真实结果时只有RAM fixture，不作为研究证据。

## 来源与 CLI

仅读取显式 summary 文件，核原字节SHA/own schema/固定分母/节点/面板/配对/支持谓词/状态字节一致性，保留上游 bindings 与 authority 作为来源元数据，不重新打开它们或替代其 full strict。本三件套SHA进入新输出，verify独立确定性重建。JSON上限128MiB、重复key/非有限值拒绝；输出排他创建本stem新文件，不覆盖。0新神经/物理/AD/训练，无producer import。

在 `/home/gbk/code/LeWorldModel/le-wm-main`：

```bash
.venv/bin/python -m pytest -q tests/test_agent_p3_horizon_activation_summary_20261029_v1.py
.venv/bin/python -m experiments.spectral_reachability.agent_p3_horizon_activation_summary_20261029_v1 analyze \
  --summary /ABS/agent_p3_shared_panel_summary_20261029_v1_MAIN01.json \
  --summary-sha ACTUAL_FINAL_SUMMARY_SHA \
  --output /ABS/agent_p3_horizon_activation_summary_20261029_v1_MAIN01.json
.venv/bin/python -m experiments.spectral_reachability.agent_p3_horizon_activation_summary_20261029_v1 verify \
  --analysis /ABS/agent_p3_horizon_activation_summary_20261029_v1_MAIN01.json \
  --analysis-sha ACTUAL_ANALYSIS_SHA
```

尚未生产消费，待 MAIN 最终summary实际SHA与审读后运行；不预填未来SHA。

## 回归与冻结

首轮20 passed /0.25s；补齐48槽未计算面板与零计数回归后，最终 **21 passed /0.24s**。覆盖全24节点、全部八对、合法多原因拒绝、边界chosen但无dispatch、H1中途终止、全节点未到达、H3超执行窗缺失、signed-zero raw/state分叉、固定分母/面板/谓词/状态篡改拒绝、真实临时JSON落盘→analyze→verify、重复key/非有限值/错误SHA/覆盖拒绝。全部是RAM/fixture，不是生产或科学效果证据。

核读的独立上游 summary 源码SHA `69eb17b59cc5430a1644a604fcff7eec79ffcbe4c2488e2eae92a33b198e0e21`；原source/tests/doc保持冻结字节。实现不执行上游summary，也不调用producer；源码快照只说明ABI审读对象，不冒充未来actual summary SHA。MAIN可用最终实际文件SHA执行上列CLI。
