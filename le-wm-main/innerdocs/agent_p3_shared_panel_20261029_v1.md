# P3 shared-paid H1/H3 panels v1

这是已冻结`agent_p3_resource_matched_followup_20261029_v1.md`的生产接线实现；共享付费消融，不是自然廉价fixed baseline、不签P1/P3/C3或freshness、不声称equalFLOPs。只写本stem源码/tests/本文，旧代码/结果不变。使用experiment-code组织实现和RAM回归；不采用训练、按结果调参或绘图步骤。

## 固定合同与MAIN批准的late-H3

known roots0/1 × fixedH1/adaptive × endpoints/midpoints，8episodes；每episode原reset/warmup10、原prefix5、真实RGB反馈b5/10/15、最终物理20。bank分别[0,1/3,2/3,1]及[1/8,3/8,5/8,7/8]，每个H面板4候选，无挑根、删重复、重采样。所有后续节点用该臂实际新RGB及实际last10 raw，不复用根J或预测child。

每个到达节点依序：H1/H3独立nominal与AD primal及192VJP；两个信号谱；H1四候选whole+segmented及独立winner replay；H3同样完整支付；最后原信号policy决定horizon。两策略都完整支付面板，即使fixed不使用H3。只释放所选赢家FIRST5，ACK之前不能重编译。候选模型Q不用于选择horizon；稳定(cost,index)仅决定各面板赢家。

MAIN已明确批准：H3在b5/10/15分别预测到20/25/30，物理包只执行到10/15/20。禁止remaining-fit拒绝、预测截断、假称同预测终点或把H3当三个真实child；本源码不调用旧fullcommit `plan_node/Session/episode/execute/consume`。复用冻结policy-free `generate`、NativeProgram/NativeJournal、native env/Meter/Store低层、moving setup/packet/roll/encode_boundary及RGB预处理严格重算，不monkeypatch模块globals。旧step5视觉编码接口仅用同一当前PNG的显式alias，mapping记录真实时间。

## policy应用范围与拒绝ABI（供Pasteur/MAIN）

`policy_decision`使用冻结`signal.runtime_signal`与`core.choose`，保留完整原`decision`的type/horizon/candidates/program_depth_limit/continue_expanding/threshold等字段。**仅horizon用于路由**；`applied_fields=['horizon']`，`actual_candidates_per_panel=4`，`original_width_and_depth_executed=false`。原4/8候选要求不伪装成已执行；这不是原policy全预算/深度实现。不调用`decide_root`生成8请求再偷偷执行4。

非progress类型（包括reconfiguration/composition）、不继续或非法horizon均在完整双面板付费后返回`node_result.status=refused`、`commands/selected_horizon=null`及明确`refusal`，不fallback H1。episode=`contract_refused`、control_returned为真实边界，后续node保持`not_reached`和null；其余预定episode仍按顺序继续。异常则`failed`/`partial_or_failed`，无重试/退费。strict的`complete=true`只表示原生执行或拒绝证据完整；必须另读`all20/completed20/refused/equal_paid_axes_full20`，不能把拒绝当成功或资源匹配。

新独立schema：`agent_p3_shared_panel_20261029_v1.prepare/.node/.node_result/.dispatch/.report/.consume`。node result含`responses/diagnostic/panels[H1,H3]/policy/commands/selected_horizon/resource_receipt`；每panel含bank/candidates[0..3]/winner/完整selected_path。commands严格5×2 raw float32，不是15步lookahead。dispatch绑定request/result/native journal SHA及FIRST5。报表固定8/24/48/192 episode/node/panel/candidate分母；未到达槽从prepare roster可重建。模型输入只允许task/context/raw10/observation哈希，不输入reward、state7、success或未来RGB；原生评价仅执行器/strict可见。

所有consume分支显式返回`prepared=str(Path(prepared).absolute())`、`prepare_sha256`及`report_sha256`，由实际已验证CLI参数绑定，不是placeholder。`per_episode_neural`含episode_id/source_slot/mode及原core全量caps/attempted/completed/phase向量；额外未独立计量的scalar_backward_calls/AD_primal_predict_API/Jacobian_elements保持null，不拿raw callback数量替代scalar-backward，不从nominal预算猜失败AD工作。没有强行引入Nash新资源模块；可由其独立适配器核查。

当前会话无可用agent消息工具，无法直接向Pasteur发消息；请MAIN将本节新ABI转发。没有创建后代或编辑Pasteur文件。

## 预算

|每完整episode资源|数量|
|---|---:|
|predict API / samples(batch1)|132 / 132|
|信号nominal / AD primal predict|12 / 12|
|双面板程序predict|108|
|AE / predictor / pred_proj调用|各132|
|predict上下文token / pred_proj行|396 / 396|
|J / VJP / grad / raw_backward|6 / 1152 / 1152 / 1152|
|signal compile / program compile / solve|6 / 6 / 6|
|array SVD（含信号谱）|12|
|完整trajectory replay API / metric / policy decision|54 / 24 / 3|
|encoder / projector / native public目标（含warmup）|4 / 4 / 30|

每节点44=8信号+9×H1+9×H3；全8目标1056predict、48J、9216VJP、32encode/32proj。与旧51不能声称同资源。forward/AD/模块嵌套费用各轴独立；inclusive操作wall不相加为总wall；FLOPs/cuda_ns/backward_engine_calls未知保持null。提前停或拒绝不补调用、不填缺帧，attempt即扣账，failed不refund。

job cap：wall7200s，输出2GiB含256MiB终态reserve、磁盘floor200MiB，CUDA allocated3GiB/reserved4GiB。native cap沿原8episode（constructor8/reset24/set_state48/goal_render24/goal_setter32/goal_bind8/public240/physics2448/explicit_render256/close8），是上限不是actual；真实reset/warmup/Space子步逐event计量。原float32模型/normalizer/Q/目标、MATH、关闭两类TF32/autocast，原source/安装class/模型buffers前后哈希；不训练/变门槛。

## MAIN命令（worker不生产执行）

在`/home/gbk/code/LeWorldModel/le-wm-main`运行。SHA占位由MAIN从实际前一步输出取得，不能填未来SHA。`prepare/verify`会完整重建冻结上游，可能耗CPU；本worker不运行它们的生产版本。

```bash
.venv/bin/python -m pytest -q tests/test_agent_p3_shared_panel_20261029_v1.py
.venv/bin/python -m experiments.spectral_reachability.agent_p3_shared_panel_20261029_v1 --help
.venv/bin/python -m experiments.spectral_reachability.agent_p3_shared_panel_20261029_v1 accounting
.venv/bin/python -m experiments.spectral_reachability.agent_p3_shared_panel_20261029_v1 prepare --output results/agent_p3_shared_panel_20261029_v1_prepare_MAIN01.json
.venv/bin/python -m experiments.spectral_reachability.agent_p3_shared_panel_20261029_v1 verify --prepared results/agent_p3_shared_panel_20261029_v1_prepare_MAIN01.json --prepare-sha ACTUAL_PREPARE_SHA
.venv/bin/python -m experiments.spectral_reachability.agent_p3_shared_panel_20261029_v1 run --prepared results/agent_p3_shared_panel_20261029_v1_prepare_MAIN01.json --prepare-sha ACTUAL_PREPARE_SHA --output results/agent_p3_shared_panel_20261029_v1_MAIN01 --authorize-known-cuda-and-physics
.venv/bin/python -m experiments.spectral_reachability.agent_p3_shared_panel_20261029_v1 strict --prepared results/agent_p3_shared_panel_20261029_v1_prepare_MAIN01.json --prepare-sha ACTUAL_PREPARE_SHA --output results/agent_p3_shared_panel_20261029_v1_MAIN01 --report-sha ACTUAL_REPORT_SHA --receipt results/agent_p3_shared_panel_20261029_v1_consume_MAIN01.json
```

`consume`与`strict`同义；stdout默认不写receipt，显式receipt仅排他新路径。strict不运行模型：重建prepare、核字节SHA/完整事件census、独立原PNG预处理逐float32位比较、native IO/J全部方向/两个候选面板/原policy/赢家FIRST5重建、原始reset/warmup/goal/state7/reward/速度角度/native提前停及原始source/model/precision闭包。不是神经数值重跑或科学证书。

## 验证与未完成项

测试仅显式RAM数组、合成Env/视觉callbacks、临时saved fixtures与冷CLI；从不加载模型、不运行Space/physics。覆盖b15 H3完整15步预测仅执行5、原reconfiguration拒绝后续槽null及其余臂继续、4/8/depth完整保存、预算不退费、module/J/赢家/PNG预处理/未知事件变异、prepare source漂移。开发中出现两项fixture错误（跨arm完整panel错误要求SHA相等；unblob只读buffer不能原地变异），已针对fixture修正，不是科学负结果。

最终实际测试结果随冻结交付消息列出。生产prepare、主机CUDA、物理与生产full strict均未由worker执行；MAIN审读后负责。known2不等fresh，P1及独立confirmation仍未放行；progress-only horizon-routing不等C3 program type/depth机制成立。新路径可能真实产生unsupported/refusal，不据旧路径outcome或预检结果改阈值。
