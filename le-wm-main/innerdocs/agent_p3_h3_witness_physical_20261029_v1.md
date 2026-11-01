# P3 b5 H3 full15：真正同动作物理见证

本实现只补已选预测对应的真实同动作轨迹，不是新planner或C3/P1放行。既有shared-panel在每个边界只执行所选预测的FIRST5；adaptive在raw11可能改动后续动作，因此不能把原MPC终态编码与b5完整H3预测作为same-action fidelity。新实验固定known roots0/1 × endpoints/midpoints四分支，不按native结果、Q或误差删槽、换根、换winner、改阈值。

只新增本stem source/tests/本文。旧源码、results与所有科学gate不变。使用experiment-code组织实现/fixture；技能旧reference路径在本机不存在，采用现有native IO ABI及本任务明确约束，不执行训练/绘图等无关步骤。worker仅合成数组Env、临时文件codec和保存元数据schema检查；不执行生产prepare、fullstrict、物理或模型。

## 固定上游与提取ABI

绑定MAIN实际shared-panel v1：

|权威|SHA256|
|---|---|
|report|668db35d79eb650419e4efa42dfc21a15cdf91725700b2f202e2a2ec975bcd6a|
|strict receipt|d23ffa48bc954306bc27a15f924da8daafffc9b4bc76a91e59aa949ab1abbafd|
|prepare|7ca85050eeab4e47ed35cf84c284ed7e08fc64bce8064d7db9450074045b643a|

`material(arguments)`先核显式report/receipt/source SHA，再真实调用冻结shared-panel `consume(output, reportSHA, prepared, prepareSHA)`，要求与完整持久化receipt相等且complete=true；随后重建其prepare与原native source。不是status-only接受，不伪造旧receipt，不调用旧producer的run。生产prepare/verify/consume均可能重复saved闭包CPU工作，MAIN运行；wall计入run开始后的verify，不声称等wall或等FLOPs。

`select_witnesses(compiled,report)`是无神经的纯schema/提取函数，不独立签发上游strict。顺序固定root0 endpoints/midpoints、root1 endpoints/midpoints。每个分支只取`episode_id=s{root}_adaptive_{bank}`、`boundary=5`：

- 原request通过冻结`validate_request`，原selected_horizon必须3；不强制把H1换成H3。
- 使用`receipt.panels[1].winner`原index，要求与原4候选对应行一致、returned且selected_path等于原whole。此处不重算argmin、不使用physical标签。
- 完整suffix来自`winner.actions`的15×2 literal float32，原result.commands必须等于其FIRST5。
- prefix严格来自原prepare layer经冻结`prefix_binding`，必须与该原episode的prefix一致。
- 原cp5的PNG/pixel必须与该request的actual observation绑定。保存完整request/result/winner/path/task/prefix/cp5及node/witness摘要供后续编码桥使用。

每个own execution_plan的`commands`是**control20**（prefix5+suffix15）；`prefix_commands`和`suffix_commands`分别为5和15，不能把commands误读为单独suffix。`source_witness`保留原H3预测、实际b5观测context、model/Q/normalizer binding及whole路径；未来编码必须对齐control10/15/20与H3三个预测端点，而非旧MPC终态。所有raw按原float32数值/bytes认证，禁止clip、tanh重建、重优化、负零丢失或动作替补。

完整bindings合并原native source、upstream prepare.sources（含模型与环境inventory）、完整upstream产物、实际receipt、直接native工具源码及本三文件。verify重新构建全部own prepare并比较canonical bytes；run/consume后再重hash所有bindings。不加载模型，也不把“物理阶段0新模型调用”误称上游没有付费。

## 物理路径与失败语义

复用冻结native的place/bind_goal/dispatch、native_hooks、off/on/off capture、原环境校验及Store基础write/emit/put。own schema/episode/execute/validate_saved完全独立，不patch globals、不调用旧physical execute/episode、不把新结果改名为旧receipt。

四臂各自新构造环境，原reset/place/warm10，逐帧与原source warmup reference匹配；原warm10 RGB三帧、raw10、native7 root/goal hash一致后才绑定原task。warmup使用原warmup goal，不能拿它的reward当新task reward。再执行完整control20，真实native earlystop优先，不padding、不在停后继续、不修改目标或默认goal_pose。每一步保存请求/实际rawfloat32 bits、latest action、native7/state/snapshot、goal/reward/terminated/truncated、资源intent/return/failure及原任务metrics。

control5结束时必须与原actual cp5 **state7、完整snapshot、off RGB pixel SHA逐字节一致**，才可发出control6（H3 suffix首步）。不匹配是工程失败，保留该分支所有证据，后续预定分支继续尝试，不restore预测context、不偷换原动作。strict也重核此gate；无效cp5不能仅因report写成partial就获得合格receipt。

每warmup0..10及control0..20都保存真正off/on/off triplet，checkpoints为0/5/10/15/20；额外cp0便于源检查。on只用于渲染，不执行动作；native默认轮廓不冒充构造task的goal。earlystop时missing_checkpoints显式列出，不复制终帧。native7含速度，原native reward为负full7距离；原success阈值不变，wrapped-angle只是诊断。

report保留4完整rows（not_started/engineering_failed/native_stopped/horizon_completed）、所有实际计账、first/secondary errors、完整journal chain和binding文件。partial不可称完整执行，unknown endpoint为null，不删除失败分母。原始invalid RGB事故bytes也保留；strict拒绝无效动作/RGB不等于丢弃其已写partial。

## 事前资源上限

|资源|完整四分支上限|
|---|---:|
|constructor / close|4 / 4|
|reset / set_state|8 / 16|
|goal_render / goal_setter / goal_bind|8 / 12 / 4|
|public_step（warm10+control20）|120|
|physics_step（原10子步+placement）|1216|
|explicit_render / PNG|384 / 384|
|新predict / encoder / projector / AD / 训练|0 / 0 / 0 / 0 / 0|

上限不是实际计量；attempts失败不退款。constructor callback内部边界继承原physical adapter，不宣称与P3 CUDA+physics driver的不同constructor计量边界相同。wall900s从run入口开始，含输入verify；输出64MiB含8MiB终态reserve，磁盘floor200MiB；prepare/receipt metadata128MiB。close可在失败/超时后清理，其他dispatch不得绕过wall/fault/floor。native7与RGB之外没有新模型或CUDA资源；所有science/freshness/equalcompute/C3/P3/P1 admission均false。

## saved strict输出与下一编码接口

`consume(output, expected_report_sha)`完整重建own prepare/upstream strict，验证native journal全部event kind/index/previous/event hash，所有嵌套资源intent/return/failure、原seed/goal绑定时序、每一步exact raw20/warm10、所有native标签与metrics、逐PNG SHA/pixels/guard、triplet不扰动、完整文件census及终态caps。未知event、额外文件、原rawbits或goal/速度/reward变异即拒绝，不filter空行；真实磁盘两event回归覆盖继承Store codec，只有单LF。

own receipt输出`physical_report_sha256`、`plans[4]`（commands是20）、`rows[4]`（完整native严格rows）、`counts/bindings/planner_authority`、`same_action_H3_fidelity_available`及missing checkpoints。plan.source_witness含实际cp5和原H3预测全部context，后续MAIN CUDA bridge可按它取control10/15/20 off PNG；未到达就是missing，不编码伪终态。`new_physics_calls=0`指saved consume本身，执行report内counts是真实物理计量。

## MAIN CLI（worker不执行）

在`/home/gbk/code/LeWorldModel/le-wm-main`。P和R由MAIN从实际前一步输出取SHA；不能预填未来hash。路径参数使用绝对路径。

```bash
.venv/bin/python -m pytest -q tests/test_agent_p3_h3_witness_physical_20261029_v1.py

.venv/bin/python -m experiments.spectral_reachability.agent_p3_h3_witness_physical_20261029_v1 prepare \
  --output /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p3_h3_witness_physical_20261029_v1_prepare_MAIN01.json \
  --planner-output /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p3_shared_panel_20261029_v1_MAIN01 \
  --planner-report-sha 668db35d79eb650419e4efa42dfc21a15cdf91725700b2f202e2a2ec975bcd6a \
  --planner-receipt /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p3_shared_panel_20261029_v1_consume_MAIN01.json \
  --planner-receipt-sha d23ffa48bc954306bc27a15f924da8daafffc9b4bc76a91e59aa949ab1abbafd \
  --planner-source-sha 8cc8f796f313d4cc3f209ce231ebcaec4808f096acc02781cd6df734800dc464

.venv/bin/python -m experiments.spectral_reachability.agent_p3_h3_witness_physical_20261029_v1 verify \
  --prepared /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p3_h3_witness_physical_20261029_v1_prepare_MAIN01.json --prepare-sha "$P"

.venv/bin/python -m experiments.spectral_reachability.agent_p3_h3_witness_physical_20261029_v1 run \
  --prepared /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p3_h3_witness_physical_20261029_v1_prepare_MAIN01.json --prepare-sha "$P" \
  --output /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p3_h3_witness_physical_20261029_v1_MAIN01 \
  --authorization MAIN_P3_H3_WITNESS4_CONTROL20_PHYSICS_ONLY

.venv/bin/python -m experiments.spectral_reachability.agent_p3_h3_witness_physical_20261029_v1 consume \
  --output /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p3_h3_witness_physical_20261029_v1_MAIN01 --report-sha "$R" \
  --receipt /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p3_h3_witness_physical_20261029_v1_consume_MAIN01.json
```

开发fixture更正：真实Store以exclusive `xb`拒绝重复写入，正确异常是`FileExistsError`。初版codec测试误期待`ValueError`，已仅修改独占测试的期待类型；生产行为、继承Store和封存文件均未改。这是执行前夹具错误，不是历史科学结果的原因。另以fixture确认修改未来native outcome不影响固定见证提取。

冻结交付消息附本三文件SHA、fixture测试终态和实际upstream source SHA。未完成的是MAIN生产prepare→物理→saved strict与另行授权终态RGB CUDA编码；本实现不声称已经取得这些观测，也不签同动作预测正确、C3或P1通过。
