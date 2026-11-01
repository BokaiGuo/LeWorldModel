# Stage C：固定十条真实延续的结果分析协议与RAM算术核

仅NEW本stem source/tests/doc。此slice是**分析协议＋纯算术核**，不是Pauli物理adapter，
不是生产saved consumer，也不生成prepare/receipt/ready manifest。worker不NN/physics/fullstrict，
不改任何冻结文件。experiment-code只用于固定分母、失败/算术fixture，不训练、拟合、择优或绘图。

上位完整协议：`innerdocs/agent_p3_secant_same_time_capability_protocol_20261029_v1.md`，
SHA `b35af390fc74e6ebd44a9879784308ec74947fb28f640ce5e795cc263e65dccf`。
原能力summary三件套保持冻结；实际saved build/verify和输出SHA由MAIN交付后才绑定，不能先填。
原Stage B完整receipt SHA已由MAIN提供：
`682a78366b10a60895c4530f2fc6cc401e6439b625f4ba0a8aa996dc2fed79a7`。
本轮不读取阶段C运行中/未来结果，以下规则不依结果变化。

## 1. 固定分母、时间、动作与对照

端点顺序固定 `(r0,C),(r0,Z),(r0,P),(r1,Z),(r1,P)`；每端点先U后Z2，共10job。
U是原Stage B单个已认证raw5，Z2是float32正零5，不重新编译、择优或根据native结果改动作。
每条应原reset/place/warm10/prefix5/role5自然重放，再probe5；Pauli负责双guard和全运输。
本分析只继承MAIN完整strict对这些事实的认证，不再次执行/消费其原native链。

固定control15/absolute25为同时间实际终点，保留cp0/5/10作为身份控制：
cp0/5/10 state、完整snapshot、PNG/pixel及原incoming raw/history须和相应原端点精确连接；
U/Z2两行共享端点来源，不共享live env，不以state-only代full guard。
没有cp15就是null；native在14或更早成功/停止不补帧、不移贴终态到15。
恰在15返回并原生终止且有真实checkpoint可用；原terminal时间/标签另列。

原C逻辑槽仍为r0H1/r0H3/r1H1/r1H3，共4；只有r0H1有C端点。
每逻辑C对Z和P各比较一次，共8；另三C no_witness的6比较保持null。
四条Z/P真实轨迹不变成缺失C的fallback，10物理分支不变成10个独立根。
只有known2、独立新根0；不择根、择h、择角色、择成功时刻或缩分母。

## 2. 主定义G与全轴完整表

原native7顺序固定agent XY、object XY、object angle mod2π、agent VX/VY。
目标速度沿原native goal7，不默认为0。对原state7−goal7独立Python float64逐轴平方与math.fsum，
全表固定列：

|列|定义及解释|
|---|---|
|native7 distance/squared|7轴全量欧氏距离/平方；不能称仅位置|
|jointXY / agentXY / objectXY|前4、前2、第2..3轴误差/平方，分别报告|
|velocity / squared|原agent速度目标误差，非object速度，不忽略|
|raw angle / squared|原角度直接相减，用于原native7分解|
|native angle term|原min(abs(delta),2π−abs(delta))，包括历史负值行为|
|wrapped angle|abs(atan2(sin(delta),cos(delta)))，仅诊断，不替换原成功门|
|native success|原实际returned标签，独立公式仅diagnostic，并列是否一致|

原指标分解 `native7²=jointXY²+rawangle²+velocity²` 的归约舍入差另列；无混单位新总分。
原成功条件`jointXY<20 AND native_angle_term<π/9`不改，wrapped/object-only成功不替代native标签。

**主D**为原实际`−native reward`，即原native7距离；保留原NumPy归约，不偷换为fsum末位结果。
每端点a的主收益固定：`G(a)=D15(a,Z2)−D15(a,U)`，正值严格表示U优于对应zero延续。
另列独立fsum native7的G及二者差异；joint/object/agentXY、velocity、raw/wrapped angle每轴
也全量列`error(Z2)−error(U)`，方向均正表示该误差减少。各轴单位分开，不能挑最好轴称机制成功。
native success U/Z2单独列tie/U-only/Z2-only/null，不由G正负代替。
比较用直接`<、>、==`，不通过乘积判号，无新epsilon、阈值、CI、p值或gate。

## 3. C对两个固定对照的delayed chain

对每个逻辑C与每个B∈{Z,P}逐项列：

1. 继承原summary的`D10(C)>D10(B)`。
2. 继承原**original solver**分支的`R10(C)<R10(B)`；不以新fsum/SVD舍入判号倒改原R。
3. `G(C)>G(B)`。
4. `G(C)>0`。
5. `D15(C,U)<D15(B,U)`。

所有比较无容差，tie不算严格通过。每对五项合取，再对Z/P两对合取；不取min/max或有利对手。
已知原C对Z/P的**即时代价双对照合取为false**，必须原样保留；
未来G或最终距离再好也不能把该协议前提false改写成“先损后益链已通过”。
本kernel不硬编码哪个对手的单项false，只逐项继承已冻结独立summary，核双对照合取一致。

三值逻辑：有任一false则合取false；无false但有null则null；全部true才true。
所以工程缺项不会抹掉已知false，也不能被当作新的模型负结果。
6个原C no_witness比较的原即时代价、R、G及最终距离全null，不能借Z/P真实值补C。
即使某known-root描述性合取true，也不等于任务成功、局部J是唯一中介、P1/P3/C3或fresh确认。
所有scientific/admission/release/independent/equalFLOPs标记始终false。

## 4. 失败与解释的最小判别表

|实际状况|保留/可描述|不能升级为|
|---|---|---|
|native在15前停止|原时间、terminal state/实际goal及success；cp15/G为null|用提前终态充同时间反超|
|恰15正常返回/原native stop|真实cp15与原success；同时间G可计算|以G代native成功|
|guard/源码/动作/运输失败|全部10分母、失败字段；已有状态诊断保留，收益比较不认证|模型机制负结果/重试挑运行|
|cp15已返回但cleanup工程失败|该状态与原reward仍留描述；eligible=false，G保持null|完整可比较终点|
|C即时代价合取false|明确该预定义前提未成立；G各轴仍全报|换objectXY或换对手挽救原链|
|R/模型Q好但真实G差/缺失|分别列预测与真实，不择优|由局部SVD保证实际能力|
|原三C no_witness|全部6比较null|0收益、tie或6个已执行失败样本|

若初始warmup早停而尚未绑定原task goal，terminal metrics必须使用该实际返回的goal，
不能把原task goal写进reset前/attach前记录。cp15若存在则必须已经是原task goal。

## 5. 必要源码接口、未来真实接线与明确停止线

本版仅stdlib，无NumPy/SVD、producer math import、模型/环境导入、文件读取或生产写入。
`protocol()`/`describe`固定分母/列/公式；`native_math(state7,goal7)`独立fsum；
`evaluate(rows10,capability_summary)`只做RAM可测的纯算术，输出明确
`authentication_performed=false, production_consumer_ready=false`，不是receipt/authority wrapper。
它绝不证明调用者输入是真实生产数据，也不能把手写records签成真实物理证据。

`rows10`每项为纯计算record：index/endpoint/root_slot/role/probe/status、原goal7、
cp15_state7/native_reward/native_success、terminal_control_step/state7/goal7/native_success、failure。
这些是kernel参数，不是新物理schema；认证层尚未实现，禁止手工填入生产结果调用当完整consumer。

已只读Pauli当前`agent_p3_same_time_capability_probe_physical_20261029_v2`源码，
其**当前、尚未由本任务获得最终冻结authority**的接缝为：

- `.consume.jobs[10]`每项`.job_consume.row`，`row.job`含原index/endpoint/root_slot/source_role/probe/raw15/warm10。
- `row.endpoint15`/`row.checkpoints`给真正cp15及cp0/5/10；对应`row.responses[step==15]`给原reward/native_success。
- `row.operations`保留public返回goal_state/raw bits和reset/goal/native完整账；`row.guards`是已认证51/66列表，
  不能把该列表本身当作再次独立证明因果guard；完整认证继承原strict。
- batch/10job report+completion外部SHA、原source_authorities、StageB receipt、五U witness，
  加MAIN本次能力summary的actual SHA，均须最终逐域核对；不在本文猜未来文件SHA。

待MAIN给actual完整receipt/report及最终ABI/SHA后，另获接线授权再实现真正saved读取/写/verify，
保留全部10行及失败；若ABI有差异报告，不改本协议指标/分母，不改Pauli文件，不回调其fullstrict。
当前不能运行生产analyze，本版没有该CLI，不能用RAM片段冒称production ready。

## 6. 当前可执行命令与测试

```bash
cd /home/gbk/code/LeWorldModel/le-wm-main
OPENBLAS_NUM_THREADS=1 .venv/bin/python -B -m pytest -q -p no:cacheprovider tests/test_agent_p3_capability_probe_result_protocol_20261029_v1.py
.venv/bin/python -B -m experiments.spectral_reachability.agent_p3_capability_probe_result_protocol_20261029_v1 describe
```

RAM覆盖固定10/5/8/4全分母、已知false不能被未来正G洗掉、tie/subnormal/null、native angle历史行为、
原native reward与独立fsum分栏、object/joint/velocity全轴、goal速度非零、原success不替换、
earlystop与恰15、工程失败后cp15、无实际返回、attach前实际goal、借C或改分母/目标/条件的拒绝。
本轮测试使用合成标量/状态，不读取任何阶段C结果，0模型/物理/AD/fullstrict。
最终回归 **36 passed /0.18s**；首轮和补齐cp15原goal一致性要求后的回归均通过。
三件套至此冻结。唯一待接线是MAIN交付的阶段C真实authority及最终ABI和能力summary actual SHA；
本版有意只交协议/纯算术核，不存在生产analyze/consumer冒充已完成的情况。
