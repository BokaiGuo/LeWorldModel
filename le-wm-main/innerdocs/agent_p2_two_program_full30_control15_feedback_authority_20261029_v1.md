# control15反馈重编译协议：实际RGB权威补记

## 状态及不变合同

本文补齐已交付协议 `agent_p2_two_program_full30_control15_feedback_protocol_20261029_v1.md`
（SHA256 `75e8604604bdd8f6a12eda24367747640a7318ca4525c6dba19c685fd85060d4`）第1节的未来RGB输入。
原协议和全部冻结源测文保持不动。本文与原协议合读为完整待MAIN批准的反馈重编译合同；
不是实现/生产授权，未运行NN、physics或fullstrict。

## 三项真实RGB输入

以下路径均为实际文件，不是拟用文件。目录前缀为
`/home/gbk/code/LeWorldModel/le-wm-main/`。

|类型|相对路径|SHA256|
|---|---|---|
|prepare|results/agent_p2_two_program_full30_rgb_encode_20261029_v1_prepare_MAIN01.json|20a248dc940188ff3858fe71769cdc2d726ec23d2e77e8e50783ff777db63527|
|report|results/agent_p2_two_program_full30_rgb_encode_20261029_v1_cuda_MAIN01/report.json|915b6760db8bf30abe16e4bfba347604e6359c17d08c1ff3f2a7a076b7714df4|
|完整receipt|results/agent_p2_two_program_full30_rgb_encode_20261029_v1_consume_MAIN01.json|42164a225310c39ff83e398f55501e36e8db9876d5b52dbc697d031e31468166|

MAIN提供完整consume成功权威。本次worker仅重新hash三文件、读取metadata与context字段，
未调用producer.verify/consume，未独立重签神经parity。receipt已认证28encoder/projector、
四anchor及24forecast；其固定分母仍为28位置/4plan/2known根。模型未变、errors=[]是继承MAIN证据，
不是本次新神经实验。未来runner必须保留 `full_upstream_rebuilt_here=False`，
独立重hash所有被引用闭包；不能把文件存在或来源指针当全artifact认证。

## 确切resolver接口

未来新prepare显式接受上述三组path+SHA以及原协议所列model/native权威。
验证receipt.schema严格等于
`agent_p2_two_program_full30_rgb_encode_20261029_v1.consume`，
status为`all28_full30_RGB_verified_NOT_admitted`；receipt.report_sha256与report本体hash一致，
report.prepared/prepare_sha256与实际prepare一致。所有科学门保持false，不能把NOT_admitted改成放行。

固定四项身份，不能按向量相同去重或调序：

|i|root_slot|arm|
|---|---|---|
|0|0|two_program_transfer|
|1|0|joint_two_round|
|2|1|two_program_transfer|
|3|1|joint_two_round|

对每个i：从 `receipt.actual_context_chains[i]` 核对plan_index/root_slot/arm；
只取其checkpoints中唯一 `step == 15` 的 `.context`，并要求 `.context_missing is False`。
不能按列表第3项盲取，不接受bool step、重复step或不完整六字段context。
真实schema六字段为layout/context_length/latent_dim/action_dim/latent_history/action_history，
完整3×192 latent及2×10 action token均须为原literal float32值，不允许静默cast修复。
本次只读确认四项cp15均具备该完整结构；不因此省略未来数值、任务和源字节验证。

未来resolver还须独立从receipt.rows中以plan_index与control step连接实际cp5/10/15向量，
并从prepare对应rows核对PNG source/time/raw/goal/normalizer，逐位重构上述context：
latent_history依次是actual cp5/10/15，action_history由实际control6..15的两段raw5恰好一次归一化。
绝对native时间15/20/25，不是绝对5/10/15，也不是control0/5/10。
先认证完整prefix15与原planner rawbits、native历史guard、task/root身份再产生模型job。
若上下游字段不符必须blocked，不借另一个arm或只保留末latent。

原I策略context仍来自同i planner完整 `contexts[2]`，原后缀u0仍为该plan原最终logit60后30维；
不能用新actual192替换原imagined字段后冒充原预测。两种context分别封存SHA。
传入新模型worker的白名单仅原task/goal、冻结prefix、u0/domain witness、所需C与源身份；
cp20/25/30实测向量、native成功与终态不进入优化。完整闭包可由prepare验证，不等于优化器可读标签。

## 预算与封存决策保持

原协议固定O/I/F × 4base slots，共12策略槽；新增仅8个I/F模型job。
固定一轮/四alpha/保incumbent，216predict/8J/1536 requested VJP/46080 J元素，
0新encoder/projector/physics；原28encode与原模型/native费用仍单列继承，不能退款。
R/剩余球与float64 chart→float32 raw合同不变，不能根据本轮预测/真实误差选择启用哪臂。
I/F新评分、全候选和独立重放均真实付费；模型guard不能宣称原生成功或真实Q改善。

MAIN批准后才可实现新kernel/runner/tests/doc、完整RAM运输预算与saved消费者；之后MAIN自行生产。
若未来做物理，先独立连续复现到cp15并验证原native/RGB字节，保存context才能复用。
不匹配必须blocked，不能在control15 reset、注入状态、默增encoder或把离线重编译称已闭环部署。
同时间cp/earlystop/null/全部两根四臂及原任务成功指标，均沿原协议第5节，不按本轮结果修改。
