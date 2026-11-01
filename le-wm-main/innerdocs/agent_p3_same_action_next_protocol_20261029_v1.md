# P3下一轮：固定H3同动作、同目标的真实history替换

状态：仅预先方案和接口准备；没有新runtime、prepare、NN、physics或生产消费。
旧encoding/decomposition/canonical文档与结果不修改。本轮为看到known-root既有实验后的
探索性诊断，不是独立确认；不得根据本轮数值改变下面的槽、预算、目标或评分。

## 1. 问题与最小干预

已有四分支为原known2 roots × endpoints/midpoints，执行原prefix5和在b5已选定的
H3 winner完整raw15。cp5/10/15/20真实RGB已经编码；原H3预测来自b5观测后连续roll。
下一轮不改动作、不重选winner、不重新优化或执行物理。问题严格限定为：
**在同一真实轨迹、同一当前control时间、同一未来raw5和同一预测终点下，
用完整真实观测history替换原rolled history，是否改变模型下一端点误差？**

固定起点s∈{10,15}，终点s+5。四分支×两起点=8配对；每对：

| 路线 | 起点context | 下一动作 | 标签/终点 | 新调用 |
| --- | --- | --- | --- | --- |
| R：原rolled | 原H3路径该边真实使用的完整context | 原full20[s:s+5] | 原真实编码a_(s+5) | 0，严格复用 |
| A：actual | encoding receipt的actual_context_chains中同s完整context | 完全相同raw5 | 完全相同a_(s+5) | 1 predict |

所有context均含完整latent_history和action_history；不能只替换最后一帧而称完整actual。
不取native预测full3前两行当端点；只取原native API末行，并正确roll成下一context。
原task、goal192、identity Q、checkpoint、normalizer、raw尺度、precision与预测API均不变。
本协议不做模型参数训练、动作优化或闭环重新部署。

## 2. 原证据与本轮不能排除的因素

原H3物理witness已将动作固定到control20，因此避免把原adaptive在raw11以后换动作的
轨迹错误当成H3同动作标签。encoding cp5原观测bitexact复核、cp时间对齐、实际rawbits、
原goal/Q/source绑定，是已有工程证据；其适用范围不扩展到所有未来runner。
decomposition的内积恒等式可解释同动作Q差如何由向量误差与目标方向组成，
但本身不证明任何误差原因被排除，也不证明新干预有效。

本轮同时间R/A将剩余预测长度和未来动作固定，故比直接比较cp10/15/20累计误差有
辨识力：它隔离**模型输入context来源的整体替换**，而非“越晚越容易”的horizon混杂。
即使A优于R，也只能说原冻结模型在这些输入上的预测对history来源敏感；不能归因于
最新槽、旧槽、速度、接触、视觉编码、动力学本体、goal/readout其中唯一一个因素。
actual history仍是有限RGB embedding，不是真实完整物理状态；模型可观测性与表征充分性
仍未识别。A差于R也不能证明反馈无用；本轮没有重新规划或部署策略。
误差降低不保证native7/jointXY/objectXY/angle/成功改善，Q误差抵消不等于向量准确。

## 3. 与P5/跨runtime工作的非重复边界

P5 latest-slot factorial锁actual5/10，交叉slot15 visual/proprio的AA/AP/PA/PP，
针对其164分支/80pair与JEPA输入结构。本轮不复制visual/proprio交叉、renderer、
最新槽消融、80响应集合，也不把P5结果迁移成此原生192模型的证据。
这里仅复用P3自己的四条固定raw20轨迹与完整history，完成8个同时间A/R下一步对照。
本轮不做Pauli的跨runtime/API/layout/实现对齐；只使用原冻结native调用链。
跨runtime结果若另有认证，只列为外部权威，不把其容差或数值改写为本轮parity保证。
在prepare阶段如发现已有**相同四分支、相同两时间、相同完整context/rawbits/输出**
的严格认证调用，必须先列明exact reuse，缺多少补多少；不能因标签相近认为重复。

## 4. 精确context和R复用接线

原plan.source_witness.request.task.root_context为C5；
plan.source_witness.winner.whole[0/1/2]为滚动后C10/C15/C20。
R起点10/15分别是whole[0]/whole[1]，参照预测端点分别是whole[1]/whole[2]
的latent_history[-1]。不能把whole同一行同时当输入和下一输出。
A起点从encoding own receipt.actual_context_chains[b].checkpoints按step精确取context。
所有槽使用原task的float32 subtract/divide一次；actual与rolled的动作history必须
逐位相同，并与相同causal raw10绑定。归一化不重复做，不能重编码真实latent。

**R零新增调用的前提**：从原planner真实IO/严格receipt确定该R下一预测调用确实由
该完整context及raw5产生；checked_path的逻辑context闭合不能代替真实模块输入认证。
将context→原predict native buffer的ABI重建并核对原保存buffer/输出末行bit模式。
若原严格证据缺某个call的完整接缝，保留该槽blocked，不能静默补一次R或称零费复用。
需要R重放时必须另行事前版本化预算（最多再8predict），由MAIN审读授权，
不得在本8调用协议内暗加replay、重试或去重退款。

## 5. 未来prepare/运行/消费接口

仅定义建议接口，尚非可运行CLI：

```
prepare_same_action_history(
  encoder_prepare_path, encoder_prepare_sha,
  encoder_report_path, encoder_report_sha,
  encoder_receipt_path, encoder_receipt_sha,
  original_planner_prepare_path, original_planner_prepare_sha,
  original_planner_report_path, original_planner_report_sha,
  original_planner_receipt_path, original_planner_receipt_sha
) -> fixed_jobs[8], inherited_bindings, resources, blocked_slots

run_host_cuda(prepared_path, prepare_sha, output_dir, explicit_authorization)
consume_saved(prepared_path, prepare_sha, output_dir, report_sha) -> paired_rows[8]
```

已实际encoder prepare SHA：`8689f3a0014d264e135ae2dbdf04f3cd2f6c91a15d6d26d14751d6daad87f9b1`；
report SHA：`c3195ea6ba35898bfb0eaef8f9dd69e7e2f4e01834c7a2a43c6c73302352c539`；
receipt SHA：`671e48b0e953e6adeb8e07381ecaa965572afa8801d29d961c344eb82a3e505e`。
此处仅引用MAIN提供值，未在本轮运行生产消费。原planner各实际路径/SHA须显式提供，
由encoder prepare/physical/source_witness继承链核对，不推测新产物或伪造未来SHA。

每job至少保存：index、plan_index、root_slot、bank、task_id、start/end control步、
selection/witness digest、原full20和rawbits、固定future_raw5及bits、原goal192、
R/A完整context与各自canonical digest、causal raw10 bits、一次normalizer绑定、
R实际旧调用/IO/末行输出指针及SHA、A输入ABI、actual endpoint标签指针。
模型输入白名单不含native outcomes；raw动作和context来自既定轨迹，不让标签进入选择。
缺必需context/call/标签全部保留对应行和reason，不用邻时刻、另一bank或预测填充。

输入认证可继承MAIN已完成的完整receipts，显式核实际文件SHA并核全部接缝；
必须披露本轮是否重建上游完整strict，不能仅验SHA后声称重放所有旧神经/物理。
新运行本身仍须完整native输入/输出、模块intent/return、事件全序、输出census、
模型/weights/buffers/source inventory前后、precision守卫与partial保存。

## 6. 固定评分与跨时间解释

每槽保存pR/pA/a/g的完整192向量，独立算两路线MSE、Qp/Qa、signed Q error、absolute
Q error、E+2〈a−g,e〉/192两项及舍入residual。配对差固定为A−R；负MSE差仅表示该槽
模型端点误差下降，不叫控制收益。Q signed差和absolute差都保留，ties不加epsilon。
全部8行输出，无择优root、bank、时间或指标，无按成功过滤。
原真实native指标在同终点旁列但两路线共用同一标签，不能据此算两个物理策略的收益。
同分支s10→s15的变化是重复测量/不同历史状态，不能当root独立配对；
不比较pA15与actual20之类错时间组合。主结论按8固定槽及2known roots描述，
不作独立样本显著性或跨bank纯调度因果宣称。

## 7. 最小追加预算、失败和独立性

新增上限8次top-level native predict，batch1，全部no-grad；每次原full3历史输入，
API调用8、batch samples8、input历史token-frame samples24；输出完整三行均落盘，
只评分末行。底层action_encoder/predictor/pred_proj按原API各8次单列实际attempt/return，
若正式接口实际调用结构不同，prepare前先修订合同而非生产中补造计数。
新增encoder/projector/J/VJP/backward/solve/SVD/训练/physics全部0；原16编码、planner和
物理成本继承列，不退款。已保存的R预测付费成本不是新执行0成本模型。
不同算子FLOPs未知，total_FLOPs=null，equal_total_compute=false。

建议先固定工程上限wall600s（入口verify/load/dispatch/final rehash全含）、输出64MiB、
final reserve8MiB、disk floor200MiB、metadata128MiB、CUDA allocated3GiB/reserved4GiB；
实际完整IO尺寸须实现前静态核算能装入此cap，不得因结果或超时偷偷扩大。
HOST CUDA only、原f32、TF32 off/autocast off/SDPA MATH、原frozen model/normalizer；
失败不重试不退费，保留8槽attempted/completed/missing和首因，partial不当科学负结果。

两个已见root不因四branch、八配对、多个时间变成新独立根。新根确认是另一个未来
预注册任务，须在看任何新根outcome之前固定cohort及选择规则；本协议不触reserved根，
不声明可沿用本cohort作fresh确认。P1/P3/C3/physical_release全部false。

## 8. 封存与停点

1. MAIN先确认原encoder/decomposition实际权威与本协议；本协议不读取其数值挑方案。
2. 后续实现只新stem，RAM fixture核context时序/完整history/action bits/同标签/缺项；
   R旧调用可复用资格及IO尺寸在生产前明确，阻塞则停，不自动扩实验。
3. 冻结source/tests/doc与全部候选8输入、预算和原模型sources，再由MAIN prepare并绑定SHA。
4. MAIN授权HOST CUDA；全部A预测与model-after先落盘seal，再解锁标签评分；原已见标签
   的知识无法抹除，seal仅防程序信息泄漏，不声称研究者盲法或独立确认。
5. 新strict完整验证本轮调用；saved分析保留全8、ties/missing、资源轴和继承证据边界。
6. 任何需要latest-slot/多模态factorial、跨runtime、额外R replay或新root的扩展，
   必须另立事前合同，不能从本轮有利结果自动触发。

未完成项：尚未实现producer/fixture/CLI，尚未核原planner每条R真实IO是否可逐一复用；
尚无本轮新模型输出或效能结果。本文是下一最小实验的设计，不是执行完成证书。
