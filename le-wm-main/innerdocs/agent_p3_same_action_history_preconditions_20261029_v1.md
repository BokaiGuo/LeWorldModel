# P3同动作history前置resolver：八槽saved prepare/inspect

本stem独占source/tests/本文，执行批准的
`agent_p3_same_action_next_protocol_20261029_v1.md`，不改任何旧件。
使用experiment-code组织接线和fixture；旧技能参考目录不可用，本任务约束优先：
无NN/physics/fullstrict/生产prepare/训练/绘图，worker仅RAM与临时保存文件。
MAIN审读后自行运行本saved CLI；没有CUDA launcher或授权运行入口。

## 输入与继承边界

显式传入六个绝对路径及实际SHA：encoder prepare/report/完整consume，
original shared-panel planner prepare/report/完整consume。没有未来SHA默认值。
encoder report内prepared与prepareSHA、encoder receipt内reportSHA逐一闭合；
planner同理，并核planner receipt的prepare绑定。
三planner文件的精确路径/SHA必须已出现在encoder prepare继承的bindings中。
固定封存encoder源码、decomposition源码、planner源码、批准协议SHA；
本三件套SHA加入新prepare，inspect完整确定性重建，拒绝源/测试/doc漂移。

原encoder schema/status必须complete；原planner schema必须own shared-panel，
完整receipt必须`complete=true`且`saved_shared_panel_verified_NOT_admitted`。
仅继承MAIN已经完成的encoder和planner完整消费，并不调用它们的consume/verify/build。
复用独占decomposition纯saved函数独立核16行向量/Q/身份/rawbits和原H3端点，
不导入producer，不签新的神经parity或物理strict，不重复全上游源/权重重hash。
`evidence_scope.full_upstream_strict_rebuilt_here=false`；模型IO数值真实性继承原receipt，
本resolver检验其实际保存bytes/接缝，不能证明模型本身预测正确。

## 固定8槽接线

顺序branch-major：s0/endpoints、s0/midpoints、s1/endpoints、s1/midpoints；
每branch起点10→15、15→20。不重选root/winner/动作，不把四branch当四独立root。
task/goal192/action_mean/action_std/model hash从原planner task逐字联接物理witness。
原full20为prefix5+H3 raw15，raw5从full20[start:end]取，causal raw10从原b5真实
raw10及已执行suffix计算；完整float32 bits固定。bool/nonfinite/非原生f32均拒绝。
normalizer严格原float32 subtract/divide一次；rawbox[-1,1]，std>0，无clip或投影。

A：独立从b5完整context及真实cp5/10/15/20编码逐次shift，核receipt的
actual_context_chains所有四时刻完整context。核每一步原causal raw10的两个action
token归一化。缺任何必要真实latent后context=null，不以预测补历史；
核schema/layout和完整3×192 latent、2×10 action tokens，不只核末帧。
R：从原winner.whole起点context取完整历史；A/R action history必须逐位相同。
输出A_predict_input包含完整latent history、future raw5及归一化后outgoing3 tokens；
act_emb必须未来原action_encoder真实产生，不能借用R act_emb冒充A执行。

## R真实IO资格：不得伪造zero-cost复用

每branch读取原planner report中对应`s{root}_adaptive_{bank}` b5 node。
node.request/result须精确等于原物理plan.source_witness.request/result；
H3 panel.winner/selected_path、原selected horizon、resource receipt全部一致。
不因另一个候选/segment存在相近预测而借用其结果。

从该node目录events.jsonl认证完整物理文件SHA、每条journal链index/previous/digest、
最后tail；canonical为compact JSON无LF参与digest，writer每record单LF。
空行/双LF、断尾、重复键、错SHA/tail均失败，不过滤空行。
整journal只做链验证；非目标scope其它实验事件的科学/完整资源验证继承MAIN receipt。
目标scope精确为slot/mode/H3/winner_index/selected，全部保留，不过滤unknown。

目标scope必须正好三边×10事件、phase=whole、step=0/1/2，逐生命周期：

1. program_packet：latent/history/raw/mean/std/token/outgoing。
2. action_encoder input/return。
3. predict input，predictor input/return，pred_proj input/return，predict return。
4. program_boundary。

所有blob严格native `<f4` shape/data/base64/SHA，完整输入/输出三行保留。
每对operation_index唯一且回到同一node资源账的正常returned/error=null操作；
资源类型、scope、input shape全部一致，不靠末尾assert猜成功。
核literal raw5、normalizer、token/outgoing、完整latent/history、实际act_emb连接、
predictor→projection reshape→full3输出、末行shift到boundary及原winner每边path。
三边均通过才给该branch两槽`R_reuse_verified=true`。
`R_proof`含原record indices、predict操作ID、输入/输出SHA、完整3行输出和末行预测；
`R_journal`指向已认证真实文件，不把逻辑checked_path当真实调用证据。

缺node/journal/必要字段、chain/IO验证失败则该branch两槽blocked并保留首因。
authority SHA/schema、A构造、全局动作/目标等冲突则拒绝prepare，而非产出ready。
R失败不退款、不重放、不fallback，`new_R_calls=0`不等于“R已经可用”。
实际标签缺失亦blocked，分母始终8，禁止subset准备后冒称全8ready。

## 输出与预算

prepare schema：`agent_p3_same_action_history_preconditions_20261029_v1.prepare`。
status是`all8_saved_preconditions_ready_NOT_execution`或`blocked_saved_preconditions`。
`jobs[8]`每项含身份、start/end、原goal/task/model/选择digest、full20/raw5/raw10及bits、
完整R/A contexts/digest、A输入recipe、R真实证据、真实label行指针及向量、缺项原因。
标签仅供saved evaluator；未来runtime应在所有A预测seal后才解锁评分，不能按标签选job。
本prepare不是部署许可，不表示新A调用执行完毕。

新增未来上限8predict、batch samples8、input token-frame samples24；原R复用新增0。
native action_encoder/predictor/pred_proj各上限8；J/VJP/backward/solve/SVD/encode/projector/
physics/training全0。原planner/encoder付费资源单列inherited_resources，不重置为0成本。
本resolver新增神经/物理调用全部0；FLOPs未知、equal_total_compute=false。

按原packet+全nativeIO+boundary共19数组，每A共6460个float32=25840B，
8A有效tensor206720B，逐blob base64实际275712B，保守上界275760B。
预计80个native数组事件，每event额外JSON metadata事前限64KiB，另计jobs序列化bytes、
8MiB final reserve，合计必须<=64MiB；metadata128MiB、floor200MiB。
输出`job_serialized_bytes`和`IO_budget.projected_bounded_output_bytes`，不是仅数predict。
未来writer必须落实这些cap；本resolver静态可装下不等于未来实现已经合规。
wall600s/allocated3GiB/reserved4GiB沿批准合同；resolver无GPU，未测运行时wall或峰值。
未来增加任何R replay或其他native数组都必须另立合同，不能暗加在本预算。

## MAIN命令

工作目录 `/home/gbk/code/LeWorldModel/le-wm-main`。以下六路径为MAIN实际绝对路径，
不能把output目录直接冒report文件。prepare/inspect仅saved，不跑完整strict。

```bash
.venv/bin/python -m pytest -q tests/test_agent_p3_same_action_history_preconditions_20261029_v1.py

.venv/bin/python -m experiments.spectral_reachability.agent_p3_same_action_history_preconditions_20261029_v1 prepare \
  --encoder-prepared "${ENCODER_PREPARED:?}" \
  --encoder-prepared-sha 8689f3a0014d264e135ae2dbdf04f3cd2f6c91a15d6d26d14751d6daad87f9b1 \
  --encoder-report "${ENCODER_REPORT:?}" \
  --encoder-report-sha c3195ea6ba35898bfb0eaef8f9dd69e7e2f4e01834c7a2a43c6c73302352c539 \
  --encoder-receipt "${ENCODER_RECEIPT:?}" \
  --encoder-receipt-sha 671e48b0e953e6adeb8e07381ecaa965572afa8801d29d961c344eb82a3e505e \
  --planner-prepared "${PLANNER_PREPARED:?}" \
  --planner-prepared-sha 7ca85050eeab4e47ed35cf84c284ed7e08fc64bce8064d7db9450074045b643a \
  --planner-report "${PLANNER_REPORT:?}" \
  --planner-report-sha 668db35d79eb650419e4efa42dfc21a15cdf91725700b2f202e2a2ec975bcd6a \
  --planner-receipt "${PLANNER_RECEIPT:?}" \
  --planner-receipt-sha d23ffa48bc954306bc27a15f924da8daafffc9b4bc76a91e59aa949ab1abbafd \
  --output /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p3_same_action_history_preconditions_20261029_v1_prepare_MAIN01.json

.venv/bin/python -m experiments.spectral_reachability.agent_p3_same_action_history_preconditions_20261029_v1 inspect \
  --prepared /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p3_same_action_history_preconditions_20261029_v1_prepare_MAIN01.json \
  --prepare-sha "${PRECONDITIONS_SHA:?actual generated prepare SHA required}"
```

写文件exclusive xb，with内flush/fsync，之后目录fsync；拒绝覆盖旧文件。
inspect确定性重建saved接线并逐字比较prepare，不能伪装whole upstream严格重跑。
blocked也保存8槽可inspect，但不是CUDA release，不允许以ready子集冒用完整实验。

## 验证与剩余项

测试仅RAM/临时文件，含真实native blob、journal writer→reader链、完整三边IO、
全六外部SHA、动作/actemb/normalizer/context/operation/unknown/phase/completion篡改，
missing保持8槽、exclusive写和source绑定篡改。没有模型加载、CPU神经或生产strict。
本实现此前仅只读查看真实schema代表字段和canonical/noLF ABI，不把此窄查看称新认证。
实际八槽能否全部ready、完整prepare byte size由MAIN生产saved调用给出；worker未执行。
最终独占fixture回归48 passed in 1.54s；prepare --help通过。测试实算19数组/调用与
全8base64 275712bytes，覆盖独立重算SHA后的错误authority接缝、固定source pin拒绝。
尚未实现后续8A HOSTCUDA runtime，本任务仅前置resolver，不声称后续实验已可运行。
全部P1/P3/C3/physical_release=false，known2诊断不是新根/成功或因果效果确认。
