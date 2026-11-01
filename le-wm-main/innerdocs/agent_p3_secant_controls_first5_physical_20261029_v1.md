# Stage-A：固定 known2 Z/P first5 原生对照

仅独占新 stem：agent_p3_secant_controls_first5_physical_20261029_v1。
MAIN 审读后执行；worker 只 NumPy/RAM mock native arrays、临时真实 Store、saved 字段测试，
不执行生产 prepare/fullstrict、NN、physics、网络或子代理。本文不是运行报告。

继承事前协议 innerdocs/agent_p3_secant_same_time_capability_protocol_20261029_v1.md，
SHA b35af390fc74e6ebd44a9879784308ec74947fb28f640ce5e795cc263e65dccf。
本交付只有阶段A；不实现 encoder、J、能力probe、对照成败分析，不改变科学门。

## 固定分母、动作、历史

四行固定顺序 (root0,Z)、(root0,P)、(root1,Z)、(root1,P)，全是 off 原生条件。
Z 是 little-endian float32 正零 (5,2)，有符号零逐位核验；P 是原 shared-panel
/layers/{root}/commands[5:10]，原 line_search_GN 已固定 continuation。
每条从原 reset/place、warm10、原 prefix5 重放，不 hydrate cp5 或共享 live env。
prefix 必须同时逐位等于原 layer commands[:5]、actual-cp5 provenance.prefix5、物理 reference。
不 clip/tanh/inverse/reoptimize/reselect；原 raw [-1,1] 合法、无损f32才可准备。

每条在 control6 intent **之前** durable accepted cp5_guard：
完整51 off/on/off帧 state/snapshot/PNG/pixel exact；真实返回 warm10/prefix5；
incoming raw10=warm[-5:]+prefix5；mean/std 恰一次原转换；原 abs5/10/15 三帧及
完整 latent3/action_history/context/task/goal 绑定。拒绝门不调用后5。
新5动作只commit至control10=absolute20；不扩大horizon，不做候选fallback。
原 native termination/truncation 不屏蔽；没有实际cp10就 missing，不把早停终点冒充cp10。
工程失败阻止后续行派发；四行总保留。native早停则继续其它固定行，不补齐动作。

C 的四个逻辑槽仅引用继承 authority：原 root0/H1 endpoint 与另三 no_witness null。
Z/P不填补 no_witness，不重跑C，不增加独立样本。12逻辑角色槽及未来8比较槽仍由原协议定义；
本文件只新采4条共享于H1/H3的控制轨迹，不把引用当重复物理采样。

## 固定来源和信任边界

相对路径根 /home/gbk/code/LeWorldModel/le-wm-main/。源码 AUTHORITIES 是绝对路径映射：

|权威|相对路径|SHA256|
|---|---|---|
|C prepare|results/agent_p3_secant_commit_physical_20261029_v1_prepare_MAIN01.json|a085c3c6a609664e61305b156ce68d1cbdf44296634a42a8bc5416e7091fe295|
|C report|results/agent_p3_secant_commit_physical_20261029_v1_physics_MAIN01/report.json|f64d2843ecf0c217a5581d25a10fa77ccf78c372bf1ddc2aca8995ab4af0a616|
|C completion|results/agent_p3_secant_commit_physical_20261029_v1_physics_MAIN01/completion.json|f68fb8ef6638a2fa3c74a8139d1b7a517a7c2d36df02122104b759998b470d75|
|C full receipt|results/agent_p3_secant_commit_physical_20261029_v1_consume_MAIN01.json|d0c5a34b14dc84ffece573d904e3950e54628df307987ee77fcb6265ca91b638|
|C frozen source|experiments/spectral_reachability/agent_p3_secant_commit_physical_20261029_v1.py|a065c8f652919d8315f7848c70e9b944ad6aec0717bc6bbe310b54b224c72aea|
|P prepare|results/agent_p3_shared_panel_20261029_v1_prepare_MAIN01.json|7ca85050eeab4e47ed35cf84c284ed7e08fc64bce8064d7db9450074045b643a|
|P full receipt|results/agent_p3_shared_panel_20261029_v1_consume_MAIN01.json|d23ffa48bc954306bc27a15f924da8daafffc9b4bc76a91e59aa949ab1abbafd|

本次以 MAIN 持久完整C receipt 为显式信任起点，不重新调用 C.consume/C.verify/上游
model consumer。核C四权威SHA、schema、四整行/plan、counts、errors、completion、prepared、
source/P绑定与所有flags；核原唯一cp10确实到达，另外三行未执行；全 closure 文件逐项SHA。
P receipt核实际schema/status/prepare身份；P原task-export的**文件SHA**与解压后gn_task
身份分开核验，绝不把gzip字节SHA当成任务JSON digest。
prepare/verify 再完整重建新prepare、核新source/tests/doc/protocol与原source/model/assets闭包。
这是继承认证，不新认证历史真实性或科学准入；实际C费用保留在 planner_authority.inherited_counts。
本job inherited=0 仅表示执行账不虚增继承操作，绝不表示C没付费。

复用范围限于原低层 constructor/place/goal绑定、warmup_commands、native_hooks、episode、
Meter操作计账、prefix/labels原生算术、原task parser和SHA/JSON codec。
新执行循环、schema、guard、Store、publication、consumer独立；不调用旧高层runner，
不 monkeypatch 冻结模块。仅 RAM tests 对 mock env 接口临时 patch，生产路径不patch。

## 整数资源与字节证明

完整4行原生资源上限，不为达到上限padding：

|资源|每行|全4|
|---|---:|---:|
|constructor / close（各）|1|4|
|reset / set_state|2 /4|8 /16|
|goal_render / goal_bind / goal_setter|2 /1 /3|8 /4 /12|
|public_step|20|80|
|physics_step|204|816|
|explicit_render / PNG（各）|66|264|

20=warm10+prefix5+control5；204=2reset×2内部步+20×10。
66=3×[(warm0..10共11)+(plan0..10共11)]。
getobs内部次数无独立hook，仍unknown，不填0；NN/训练新调用为0。

RGB8 224×224：scan=224×(224×3+1)=150752 B；
zlib compressBound 加 1024 B PNG headers/chunks 保守余量得到每PNG上限151834 B。
264×151834=40084176 B；所有非PNG合计硬配额17MiB=17825792 B；
二者57909968 B，加8MiB收尾reserve=66298576 B，
严格小于64MiB=67108864 B，余810288 B。不扩cap。

PNG实际编码长度/数量、所有非PNG累计长度分别在每次写入前硬核；
64MiB总cap、非收尾预留8MiB、disk free floor200MiB也每次核。
17MiB不是“12份copy”猜测：完整真实Store测试直接序列化全树并量字节，且生产硬限制累计
包含report/completion/journal/binding，不能把某类副本排除。任何实际metadata超额必工程失败，
保partial，不删context/字段/降精度/分母；该预算不保证未来未知错误栈必能收尾。

全4实际RAM lifecycle 的整数持久副本：
1216 operation receipts=完整caps各资源总和，journal intent+terminal 共2432事件，
report再持久1216份（共三份，不计作三次native）；264 fresh_observation事件，
204 prefix帧=4×51，随 branch_closed 与 report.rows 两份；
4 branch_closed全行+report.rows全4行，完整plan/context/provenance均已含在行序列化中；
40 own_response事件及响应行，80 returned_transition事件；
trajectory/checkpoints/last_observation/endpoints嵌套副本也包含在实测journal/report字节，
不是只测prefix后乘任意因子。execution_binding、contract、C引用的重复字段同样入配额。

RAM fixture使用真实临时Store、fsync、完整journal/hashchain、全部264 PNG解码和consumer；
另测264张不可压缩noise PNG与整17MiB metadata同目录，证明硬预算可共存。
fixture不是生产physics或真实结果认证。精确实测数与最终测试结果见交付消息；
源码 consume 返回 actual_serialization_census，MAIN生产后可独立核真实字节和事件数量。

执行wall900s从 execute 的Meter起（包括native/IO、最终report publication/hash），
不把独立saved prepare/verify算成物理执行wall；completion的inclusive wall绑定report fsync及
全artifact hash。completion自身fsync/hash另做最终900s检查，失败durable completion_failure，
consumer拒绝。自绑定不能声称覆盖completion自身最终hash，scope明确保存。

## Report / consume ABI

flat目录：execution_binding.json、canonical exactly-one-newline journal.jsonl、
264以内独立RGB224 PNG、report.json、completion.json；失败可能有completion_failure.json。
不产NPY，不写其它stem或MAIN旧产物。成功run stdout只输出路径、report/completion SHA、
status与计数（小于2KiB），完整report只落盘。

report.schema = agent_p3_secant_controls_first5_physical_20261029_v1.report，
成功状态 all4_ZP_first5_native_recorded_NOT_admitted。
report.rows[0..3] 保完整原生轨迹、prefix51、native7 metrics/reward/success、raw动作、
checkpoints、missing_checkpoints和状态；未到终点保持null。
report.endpoints[0..3] 固定同序，每项 {index,root_slot,role,control_step:10,
absolute_step:20,available,checkpoint,missing_reason}；checkpoint只来自真实step10。
保留实际早停final_state不等于可用cp10。新encoding应只接通过本consumer的available端点，
本切片不授权/运行任何编码。

消费API consume(output, expected_report_sha, completion_sha) 返回 own .consume，
全4 rows、endpoints、counts、errors、C引用、完整bindings、actual_serialization_census、
saved_PNG_verified，全部科学门false。consumer 独立重建全部新journal、operation hashchain/
nested intent-return/caps、原51 guard顺序、raw bits、所有native labels及goalphase、PNG file/pixel/
state/snapshot、原off/on/off恢复、完整文件集合与completion/wall、完整closure终态SHA。
读取输出前先核总文件数/大小避免无界读入。receipt不伪装旧C成功或新科学证明。
partial报告可审计；缺失或publication失败的完整strict必须拒绝，不能用status替代完整证据。

## MAIN 命令（worker不执行）

以下只创建新MAIN01；如存在须另选新suffix，不覆盖。

~~~bash
cd /home/gbk/code/LeWorldModel/le-wm-main
.venv/bin/python -m pytest -q -s tests/test_agent_p3_secant_controls_first5_physical_20261029_v1.py
M=experiments.spectral_reachability.agent_p3_secant_controls_first5_physical_20261029_v1
P=/home/gbk/code/LeWorldModel/le-wm-main/results/agent_p3_secant_controls_first5_physical_20261029_v1_prepare_MAIN01.json
O=/home/gbk/code/LeWorldModel/le-wm-main/results/agent_p3_secant_controls_first5_physical_20261029_v1_physics_MAIN01
R=/home/gbk/code/LeWorldModel/le-wm-main/results/agent_p3_secant_controls_first5_physical_20261029_v1_consume_MAIN01.json
PREP_SHA=$(.venv/bin/python -m "$M" prepare --fixed-authorities --output "$P" | jq -er .sha256)
.venv/bin/python -m "$M" verify --prepared "$P" --prepare-sha "$PREP_SHA"
.venv/bin/python -m "$M" run --prepared "$P" --prepare-sha "$PREP_SHA" --output "$O" --authorization MAIN_P3_ZP_FIRST5_PHYSICS_ONLY
REPORT_SHA=$(sha256sum "$O/report.json" | cut -d' ' -f1)
COMPLETION_SHA=$(sha256sum "$O/completion.json" | cut -d' ' -f1)
.venv/bin/python -m "$M" consume --output "$O" --report-sha "$REPORT_SHA" --completion-sha "$COMPLETION_SHA" --receipt "$R"
~~~

这些变量只从实际文件派生，不预填未来SHA；MAIN须确认run/consume exitcode，不只读取stdout。
待MAIN实际guard/physics/strict，当前交付未认证新Z/P终点，也不能证明非贪心收益或能力改变。
P1/P3/full_C3/scientific/freshness/independent_confirmation/physical_release 全false。

