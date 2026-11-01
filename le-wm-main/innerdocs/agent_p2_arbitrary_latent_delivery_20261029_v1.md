# Exp1/P2 known2 arbitrary FIRST：真实输入绑定与纯数值交付

独占新source/tests/doc；依MAIN批准的`agent_p2_arbitrary_latent_waypoint_protocol_20261029_v1`
（SHA179530bcfe842a678546daf66906d626d20765d1ba8e6f56c9abfd15ec3b540c）。
不覆盖旧delivery的arbitrary missing，不NN/physics/fullstrict/训练/子代。
experiment-code技能仅用于固定循环、完整失败账及回归；不采用其训练/调参/按效果择优建议。

## 算法与三阶段

`prepare`只读并认证固定MAIN delivery SHA82b36fa3030f8ebb2da37bdc91ba3e8d582f129d6bc9e5c9f8e774b363ef3f62，
导入其parity/tracker完整已存receipt，核report→prepare、receipt→report及全部delivery sources字节。
仅读取选定parity history/action/真实goal三个NPY/根，独立检查header/EOF/shape/dtype/finite/数据及文件SHA。
goal固定来自`rows[root].pixels[3].paths.hi.output`，绝不是prior target或program endpoint。
原六槽、原missing、known2/off、task/RGB及四分支共同z/a均核，保留两个输入slot。

`optimize`须MAIN显式`--authorization MAIN_NUMERICAL_ONLY`，先完整重建自己的prepare，
再固定两根完整执行原geometry surrogate。只调用冻结core.objective，使用原循环/运算次序：
三latent变量以当前末帧重复初始化，64步eta16；loss=原identity192 goal MSE+相邻平滑MSE。
不调用head/J/SVD/solve。保留0..64全部history/loss/gradient0..63；固定最终FIRST，
只在最终交付一次float64→float32 cast，不投影/再归一化/截断/回退。
数值循环与原core.optimize的完整输出history/targets逐值回归，未猴补任何旧globals。

`consume`重新解析自己的prepare绑定，不调用旧producer consumer/fullstrict。
从原生新journal完整重算每个实际返回的loss/gradient/update，独立核FIRST/context/goal/传输bytes，
不只核stored status；未知事件/全生命周期移位/额外文件或payload一律拒绝。
完整失败前缀可验收为partial，不称成功，不模拟失败原因已被独立复现；重新算数的CPU费用另行披露。
receipt的verification_cpu_calls逐轴给实际复算loss/analytic_gradient/update次数；失败但未返回的原操作不重做。

## 来源与科学边界

冻结旧optimizer三件套、旧delivery三件套、批准协议及原父subgoal协议source均有源码内literal pins；
本新三件套动态SHA封入prepare并前后复核。实际运行时不改旧source或结果。
模型、encoder、preprocess、normalizer、Q、raw-domain等coordinate_refs指向原tracker prepare完整metadata，
并继承425项expected_state摘要；context/goal绑定原parity authority和所选数组真实bytes。
rehash确实读取每个继承binding的原字节，不应概括为“所有来源仅metadata、不重读任何权重”。
本轮只读核178条delivery sources路径：含14份已运行输入/输出NPY，其余为源码/文本/JSON；
未见checkpoint/safetensors权重文件。这是实际路径清单核对，不是新的权重inventory认证。
代码对全部bindings一视同仁做字节hash，不加载/反序列化模型；即使上游绑定包含权重字节，
hash也不等于重建模型或训练身份。不独立重建原神经parity或完整upstream strict。
Q只允许原tracker的identity192声明，不拟合Q；既有z/a不再normalize，原raw inverse缺口不修补。

优化器只收到旧core白名单task：root/off、完整context、真实goal及coordinate_refs。
历史action是过去causal token，不是未来见证。外层parent metadata虽含其它臂，但未来commands/
prior目标/程序endpoint/native标签不传优化器；payload严格只有z/a/target。
本geometry loss实际只依赖最后latent，不能称利用全部history的模型推理。

原core的checkpoint标签5/10/15不是本FIRST的物理时间证明；新row的source_time=null。
prior时间未知、program首目标来自raw5、共同tracker H3不同：time_alignment/same_generation_budget/
equal_FLOPs/P1/P2/E1/scientific_admission均false。目标优化不是可达性/价值head，不承诺效能。

## 真实新Store及资源/失败

自己的canonical compact JSON没有LF；Store.emit只附一个LF。event包含index/previous/kind/payload，
hash=SHA256(canonical(event))；reader核原字节canonical、每条链、全事件顺序和census，不filter空行。
每个intent/return及put文件均flush+文件fsync后再进行后续操作；新文件还fsync目录，排他xb不覆盖。
全作业run_started→两root_started→固定loss/update intent/return或failed→root_finished→run_finished。
在当前root失败仍保留另一root槽，不能择根/跳过失败分母。

每根正常65loss/64解析梯度/64更新；两根130/128/128，均分attempted与captured returned。
loss和gradient在同一次core.objective执行，分别记逻辑算子，不虚称两次神经调用。
失败intent不退款；未能持久写intent则无算数dispatch/无debit，failed事件明确intent_recorded=false。
partial完整history及最后params保留，target/payload=null，禁止回退goal或别的iterate。
模块返回但return日志未保存的情形按未捕获返回处理，不能推断未保存的实际内部完成数。
未执行NN/encoder/projector/predict/J/VJP/backward/physics/controller/训练，新调用全0。
继承费用不退款；FLOPs未知。原共同controller的未来预算不在本工具执行或冒领。

输入metadata每文件16MiB，原NPY每文件4MiB；全output16MiB、final reserve1MiB、disk floor200MiB，
数值stage协作wall300秒。起点在run调用verify之前，贯穿完整prepare/source/所选数组重读、
Store、算术、cast运输、全部payload/journal落盘、后置rehash及最终report文件fsync与SHA。
正常事件先查wall；failed/final仍受总byte/floor但允许wall超限后写结尾，不把timeout写成complete。
verify已耗尽预算时，两根都保留partial，零优化dispatch/debit，不在verify后重置时钟。
report的whole_numerical_wall是起点至report写入前的快照；随后completion.json记录
whole_stage_wall_through_report_fsync及完整状态，绑定report SHA，计入report实际写入时间。
这个末尾acknowledgement自身写入不在已记录边界内，consumer核其精确边界；无自引用计时声明。
consumer同时核两处wall，只有两根数值完成且inclusive wall<=300才给complete/all2 status。
末尾report写盘超时即使算术两根已完整，receipt也仅partial，不将完整算术目标称为预算内完整作业。
wall是保存账的边界核对，不是独立外部可信硬件计时认证。无硬实时抢占保证。
单独prepare/verify命令及consume复算的额外CPU费用另列，不能当零成本或退款。
进程被杀、磁盘满或绑定漂移可能只留下部分文件，无完整report/receipt，必须failclosed，不能伪造complete。

## MAIN准确CLI（worker仅fixture，不对actual调用以下优化/消费）

```bash
cd /home/gbk/code/LeWorldModel/le-wm-main
OPENBLAS_NUM_THREADS=1 .venv/bin/python -B -m pytest -q -p no:cacheprovider tests/test_agent_p2_arbitrary_latent_delivery_20261029_v1.py
.venv/bin/python -B -m experiments.spectral_reachability.agent_p2_arbitrary_latent_delivery_20261029_v1 contract
.venv/bin/python -B -m experiments.spectral_reachability.agent_p2_arbitrary_latent_delivery_20261029_v1 prepare \
 --output results/agent_p2_arbitrary_latent_delivery_20261029_v1_prepare_MAIN01.json
# PH替换为上一步真实SHA；不预填未来值。
.venv/bin/python -B -m experiments.spectral_reachability.agent_p2_arbitrary_latent_delivery_20261029_v1 verify \
 --prepared results/agent_p2_arbitrary_latent_delivery_20261029_v1_prepare_MAIN01.json --prepare-sha "$PH"
.venv/bin/python -B -m experiments.spectral_reachability.agent_p2_arbitrary_latent_delivery_20261029_v1 optimize \
 --prepared results/agent_p2_arbitrary_latent_delivery_20261029_v1_prepare_MAIN01.json --prepare-sha "$PH" \
 --output results/agent_p2_arbitrary_latent_delivery_20261029_v1_numerical_MAIN01 --authorization MAIN_NUMERICAL_ONLY
# RH替换为实际report SHA。
.venv/bin/python -B -m experiments.spectral_reachability.agent_p2_arbitrary_latent_delivery_20261029_v1 consume \
 --prepared results/agent_p2_arbitrary_latent_delivery_20261029_v1_prepare_MAIN01.json --prepare-sha "$PH" \
 --output results/agent_p2_arbitrary_latent_delivery_20261029_v1_numerical_MAIN01 --report-sha "$RH" \
 --receipt results/agent_p2_arbitrary_latent_delivery_20261029_v1_consume_MAIN01.json
```

receipt的plans[2]为新arbitrary目标交付：含原task/history/action/goal摘要、完整numerical文件SHA、
target64、target32 bytes摘要、z/a/target payload、资源和failure；不是旧schema receipt。
实际tracker/HOSTCUDA/physics仍需MAIN另行授权；本工具没有此入口。原missing不改写。
测试结果及冻结三SHA在交付消息列出；本轮未运行真实prepare/优化/consume。

最终fixture验收：新59项加旧冻结数值核28项共87项通过；包含CLI相对路径完整链、
verify耗尽预算零dispatch、report fsync超时partial及重封SHA后的伪造超wall完整状态拒绝。
所选实际metadata只查看尺寸，127KiB/330KiB/931KiB/403KiB，均在16MiB reader界内，未调用实际build。
