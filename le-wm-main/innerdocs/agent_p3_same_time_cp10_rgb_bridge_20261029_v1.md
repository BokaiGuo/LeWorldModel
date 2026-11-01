# 五个实际 cp10 端点 RGB → actual-history task 桥

独占 agent_p3_same_time_cp10_rgb_bridge_20261029_v1 source/tests/innerdoc。
仅阶段A编码/任务接线，不实现7J/五U/probe，不重复物理、不训练、不拟合Q、不计算新Q或排序。
MAIN独立审读后HOST CUDA运行；worker只RAM mocks、saved读取、临时磁盘ABI回归。

沿已批准 same_time_capability 协议
innerdocs/agent_p3_secant_same_time_capability_protocol_20261029_v1.md，
SHA b35af390fc74e6ebd44a9879784308ec74947fb28f640ce5e795cc263e65dccf。
按 experiment-code 技能组织来源、日志、失败分母和RAM测试；不采用训练/调参/择优运行建议。

## 固定来源与信任边界

固定顺序 (root0,C)、(root0,Z)、(root0,P)、(root1,Z)、(root1,P)，均为原off条件。
原C四组分母另外三no_witness保持null；本桥不将Z/P解释为这三组的fallback。
五端点全部已到达实际control10=absolute20，本版本prepare缺任一即拒绝，不筛根或补早停帧。

|来源|实际SHA256|
|---|---|
|C prepare|a085c3c6a609664e61305b156ce68d1cbdf44296634a42a8bc5416e7091fe295|
|C report|f64d2843ecf0c217a5581d25a10fa77ccf78c372bf1ddc2aca8995ab4af0a616|
|C completion|f68fb8ef6638a2fa3c74a8139d1b7a517a7c2d36df02122104b759998b470d75|
|C full receipt|d0c5a34b14dc84ffece573d904e3950e54628df307987ee77fcb6265ca91b638|
|ZP prepare|54adc4f5c5183815f9e78c660ec0c3026f52f5ecab8056c322307d6c5fad7f5a|
|ZP report|459c30ef31d2b062fa8ddcafbddaa6d125972706edd7458811bab51a654bc646|
|ZP completion|4dfd3f22f78469a01d6b71dbba8fbf921a8bd35730727cc7d8ff844123c87eef|
|ZP full receipt|c58ca1e43439841a3e592edb1128f95e2b5b8375af8f686052ab662930e74b50|
|原已审encoder runtime source|c3c7f4421ffe843acc845ba9ef1bc19424ae66e4d08fa96b7a3fc63f22bb4748|
|冻结ZP source|e9403bd9608dcd43bde780b538e60d24711c51ca568e012083d855d47e47096c|

完整绝对路径来自源码 descriptors()；根为 /home/gbk/code/LeWorldModel/le-wm-main/。
C stem=agent_p3_secant_commit_physical_20261029_v1；
ZP stem=agent_p3_secant_controls_first5_physical_20261029_v1；
prepare为 results/{stem}_prepare_MAIN01.json，report/completion在
results/{stem}_physics_MAIN01/，receipt为 results/{stem}_consume_MAIN01.json。
encoder source 为 experiments/spectral_reachability/agent_p3_h3_witness_encoding_20261029_v1.py。

以MAIN这两份完整持久receipt为信任起点，核所有外部SHA、schema、完整rows/plan/counts/errors、
completion/prepare/source绑定、C+ZP同root/goal/history身份、原51 exact guard通过字段与所有flags；
合并完整source/model/artifact闭包逐文件SHA，另独立解五张PNG和三帧history文件SHA。
**不递归调用旧physical consume/verify/fullstrict**，不声称在本桥重新认证全部历史物理。
新编码的全部native IO、guard chronology、task/context、输出和资源由本consumer完整重建。
upstream_fullstrict_reconstructed_here=false；这不是削弱旧门或伪造旧strict。

## 真正的 history 与新task ABI

每条旧cp5任务的latent3来自abs5/10/15。新任务严格取旧latent_history[1:3]（实际abs10/15），
追加该角色原始实际cp10 off PNG的新native192编码（abs20），不使用预测latent或编码native7状态。
历史RGB的abs10/15 source file/pixel/state/snapshot须与该实际轨迹plan0/5完全相等。
保原context元字段，shape=(3,192)，layout=states_with_between_state_actions_v1。
incoming raw10=**原prefix5 + 该角色实际commit5**；不是warmup tail。
对这10步用原f32 mean/std恰一次 subtract/divide，reshape(2,10)，连接原action_history ABI。
无额外 /100、clip、tanh、inverse或Hi scaler。goal/projection/mean/std/model/root语义原样保留。

输出task仍严格原六字段：
binding、root_context、goal、projection、action_mean、action_std。
binding保原root/model/goal/projection/normalizer/action_domain/blind身份；
context_sha256为新context；task_id、task_sha256、task_contract_sha256用本stem命名空间、
旧cp5 task、角色、真实三帧source、raw10、两份物理authority与新latent NPY收据重新生成。
cp10 task_id必须不同于旧cp5 ID；不声称这是新fresh root或新物理反事实。

### 可交给阶段B的 output/task ABI

flat目录 cp10_tasks.json 是固定5项数组；只有本桥完整consume通过后可使用。
每项包含：

- index/root_slot/role（固定顺序如上），absolute_steps=[10,15,20]；
- task（六字段原ABI）、task_sha256（解析task的原compact producer digest）；
- identity（namespace、本角色、history_sources[3]、raw10 SHA、旧cp5 task SHA、
  原source_report/source_receipt path+SHA、新encoded_latent完整NPY收据）；
- base_task_id、old_cp5_task_id、source（实际cp10 PNG/native checkpoint）、
  causal_raw10（原literal 10×2）、全部科学flags=false。

task.root_context.latent_history=(3,192)，action_history=(2,10)，其余元字段保持原context ABI。
cp10_tasks.json **文件SHA**是report.tasks_sha256；与每项解析task_sha256不是同一个digest域。
同role即使PNG或编码偶然相同也各付一次新编码，不据结果去重。
后续阶段B不能仅信task JSON，须显式接本桥report+completion+完整consume的外部实际SHA；
consumer输出tasks[5]、全部bindings、source_authorities和counts，可作为明确消费接口。
本桥没有任何J、solve、U、prediction/path、编译资格或physical release。

## 编码、封存、限制

直接复用已审原moving模型loader/model.encode、图像预处理、native Trace/forbidden guard，
不改旧source或模型；TF32off、noautocast、inference_mode，sdpa_kernel(MATH)请求并核实际开关。
保存完整encoder输入(1,3,224,224)/hidden(1,257,192)、CLS(1,192)、projector输出(1,192)、
外部返回192、原RGB与预处理(1,1,3,224,224)。
模型tensor hash和所有class source inventory前后不变、policy/config/weights文件SHA完整绑定。
MATH只为请求+observed flags guard，不包装为实际GPU kernel dispatch证书。

新费用严格各5 encoder/projector API、samples、frames；预测/AE/AD/physics/训练为0。
native Guard原底层以4n容量定义，独立Ledger在其前额外限制**恰5图/10module entries**；
不是允许第6图，也不把底层预分配容量当调用数。全部禁止API保持原guard并finally恢复。

五个新编码全部持久 native_seal.json **先封存**，才构造任何新task。
事件50为 all5_native_sealed；事件51为 cp10_tasks_published。
编码失败保五行roster，未成功槽不伪造latent；不重试，不出可用partial task子集。
上游原goal本就已知，未将新native终点指标或未来label用于任何编译/筛选。
native_response仅继承并分栏作身份/原生观察，不作为task输入；不把RGB Q当成功。

## 工程 shape/byte 证明

沿已有RGB工程上限64MiB（8MiB reserve）、floor200MiB、wall600s、GPU allocated3GiB/reserved4GiB。
这不是科学预算匹配或新算法费用优势。shape_census()先验整数计算：

|每图无损保存数组|shape|原始B|
|---|---|---:|
|RGB|224×224×3 uint8|150528|
|preprocessed|1×1×3×224×224 f32|602112|
|encoder input|1×3×224×224 f32|602112|
|encoder hidden|1×257×192 f32|197376|
|projector input/output，各|1×192 f32|768各|
|外部latent|192 f32|768|

每图7份，共35份，raw stored array bytes=7,772,160。
每NPY实际校验128B头，gzip含wrapper保守compressBound；35文件合计上界7,780,080 B。
全部JSON累计硬配额16MiB=16,777,216 B，覆盖event/roster/seal/tasks/report/completion与其内嵌副本，
不凭任意copy factor猜测。加8MiB reserve，合计32,945,904 B <67,108,864 B。
真实module IO单独为4,005,120 B（encoder input/output+projector input/output），不可与所有保存副本混同。
metadata超限直接工程失败，不能减精度/删字段/改分母；不保证未知错误栈在任意情况下必能完成publication。

Store每次核实际cap/reserve/floor、数组shape/dtype/35数量、gzip上界和metadata累计。
完整正常文件35 arrays +52 events +binding/roster/native_seal/tasks/report/completion=93。
完成report fsync及全artifact hash后发布completion；再核completion自身hash/fsync后的最终wall。
越界保存completion_failure并拒绝strict，consumer独立核全文件/canonical bytes/census。
模型/源/序列精确检查不过则partial，不松动门。

## RAM验证范围

tests含真实已存source身份与三帧chronology检查，不执行旧完整consumer；
完整build在临时磁盘构造显式fixture authority路径/SHA、保原actual PNG和任务结构，
原policy只替换为明确fixture字节（**从不加载模型**），不制造生产receipt。
走完整build→verify→真实Store→fake CUDA-shaped NumPy hooks→native seal→task publication→完整consume，
不patch掉本桥consumer、physical_authority或task构造验证。
另做噪声全35数组+16MiB JSON quota共存、禁用predict/AE/AD、错误197/256tokens、漏槽、
history/源/动作/guard/seal时序/文件/模型/precision/资源篡改拒绝与cold import。
fake tensor device标签是fixture，不是CUDA执行证据；真实主机编码/模型加载/wall仍由MAIN认证。

## MAIN CLI（worker不执行）

~~~bash
cd /home/gbk/code/LeWorldModel/le-wm-main
.venv/bin/python -m pytest -q -s tests/test_agent_p3_same_time_cp10_rgb_bridge_20261029_v1.py
M=experiments.spectral_reachability.agent_p3_same_time_cp10_rgb_bridge_20261029_v1
P=/home/gbk/code/LeWorldModel/le-wm-main/results/agent_p3_same_time_cp10_rgb_bridge_20261029_v1_prepare_MAIN01.json
O=/home/gbk/code/LeWorldModel/le-wm-main/results/agent_p3_same_time_cp10_rgb_bridge_20261029_v1_cuda_MAIN01
R=/home/gbk/code/LeWorldModel/le-wm-main/results/agent_p3_same_time_cp10_rgb_bridge_20261029_v1_consume_MAIN01.json
PREP_SHA=$(.venv/bin/python -m "$M" prepare --fixed-authorities --output "$P" | jq -er .sha256)
.venv/bin/python -m "$M" verify --prepared "$P" --prepare-sha "$PREP_SHA"
.venv/bin/python -m "$M" run --prepared "$P" --prepare-sha "$PREP_SHA" --output "$O" --authorize-host-cuda
REPORT_SHA=$(sha256sum "$O/report.json" | cut -d' ' -f1)
COMPLETION_SHA=$(sha256sum "$O/completion.json" | cut -d' ' -f1)
.venv/bin/python -m "$M" consume --output "$O" --report-sha "$REPORT_SHA" --completion-sha "$COMPLETION_SHA" --receipt "$R"
~~~

存在则改新suffix，绝不覆盖；所有未来SHA只取实际输出，MAIN须核各命令exitcode。
本桥未运行生产。P1/P3/full_C3/scientific/freshness/physical_release/independent_confirmation全false；
J/probe/能力或非贪心收益没有在这里实现或认证。作者值与本地结果不混排。

