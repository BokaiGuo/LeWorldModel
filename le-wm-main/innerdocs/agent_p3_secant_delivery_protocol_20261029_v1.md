# actual-cp5 reach-secant：下一阶段模型delivery验证协议

仅协议，待MAIN批准实现；无runner、生产授权或新receipt。本轮只读取已封存实际回执、
prepare/report元数据和冻结源码，不调用NN/physics/任何fullstrict，不新增子代。
使用experiment-design skill组织控制和计账，但不采用其训练、调参、择优扩展建议。
已知4/0/0/0结果后提出本协议，属于探索性工程交付设计，不冒称盲预注册或独立确认。

## 1. 精确来源与目前证据边界

下表路径均相对于固定绝对根 `/home/gbk/code/LeWorldModel/le-wm-main/`。
未来prepare须使用这些精确路径和SHA，不按latest或文件名猜身份。

|来源|路径|本轮实际SHA256|
|---|---|---|
| secant实际prepare | results/agent_p3_reach_secant_actual_cp5_cuda_20261029_v1_prepare_MAIN01.json | b3cc19fe7361e6bcc03ebf533a9a5acc115438cb6efda2f3c5655e7f680799e6 |
| secant实际report | results/agent_p3_reach_secant_actual_cp5_cuda_20261029_v1_cuda_MAIN01/report.json | 191684824c95b39fcaf4cd2683ad3f3f7cd071009cdaaa076014bd9f67df91e7 |
| secant完整consume | results/agent_p3_reach_secant_actual_cp5_cuda_20261029_v1_consume_MAIN01.json | d96f97dd076894901edea43f6223cf8cb32da4dc630968fea8fae7834b143151 |
| frozen runner | experiments/spectral_reachability/agent_p3_reach_secant_actual_cp5_cuda_20261029_v1.py | a7ecbb3fba3957af2382aecf363fbf6dd996204f8b39aedae1a486279a0b3c08 |
| frozen secant kernel | experiments/spectral_reachability/agent_p3_reach_secant_kernel_20261029_v1.py | 8fd9e2210d3d9ed6814cf07b2e9f85501156eb9733ab29ec2bfac908fbca2468 |
| 原delivery ABI/原生replay实现 | experiments/spectral_reachability/agent_p3_actual_cp5_r_delivery_20261029_v1.py | 09af91981549547c5f770ad3d0ffa42ad92b48e37daefd612191d18ec9755a26 |
| 仅供后续物理接线参考，非secant可直接调用接口 | experiments/spectral_reachability/agent_p3_actual_cp5_commit_physical_20261029_v2.py | a17ee3512ebb3289fdc862c9651641fa0e61a1670689518786701f3e64021776 |

secant回执status为`all4_actual_cp5_secant_groups_verified_NOT_delivery`，groups=4，
candidate_denominator=16，继承32候选+4 paid-zero；`delivery=null`、model_delivery_verified=false。
本轮读取持久authority不等于重新执行其完整consume；未来实现须把信任边界明确写入prepare：
固定完整持久receipt为起点，逐字段核prepare/report/receipt三向SHA、四arms、完整来源闭包、
原生call/选定行引用；若MAIN要求重新完整consume，由MAIN执行并整份canonical相等，不能虚称已重建。

actual-cp5 builder：`results/agent_p3_actual_cp5_task_builder_20261029_v1_prepare_MAIN01.json`，
SHA `ce6670d84f6f0f712bdfe5e1b7471be65b0441338c18010f79b3967ef28397c7`。
模型policy原`/home/gbk/.stable_worldmodel/checkpoints/pusht/lewm`，tensor hash
`4b12c01f5859f4b46bea478f3a8d5e2d3f516df456a622a3feec31bde7e33546`。
继承prepare全部source/class/weights/config/precision绑定；文件SHA、tensor hash不冒充
训练lineage或任意历史loaded-buffer证书，requested MATH不冒充实际kernel证明。

## 2. 固定全四组、时间轴和候选选择

|slot|root|h|原eligible indices|下一模型验证的逻辑槽|
|---:|---:|---:|---|---|
|0|0|1|0,1,2,3|选定一项，whole及segmented各独立重放一次|
|1|0|3|空|no_witness，两个验证槽明确not_applicable，不派发|
|2|1|1|空|no_witness，同上|
|3|1|3|空|no_witness，同上|

四组/16候选/原32+4control完整保存；不是把slot0拆为四个独立成功根，仍仅两个known roots。
不删除其他三组，不额外试根、换h/半径/方向，不让no_witness出具零动作fallback。
no_witness是原exact gate无合格项，不是物理失败或全空间不可行证明。

固定源路径指针：`report./rows/{slot}/result/rows/{index}`；资格列表来自
`receipt./arms/{slot}/eligible_indices`，并逐项核原saved值：
`row.immediate > prepare.source_groups[slot].zero.immediate` 且
`row.reach < prepare.source_groups[slot].zero.reach`，无epsilon、无新增阈值。
R是原postcontext-J/bounded_ridge own_reach_value，不是suffix非线性终距。

选择沿冻结原delivery `ranked`规则：全eligible按`(immediate, candidate_index)`升序，
保存全部order，唯一selected为order[0]。不按reach、bounded_suffix_cost、物理结果或
whole/seg是否通过重新挑候选；选定项失败则该组blocked，不尝试order[1]。
原四个半径评分完整保留，不因只交付一项退款或称四次独立成功。

完整task/provenance采用原prepare.tasks[root]/provenance[root]，role严格为
`actual_observed_abs5_10_15_control_minus5_0_5`。
actual latent历史abs5/10/15对应control−5/0/5；原control0历史是abs0/5/10，不能冒认相同task。
root0/1的local task_id分别
`98964c0033a0d06bb7b4b5b5f45c52375b8fc954a2137deb5aa2aa230cca8a63` /
`0edc3a2b4d147dbc6c187124b12ad57d6776ff249815ee3f39c933a6616f52f0`。
原goal/Q/mean/std、binding及base/local不同task_id全部保留；不另编码/换normalizer。

## 3. raw/action witness与独立whole/segmented合同

先seal selection、选定原raw edge、来源完整行hash及字节hash，再实际从saved bytes重载。
raw必须有限、精确float32可表示、shape=(5h,2)、逐元素在[-1,1]；逐位等于选定源row.actions。
同时核源params/radius与原secant kernel映射来源，不重做方向/映射来替换实际raw。
禁止clip、二次tanh、inverse roundtrip后沿用旧评分、重选、padding或添加诊断suffix。
保存little-endian f32 edge/commit/lookahead各SHA，normalized token另存实际字节。

从**同一完整actual-cp5 root_context**各独立初始化两次，复用冻结NativeDelivery.replay：

- whole：H1为[5]；H3为[15]，逐packet推进原生full3。
- segmented：H1仍[5]但必须是第二次实际调用，不能引用whole的输出；
  H3为[5,10]，first5结束后从自己的完整boundary context重建，再连续执行剩余10。
  不是改为[5,5,5]或旧cached-rollout的不同ABI。

每步保存原raw5、原mean/std、一次f32 normalize的token、incoming2、outgoing3、
实际AE/predictor/proj输入输出与完整postcontext/action_history。下一状态只能由本调用
自己的末帧和history移窗形成，不借源终点/future RGB或另一layout状态作为输入。
whole/seg完整path都必须与源选定row.path canonical逐字段精确相等，且array原dtype/shape/bytes
逐位相等；原h个边界全核，不只比末帧。无allclose，无后加ULP容差。
bitwise/maxabs/ULP可作拒绝后的诊断，非改门工具。

两次输出及IO先全部durable seal才做一致性判定；不额外metric/J/solve或新资格评价。
完整新strict需要验证所有原动作/调用/文件SHA、事件intent/return/failed、hashchain、
选定索引/原gate、因果context、模块费用、source/model前后，以及四组分母。
失败保留partial和两验证槽，不重试、不继续挑项；后续组保留blocked_after_failure。
new own-schema建议`agent_p3_secant_delivery_20261029_v1`，不是旧R delivery schema。
合格模型delivery可以标model_delivery_verified=true，但所有P1/P3/scientific/freshness/
physical_release/execution_authorized/full_C3等原关闭门保持false。

## 4. 新付费上限、实际可派发与继承成本

只付独立验证，不重跑secant/父J/候选搜索。每有选定项的组需要2h个packet predict，
2个replay call；h=1为2，h=3为6。没有候选的组不制造NN调用凑预算。

|分轴|全4组预留最大|按当前冻结资格可派发上限|
|---|---:|---:|
|whole/seg replay calls|8|2|
|predict_API / predict_samples（各）|16|2|
|AE calls / samples、predictor calls / samples、pred_proj calls（各）|16|2|
|predict context tokens、AE context tokens、predictor context tokens、proj rows（各）|48|6|
|新J/signal_compile/VJP/grad/raw_backward/solve/SVD/metric/program_compile|0|0|
|新encoder/projector/physics/training|0|0|

selection/哈希/saved校验是CPU元数据开销，记录wall，不声称0FLOPs。
工程提议沿secant output2GiB含512MiB收尾reserve、floor200MiB、whole3600s/group1800s、
CUDA allocated3GiB/reserved4GiB；不复制历史大文件，引用必须hash绑定。实现前仍需完整shape/byte
census证明足够，不缩分母/精度。上限是保护，不是新科学配额优势；runtime需MAIN审读批准。

继承累计实际276predict/56J/10752 VJP方向/104replay/60solve/60arraySVD/60programcompile/
104metric；AD-primal60及J元素115200另轴，不与predict_API合成总NN计数。
其中原阶段196predict/40J、本secant80predict/16J分栏保留；原receipt attempted=completed。
若当前唯一selected顺利验证，累计为278predict/56J/106replay，其它继承轴不变；
全4组均有selected的设计上限为292predict/112replay，但不能对当前资产虚称已付。
scalar backward engine/FLOPs仍unknown；不称equal-cap/equal-FLOPs或新的fresh cohort。

## 5. 交付字段与物理之前的停止线

未来new prepare应绑定本协议、原三authority、完整tasks/provenance/source/model，
四行selection（含null/order/原因）、原row pointer/hash、raw/normalized/context witness及限额。
new report/strict保留四行，每行两个verification槽；当前预期1组需验证、3组no_witness，
这不是提前宣称1组通过。验证失败仍可记录工程完整性，但不得输出verified delivery。
不得把新secant对象塞入旧R/goalbank物理consumer来冒充其producer。

后续物理需要MAIN单独授权和新secant own-schema adapter/consumer，不是本协议授权运行。
可参考冻结physical v2的validate_delivery/assemble/guard_proof/Meter，但其PRODUCER、
schema、protocol和status写死旧R，**现状不能直接接受secant**。必须新增绑定，不改旧文件。
前置包括：

1. 新模型完整strict实际report/receipt SHA，四组状态、唯一selected规则与原raw witness均认证。
2. 原native root/seed/constructor/reset recipe/goal/off、warm10、原prefix5完整来源；
   cp5 builder原provenance/raw10和物理bridge全三实际history/PNG/state/SHA逐项一致。
3. 真实从原reset重放warm10+prefix5，不能reset/hydrate到imagined poststate。
   full51 prefix以及cp5状态/snapshot/RGB/history/source全部exact，durable accepted guard
   必须在control6第一个dispatch intent之前。若原门不满足，保留失败，不降低比较强度。
4. 窄commit沿原FIRST5：只部署selected.raw_edge[:5]，control6..10/absolute16..20。
   H3余10仅lookahead，不自动执行；诊断suffix永不拼入计划。若要全H3实物终点，另立协议及预算。
   物理返回的native任务agent+object/角度success、native state/reward/PNG全保存，不能以RGB Q或
   object-only投影代替原成功定义。no_witness三行不构造env、不跑zero，留完整分母。
5. 物理资源另列真实constructor/reset/set_state/goal/render/getobs/public/physics/close收据。
   参考v2完整FIRST5路径每执行行20public、204physics；全4行上限80/816，当前最多1行20/204，
   来自10warm+5prefix+5commit和2reset×2内部step，不是本轮已付账。未来adapter若原生额外操作
   不同，必须先封存完整资源合同，不能漏goal bodies/setup或沿用口头数字。全部失败/earlystop保留。

## 6. 明确阻断及实施验收

以下任一项即blocked，不回退：authority/source/model/精度漂移；任何四组/16槽缺失；
原gate/order不符；无候选却生成动作；raw域外/非finite/精度丢失；base/local身份或abs/control
时序混接；normalizer/goal/Q改变；whole/seg action/context/native IO不精确；cap不足或调用超限；
未来physics前置任一不认证。不得失败后改半径/门槛/排序、重试挑运行或换根。

未来最小RAM测试应覆盖全四组4/0/0/0、tie用index、选择失败不fallback、H1独立双调用、
H3精确5+10重建、raw/f32/history/goal/partial/事件barrier篡改、成本与文件census。
本轮仅写协议：未实现/运行这些测试，无新的可执行delivery CLI、无未来receipt SHA。
下一可执行工作是MAIN审读本协议后授权新stem实现；旧runner的consume不是本协议执行入口。
模型exact witness若未来成立也不证明真实reach/非贪心控制收益、完整policy tree、P1passed或P3准入。
