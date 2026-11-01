# P3等资源后续：最小R-aware真实闭环slice与固定预算接口

仅新增本文，不修改canonical、旧协议、source或封存产物。comparison v2三件套保持冻结；未读其新生产分析结果来定规则。使用experiment-design技能按“先闭合实现、后独立评价”组织前置，不采用择优改参/训练/sweep。本文是下一实现建议，不是已完成的预注册运行、生产授权或P1/P3/C3证书；worker不运行NN/physics/fullstrict。

## 1. 当前可复用与真正缺口

路径均相对`/home/gbk/code/LeWorldModel/le-wm-main`。本轮只读核对：

|冻结资产|SHA256|能力及不能代替的部分|
|---|---|---|
|`innerdocs/main_p3_r_spectral8_variant_protocol_20261029_v1.md`|`3a9bddc658204ca837892cf17d69eafe205f7c3f3629c383df81bcadae38792e`|已固定spectral8、独立zero、全部eligible进入原modelQ beam、no_witness停枝；无需再选择bank|
|`experiments/spectral_reachability/agent_p3_r_spectral8_bank_20261029_v1.py`|`e254f08db87b1e8b6985b6afd6ba3f3ed8f98e8f1af0d87f9edcd14021f2786d`|parent已付真实raw-J来源到8raw的NumPy生成与authority；不是生产orchestrator|
|`experiments/spectral_reachability/agent_p3_parent_r_bank_evaluator_20261029_v1.py`|`71e4e178eb180a0d9b740697c98538d6f9afd0ba50d35efb03aa23f91378e04d`|完整9项RAM评价与严格资格；禁止把HOST改标RAM调用|
|`experiments/spectral_reachability/agent_p3_h3_witness_physical_20261029_v1.py`|`f06772639f5477d9bd9bbe962a7b37ece34878226c30b0b43a0efa3a30eb7fe0`|已审原生reset/warmup/逐步dispatch/cp5 guard/标签/图像审验可复用；输入与raw20固定，不能直接冒充新tree释放合同|

`agent_p3_parent_r_native_components_20261029_v1`已有postcontext H1/suffix公共primitive、真实NativeTree接缝及nominal boundary无额外predict恢复，细节见其冻结innerdoc；本次不重写。仍缺：生产九槽整组预检与计量/authority认证、R-aware parent历史edge与eligible beam、完整保存消费、tree输出到物理action release。旧progress-only adapter在validate_parent/atomic_queue/generate仍拒绝R，不能靠新bank单独宣称它已支持R。原R-aware tree实现和新physical release必须独占新schema，不改旧白名单或旧FLAGS。

equal-cap生产report `8c65d3c5c54ba447b279fd0ad938edfab6eb42f7e1b3dbb4a1624987153cc58d`及MAIN新完整strict receipt `2ca340e108cd0c82fea6f9378a95b47f51aad6c49599f9f6cf113d8b32b3cb82`仅证明该封存progress合同的工程闭包；reader64→256MiB兼容不是科学负结果或R实现证据。comparison v2不提供physical release授权。

## 2. 下一代码只做一个slice：实际cp5观测→R-aware树→首packet→实际cp10

这是最小**一次反馈闭环接通**，不是完整20步MPC，也不是先把所有full-policy/cost-matching/独立根工程堆齐才工作。固定原两known roots/off、每根fixed与adaptive四槽；每槽从原root独立native reset/warm10/原prefix5，cp5须对既有该root authority精确state7/snapshot/PNG/动作history。cp5真实RGB经冻结encoder得到完整latent3/between2，不能把模型imagined parent换名actual；若现有cp5三帧未充分封存，准备阶段列出缺项并由MAIN新增固定采集/编码，不伪造旧报告。

1. 编译：原choose/阈值/beam/depth/144请求位不改，新R-aware orchestrator保留progress实现，R按已冻结spectral8协议、zero+8逐项postcontext H1/J与suffix lookahead，全部eligible交原modelQ beam。无资格停该分支，无progress/zero fallback，不能用(reach,index)诊断winner替代beam。全过程只接RGB/latent/history/raw/goal；native state/reward不进planner。
2. 封存：完整model plan必须root whole+segmented双replay通过，且最终timer/model/source guard通过。导出整计划、首5raw、完整task/context/goal/normalizer/plan/report/source SHA及显式`execution_policy=commit_first_packet_once_v1`。这是新动作提交政策，不假称原open-loop全程序都被实际执行。没有完整release则首packet=null；不从diagnostic_best救援。八个R槽+zero和144队列分母不丢。
3. 物理：四槽按root0 fixed/adaptive、root1 fixed/adaptive固定顺序；模型选择全部封存后再任何新物理评价。从认证cp5执行**恰好首packet5**到cp10，不padding、不把诊断suffix当真动作。若native早停就停止，cp10缺失null，原生成功、XY、wrapped angle、velocity/state_dist逐轴独立报告，不以latent Q替代。新driver对public_step6派发前执行causal cp5 guard，失败无suffix动作。
4. 闭环末端：封存cp10实际RGB/state及逐步returned labels；独立RGB编码仅后验比较同首5raw预测与actual。此时完成一次“观察→决策→执行→再观察”，尚未证明第二次重规划、长程收益或全模型plan物理正确。后续cp10重新规划再执行，是另一个须事前固定总预算的扩展，不在本slice自动支出。

四槽若某个无release，保留null及原因，不执行zero凑一个轨迹、不改分母；不以只剩成功export的槽计算“全体方法收益”。非零资源失败已收费，未dispatch物理费用记录未执行，不冒充任务失败/任务成功。renderer/goal在同pose下精确来源绑定，source/loadedbytes/resetledger不齐先拒绝认证，不归因理论。

## 3. 固定预算ABI：先保留自然策略equal-cap，再明确actual不保证匹配

建议下一prepare直接固定既有A的完整27轴共同`B=parent.default_caps('adaptive')`，每槽仅一次树编译、144请求位、同步W=3600s，全四槽576；不从新comparison输出定B。沿用whole CUDA job18000s、输出1GiB含256MiBreserve、disk floor200MiB、allocated3GiB/reserved4GiB。新增R内部工作要在**同B内**原子预检；装不下可诚实拒绝，不暗增562predict/34J等旧上限。这是共同cap可行性slice，不承诺R实际激活或四计划都有release。

prepare至少固定以下own-schema接口（不是要求另建审批框架）：

```text
budget = {
  axes: exact_original_27_names,
  cap_by_arm: {slot0:B, slot1:B, slot2:B, slot3:B},
  arm_wall_seconds:3600, max_decisions_per_arm:1,
  request_slots_per_arm:144, request_slots_total:576,
  controls:{R_candidates:8, zero:1, suffix_lookahead_H:1},
  reserve_rule: whole_and_segmented(root_to_candidate_complete_chain),
  atom_rule: parent_signal_then_entire_bank_or_no_bank,
  physical_commit: first_packet_5_only,
  FLOPs:null, matching_axes:[], no_padding:true, no_refund:true
}
preflight_group(parent, decision, used27, B, worst_case_verify27)
  -> allow | resource_cap_before_bank; full_required27 + scope/shape/source
```

R额外费用不能沿progress最低数冒签：h1/h3九项edge+postJ+suffix的前向分别36/54、J9/VJP1728；另有bank真实SVD1/compile1，九项solve/SVD、metric/replay/wrapper按新实际调用图完整计。outer-parent H1/H3已付pair、最终所选完整链双replay再单列，不能与RAM tariff/native ledger双计。上面数字是既有数值合同的派发前推导，不是HOST实测；prepare须由无NN成本编译器输出完整各轴/shape，任何缺轴或未知必须标记并拒绝matched声明。真正native调用的attempted/completed及AD/J派生量由实际事件计，不用数字反填。

可预检reserve取该decision下所有合法返回计划长度的上界，不能先看winner长度再释放已预留预算；reserve不算已花费，也不要求把未用reserve花完。九条postcontext信号不能借root J。超时可能在kernel完成后才发现，返回工作仍计completed，guard失败禁止release；日志证据缺失的partial不能签完整strict。

物理/编码分账户：prepare从既有native调用模板静态编译每槽reset/warm10/prefix5/commit5的完整reset/public/physics/render/PNG上限和全部固定checkpoint，不照抄旧20步资源数字；encoder按实际cp5输入缺项/终端槽另列，0predict/AD，不藏进planner cap。未得到精确原生静态上界、磁盘/输出限额和错误unwind预算前不得生产物理。这里不伪造未来driver CLI或未来report SHA。

## 4. 如果目标必须是“事前实际匹配”，另固定matched-work，不padding自然树

自然策略有R/no_witness/child stop/不同H/不同replay长度：共同B无法事前保证actual27相同。不能事后选恰好匹配的根、给fixed补无用计算、给某根追加候选、退款失败或把预测/J计数两轴相同称总计算匹配。

最小有用的严格工作对照是**同一完整R-aware候选库的决策对照**，不是宣称原动态策略不变：每个固定actual cp5 parent，事前请求H1和H3完整signal；分别完整progress4库、完整R8+zero库及九项postJ/lookup suffix，所有候选真实评分入库。两套封存选择规则面对同一库，在任何物理反馈前同时提交选择；两套规则都有权限使用整库，标明各项真正用途。若一方实际不使用R信息，必须披露那是共享库/未使用信息成本，不解释为自然策略效率。

执行前RAM成本编译产生literal workload（operation、context/packet shape、H、候选index、source、完整27增量、最大replay reserve）及SHA。共同输入只生成一次的共享库须单列`shared_acquisition`，不能在两个独立runtime账中虚写各花一次；若要独立runtime的matched actual，两个arm各独立完成同一有用库，保存各自实际事件和model checks。选择后的whole/segmented双replay按首packet固定5raw验证，长度均1packet；不能把整条不同长program的验证强塞进“matched”的等长账，整计划验证若存在另列且会破坏完整27实际匹配。

这里新增library-selector/commit5政策是不同算法和较窄识别问题，不能与原tree等价；已有shared-panel主要是progress/H面板，R九postcontext库才是本路线新增覆盖。不要为了做新实验再重复旧progress-only panel。最小下一代码优先完成第2节R-aware自然树接通；matched-work可待接通后单独授权，不同时开发两条runner。

实际匹配声明预先指定K=完整已测27轴**两相**attempted/completed，连同sample/token/H/J形状、精度/模型身份。成功完备的固定workload理论上可匹配；runtime/早停/失败使不相等时，保留所有槽并报告“不满足matched”，不筛掉后签成立。wall是响应变量，只有共同cap，不要求逐位相等；FLOPs/backward engine未知仍null。shared inputs或独立重复的计账口径必须在prepare选定，不能运行后切换。

## 5. 独立根与结果判别边界

两known roots无论展开多少parent/候选、换renderer、四方法/多预算，独立根数仍2。此slice只校准接线/可执行R/模型动作响应和任务轴，不从它得到freshness、泛化置信区间或P1/C3。真正独立任务根需要主线现有准入合同的预先锁定cohort、loaded bytes/seed/reset/task/goal身份与保留集合、无选择性补根、先封存模型/normalizer/选择器/预算再开标签；不得由本文替代原最低样本数或统计门。本文不打开reserved、不新增根挑选规则。

|保存事实|准确判别|下一动作|
|---|---|---|
|root/postcontext/history/goal/模型/ledger不闭合|数据或实现完整性故障|修新driver或新增consumer，保留原失败；不签模型负结果|
|R请求仍unsupported，或只有RAM evaluator|合同覆盖未实现|接生产R-aware evaluator/tree，不加一次同progress实验当替代|
|预算整组拒绝|当前固定B下不可派发|报告cap边界；不事后加B救根，不当R执行无益|
|九项全完成但无eligible|当前模型/冻结bank的严格witness不存在|原样no_witness；非全局不存在、非真实物理理论证伪|
|R执行、资格/beam/replay/release都通过，实际任务轴不改善|此封存known-root实现的有效负证据|按native success/XY/angle/velocity等固定轴报告，不单因归咎encoder或理论|
|任务轴改善但没有freshness/强基线统计/完整R策略覆盖|工程/有限条件正证据|不签P1/P3/C3；独立cohort与原gate仍另需满足|

## 6. 下一实施交付及停止点

只建议一个新R-aware producer/consumer+RAMtests，显式复用已冻结bank/components/模型原生API，自有schema；之后一个最小commit5物理adapter+RGB后验接缝。先RAM覆盖九项身份/资格/整组预算/无fallback、全部eligible beam、R历史edge/原144、失败不退款、whole/segmented/完整release、真实持久writer→reader及超长arm行边界，再由MAIN审读封存并串行CUDA。physical prepare须认证模型release/actual cp5，public step6 guard与native早停/完整日志先RAM通过，再由MAIN执行。

本轮交付只有本文；未创建上述runner、未运行测试/生产、未改任何冻结文件，不承诺未知HOST tariff或物理资源已获得认证。comparison v2源码/tests/doc SHA本轮只读重核仍为 `d08fa4a5b5224b151f43e5e63805eb2cabed93df456d7f7ed3fd469bd3a0641d` / `8d8485e7dcd4edf980cea1512e4f6cc2a2757c57a2060778f2cdb204f02b1be7` / `acddbb33d3bb35fb2ec3120d7d1a738d0999d9603a305f74cb4c542a3427d3a7`。
