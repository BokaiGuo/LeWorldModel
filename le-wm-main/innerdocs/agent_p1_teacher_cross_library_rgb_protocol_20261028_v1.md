# Teacher共同库真实RGB：固定40分母、缺失终态不可release

本文件仅固定下一实现合同；不改任何封存producer/consumer/结果，不启动模型/物理，不补图、不重渲染、不重新派发动作。MAIN负责审读、严格消费与实际HOST CUDA。状态为协议交付，不是已实现RGB runner或已完成编码。

## 绑定的实际材料

- 物理report：`results/agent_p1_teacher_cross_library_physical_20261028_v1_MAIN02/report.json`，外部SHA256 `464656755bb83cec14b7acdd457641592b11be6d335f285166687df15368bc01`。
- compiler report：`results/agent_p1_teacher_cross_library_20261028_v1_cuda_MAIN01/report.json`，SHA256 `d5ab99d55bacd73c8d9cf0fb774819bb46af63100053198858e639d8fa746738`。
- compiler严格回执：`results/consume_teacher_cross_MAIN_20261028_v1_MAIN01.json`，SHA256 `d82109eb8de582e59a4f0485307c01c9b91f9f84551c218c61cca4da24335213`，全部40 raw10 plan，不重跑96predict/8J。
- physical prepare/source、原task/goal/model/normalizer、artifact/journal均从已封存execution binding及prepare完整closure核对。physical严格消费回执尚待MAIN给实际路径与外部SHA；**不得编造**，缺失则manifest只能为`blocked_pending_physical_strict_receipt`，不能神经执行。后续consumer应接MAIN已验收的partial回执，不把partial伪装complete或重复物理作业。

本轮仅read-only核report原文件SHA及JSON：40行、39 `horizon_completed`、1 `failed`；800public/8160internal physics/40close均attempted=completed；explicit renders2638而非满额2640。index12为root_slot0、condition=on、role=Z、library_index2/-uZ（report task整数为1）。其suffix5实际已返回，agent XY `[388.785817139554,514.7195833901377]` 越512canvas；随后off图`b12_suffix_05_0.png`持久化后被fresh RGB/canvas guard拒绝，on与off-replay没有生成。该行缺checkpoint10/end_state，不能用last_state填充有效图像终态。off incident文件SHA `66f87027b85db655cc5efad958570fd6e19897c26ceb205e348d9c77f0474ae3` 仅保失败证据，不进入编码队列。

## 固定manifest及准入规则

固定排序不变：四task `(0,off),(0,on),(1,off),(1,on)` ×角色`Z,S` ×库`zero,uZ,-uZ,uS,-uS`。保留全部40行，不能删掉失败项、改分母、因预测Q/真实效果或动作重复筛选。以`index,job_digest,anchor_id,task_id,blind_sha256,root_slot,condition,role,library_index,library_label,plan_digest`完整身份join，不能猜文件索引或仅按列表位置拼接。

每行保留`physical_status,public_attempted/completed,goal_binding,raw10_digest,checkpoint_requested=10,absolute_step_requested=20,frame_status,missing_reason,eligible_for_encoding,actual_latent,actual_Q,actual_model_error`。缺失字段真实为null，不设0、不插值、不以valid39的平均填补。完整分母40、预期可编码39是不同字段；若MAIN严格消费发现额外缺失，进一步blocked并保留实际数量，不能强行达到39。

有效terminal图必须同时满足：

1. 行为`horizon_completed`、完整10 release原动作和warm10字节/原prefix51帧身份门已通过；原goal及原condition不变。
2. checkpoint step10/absolute20存在，和journal同一job的suffix5实际选定condition帧完整join；off取overlay_index0，on取1。仅按同一condition，绝不拿off给on。
3. PNG文件/像素SHA、shape224×224×3 uint8、snapshot/state7/pre/post invariant、原canvas/RGB/chronology guard和off/on/off要求全部通过原strict consumer。新层只能进一步拒绝，不能放宽producer要求。完整行goal/renderer/raw histories不因缺失修补。
4. 外部report/prepare/source/strict receipt及所有实际图像/目标/原编码/model closure均精确匹配，路径无traversal/symlink替换，输入前后复hash。

index12必须是`frame_status=missing_condition_terminal`、`missing_reason=guard_rejected_off_before_required_on_capture`、`eligible_for_encoding=false`，`actual_latent/actual_Q/actual_model_error=null`；incident_png可另列但不成为encoder input。其原40计划/预测值仍可保留作来源，不能冒称真实反馈已评。禁止使用off incident、旧probe20、teacher15、最近有效帧、另一role/condition、缓存预测latent、privileged state渲染或新模拟来替代。

## 两层release与固定神经预算

**默认full40路径fail-closed**：manifest恒有40槽，但当前`full40_ready=false`、`full40_released=false`、`status=blocked_missing_one_condition_terminal`。完整40真实latent/Q比较无法发布，默认full40 prepare/run应在加载模型和创建执行output前拒绝；拒绝不是模型负结果。

**可选partial39仅供MAIN另行明确授权**：若MAIN批准诊断性已有图像编码，新增自身schema的`partial_observed_only`模式，40槽manifest照保留；按固定身份顺序逐项编码全部有效terminal39张，无dedup、无选择赢家。39次encoder+39次projector（每张独立batch1），0predict/AE/pred_proj/AD/J/SVD/physics/训练/Qfit。missing index12只占槽且资源为0，不用空跑padding补成40；同图重复亦不退款。模式与缺失集合在首次模型调用前封存，不能运行后转为full40。失败attempt/return分别记，运行中故障不重试、不替图，全40状态仍保留。仅本终态评分阶段新增39编码，原teacher8编码/96predict/8J、原任务history/goal编码及物理800steps费用单列继承，不称等总算力。

partial模式成功状态为`partial39_terminals_encoded_full40_blocked_not_admitted`，不得用`complete40`等易误解tag。full40状态不是“39/39成功”；统计同时报告`expected40,eligible39,missing1,attempted,completed`，按root/condition/role/library保留缺失。支持值/胜率/平均Q不能默默以39替代40；若列available-only描述量必须标注可观察性选择偏差。index12的native实际记录可另列，不能将缺失latent当latent失败/成功，也不能由其越界单项推导全部任务理论结论。

## 原Q与纯saved比较

terminal39 encode使用冻结native192 encoder/projector、原checkpoint/meanstd、CUDA-only/MATH/TF32=false/无autocast、完整原精度module input/output与before/after tensorhash。不重编码目标/teacher上下文：仅在逐SHA绑定的原同condition goal192/identity192 Q及teacher context真实可复用时读saved数组；缺任一绑定则blocked，不自行加神经预算。无新goal、Q fitting、归一化再除100、阈值或评分选择。

每有效行按原compiler同一公式复算actual Q及同一candidate预测到真实terminal的latent误差，公式/原数组必须从冻结compiler已核saved receipt绑定，而不是发明新度量；index12均null。原common-bank模型预测/replay只读复用，不新predict。可输出同task/role库内的逐项predicted/actual对照，但有缺失的集合不可给完整库rank/argmin/winner；不由available-only最佳值选择动作或补做物理。无需重复终距/无符号谱当能力结论。

## 工程接口与未完成项

下一实现采用新独占stem（例如`agent_p1_teacher_cross_library_rgb_20261028_v1`），`manifest`纯saved、`prepare --mode full40|partial_observed_only`、`verify`、MAIN CUDA `run`、strict saved `consume`；实际CLI以实现审读后冻结，本文不冒充已有命令。显式参数必须含physical output/report SHA/strict receipt path+SHA、compiler output/report SHA/strict receipt path+SHA，以及独占output。MAIN01/MAIN02物理及任何旧文件不覆盖。

建议工程硬限：prepare64MiB以保完整来源closure，output64MiB/final reserve16MiB，journal未压缩32MiB，report8MiB，diskfloor200MiB，wall600s；**实现时先做完整39张pixels/ViT257×192/projector trace与全元数据的保守容量回归**，未过则先报工程不可达，不减精度/删模块强行过cap。fsync/no-overwrite/O(1)实际部分写计费及终态目录census保留。完整raw数组和file/NPY/array三层hash，不pickle，不下载。

待实现RAM测试至少包括：40全量身份、不以Q筛根；index12无on且off存在仍不可编码；任意旧图/跨condition/teacher15替换拒绝；canvas/agent guard不放宽；report/receipt/goal/closure/hashchain篡改拒绝；full40预先拒绝且模型factory零调用；显式partial恰39独立编码/无新predict且全40/null保留；重复图片不dedup；首错/finally错误分别保存、部分write计量和unknown额外文件拒收；同Q算术用RAM已知数组验证。

当前只交协议，未实现新RGB source/tests、未prepare、未调用神经/物理；physical严格回执的真实外部SHA待MAIN。known-root校准、state/velocity/contact中介未隔离、S仍stage-greedy、C2前提不足，P1=false。原负结果和缺失均保留，不构成独立新根确认或理论否定。
