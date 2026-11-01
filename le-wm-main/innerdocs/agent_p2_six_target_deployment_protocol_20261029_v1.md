# P2 six-target：实际部署 token replay → 六个原生物理分支协议 v1

本文件只固定后续实现合同与RAM成本公式；不是prepare、driver、ready receipt或物理许可。six-target新job尚未交付本协议所需完整实际authority，所有其prepare/report/receipt路径与SHA均待MAIN提供，不填写未来hash。不得干预Pauli当前实现。experiment-design技能仅用于事前固定顺序、付费边界和失败分母，不采用训练、调参或按结果选择。所有旧封存保持不变。

## 1. 固定分母与依赖顺序

固定顺序0→5；root0/off和root1/off各有prior_FIRST、program_first_postcondition、arbitrary_FIRST三槽。六个新controller returned mean均来自**同一个新six-target job**；旧四槽不是新job的前四条结果。前四槽mean即使字节相同也不减免新增调用、环境或日志。

工程依赖为：

`six-target实际完整controller authority → saved raw-domain核对 → 新六槽HOST CUDA inverse/re-normalize + whole/segment → 新完整paid receipt → MAIN批准物理 → 六个新env原生full15 → own full consume`。

六槽不是两根之外的新根；arbitrary以前缺失是历史事实，新实验不回填旧missing。每阶段自有six schema、全部六条身份和独立文件清单，不把旧4与新2的metadata、预测或receipt拼成六槽新运行。

## 2. 可复用的证据与不可冒用的执行

以下路径相对 `/home/gbk/code/LeWorldModel/le-wm-main/`；SHA本次只读核过。

| 已封存来源 | SHA256 | 本次仅可复用什么 |
|---|---|---|
| innerdocs/agent_p2_six_target_controller_protocol_20261029_v1.md | 306305ea86638d120760f7a497859319e3c716cee2b61ddf2522a46af18b7f1e | 六输入目标来源、统一solver/顺序及新job预算定义；不代表实际完成 |
| innerdocs/agent_p2_commoncontroller_physical_protocol_20261029_v1.md | 42e082498484f8bb399f819d31531c2a77f73b161e0bd2e15bb8ec900f5e8de6 | reset/place/warm/guard/原生指标合同 |
| experiments/spectral_reachability/agent_p2_delivered_mean_paid_replay_cuda_20261029_v1.py | 19b8b36337de68233e504a647b66b468e0fde95608a655e8e57803886b938481 | 纯float32四操作、低层rollout/capture/ledger验证算法；不调用其fixed4外壳冒充six |
| results/agent_p2_delivered_mean_paid_replay_cuda_20261029_v1_consume_MAIN01.json | a8fe4dd9c8c90ed14a5bbecdcb226ec4d7626c786fe4db1223194f7e3f6d87b1 | 旧四mean的已付费证据和历史对照，不作为新六槽付费凭证 |
| experiments/spectral_reachability/agent_p2_commoncontroller_physical_20261029_v1.py | 7e715a9005a4408fbc7899184f338aa4ae6ee6b20dd414356744aa830ceff5f0 | reviewed episode、initial proof、native hooks、renderer/labels等低层接点 |
| results/agent_p2_commoncontroller_physical_20261029_v1_consume_MAIN01.json | bf976c4f28f3b4634d2d78594779cc74933bf8870d78aef2db8cb182a49342b3 | 原真实history/task/goal及已完成四分支的历史参照；不继承任何新物理row |

已有完整RGB→latent、warm10/history、native root/goal、model425/source、公共mean/std**值绑定**可作为原authority整体引用，按实际字段/路径/SHA追溯；不得把值绑定升级为Hi训练scaler来源认证。新arbitrary来源必须沿six controller原封目标authority认证，不从旧missing造输入。

不可复用作新增完成量：旧inverse/re-normalize数组、旧24predict、旧4env/100public/324PNG、旧checkpoint/terminal、旧425 before/after。新controller本身的normalized-mean双replay也不能代替部署token重放；两次重放的输入坐标不同。

## 3. 新六槽接线硬前置，不猜flags

实现时MAIN显式提供真实six controller prepare/report/完整receipt路径与各SHA，按其最终冻结ABI核完整source closure、所有六槽row、27轴资源、数组与constructor/model425 identity，不凭status。固定绑定每槽slot/root/condition/arm、task与blind身份、z/a/target、原goal192/native goal/RGB、真实history、native returned mean/selected mean/actions的字节关系。native mean形状须是生产者实际 `[1,3,10]`；若最终ABI不同，阻塞并显式适配，不静默reshape错误布局。

不得要求猜测的physical_release或沿旧missing flags补字段，也不得更改上游false声明。新物理授权来自MAIN批准新协议/driver及完整paid authority，而不是将上游raw_deployment_authorized或science flag改true。

旧raw-domain的固定AUTH/ACTIVE不能用于六槽。需新的saved核对：从每槽**实际NPY**取native mean和incoming a，记录原file/data SHA、dtype/stride/shape/时间顺序，公共mean/std只cast一次float32，CPU NumPy明确分步mul/add/sub/div仅作事前字节参照和domain检查，不是CUDA执行证据。原native闭域[-1,1]、finite和std>0须全满足；不clip/tanh/缩放/重新选择。

若任槽mean缺失、布局不符、超界或authority不完整：保六槽分母和逐坐标原因，批次BLOCKED，不启动任何物理env；已完成的其他槽也不获部署。事前domain阻塞不强行付神经成本；实际CUDA发现差异则保留已发生的attempt和原始数组，不退款，不用CPU覆盖。

## 4. 新六槽HOST CUDA实际部署token付费

每槽固定S1/H3，原z三帧 `[1,3,192]`、a两条history `[1,2,10]`，同一冻结graph与公共normalizer；无搜索、候选、重选、encoder、AD或训练。只使用实际returned mean，不能换elite/best sample。

时间顺序把mean重排为 `[15,2]`，CUDA eager float32依次保存：

1. `p = mul_f32(mean, std_xy)`；
2. `raw = add_f32(p, mean_xy)`；
3. `centered = sub_f32(raw, mean_xy)`；
4. `tokens = div_f32(centered, std_xy)`，分为3个连续5步token。

禁止FMA融合、额外/100、第二次inverse、把raw直接当normalized、用旧mean替换新tokens；raw120bytes须与本次saved分步参照完全相同，包括signed zero，domain同样复核。tokens不要求等于旧mean，roundtrip位差/ULP独立报告，不用误差阈值改变部署输入。incoming a必须来自同root实际warm_raw10按原sub/div一次得到的两条chronological tokens，并精确等于six controller封存a；不把未来动作塞进过去history。

在同一新job按0→5逐槽执行whole H3一次、segmented[1,2]一次。段间传递完整真实模型IO/history，不重置z/a、不用teacher或native状态。每槽6次AE/predict/lowproj，完整预测/shifted context/模块IO/事件/tokens落盘，whole/segment原exact replay关系全核。六槽预测全部durable封存后才计算target或原goal的后验Q；任何物理labels不得进入forward或评分选择。新raw动作整体封存后不可改变。

425参数buffer/source/input/export/constructor全量before/after；deterministic、TF32/autocast off、MATH SDPA、HOST CUDA only，无CPU neural fallback。所有attempt/dispatched/returned/captured分开记账；partial/超时/IO失败不能用planned补actual。全部六槽实际返回且完整consume后才具备进入物理审批的工程前置。

## 5. 六个新物理分支与因果首动作guard

新own schema固定六槽，active=0..5、missing=[]仅在完整paid六槽认证后成立。旧四runner的ACTIVE、`closed<4`、8reset、100public、324PNG、missing和branch closure不可直接继承；不得monkeypatch globals或伪造旧三程序plan/edges。低层函数仅在签名/身份适配清楚时复用。

每槽新env，不继承旧row、不去重相同动作。原native constructor/reset_seed/place recipe，原两次reset和真实shape authority；执行原warm10，核warm0/5/10 off RGB与sealed history、十条requested/actual raw bits、native state/snapshot/root/renderer。禁止imagined restore或额外state setter。warmup原生早停留失败，不pad。

native goal按原phase：warm阶段placement goal，绑定后原task goal。latent target绝不是native goal。goal bind前后用原真实invariant语义核：state/完整snapshot、renderer/goal_pose保持相应不变量，而goal_state/goal_rgb发生原合同允许的挂接；不能拿state-only证明完整snapshot或把所有pre/post字段盲等。actual cp0/current RGB、goal pixels/goal192 source和history/a均需对应本槽模型输入。

在plan public step1的intent和callback**之前**，真实观测形成initial_history_guard并durable fsync；写失败/拒绝绝无step1 intent/callback。之后原raw15 bytes一次执行，无clip/再normalize/重规划；native原step内部action_scale=100不变，外层不额外乘除100。逐步保native7、reward/terminated/truncated、requested/actual rawbits、snapshot及off/on/off PNG；cp0/5/10/15仅真实存在才记录。提前终止立即停，后cp=null，native terminal实际步数单列，绝不把终点补作cp15。

新strict需自有六branch hashchain、全白名单、resource intent/return嵌套、guard时序、phase labels、全部PNG/source/action/task绑定、文件全集与真实census。不得调用旧validate_saved/audit_chain假装六分母已验。保留失败槽；缺帧或日志未闭合不能签完整receipt。worker不执行新旧fullstrict。

## 6. RAM资源公式：新增付费与历史分栏

以下均是六槽全合法/full15时的**计划**，不是已发生值。设合法且完整replay槽数n≤6，完整物理槽数m≤6；物理开始条件是n=6，不能据n挑子集。

| 新阶段/轴 | 每槽 | 六槽全完成 |
|---|---:|---:|
| CUDA独立mul/add/sub/div | 4 | 24（每类6） |
| deployment whole/segment trajectories | 2 | 12 |
| predict API / samples | 6 / 6 | 36 / 36 |
| AE / predictor / lowproj calls | 各6 | 各36 |
| AE full-history tokens | 24 | 144 |
| predictor context tokens / lowproj rows | 18 / 18 | 108 / 108 |
| physical constructor / close | 1 / 1 | 6 / 6 |
| reset / set_state / goal_setter | 2 / 4 / 3 | 12 / 24 / 18 |
| goal_render / goal_bind | 2 / 1 | 12 / 6 |
| public steps warm+plan | 10+15 | 150 |
| physics steps reset+public | 4+10×25 | 1524 |
| explicit renders / PNG | 3×(11+16) | 486 |

AE每trajectory history长度3+4+5=12，所以双replay=24；predictor每步last3，所以双replay=18。物理每次public原10个physics；reset额外4。cp0与warm10各有实际triplet，不能去重。

完整six controller自身计划72predict/324samples、1296AE tokens、972predictor context tokens/lowproj rows，作为**不同上游阶段**导入实际账。只有完整实测确认后，两新增模型阶段算术总计可为108predict/360samples、108各模块调用、1440AE tokens、1080predictor context tokens/lowproj rows；不把旧四controller或旧24replay并成新付费。

两新阶段image encoder/high/AD/J/VJP/backward/training=0；物理阶段模型调用=0；FLOPs=null，完整CPU capture次数若原接口未测仍null。上游目标生成/认证/模型构造/hash/capture/wall分别保留，不称免费或equalFLOPs。失败attempt不退款，部分完成实际资源不能用n/m简式代替逐事件账。

预先固定建议工程cap（待MAIN批准新实现；不是修改旧cap）：paid job总768MiB含96MiB终态reserve、单slot64MiB trace payload、单root384MiB、floor200MiB；每slot180s、whole1350s，含认证/构造/保存/同步，allocated3GiB/reserved4GiB。physical总96MiB含12MiB失败/终态reserve、prepare128MiB、floor200MiB、whole1350s，无GPU。开始时磁盘须分别满足cap+floor。逐阶段独立时钟和完整wall scope，不把等待MAIN审批/GPU排队当kernel时间；不得执行后扩cap掩盖失败。cap不足先BLOCKED并另版审批，不删除实际日志。reserve耗尽/kill/磁盘故障可能仅partial，不声称均可完整consume。

## 7. 固定读法、阻塞与下一最小slice

每root完整三臂cp0/5/10/15/native terminal表；固定program−prior、arbitrary−prior、arbitrary−program三对，共两根×五列×三对=30对，六臂×五列=30行。缺checkpoint保null，不挑success/根/时刻。native7²、jointXY²、raw_angle²、velocity²及其恒等分解全表；原native angle行为和wrapped诊断并列不替换，success/XY/angle/velocity分别解释，不称统一原因。只有另行批准真实RGB编码后才有real latent，不拿native状态填latent。

当前真正阻塞：新six controller最终完整actual authority尚未在本协议中提供；新six raw-domain/paid adapter及own consumer尚未授权实现/实际执行；新six physical driver及真实first-step guard尚未执行。domain超界是部署工程阻塞，source/receipt/schema/IO失配是证据链阻塞，不能统称科学负结果。新目标/normalizer来源仍可能限制解释，不以工程闭合认证训练身份。

下一最小授权slice仅为：在Pauli真实six ABI冻结并MAIN完整consume后，新增own-six saved raw-domain + HOST CUDA paid replay source/tests/doc，MAIN先审RAM及source再实跑/全消费；之后才单独授权六分支物理adapter。此文件不提供不存在runner的CLI或ready wrapper，不填写未来hash，不要求重跑旧四物理来证明新六。

P1/P2/scientific admission、freshness、independent confirmation保持不成立；两known roots不变，prior物理时距未知、三来源生成预算不同。仅共同controller工程，不称时间公平、等总预算或科学gate通过。本轮仅新文档、只读源/SHA与stdlib RAM资源公式；无prepare、模型加载、NN、环境、physics、fullstrict或子代。
