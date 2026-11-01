# P5：不依赖历史 attempt ledger 的前瞻来源分离方案

状态：**proposal / BLOCKED，不是执行授权或 fresh 认证**。本文件在已知旧 P5 诊断后提出，只固定下一次拟议工程试点，不称对既有结果盲态预注册。不得打开预留 outcome、按任何已有或未来结果挑源、根、阈值或分母；不修改 P1/P5 科学 gate。当前没有可验证可用的新来源、真实新实例、外部随机性或 custody receipt，故没有运行命令。

## 1. 哪一种“互斥”可以不依赖旧 ledger

| 命题 | 可行证据 | 仍不能推出 |
|---|---|---|
| 采集事件／实例不同 | 新外部保管者签发实例 UUID、创建与采集时间、独占存储清单、不可回填的顺序日志；协议及代码 hash 在采集前第三方时间戳 | 状态、画面、环境分布与旧数据互斥；统计独立或训练未见 |
| 数据条目命名空间不重叠 | 新数据集源、采集批次 UUID、episode UUID；来源方绑定原始对象字节、创建记录和无拷贝承诺，身份按 `(source UUID, episode UUID)` | 不同 namespace 不证明像素/状态不同，也不能证明未从旧集拷贝；凭自签字符串不成立 |
| 物理状态集合严格互斥 | 有证据的旧全范围外包络与新严格不相交支持集，定义坐标系和全变量并证明；或物理上新对象/环境语义加入不可与旧实例相同的属性，且旧语义范围被可信覆盖 | 不完整旧 ledger 本身不提供全范围；换机器、种子、来源 ID 或更改外观不自动实现同任务状态集互斥 |
| 新实验抽样独立于历史结果 | 固定算法后外部产生未预知随机熵；独立保管者确认未看旧 outcomes、不以响应筛选，记录过程与失败 | 一次可复验 draw 不能证明其 RNG/保管者确实独立；样本量4不能证明总体独立或泛化 |
| 未进入冻结模型训练 | 训练数据及 checkpoint 血缘完整认证，或有独立可证明的训练完成后新采集且无旧数据复用 | checkpoint hash、时间戳、dataset UUID 或 train split 名单不足以独立证明训练全来源 |

因此本方案替代的是**依赖旧尝试枚举的严格身份排重路径**，只在定义的新实验事件／来源级问题上推进；不绕过原 fresh-root 合同，不能将未知状态碰撞标为 false 或签旧 gate。若要求证明旧物理状态集绝对不重叠，且没有完整历史范围或其他可信分离证据，仍 BLOCKED。

## 2. 本地依据和来源候选的诚实结论

以下为当前只读源码/协议，不视为新的训练血缘证明：

* `config/eval/pusht.yaml` SHA `66eca39ebc1034d0cf746fef80e1c6309c07dc1fbd6fa067779ac97c446d2856` 明示 `swm/PushT-v1`、`pusht_expert_train`、从数据 state/goal 设置环境。
* `config/train/data/pusht.yaml` SHA `dcdd082ebdc628cdc8a88f15cb647fce6eec87343f9fe190271946ff5d52f757` 明示 `pusht_expert_train.lance` 与 pixels/action/proprio/state。训练 provenance 未认证，不能把现有 train 数据换文件名作为 fresh，或由另一任务的数据 split 冒称 PushT 互斥。
* `innerdocs/agent_p5_freshness_attempt_trace_20261029_v1.md` SHA `16a68408bbf5c1c3921c27f1ef02ccab60fe2a08df1766890b5ebb8353b9a6fa` 记录十条已发现 dispatch、原载入源码及历史完整 attempt 的缺口。
* `innerdocs/agent_p5_raw164_jepa_h2_cuda_20261029_v2.md` SHA `9245e43f2b9f18f693dcb24d04d6db08cb6728026ae298974f49efa393b993c4` 记原 frozen JEPA 数据接缝与限制。现有 164 分支来自两个 known root，并非新数据源。

本次没有发现并认证可立即使用的外部新 PushT 数据或全新环境；非穷尽检索不证明不存在。已有公开 archive、仓库的训练/测试名、同模拟器新 seed 都不自动达成上述条件。若改对象几何、动力学或坐标域获得可证语义差异，则成为**新域迁移实验**，必须重新审计模型输入/动作/任务与评估意义，不是原同分布 P5 fresh confirmation。

## 3. 最小可执行拟议 slice：事后冻结方法，采集前固定四槽

此处是需 MAIN 先批准并与外部保管者约定的唯一候选设计，不是已经批准的正式 P5 样本量。四槽仅是工程试点；不依据根成败增减、补采、筛选或声称足够统计功效。

1. MAIN 在接触新源的任何状态/结果前封存：本协议、原生 PushT 环境与 wrapper 实际代码/配置/依赖 SHA、renderer 和任务 goal7 的原语义、冻结模型/normalizer/预处理、时间单位、总资源上限和失败规则。若要保持同任务，执行源必须是独立新实例但保证环境动作/renderer ABI 与原 P5 逐字段一致；任何有意义语义变化须先停止并另立迁移协议，不修成看似相同。
2. 外部保管者在承诺后产生 256-bit 随机值 R：采用独立可信来源的公开随机信标，指定其未来轮次、证据签名和最迟可用时间；信标不可用则整批 BLOCKED，禁止换轮次或自选值。固定 4 槽，按索引 `i=0..3` 计算 `HMAC-SHA256(key=R, message=UTF8('P5-independent-pilot-v1/root/') || uint32_be(i))`，取前 4 字节为无符号大端 seed；seed 重复则整批 BLOCKED、不重抽。R 只有在承诺后才产生；不得将时间戳或 hash 自身当作不可预测性证明。审读者验证签名、轮次、时间顺序和重算全部 seeds。
3. 每槽一个 fresh env，只执行一次 `reset(seed=seed_i)`；目标与初态由冻结环境 native reset 产生，不从旧数据加载 state、goal、RGB、动作或接触缓存。一次 reset 异常、native 非有限值、早停或 ABI 不符均保留该槽失败，不补其他 seed、不更换 source；只允许事前固定的可执行性检查，不能根据模型/物理响应挑槽。所采源原始对象、state/goal/snapshot/RGB及 history 的源文件 SHA、环境版本和随机 API 接受值需完整记录。一个 episode 对应一个槽：branches 和同源子窗口不增大 root N。
4. 同一槽的比较在完全相同的已记录起点按事前固定规则产生，不复制或伪造动态缓存。若需要多分支物理控制，须另行实现 fresh-env 原生 replay，并在第一处差异前核 state、完整可见 snapshot、RGB、goal 和动作历史；RNG/contact cache 未见证则标 unknown，不能称完全相同底层状态。所有 replay/guard/资源费用单列，失败与早停不重试不退款。
5. P5 模型的视觉/agent-proprio/动作单位及原 normalizer 需沿冻结 ABI；固定真实时间顺序和目标标签，先封全部预测再读取本次标签做分析。原 nonzero goal agent velocity 和原 native success 语义分别保留。时间、历史和 labels 必须来自新同槽真实轨迹；没有新模型/训练/拟合，也不假设新 reset 自动支持此前 164×41 probe 实验。若该 factorial 对照被批准，动作及分母必须在接收 R 和任何新观察前另行明确；本协议不自动继承 41 probes 或旧旧 outcome。

试点分母始终 4：一槽不合法则对应 missing/失败，整批不以幸存槽宣称完整；任何后续重复收集需新协议与新独立承诺，而非补这个批次。所有模型和物理成本待未来实际 adapter 与资源上界事前确定；不能仅计 predictor 或伪称等 FLOPs。

## 4. 可验证的 fail-closed 前置检查（无 self-certification）

执行者应先把每一项写成外部路径、文件 SHA、JSON pointer/字节范围和检查结果；证据缺失、矛盾或不适用均为 `unknown/BLOCKED`，绝不默认 false 或由执行者自签替代。所有证据仅 metadata/身份；不要求读取旧预留 outcome。

| 顺序 | 检查对象 | 可复核检查 / 拒绝条件 |
|---|---|---|
| A | 方案承诺 | 原始协议+代码/config/预算 SHA；由不受执行者事后控制的第三方证明先于信标生成及源采集；缺时间顺序或原始 bytes 则 BLOCKED。 |
| B | 外部随机性 | 固定信标轮次、签名/证书、公开值 R 与发布时间；重算4个 HMAC、检查不重复及无额外 salt/retry；签名或排序失败整批 BLOCKED。 |
| C | 来源及保管链 | 外部机构/操作人的可核身份、实例/批次唯一ID、创建事件、代码镜像和配置 SHA、采集前空存储/访问控制、按序 append-only 日志和原始对象完整字节 SHA；声明独立不等于证据，缺其中一项 BLOCKED。具体可信机构、时间戳及签名验证路径由 MAIN 事前指定，本文没有指定或认证。 |
| D | 原生合同 | environment/physics/wrapper/renderer/goal7/actions/frameskip、RGB预处理和proprio、模型/normalizer、source-hash与旧合同的逐字段兼容表；不一致或未知不签原同任务；不得通过静态配置相同宣称实际运行一致。 |
| E | 样本身份和盲态 | 四个固定槽全部 reset intent/return/failure、episode/source/seed/state7/goal7/snapshot/RGB3/raw history，逐槽 file SHA 和日志时序；未实际发生为 null。预测 seal 前不读取新标签做选择；审计日志与访问控制只支持可核盲态，不保证操作者心理状态。 |
| F | 已知历史冲突 | 按旧已认证 identities/seed spans 和新四槽做确定性已知冲突检查；历史不完整保留 unknown，不能叫全历史无重叠。严格物理支持不交的额外证据如不存在则此项 `state_set_disjoint=unknown`。 |
| G | 学习与独立性 | checkpoint 实际 SHA 和训练来源证明；采集后产生/无旧拷贝/无反馈的独立证据；分别给 `training_lineage`, `event_new`, `state_set_disjoint`, `sampling_independence` 四状态，不能由其中一项推断其他项。 |
| H | 科学和执行许可 | MAIN 另外明确审批 P5 确认任务及预算；此处即使 A–G 工程通过，也只允许在新协议限定问题上继续，原 P1/P5 gate、全历史 freshness 与 physical release 均不自动通过。 |

本次只读 metadata/源码，没有抽取随机值、建实例、运行环境、NN、物理或 fullstrict；也没有建立外部保管链。若外部来源无法提供可审计生成记录，最小下一步是先解决保管者/信标/提交时间戳选择，不是写一个自我认证 wrapper。
