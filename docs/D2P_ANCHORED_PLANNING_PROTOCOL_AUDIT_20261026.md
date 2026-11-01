# Anchored Planning：身份、协议与接入审计

核验日期：2026-10-26。状态：**一手来源核验未完成，不可据此引用 AP 数值或宣称已复现**。

本次仅新增本文；没有修改 runner、历史报告、checkpoint 或环境，没有启动 GPU、神经推理、训练或下载大型数据。以下严格区分附件线索、本地可验证事实和 unknown；检索失败不是论文不存在的证据。

## 1. 来源与版本登记

用户附件已完整阅读：`/home/gbk/.codex/attachments/380fd905-cc8f-4924-a507-d9292f925635/已粘贴的文本.txt`。
附件 SHA256：`893dbbd2b1ffa793233b88947911f179beb298df7a8ee1510eb63bec3dadc4de`。
它是待核验研究建议，不是论文原文，且所列 arxiv.org 图标没有提供可唯一定位 AP 的文章 ID。

| 对象 | URL / 定位 | 版本、表、页 | 本次证据状态 |
| --- | --- | --- | --- |
| AP 标题 | 附件称 *Aim Short to Reach Far: Your Frozen World Model Can Plan Better Than You Think* | 作者、年份、arXiv ID、版本日期、DOI、会议均 unknown | 未完成独立身份核验 |
| AP 原文 | unknown；不得推测 arXiv ID 或仓库名 | 附件称 Table 1；原文页码 unknown | 未取得正文/PDF，不能转录为 reported verified |
| AP 官方项目/代码 | unknown | release / commit / license unknown | 未确认作者到代码的官方链接链 |
| AP checkpoint | unknown | revision、config、权重 SHA256 unknown | 不能用本地 LeWM 存在代替 AP 精确模型绑定 |
| AP evaluation / memory manifests | unknown | 文件路径、revision、哈希 unknown | 未定位官方 queries、perturbations、记忆拆分 |
| 原 LeWM 论文（不是 AP） | [PDF](https://arxiv.org/pdf/2603.19312v1) | URL 明确为 v1；本次未重新读取 PDF | 本地 README 提供的链接，不构成 AP 原文证据 |
| 原 LeWM 资产入口（不是 AP） | [模型与数据 collection](https://huggingface.co/collections/quentinll/lewm)、[项目主页](https://le-wm.github.io/) | 远端当前 revision unknown | 本地 README 可验证这些字符串，未完成在线复核 |
| 通用运行库（不是 AP） | [stable-worldmodel](https://github.com/galilai-group/stable-worldmodel) | 本地安装包 metadata：0.1.0；上游 commit unknown | 本地 metadata 提供 Homepage；不是 AP 官方实现 |

本次检索记录：

- arXiv MCP 精确标题查询 `ti:"Aim Short to Reach Far"` 返回 `arXiv is rate limiting this IP`，没有提供论文结果。
- 直接 GET [arXiv 标题检索](https://arxiv.org/search/?query=Aim+Short+to+Reach+Far&searchtype=title&abstracts=show&order=-announced_date_first&size=50) 在沙箱内失败：`curl: (6) Could not resolve host: arxiv.org`。
- 按权限请求联网重试，首次及唯一一次再次申请均因自动审批超时而未执行。没有继续重复申请，也未绕过权限。此处不将网络请求申请当作成功读取；需要后续获得有效联网授权或用户提供一手论文/官方链接。
- 本地 `docs/`、主 Python 工作树的 Markdown/manifest 文件以及已安装 stable_worldmodel Python 源码中检索 AP 标题、AP-CEM、AP-rank，未定位 AP 一手实现或原文；已有 baseline policy 仅记录对照要求，不是外部来源。

## 2. Table 1：附件数字全部保持未核验

下表只是附件主张的核验清单，不是已读原表的转录，不可进入论文的 verified reported-results 列。

| 附件称 Table 1 standard starts | 原 LeWM | AP-CEM | AP-rank | 论文版本 / 原表页 / 独立核验 |
| --- | ---: | ---: | ---: | --- |
| Cube | 15.6 | 46.9 | 73.4 | unknown / unknown / 未核验 |
| PushT | 7.0 | 69.5 | 66.4 | unknown / unknown / 未核验 |
| Reacher | 66.4 | 75.8 | 96.9 | unknown / unknown / 未核验 |
| TwoRoom | 17.2 | 46.9 | 98.4 | unknown / unknown / 未核验 |
| 四任务宏平均 | 26.6 | 59.8 | 83.8 | unknown / unknown / 未核验 |

附件还声称每任务 128 paired queries、指标为 success rate (%)。这两个定义及 standard starts 的含义均未由原文确认；不能据四舍五入百分比反推确切成功次数、样本量、误差或显著性。宏平均算术上相容也不能验证论文身份或实验真实性。

## 3. AP-CEM 与 AP-rank：权限必须分开登记

| 协议维度 | AP-CEM：附件线索 | AP-rank：附件线索 | 必须补齐的一手定义 |
| --- | --- | --- | --- |
| 模型训练 | 冻结 LeWM，无额外网络训练 | 同左 | checkpoint/config/revision；冻结范围，是否任何缓存/读出学习 |
| 观察记忆 | 从已有轨迹检索中间观测作为局部目标 | 也使用观察记忆 | 记忆来源、规模、观测历史、索引构造、距离、邻居数、goal/query exclusion |
| 记录动作访问 | 声称由 CEM 合成动作；不能据此断言代码从不读历史动作 | 声称额外使用记录动作片段排序/选择 | 实际数据访问路径；片段长度、起始对齐、候选数、是否拼接/直接 replay、动作归一化 |
| 推理输入 | 不能确认是否只有 RGB | 不能确认是否另有状态/动作辅助 | observation-only、history、proprio、simulator state 的精确定义 |
| 起点 | 附件称 Table 1 standard starts | 同左 | 从记忆轨迹抽取还是独立重置？起点是否恰好对应被检索片段？ |
| 扰动起点 | 附件建议另测，但无参数 | 同左 | 扰动变量/分布/幅度/随机种子、是否共同 query、是否重检索、目标是否固定 |
| 资源 | 仅给算法名称，无可核验预算 | 候选片段与检索工作可能额外存在 | horizon、action_block、replan、执行预算、CEM iterations/samples、实际调用与编码/检索成本 |
| 成功 | 附件称 success rate | 同左 | 四环境各自 predicate、阈值、状态坐标、终点或 any-time 成功、超时/失败处理 |

以上算法解释仅是附件线索；没有原文/代码支持时不能写成“AP 官方协议已确认”。尤其不能把 AP-rank 与无动作记忆的 D2P 混称同信息预算。若允许 AP-rank 访问动作片段，应给 D2P 同权限，或明确标为 information-privileged baseline。

起点扰动必须独立于 memory 构建、阈值校准与 outcome 选根。禁止观察结果后筛掉 AP 无法 replay 或 D2P 不利的 query；standard 与 perturbed starts 必须分表，并共享预先封存的 paired query 身份。

## 4. 本地资产：存在不等于 AP 兼容

下列是本次实际读取文件目录与计算权重哈希所得，不依赖附件数值。没有加载模型。

| 环境 | 本地 checkpoint 目录 | weights.pt SHA256 |
| --- | --- | --- |
| Reacher | `/home/gbk/.stable_worldmodel/checkpoints/reacher/lewm` | `eb70b1fd5409f8f81875d62f5ee5a20dd220a3128a477de66b5760f475f0f469` |
| PushT | `/home/gbk/.stable_worldmodel/checkpoints/pusht/lewm` | `48938400ae3464c9680731287f583a9cb516f55a8ec64ea13a91be47fb15b607` |
| Cube | `/home/gbk/.stable_worldmodel/checkpoints/cube/lewm` | `2839a907362f403f9136383016e91774373a295d958ae75121791f22a9fddf89` |
| TwoRoom | `/home/gbk/.stable_worldmodel/checkpoints/tworoom/lewm` | `566f223624ea4bfb39dbfe6ae731198dd6ea73b7b8919fed6b1ecafca810f7dd` |

四目录均存在 `config.json`、`README.md`、约 69 MiB `weights.pt`。正确本地目录是 `tworoom`，不是 `tworooms`。本地 README 链接到 [Reacher](https://huggingface.co/quentinll/lewm-reacher)、[PushT](https://huggingface.co/quentinll/lewm-pusht)、[Cube](https://huggingface.co/quentinll/lewm-cube)、[TwoRooms](https://huggingface.co/quentinll/lewm-tworooms)；这些是原 LeWM 资产定位线索，不是 AP 的模型使用证明。`cube/lewm_d2p_repair_*` 等修复模型不得替代原冻结模型混入 AP 同 checkpoint 比较。

本地数据与组件：

- `/home/gbk/.stable_worldmodel/datasets/pusht_expert_train.h5`、`tworoom.h5` 实际存在；本次只查目录，未扫描全体数据、重算大型文件哈希或验证 AP split。
- `/home/gbk/code/LeWorldModel/le-wm-main/data/reacher_official_source_20261024/extracted/reacher.h5` 实际存在，`ls` 显示约 93 GiB。相邻 `official_source_manifest.json` 历史记录大小 98,905,882,624 bytes、SHA256 `85a7dddfa1801302abcb175a80a23bb69c78291dd977ce40d69aedcb9123da06`，源 repo `quentinll/lewm-reacher`。该哈希本次没有重算；manifest 记录的压缩 archive 不因记录存在就认定仍在磁盘。
- 未在本次限定检查中定位 Cube 原始轨迹库或 AP 官方 observation/action memory；不能声称整个系统上不存在。
- 解释器：`/home/gbk/code/LeWorldModel/le-wm-main/.venv/bin/python`。本地安装 metadata 显示 stable-worldmodel 0.1.0，包含 CEM/GD/MPPI 求解器及 PushT、Reacher、TwoRoom、OGBench Cube 环境源码；仅源码/metadata 检查，不保证所有环境运行依赖已经通过 reset。
- `experiments/checkpoint_compat.py` 提供现有 checkpoint 兼容加载入口；`experiments/spectral_reachability/d2p_model_call_instrumentation.py` 提供 planning/evaluation 分相实际 predict 计数；`d2p_stage_admission.py` 中新增预算门只验证 planning predict samples，不是 AP 或 D2P 总计算量匹配证明。

## 5. 现有配置与 success：不可直接继承为 AP

本地 `le-wm-main/config/eval/reacher.yaml` 和 `pusht.yaml` 的默认值：50 queries，goal offset 25，执行预算 50；planning horizon/receding horizon/action_block 均为 5，图像尺寸 224。`config/eval/solver/cem.yaml` 为 300 samples、30 steps、topk 30。这些是本地原生评估配置，不是 AP Table 1 的已核验参数；更不是按 300×30 就能计算的实际 predict 总量。

一个必须保留的 success 警告：本地 `stable_worldmodel/envs/pusht/env.py:346` 的 `eval_state` 实际计算 `norm(goal_state[:4]-cur_state[:4]) < 20` 且角度差 `< pi/9`，并不是简单 object-only xy 阈值；同文件另有 `success_threshold=0.95` 的覆盖率逻辑。当前 D2P suffix 分析中的 object pose 连续成本、上述 env predicate、coverage success 三者不能混用。AP 使用哪一个，以及评估 wrapper 如何汇总，仍为 unknown。Reacher、Cube、TwoRoom 的 AP 成功阈值与终止语义也全部 unknown。

## 6. 接入的最小依赖与 admission 清单

此处为待完成的接口要求，不是授权启动后续阶段；正式 P1 之前不打开 P2/P3/P5。

1. 先取得唯一论文身份与原文版本：作者、ID、vN、日期、Table 1 原文页、完整 caption 和 appendix protocol；从论文/作者主页确认官方代码链接，不猜仓库。
2. 锁定小型官方代码快照 commit/license、依赖 lock 和精确配置；确认官方 checkpoint revision/config/hash 与上表资产是否完全一致。若不同，不能直接宣称 matched backbone。
3. 取得官方小型 manifests：query IDs、trajectory/time indices、goal offsets、standard/perturbed starts、扰动 seeds、memory trajectory split、observation/action 权限及其 hashes。原始大数据如需新增必须另获授权，不在本任务下载。
4. 复用现有冻结加载与环境接口，但核对 RGB preprocessing、history/action tokens、normalizer、frame skip、环境版本和 reset snapshot；AP-CEM observation memory adapter 与 AP-rank action-segment adapter 必须分接口及独立权限审计。
5. 预注册并封存统一 success predicate、最大实际 env steps、replan interval、终止/失败/缺失 query 处理；对 memory/query 泄漏做 episode/root 级排除，而不是只比较 seed。
6. 记录 memory encoding/index 建立成本和缓存策略、retrieval、online encoding、predict calls/samples、backward/JVP/VJP、wall-clock、峰值显存及实际执行步数。只匹配 predict samples 时必须逐字限定该范围，不宣称 total compute matched。
7. 在父任务明确批准的阶段内先做不含推理的 manifest/schema/权限检查；GPU 与实验由父任务统一调度。本次没有接入 AP、复现 Table 1 或产生新的 outcome。

## 7. 最终判定与 unknown

**本地接入基础部分可用；AP 身份、官方代码/模型/manifests、Table 1 与关键协议均未完成一手核验。** 当前只能将 AP 作为必须核实的候选竞争基线，不能将附件数字升级为已确认文献结果，更不能与 D2P 当前校准结果拼成公平排行榜。

待补的关键 unknown：论文作者/ID/版本/页码；官方仓库及 commit/license；模型 revision/config/hash；memory 数量与 episode split；AP-CEM 是否访问记录动作；AP-rank 片段提取与执行方式；扰动起点分布；paired query manifests；每环境 success 与终止语义；完整资源成本；Table 1 数值及其误差/样本口径。任何一项未取得来源，就保留 unknown，不补造 URL、页码或默认值。
