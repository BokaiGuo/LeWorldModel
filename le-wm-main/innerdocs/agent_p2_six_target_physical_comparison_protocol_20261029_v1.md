# 六目标共同 controller：后续物理比较协议（待 MAIN 批准）

本文件只预注册工程比较及报告合同，不实现 runner，不授权生产、不改变已封存结果或任何科学门。固定 known root0/1、condition=off；不新增根、不读取后续物理 outcome 选择目标、动作、阈值或报告时间。Pasteur 独占 raw deployment 认证；此处仅规定必须消费其最终真实 authority 的接口，不另造 inverse/scaler/replay 验证器。experiment-design 技能只用于固定设计和失败报告，不采用训练、调参或效果择优建议。

## 1. 已有 authority 与唯一输入

以下路径均相对 `/home/gbk/code/LeWorldModel/le-wm-main/`，须连同 source closure 与完整 receipt 重建认证，不只检查 status。

| authority | 路径 | SHA256 |
|---|---|---|
| six prepare | results/agent_p2_six_target_controller_cuda_20261029_v1_prepare_MAIN01.json | cccfadc20911dc1be008a047c57bca7400272f6602deb435fb0341a0c8bbc8f9 |
| six report | results/agent_p2_six_target_controller_cuda_20261029_v1_cuda_MAIN01/report.json | b81f12ee3602835bc14fd2e22d7e3644b8376f7ba8b0b265948bfb18429b6934 |
| six full consume | results/agent_p2_six_target_controller_cuda_20261029_v1_consume_MAIN01.json | 592062bded7d6eb1716594b21fac4b5ef6255cf93d5285c2361f0ec2686c0103 |

固定顺序：slot0 root0 prior_FIRST、1 root0 program_first_postcondition、2 root0 arbitrary_FIRST、3 root1 prior_FIRST、4 root1 program_first_postcondition、5 root1 arbitrary_FIRST。每槽只执行本次 six 新 dispatch 的最终 returned mean 所对应已认证 raw15，不复用旧四槽计划/物理，不传 privileged witness，不重新搜索或反馈重规划。共同 controller 的 H3/N8/I2/topk2/var1/seed0、whole/segmented[1,2] 不再改变；72 predict API/324 samples 是原 normalized job 成本，不能充当部署实际 token 的预测费用。

Pasteur 将来提供的 prepare/report/full receipt/source pins 由 CLI 显式输入，不填未来 SHA。必须逐槽连接 `six slot identity → normalized mean bytes → raw15 bytes → actual deployment tokens → paid whole/segmented prediction`，保留两种 token/成本版本；如果 actual token 与旧 token 非 bitexact，后续物理模型误差只用已付费 actual-token 预测，旧预测永久保留但不能换标签。它的认证范围、raw 部署工程授权与 MAIN 显式运行授权均须独立明确；旧四槽 raw-domain/paid-replay 证据不自动覆盖此次六槽。

## 2. 前置硬门与禁止替代

1. six 实际整份 receipt/source/model/input/context/目标与 completion 精确复核；六槽均 returned，原目标和完整 history3/action2 未变。同 root 三槽的原 task ID/SHA、goal、native source episode/reset seed、placement、warmup、normalizer/Jacobian 映射和 action_space 必须逐位/逐字段 join。不能仅凭同192维或相似图像配对。
2. Pasteur 的实际六槽部署 authority 必须对完整15×2有序 raw 数组、domain、normalizer 来源、真实执行 token 和其预测适用性作明确认证。任何槽不满足即保留 blocked 状态；没有替代计划、clip/tanh、补动作、重选 mean、借用 Hi scaler。完整六槽比较仍 blocked，不能只汇报可用子集为完整实验。
3. 原 native 源/环境/runtime/renderer 与 source task→goal/state7/RGB/warmup 映射必须已认证。准备时连接冻结 `agent_p2_common_resource_physical_20261029_v1` 与 `agent_p2_three_program_physical_20261029_v1` 的真实 recipe，不直接借其旧输出。若 six known root 与这些 recipe 的 root/state/history/RGB 不能精确闭合，列出具体失败字段，停止该槽；不得猜 seed/prefix 或恢复 imagined state。
4. 模型与 RGB preprocess 复用已封 pairing/computation parity 的原425 frozen state、原 encoder/projector 和 identity192 goal 坐标，来源/完整 receipt 显式绑定。raw prior 时间语义未知仍 null。没有这些证据不出真实 encoded-Q，而不是用预测补图。
5. 不把 P1 科学门 false 自创成拒绝已授权工程执行的理由；但 MAIN 批准此协议不等于原 native/raw/完整性门已通过，亦不等于 P1/P2 科学准入。

## 3. 同任务、同时间原生执行

六个独立新 env，按固定 slot 顺序运行，复用原 constructor/reset/placement/warm10/goal-bind 流程与每步原 hooks。warm10 不算计划 horizon；cp0 为真实决策根，随后按原顺序完整执行 raw15，cp5/10/15 为从根起的实际公开控制步数。只要前置 recipe 精确闭合，不另加 prefix 或空动作。初始化失败、native done/truncated、工程异常均逐槽留存；done 不改、native 停立即停，不 padding、不自动重启、不将短执行称15步。

每个 public return 保留 requested/actual raw bytes、计数、完整 state7、body snapshot、reward、terminated/truncated、原生指标与 renderer/RGB 来源。保留 warm 源 trace 和所有 plan1..15 轨迹，不只终点。cp0/5/10/15 同时 join 原 task goal、真实状态和 canonical off PNG；同一步的 snapshot/RGB/state 必须完全一致。off/on/off triplet 只按原 recipe 的真实性守卫用途保存，不切换截图来选更好编码。

全六槽×四时间=24个固定记录位置；缺到达的 cp 保持 null/status/reason，不沿用最后状态、不插值，不移到各自终点比较。每 root 固定三对×四时间=24个 pair；只有两侧均到达同一 cp 才给数值差，pair 总分母始终24，并列有效数。跨 root 不当独立6样本、不做独立样本显著性检验。native 早停是真实状态，不与工程失败混为一类。

## 4. 原 native 指标与物理分解

复用并独立 saved 算术核验 `experiments/spectral_reachability/agent_p2_common_resource_physical_summary_20261029_v1.py:63 scalars(state, goal)` 的 state 顺序：`[agent_x, agent_y, block_x, block_y, block_angle_mod2pi, agent_vx, agent_vy]`。后续实现须 pin 该真实源码 SHA，不修改旧函数。

令 d=state−original_goal，均用原保存数值转 float64，不改原 goal：

| 栏 | 必须保存/复算 | 单位与解释 |
|---|---|---|
| 原native | reward=−norm(d7)、native7 distance/squared、原success/done/truncated | 原混单位量保留，不新解释为统一物理误差 |
| XY | agent XY、block XY、joint XY 各 error/squared，及相对cp0的变化 | 原生像素坐标；物体与agent分栏 |
| velocity | d[5:7]、agent velocity norm/squared，原速度向量 | agent速度，不标成block速度或物体机制 |
| angle | 原angle/state、raw angle squared、原 native min-angle项、wrapped atan2(sin,cos)误差、相对cp0 wrapped变化 | radians；wrapped诊断不能替换原native success/reward |

原 native success 保持既有 jointXY/angle 合同，消费者核原实现而不是另设阈值；native min-angle 的边界/可能负值原样披露，不悄悄修成wrapped。每项收益明确方向：error_gain=cp0_error−cp_error，reward_gain=cp_reward−cp0_reward。全部单项并列，不混单位加权排名、不以 agent 惯性/仅agent位移认作block能力。若调用原 metrics 认证，同时独立从 state/goal 重算每个标量，避免自引用 producer 的 summary 字段。

## 5. 三层断点与真实 RGB 合同

固定24张 canonical off PNG（六槽各cp0/5/10/15），每张各做一次同冻结 encoder/projector batch1，不去重，包括同根看似重复的cp0。原 `agent_p2_three_program_rgb_20261029_v1` 的全tokens/CLS/projector/preprocess 保存接点可复用，但必须新独立six schema与source/task/action/time join，不能借其四槽结果改绑。goal 使用已认证原 sealed goal192，不再编码。所有输入先封存，生产神经仅 MAIN host CUDA，禁止 CPU fallback。

逐槽逐cp保存以下独立栏（identity meanMSE，f32输入转f64，明确固定归约次序；推荐沿已封six comparator的 math.fsum/192）：

- `model_to_tracking_target_Q`、`model_to_original_task_goal_Q`：本次部署 actual tokens 已付费whole/seg一致预测；cp0是observed history末帧，非未来预测。
- `actual_encoded_to_tracking_target_Q`、`actual_encoded_to_original_task_goal_Q`：真实PNG编码。cp0新编码必须与原根RGB/预处理/latent精确核对；若不一致保留失败，不校正输出。
- `model_vs_actual_encoded_MSE`、signed `model_task_Q−actual_task_Q`：预测/执行失配诊断；不拿它当原生任务收益。
- 本节每一行精确连接PNG文件/像素SHA、actualstep、state7、request/actual动作前缀SHA、latent/target/goal SHA、full history、模型/预处理源；所有原native指标并列。未到达cp不调用encoder、不伪分数。

解释边界：共同 controller 在实际编码空间未接近tracking target，可能是控制器可达性/优化误差或模型误差，不能唯一归因；预测好但真实tracking差是可检验断点，不直接证明target差。真实tracking好但原task/native差，可提示目标相关性或feature-goal metric错配，仍非唯一因果解释。没有oracle controller、原prior时距认证与生成预算对齐，不识别独立“target质量效应”。不同tracking target的损失禁止跨槽直接排名；物理结果不由 latent loss 决定winner，固定各native量与同时间差值如实报告，不事后选主指标。

## 6. 预算、账本与封存 guard（待批准的精确上限）

以下物理上限对应**原双reset/warm10/raw15 recipe完全匹配**时的旧六分支实际代码预算，不是新run实测值。若接线发现该recipe计数不适用，prepare拒绝并提交具体差异，不能悄悄增步/扩cap：

| 阶段 | 固定满程预算/上限 |
|---|---|
| native | 6 constructor/close、12 reset、150 public returns（60warm+90plan）、1524 Space.step、24 state setter、18 goal_setter、12 goal_render、6 goal_bind、486 explicit render/PNG |
| native资源 | wall900s（含verify/final hash，实际保存scope起止）；128MiB含16MiB终态reserve；floor200MiB；metadata每文件128MiB |
| RGB measurement | encoder24 API/24samples、projector24 API/24samples；AE/predict/AD/新增physics全0；不新goal encode |
| RGB资源 | wall900s（含认证/构造/捕获/落盘/final hash，不含此前native job）；256MiB含32MiB终态reserve；floor200MiB；CUDA allocated3GiB/reserved4GiB，batch1、不重试 |
| 原controller | 已付72predict API/324samples及原27轴完整导入，不重复计成新增calls |
| 部署paid replay | Pasteur 实际receipt全轴/时间独立导入，计划数未知不预填，禁止用normalized旧费用抵扣 |

phys attempts/dispatch/returned/failed、reset内部/公开步/Space子步、render/PNG/write/close/restore与清理异常全列；hooks安装前constructor内部不可观测的量明确unknown。RGB encoder/projector的attempt/return/capture及全tokens、CPU transfer/capture bytes和wall scope单列；能观测多少记录多少，不以数组个数冒充所有transfer calls。首次失败不退款，missing无dispatch，partial保留已付费用，不补足预算声称完成。proposal generation、原controller、deployment replay、native、RGB、saved认证CPU/IO分别记账，不混重复嵌套费用；总FLOPs=null，不称equal compute或equal generation budget。

新结果目录必须独占xb/fsync，不覆盖旧数据；执行前全部六槽task/input/raw/action sequence和来源封存；完整SOURCE/model/normalizer/runtime及原资产before/after hash强等，requested/actual动作逐bit核（包括signed zero）。PNG尺寸/模式/EOF/pixel hash、数组dtype/shape/bytes/hash、完整日志intent-return/hashchain/文件census与无隐藏额外调用均由consumer检查；每阶段严格receipt实际SHA独立写出并串联，不签未来证书。意外文件、来源漂移、partial或完整门失败保留失败产物，不颁完整比较receipt。资源cap允许留下有界失败footer，不隐藏超时/磁盘写失败。

## 7. 批准后的最短实现顺序与当前状态

1. MAIN 批准本协议；取得 Pasteur **本六槽**完整部署 authority/真实tokens预测与工程许可，逐字段核原 task/source/reset/warm recipe是否闭合。
2. 另开新stem实现 six native adapter+RAM mutation tests，复用旧已审hooks而非重复环境协议；prepare/verify/run/consume 完整接口由MAIN审读后执行。六槽缺一仍完整列出，完整比较阻断。
3. native actual full receipt 后，另新RGB adapter固定24图和三层join；由MAIN审读后host CUDA。saved scalar比较只消费两阶段整份receipt，保留全部24位置/24pair及所有missing。

本轮仅新协议文档，无源码/runner新增、无RAM实验或生产测试、无NN/physics/fullstrict/子代。未读取任何未来物理结果。现有six consume认证仅证明共同normalized controller六槽返回；不等raw部署或物理可执行authority。prior source_time仍unknown、generation费用不公平、known2非fresh/独立确认；P1前置未过，P2机制/scientific_admission保持false。MAIN批准前不实施、不执行。
