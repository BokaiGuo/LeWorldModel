# Fixed-prefix observed-feedback 多边界非线性后缀：实施前协议

仅独占设计文档，不授权实现/运行，不改任何旧 contract/gate。采用 experiment-design 的先固定对照/分母/费用原则，不采用其训练、调参或 sweep 建议。等待 MAIN 严格消费 observed_recompile v2 及实际物理结果后，由 MAIN 决定是否实施；正、零、负结果均不得改变本文 eta、预算、槽或评价口径。若要改变，须另版事前声明，不能覆盖本文。

## 已有证据与本支最小差异

当前 observed_recompile v2 是四个原 selected prefix5 后，在真实 cp5 的完整 RGB history3 上仅重编一次 suffix5，总控制10步。MAIN 已报告 host CUDA errors=[]/4sealed，report SHA `3de58e380acbef66269e69150cab7bca312604b7df88a0f1ae5663764d691b45`；收到本任务时 strict 正在 MAIN 消费，本文不代替验收，也不推断物理收益。

已存在 P4 moving_multiboundary 的 I/R 三反馈边界 driver，以及 P3 feedback_gradient_mpc20。不能把本支称为首次闭环：旧 P4 使用 bounded-ridge/H1 编译；P3 从零初始化 full10、32更新，执行前5、每边界重规划。**本支唯一新增对照**是从完整 nonlinear-library 所选固定 prefix5 出发，使用同一 incumbent-preserving 8步非线性 H1 优化器，逐次比较实际 RGB history3 与自身 imagined history3；后两次反馈接在各臂自己真实执行的新动作之后。不是又一次旧 suffix 重预测，也不是 P3 32步 fullH2 的重命名。

“多步”在此明确为 **多反馈边界**，不是增加梯度步数：原 native H1 预测一步对应5个 public steps，不支持宣称逐 public-step 模型反馈。总控制长度从既有10扩展为20；因此旧10步终态不能充当本支20步无反馈对照。

## 固定分母、对照与时间线

原 known2 roots × off/on =4任务槽，原 full168 library 冻结选择的4个 literal prefix5 全保留；不重新选 prefix、word、root 或 renderer。每任务两臂，固定顺序 `imagined_I`、`observed_R`，共8 episodes；三决策边界 cp5/cp10/cp15，共24节点。各臂原 prefix5 相同，之后分别执行本臂编译的三个5步包，最大控制20步，warmup10另计。known roots=2、fresh=0，不能计作8独立样本。

两臂都从原 fullsource/root 恢复，复现原 native reset/warmup/goal bind/renderer，核 root、raw10、原 prefix5、goal 和 exact cp5 start。不可直接 setter 近似拼接 cp5。warmup 的旧goal reward 与 bind 后 taskgoal 指标分开，不拿旧 reward作当前任务得分；with_target 只按原 off/on，不改 goal_pose/state 制造图。

cp5：I 的 latent3 为原 root history 移位并接原 selected prefix 的已保存 native predicted successor（原 candidate.original_predicted_prefix；需原实际 prefix replay 验证），R 为自己 warmup5/root0/cp5 三张原条件 RGB 的编码。两臂 action_history 均来自各自实际 executed_raw 最近10步，不能用预测动作替代 actual。

cp10/cp15：I 从自己上一节点 selected native successor 滚动 latent3；R 从自己最近三个实际5步边界的完整 RGB history3 编码，分别对应 root0/cp5/cp10、cp5/cp10/cp15。任何 native3 全输出只能取末行作 future successor，前两行不冒充新增未来状态。两臂都读取/编码同定义的自身 actual history3供因果诊断；I 的优化器不读取这些新编码，R 才替换 latent3。不得跨臂借 PNG、state、动作或模型响应。

terminal20 仅编码/评价，不成为第四决策。原 native success/termination 合同不变；成功或终止立即保留实际停止，失败不补步、不补根。全部8 episode/24节点/三个边界分母保留，未到达为 null，不能称早停后 actual compute 或执行步数仍完全相等。

## 同一编译器、incumbent 与合法域

两臂使用原 checkpoint、identity192 mean terminalQ、goal192、normalizer、float32 raw链、MATH/TF32off/autocastoff，参数全冻结。优化输入白名单只有 model context、确切 raw history、原 goal/model/normalizer 与 provenance；native state/coverage/reward/success 不得入 selector。

cp5 的 iterate0 恰为原 selected suffix 的保存 logits/rawbits，保留 parent selection/result/solver witness。cp10/cp15 的 iterate0 预定为该臂**上一节点选中 packet 的相同 logits/rawbits**，作为重复动作 warm start；不是假称原 library 已提供未来15/20步计划。它在新的本臂 context 上重新评估，不继承旧Q。不得 inverse tanh、zero重置、按效果换 warm start、或按物理标签选择候选。

每节点固定8 scalar reverse、eta=1，冻结 tanh chain 一次、原 inward projection，stable(Q,iteration) 从 incumbent+8 iterates 中选。独立 H1 winner replay一次，完整18predict/8reverse。原 float64 solver→float32 incumbent norm 略超 cap 合法性沿原 witness，不能新设 postcast gate、epsilon或投影修改 iterate0；后续新迭代沿冻结 projection。若上一选择仍为原 overcast incumbent，保存完整祖先见证，不能将其重新归类为“新合法域外”。cap 是原每packet域，不凭三次接续宣称原full20总范数受旧full10合同认证。

已执行 prefix 和先前 packet 永不回写。每个新 packet 都有自己的 contextSHA/goalSHA/rawbits/newQ/new_terminal_context/witness/parent seal，绝不沿用旧 predicted_Q 标签。预测Q改善只对本节点同一 context 的 incumbent成立，不保证跨节点Q单调，也不保证native任务改善。

## 预声明资源与持久化

完整调度上限：24节点 ×18predict = **432 predict**；24×8 = **192 scalar reverse**。各432 action_encoder/predictor/pred_proj，1296 temporal rows；192 grad/VJP/raw callback（同次reverse的不同轴，不相加冒充独立计算）。新视觉固定24个控制节点 full3 encode +8个终点评价 full3 encode =32 encoder/projector calls、96 frames；图像是各节点自身真实 history，不用复制缓存冒新增 encode。warmup/source/native render额外成本逐项另计，未知 FLOPs/kernel/继承成本保持 null，不宣称等 FLOPs/等 wall。

两臂完整调度的模型调用额度相同，实际 attempted/completed 分栏；保留失败/早停差异。原 compiler、midpoint、observed-v2既有成本单列继承，不能写成此次 fresh执行。若后续选择合法复用已封存 cp5 编译或物理记录，必须事前另列 reused/new 账与 exact context/raw/start/goal匹配，不悄悄把复用当本协议24次实际调用或等算力。本协议尚不授权这种执行优化。

若实施，需复用现有 Handler/Store/Meter/native bridge 和冻结数值核心，新增窄 rolling adapter，不 monkeypatch globals、不伪造旧receipt。输入图像/state/raw trace/模型源与权重/原全库选择/新source/tests/doc均SHA闭包；每节点 durable reserve→intent→entry/moduleIO/AD→return→selection→replay→dispatch→actual action/native return 双ledger完整对账。history10必须等于原 warm10+本臂 executed_raw[:boundary] 的末10；所有PNG原bytes与encoder输出绑定。未知事件、缺帧、goal/start/动作不匹配 fail closed，不能filter后通过。

暂定实施资源封顶需在未来可执行源码冻结前据实际 tensor/journal ABI核算并明确：不在本文伪称已有32视觉/432predict driver或已验证wall/disk上限。未通过工程检查则不运行，不能为装下结果事后改配额。

## 报告、可识别性与进入实施的必要条件

全8 episode、全24节点按原槽/arm/boundary输出：incumbent与selected预测Q、所有9候选、raw/logit见证、当次实际history重编码与imagined context偏差、后续实际terminal latentQ及预测误差（已实际编码时），原 native success/whole agent+object XY、单列agent/object XY、wrapped angle/velocity及coverage。metric定义从原权威复用，不拿object-only coverage代替原whole-body success；未知null，ties保留，不挑winner或有利子模式。

cp5的I/R使用同根同prefix、相同suffix初始化，可描述context替换的局部算法差异；cp10/cp15起，两臂状态/动作已经不同，最终差异是整个反馈policy的配对结果，不是同一个中间state的纯context因果效应。不得靠终态宣称可控集合扩大、非贪心能力改变、全局消除模型误差或P1成立。

实施前必须由MAIN接受：observed-v2 strict真实receipt、固定4槽的实际物理/strict closure、当前原goal/native/start合同和后续新source/tests/doc。尚未存在的receipt SHA保持未知，不生成未来SHA。负物理结果不会导致删槽、改eta/步数/门槛；MAIN可以决定不实施，须保留本协议及原因。无论是否实施，本支known诊断、P1/freshness/physical release/scientific admission均不因此自动变真。

本交付仅文档。未新增可执行源码/测试、未调用prepare/consume/模型/物理/网络，也未触碰reserved fresh roots或旧三件套。下一实际动作仅MAIN完成当前strict与物理后决定是否授权上述窄driver实现；现在没有本协议的可执行实验CLI。
