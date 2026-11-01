# Six saved差分：实际完整authority及三断点协议补充

仅新文档，等待MAIN批准实现；不修改已冻结文件，不实现/运行分解器，不NN/encode/预测/physics/fullstrict或子代。承接分解协议 `agent_p2_six_target_error_decomposition_protocol_20261029_v1.md` SHA `74f1d02c3282ebfaf7b30828156137dd9e1a4a207527d5694c0c7abe13ff59d9`，以及actual RGB ABI补充 `agent_p2_six_target_error_decomposition_rgb_abi_20261029_v1.md` SHA `16c32ff20d249fdace43761f2c3da7c224826f310d04cbe4cd4f51c28cf2036f`。后者的receipt pending是当时状态，本新文档补入最终实际文件，不回写历史。

## 完整实际authority

以下路径相对 `/home/gbk/code/LeWorldModel/le-wm-main/`：

| 角色 | 路径 | 实际SHA256 |
|---|---|---|
| RGB prepare | results/agent_p2_six_target_rgb_20261029_v1_prepare_MAIN01.json | 14aace0fad5c941fce475a41d45bb1dfa1bad9388cba9f767bdd4eb42aa4f393 |
| RGB report | results/agent_p2_six_target_rgb_20261029_v1_cuda_MAIN01/report.json | 26e00d955ee2c106e4452b97ddca26a7731d721f7e00605f5750e3a38981460f |
| RGB completion | results/agent_p2_six_target_rgb_20261029_v1_cuda_MAIN01/completion.json | f54b1ba7fc7657db1b738f1fe8be5ca280d790a14d5fae0a1a5326d0ac5bcb23 |
| RGB full consume | results/agent_p2_six_target_rgb_20261029_v1_consume_MAIN01.json | c16d99ba82c8d3adeb68515e2035cc191bd060ab6c273e07ff21cef9a23d870b |

本轮实际核最后receipt文件SHA及metadata：schema=`agent_p2_six_target_rgb_20261029_v1.consume`；status=`all24_saved_RGB_positions_verified_science_blocked`；report/prepare/completion三个SHA均与上表相同；denominator={slots:6,rows:24,pairs:24}。MAIN已完整消费，包括cp0 exact、全来源/数组/事件。whole wall通过report fsync/hash为18.69418140104972秒；不是公平运行时间比较。known_roots=2/fresh=false/scientific_admission=false；receipt的新predict/physics/AD均0。

这一步只校对实际metadata和MAIN交接，不冒worker独立fullstrict。后续saved分解仅局部SHA/身份/数组算术核验，不重新调用原consume/verify或任何上游fullstrict，不签科学准入。

## 固定24pair与三个不同断点

固定每root三对prior/program、prior/arbitrary、program/arbitrary，cp0/5/10/15；仍24pair，不因为ties或重复时序丢分母。每pair方向right−left，输入统一为越小越好的原goal误差：

| 断点 | 第一轴 | 第二轴 | 解释边界 |
|---|---|---|---|
| A：model→actual encoded | Δmodel_task_Q | Δactual_task_Q | 相同原goal、actual-token预测与真实编码次序不一致；结合残差能量/cross term，不能只拿target tracking loss比较 |
| B：actual encoded→jointXY | Δactual_task_Q | Δjoint_xy_error | 编码goal度量与真实jointXY次序不一致；不是A中的模型误差 |
| C：actual encoded→native7 | Δactual_task_Q | Δnative7_distance | 编码goal度量与原混单位7维误差次序不一致；速度/角度项可影响native7，不能将其全部归入模型误差 |

模型Q和实际编码Q始终对**同一原taskgoal**，不是不同tracking targets。native用正向distance，不能将native7_reward直接混入越小越好判号。jointXY使用原joint_xy_error；native7_distance保持原7维量，不拿wrapped angle重定义。

严格reversal定义：两轴均有限非零且一负一正。实现用明确比较 `(a<0 and b>0) or (a>0 and b<0)`，不乘积以免underflow，不设epsilon/threshold、不按结果调容差。任一轴恰为零（含signedzero）记tie，不当严格reversal；另列one-axis-zero/both-zero字段。任一为null记null，不填0；非有限数为invalid而非普通null。每断点分别列same_nonzero_sign/strict_reversal/tie/null及其24总和，不能只报告有利子集。

## MAIN已报告的描述性计数（非本worker新复算）

MAIN直接读取同一实际receipt的固定24pair后报告：

- A strict reversals=3：root1 prior/program在cp5、cp10；root1 prior/arbitrary在cp5。
- B strict reversals=1：root1 prior/program在cp5。
- C strict reversals=7。
- MAIN摘要同时给出ties=6、null=0；实现必须对三个断点各自输出实际tie/null列及具体pair身份，不把一个全局摘要不加核对复制为每轴的统计，也不把cp0 ties移出24分母。若复算与摘要不一致，只报告具体断点/字段/归约差异，不能为匹配计数改阈值。

这些数字仅是MAIN对同批known roots的描述，不是新独立确认、策略winner、失败率估计或对每pair独立的显著性结论。本补充没有读取pair数值或自行产生这些计数。

## 物理分解与禁止混因

C断点每一pair保留原native7 squared及agent XY squared、block XY squared、raw angle squared、agent velocity squared的right−left分量；四组和与原量的roundoff residual照上位协议保留。B断点另列agent/block/jointXY，避免只有agent位置变化也被称物体收益。角度同时保留原native min-angle及wrapped诊断，但raw angle squared才是native7 squared中的角度项；agent_vx/vy不是block速度。

即使C reversal多于B，也不能未经逐pair分解就宣称“全由velocity导致”；反之native7包含velocity带来的排序变化不等模型预测误差。需要逐pair原分量证据，且这些仍是混单位范数的代数描述，不导出唯一因果。跨断点交集固定枚举全部24身份和布尔/nullable标签，不只挑reversal样本或region。

## 实施边界与交付状态

MAIN批准后另建独占source/tests/doc，局部精确绑定上表四文件、原24行/24pair及p/z/t/g/source/native身份，按已封协议重算向量分解和本补充三个strict-sign断点。新函数不重encode/执行动作/跑模型/调用fullstrict；无weights加载，无新预测分数选择。先RAM mutation，再由MAIN实际saved build/verify。输出只描述24固定pair及其所需原24位置，不扩根/样本/参数搜索。

本轮仅新增协议补充；实际检查是receipt的sha256sum和jq元数据，无新测试、差分报告、认证结果或生产调用。所有旧冻结文件保持不变。生成预算不公平、prior时距unknown、known2非fresh及P1/P2科学门false继续保留。
