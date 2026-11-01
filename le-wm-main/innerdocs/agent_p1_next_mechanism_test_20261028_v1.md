# 下一最小机制校准：共同对称 suffix 库的 prefix 交叉派发

日期2026-10-28。仅预注册实施合同；尚无新runner/prepare/神经/物理执行。只新增本文，不修改已封存比较器、结果或MAIN文档。所有已知根、全部槽位和零/负结果保留；P1=false。

## 1. 已有证据到底排除了什么

MAIN已完成zero5同时间控制与selected-prefix5的实际比较：

- `results/agent_mixedmode_zero_signed_comparison_20261028_v1_MAIN01.json`，SHA `7dff71c3977758da3174f2b4650860ddbdea972685b96127fd0fc765779368d6`。
- 独立112项changes复算回执 `results/audit_zero_signed_MAIN_20261028_v1_MAIN01.json`，SHA `35e1c434e9e8db7b5a276f93a1d164dbf92f643f8250f4107b9fe1ee688312b9`。

本次只读取上述saved产物，不重新调用其consumer或重做112项算术。原四层jointXY/agentXY/objectXY/wrapped angle和五探针全部保留。jointXY的selected-minus-zero中心secant差，root0 off/on的x分别约−40.765/−41.732，root1 off/on的y约+58.608/+56.672；同时实际−x或+y有限响应各自也有不同变化。这些是**响应映射改变**，不能按某一列正负给“能力增减”结论。负导数可通过反向动作利用，方向旋转也不等于可达集合扩张；不再重复这种符号清单实验。

zero控制已匹配probe前15个public steps，但四项start state及snapshot均不同。因此纯elapsed差异不再是充分解释，位置/速度/接触及未导出的solver cache中介仍存在。本文不试图用`reset(savedpoststate)`伪造状态匹配。

更关键的前提来自 `docs/D2P_MIXEDMODE_AND_FD_RESULT_MAIN_20261028_v1.md`：222库四项所选均为stage-greedy且首段预测已改善。故当前prefix即使有真实有效的条件响应，也不是C2的“先付短期代价，换取后续有用能力”的非贪心见证。新实验先辨认**后续能力是否可被冻结模型实际利用**；不能回填这个缺失前提。

## 2. 单一问题及最小替代解释

问题：在同一原目标、同一动作域和相同20步时域下，selected prefix之后，是否有**共同、预先封存动作库**中的可利用任务进展，以及冻结模型的条件suffix是否真正利用了它？

最小替代解释及本实验的辨识范围：

| 替代解释 | 事前固定干预/判别 | 不能越过的边界 |
|---|---|---|
| 只是动作取反，某一secant列换符号 | 两个模型suffix及其真实反向动作均入共同库，两个prefix后全量交叉派发 | 只覆盖此有限库，不穷尽任意旋转/整个10维动作球 |
| 只是首段已经更靠近目标或zero惯性漂移 | 每个prefix有自身真实zero suffix；同时报告zero-relative原公共方向响应与原任务终误差，分列prefix阶段代价 | 非线性残差/接触位置改变仍可能产生收益，不称状态中介已隔离 |
| 有有利响应但模型不会选/预测错 | 各自条件模型生成suffix，并在对方prefix下交叉执行；保存全部预测与actual的排序/差值 | 库内物理oracle上包络不等于可部署选择器 |

这是一个有辨识力的**known-root机制校准**，不是“寻找一根成功再宣称C2”的新筛查。

## 3. 固定输入与两个真实 teacher context

固定原四项 `(root_slot,condition)=(0,off),(0,on),(1,off),(1,on)`，不增删根、条件或goal。prefix角色固定 `Z=实际zero5`、`S=原mixed selected-prefix5`；S仅从已封存mixed选择导出，不从probe正向响应挑动作。

严格绑定原probe prepare `e8e4554ea2448a8c4bcf4e916934800fa8c7afe3eda335174056a551620da3d5`，原40报告 `57ac436ff6cd7fcffcda62e0a165782b719c0383404647f2f87c1fa97ab009c8`、zero20报告 `dfba2b92f1e4c7a4f56a8d8168fca427c0595236a21588aea1a97cae0f897e8e`及已验收strict回执。task/context/root7/goal/renderer/原raw history必须同身份join，不能只按索引或PNG名字猜。

每个task×prefix固定使用`probe_id=zero`分支的**probe-start15**真实同condition PNG作编码代表；先验选择这个代表仅因它是固定库第一项，不能用其20步终点冒充15步图。另核同角色五次独立重建在15步的state/snapshot/pixel hash完全相同；不一致即工程拒收，不换代表或挑最接近者。

8个真实15步图各独立encode/projector一次；原任务真实5/10步latent和同condition原goal latent直接按其封存receipt复用、完整计历史成本。新context为`[z5,z10,z15_actual]`，incoming两包raw为`原warmup后5 + 本角色实际prefix5`，按原mean/std做一次原float32 token算术。Z与S绝不交叉喂latent/goal图，也不用imagined后继替teacher。部署输入只含获批RGB、动作历史及原goal；native state/probe endpoint仅作evaluator身份和评分，不能进入suffix求解器。

## 4. 候选生成：各一解，但共同库包含正反两侧

每个真实context各用相同冻结`d2p.TorchBackend`和`compiler.bounded_ridge`求一次H1 suffix：独立zero nominal、fresh192×10 J、原Q、原damping=.002、原logit球cap=`.75*sqrt(5)`。模型/normalizer/goal/Q/β均不变；不根据probe结果改幅度、J、方向、半径或λ，不多启动/线搜索/失败重试。

记两个求解结果为`u_Z,u_S`。每个task事前固定共同库，顺序为：

`B=[zero5, u_Z, inverse(u_Z), u_S, inverse(u_S)]`。

原数学路径`raw=tanh(float32(logits))`仅执行一次生成literal float32权威buffer；`inverse`对该buffer做float32精确符号翻转，**不是时间反演**，不逆序五条命令，不重新tanh、不再除100。原logit球关于0对称，−logits范数相同，故反向动作不扩大域；同时封存原logits/半径/正反raw bits与该域见证。若原动作域或映射实际不满足这项对称性，prepare须拒绝，不能clip或偷偷换域。

两个prefix之后都预测并执行同一个B，保留角色碰撞/重复动作，既不去重退款，也不把反向动作当已验证预测的“负输出”。所有输出由实际模型重新评价，所有动作由真实环境独立执行。全库先封存后物理；不按后续效果换suffix、选最有利根或自增库。

## 5. 固定40分支与非符号式评价

物理分母为`4 tasks ×2 prefixes ×5 suffix slots=40`，顺序固定task→Z/S→B上述顺序。每支fresh原seed+sample_state placement，真实执行`warm10 → prefix5 → suffix5`，不重置saved poststate，不跳zero、不复用旧轨迹。完整fresh RGB guard、off/on/off snapshot不变性、native earlystop/失败/未开始槽位均保留，不补根。

所有五个suffix都执行，所以下述“库内上包络”只是evaluation定义，**不产生物理winner派发或新方法成功率**：

1. 固定每个原四层空间的原root→goal公共单位方向`d`。角色p的有限响应为`r_p(b)=feature(end(p,b))-feature(end(p,zero))`；角度差单独wrap。
2. 同一共同库的支持值`H_p(d)=max_{b∈B} dot(d,r_p(b))`，包含zero；完整五值与确定性ties全部保留。方向为0则H=null，不从after residual补方向。该有限库支持值不会仅因交换某个已配对的正反标签而改变；但仍不是连续动作域的可控性证书。
3. 输出`H_S-H_Z`与完整共同行为交叉表，不仅输出其符号。四层全列，jointXY为原任务位置轴，object-only不能替代原agent+object成功；不将px/rad/velocity拟合为新综合分。
4. 单列两条**模型直接生成且不按物理重选**的部署对角线`(Z,u_Z)`、`(S,u_S)`，以及交叉项`(Z,u_S)`、`(S,u_Z)`。用原联合位置误差/原生success和wrapped角诊断检验实际可利用性；原Q预测值只作预测，不能冒充真实收益。还要列自身zero对照，区分首段收益与suffix增量。
5. step10/15/20均保留原native任务轴：joint/agent/object距离、官方reward/success、wrapped angle与完整7维。官方success无速度条件，reward含state7；native raw-angle异常原样记录，不在本轮修环境或重贴标签。起始body速度/接触相关snapshot完整留存，但不能从末态推断全程接触。

无事后“显著增益”门槛，不把精确浮点差>0叫科学准入。existing数值/统计authority若不足，变化只描述数值与不确定性，缺失标unknown。失败分支不移出分母；支持值缺任何应执行槽位即null而非对已完成子集取max。

## 6. 资源上限：先验完整计账，不是等总算力

8个真实prefix图各encode/projector一次，共8/8；复用原history/goal的历史编码另账。8个context各nominal1、AD forward1；两解封存后，每context对共同5候选各predict1+独立fresh replay1。因此新模型预算：

| 资源轴 | 新增上限 |
|---|---:|
| encoder / encoder projector | 8 / 8 |
| predictor API / samples / action_encoder / pred_proj | 各96（batch1） |
| AD forward（包含在96内）/ H1 Jacobian | 8 / 8 |
| 数学VJP方向 | 1536 |
| bounded-ridge solve / helper SVD | 8 / 8 |
| 训练、Q拟合 | 0 |

该96不是原observed-suffix helper的5次预算；它包含共同库正反动作各自实际评价与独立重放，不能复用nominal输出免去zero候选费用。8个solve都用相同配置，不做winner额外replay或物理反馈再优化。保存完整原精度模块输入/输出/AD/J/actions/context及模型前后tensor hash，CUDA-only、MATH、TF32=false、无CPU neural fallback。

若沿用已验收zero20相同逐步triplet/原生hooks，40支完整物理的预期硬上限为：constructor/close各40、reset80、public800（warm400+prefix200+suffix200）、internal physics8160（public8000+reset160）、set_state160、goal_setter120、goal_render80、goal_bind40、explicit_render2640。需新runner在prepare中封存这些轴，并以同接口mock核齐；任何额外render/构造步骤先改新合同而非偷藏成本。

工程容量建议有界：metadata/prepare64MiB、physical输出1GiB、terminal reserve64MiB、uncompressed journal128MiB、final report16MiB、disk floor200MiB；物理wall3600s，神经wall600s。密集trace与terminal重复序列化必须RAM尺寸回归后MAIN才能批准，超cap不是科学负结果，不删阵列/降精度通过。失败保存首错、secondary/close错误、所有attempt/return和全分母；无自动重试/补branch。

## 7. 如何判别，而不是怎么宣布通过

- 若只出现中心secant换符号，而共同对称库的任务支持/部署对角线未改善：不支持“可利用能力增加”；保留旋转、选择器与有限库不足解释。
- 若库支持值提高但模型部署对角线不改善：出现有限库内可用响应，但冻结模型没有被证实能利用；物理oracle不能替换planner。
- 若模型对角线有收益但zero-relative支持没有扩张：可能仅为残差变小/动作重新定向/更优有限策略，不称能力集扩张。
- 即使两者均改善：最多支持此共同库、此目标和此prefix引起的条件可用性变化。位置/速度/接触中介未被独立消融；不称“模型本体能力被改变”。

**C2仍缺同根完整链**：被事前冻结选择器真正选择的非贪心prefix、真实原任务轴上的短期代价、可利用条件能力改变与随后真实任务收益，以及完整direct/强基线和fresh确认。当前mixed四winner为stage-greedy，不能把本轮任何正结果改记为非贪心P1见证，也不能事后从222库挑一个物理响应好的prefix顶替winner。若要独立确认，必须在本校准协议之外先封存新source family/历史排除/freshness/有限队列与authority；缺者继续blocked，不捏造新seed或扩大样本数。

## 8. 最小可实施入口与未完成项

下一实现不是新增符号报告，而是**新8-anchor teacher-H1 compiler + 40-branch共同库交叉physical adapter**。可复用的公开接点：

- `agent_mixedmode_zero_time_control_20261028_v1`的原placement、native hooks、RGB guard、20步episode基元与显式计账，但新40 schema/limits/循环必须自有，不能改旧globals或伪装旧zero干预。
- `moving_history_observed_suffix_cuda_20261027_v1.evaluate`的fresh nominal/J、`d2p.TorchBackend`及`compiler.bounded_ridge`数学；其`branch_input`/build绑定旧GN/D2P四prefix，**不能直接用**。新adapter必须按本合同8角色实际15步PNG及完整raw history构造自己的输入和盲签名。
- `agent_mixedmode_zero_signed_comparison_20261028_v1`的四层feature/wrap/原公共方向算术，仅复用纯数学，不改该封存文件，也不再次启动已有MAIN分析。

实施顺序：saved-only身份/15步acquisition→RAM上下文/正反buffer/预算/密集容量回归→MAIN审读并新prepare→MAIN CUDA封存全部B和预测→MAIN新physical prepare/run→独立saved配对。旧5探针端点只能用于核身份及既有描述，不能当作新B执行结果。未来新stem及CLI须随实际runner交付；本文不提供不存在的run命令或虚构SHA。

本交付仅文档静态与资源算术核验；实际新acquisition/runner/tests/prepare/CUDA/40支物理、freshness/authority均未完成。本合同在已读known-root结果后提出，是下一轮固定校准而非追溯性预注册/科学通过。
