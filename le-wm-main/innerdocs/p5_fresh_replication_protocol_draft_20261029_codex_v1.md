# P5：冻结模型跨架构复制的前瞻协议草稿

状态：**草稿／未冻结／未批准／不可执行**。本文件依据已打开的历史诊断编写，
不是对那些诊断的事前注册，也不自动授权采集、GPU、物理执行或打开任何预留
outcome。`未定` 表示没有实际批准材料，不等于零、通过或不适用。本文不改变
P0–P5科学门槛，不实现或重跑历史排重；旧权威只作为依赖引用。

## 1. 问题、研究对象和范围

拟议问题：在原核心算法及适用组件经原项目要求冻结和准入后，能否在另一冻结
架构的原生任务接口中，按事前批准的同任务对照，复现核心机制与真实任务收益？
模型接口/P0与任务前提需在该模型上重新验证；跨架构复制与同一checkpoint内
多任务是两项不同工作 [A]。这只是问题表述，主比较、估计量、统计承诺和具体
新源均未批准。已有checkpoint和模型响应不是机制或任务收益的证明。

候选资产是 PushT 的 JEPA-WM 与既有 LeWM；选用哪一checkpoint、哪一种复制组件、
是否属于同信息比较，以及实际模型文件/加载字节由MAIN在正式协议确定，**本草稿
不冻结选择**。JEPA三任务checkpoint同属一个架构，不是三种独立架构 [B]。
单checkpoint多任务 E3 的 goal×horizon×动作限制另立矩阵，不合并成独立架构数 [A]。

## 2. 已知资产与有限复核

本次只读取指定小型JSON及文档并核SHA，未重跑各producer/consumer或认证其引用的
所有张量。以下文件在 `I=/home/gbk/code/LeWorldModel/le-wm-main`，`O`为其父目录。

| ID | 原始证据与本次核对的SHA256 | 可使用的事实 / 不能推出 |
|---|---|---|
| A | `O/docs/D2P_RESEARCH_REQUIREMENT_EVIDENCE_MATRIX_20261026.md` `b8fcbb9b27a4d0b41e25376cf49f6b6f2f14cb10b1067276000a16532d97f388` | 项目P5要求核心冻结后新架构重新做接口/P0、任务前提、机制和任务收益；原阶段门不变。 |
| B | `I/results/p5_architecture_provenance_20261027_v1.json` `6c169f0685bb227913ae6deb821d74093e15eed29f063584e919bae892cde7b0` | 本地JEPA合同带明确unknown；native C1记录RGB+agent `[x,y,vx,vy]`，LeWM相关比较为RGB-only，权限不等；远端权重真实性、许可使用、normalizer统计血缘及实际预训练成本等仍待核。历史合同不是新运行的loaded-bytes证明。 |
| C | `I/results/agent_observed_raw_coordinate_physical_20261028_v1_consume_MAIN01.json` `8df105747cf6835056d41ca1b1fd80f7f5b4170f7f527ec538213addc4c9d0bb`；RGB receipt `I/results/agent_observed_raw_coordinate_rgb_20261028_v1_consume_MAIN01.json` `d6b6ed08732d7595e740f98e0dc2450adf439e726aa3078854badb78f7d8b4b2` | 保存面板为164物理分支、2个known roots、0个fresh roots；可作工程对照来源，不是164个独立根。 |
| D | `I/results/agent_p5_raw164_jepa_cuda_20261029_v1_consume_MAIN01.json` `4f6bbed041f077818627e68157b0cfab99ddba2e38100827db10f46c4aa1b759`；H2 `I/results/agent_p5_raw164_jepa_h2_cuda_20261029_v2_consume_MAIN01.json` `783b65f67899791b5697bfe0819ccfb3231d57212b9e3e85aefc02419459c0ea` | 两receipt各记164请求、80对、P1/P5=false及RGB_only=false；分别继承报告SHA `e7ac4ef9eb7491f619de25b04c55f5e56560e2833a840dd5305f3db6207b3a5d` / `e5a561215bfe7b6f1c1c224e7661211e3f237fb16458a6a409f5a1da0fc08579`。不是matched cross-model性能回执；不同历史时距不作单因素因果比较。 |
| E | `I/results/agent_p5_fresh_confirmation_preconditions_20261029_v1_MAIN01.json` `d453e4cf70518839844916566fa274959c515d05310ae090fb5699e589cf3133` | 该固定快照 `blocked_no_certified_fresh_cohort`，认证数0，P5分母null；不证明其他地方绝无后续证据。 |
| F | `I/results/p5_closure_readiness_20261027_v1.json` `07cd48053f31457ec390f99b03a728ece2c32376cd770c25886161b4b8381980` | 旧closure有限快照，不是对10月29日所有资产的全局盘点；旧未绑定不等于后来不存在相关工程产物。 |
| G | `I/innerdocs/agent_p5_prospective_source_separation_protocol_20261029_v1.md` `c1e4c1424c5a75915e364b59e389e10dd228d0509f1d842b12606a2474ad6235` | 分离事件新、状态集不交、采样独立、训练血缘四命题；其四槽试点仅proposal，不能替代原fresh-root gate，也不是本草稿已批准的N。 |

已有saved-only身份连接器 `experiments/spectral_reachability/agent_p5_fixed164_physical_model_join_20261029_v1.py`
与 `experiments/spectral_reachability/agent_p5_h1_h2_saved_identity_join_20261029_v1.py`
可用于历史工程核对；它们不生成新根、不重跑fullstrict，也不等于新模型原生任务执行。
准确证据仍须以各自来源SHA及当前实际读取为准，不从文件名或164/80行数推断相同集合。

## 3. 决策表：实填约束与待审批空项

| 项 | 已知约束 / 已批准内容 | 需填写的协议决定与证据（空项） |
|---|---|---|
| 复制问题与范围 | 项目要求训练无关的冻结核心与预声明复制组件 [A] | 主问题：____；选定P2/P3/P4组件及排除项：____；核心freeze path/SHA：____；审批者/时间：____ |
| cohort | 当前固定快照无认证fresh，P5分母null [E] | P5样本量N：____；候选排序及抽样分布：____；seed/固定规则：____；source与episode划分：____；独立单位：____ |
| source与资格 | 已知历史164属于2 known roots [C]；原fresh排除与外部批准由既有权威负责 | 新source和实际cohort权威 path/SHA：____；coverage批准及cutoff path/SHA：____；每槽身份与排除receipt：____；原始来源真实性审核：____ |
| 模型/任务 | 候选JEPA与LeWM有不同观察权限 [B] | 每模型checkpoint/源码/config/normalizer真实path和SHA：____；数据及许可：____；任务/goal与native success predicate：____；动作单位、范围和时序：____ |
| 输入与标签 | 历史JEPA含agent位置/速度，非RGB-only [B,D]；历史H1/H2时间不同 | 各臂RGB/proprio/目标权限：____；history时刻/长度：____；动作packet及归一化：____；label来源和prediction-seal规则：____；不同权限的解释：____ |
| arms与反馈 | 历史报告并不证明matched控制性能 [D]；相关组件和强对照须按[A]适用项确定 | 主臂/对照身份及实现来源：____；统一初始状态/目标/动作支持：____；每臂反馈频率及可见信息：____；不等权限的处理：____ |
| 资源与训练 | 训练配置不是实测预训练成本；历史normalizer的统计血缘未知 [B] | 每臂encode/predict/JVP/VJP/backward/solver/物理及wall预算和计费边界：____；预训练与任何额外学习成本证据：____；缺失成本如何报告：____ |
| 失败与停止 | 已见根、分支或额外目标不增加独立样本 [C] | 全部attempt记录：____；失败与缺失在固定分母中的处理：____；停止/中断/补采规则：____；批准的输出与审计：____ |
| 主终点与统计 | 原科学门不能降级 [A] | 真实native成功及成本端点：____；主对照/方向/估计量：____；配对单位：____；区间/多重比较与既有门的绑定：____；非显著/失败解释：____ |
| approval和执行 | 本文与 [E] 均不授权source run；历史receipt不替代本次认证 | 协议版本与外部SHA：____；MAIN独立审批与权限：____；运行环境及资源预留：____；运行前再核实际权威：____ |

这些空项不由历史known-root表现、被观察的效果或输出臂排名反推。旧LeWM四槽
seed/顺序不能自动迁移为P5方案 [E]。若不能确认成本、训练血缘或许可，保持unknown，
不填零；不将不同latent维度的MSE直接作为跨模型性能优劣。

## 4. 拟议接线与验证顺序（非新的通过条件）

1. MAIN先确定问题、复制范围、目标和原科学门对应版本；以原实验合同说明何处
   比较信息权限相同，何处为不同权限的限定性比较。冻结具体模型和原native任务
   predicate，保留JEPA proprio差异；不得用重新训练的adapter/readout冒充主线。
2. 在接触新确认结果前，MAIN补完第3节协议、控制臂及逐轴预算/统计承诺并独立
   审批。可引用已有资产作背景，但本草稿是在历史结果之后写成，不称完全盲态。
3. freshness agent和原保管者依据原authority处理来源闭包、范围和身份；本协议只
   接受经过原流程及MAIN审阅的实际引用，不自行重新排重或认证。新来源建议 [G]
   中四种分离性质必须分别留状态，不能由新seed、新事件或自签ID推断状态不交、
   训练未见或统计独立。
4. 获得相应权限后，再验证所选模型的接口/P0、任务可达性、核心与对照的真实
   路径、预算和封存：报告实际loaded bytes、source/normalizer、动作和观察时间、
   attempted/returned、所有失败与完整成本。模型预测先封存，再按协议使用真实
   同槽标签；控制收益须真实native执行，保存全过程，不由预测MSE替代。
5. 按批准的分母、主比较和原统计门报告全部失败及不确定性；同模型多任务另报。
   缺实际前置时停在blocked，不重命名旧receipt、补根、调整门槛或生成运行命令。

验证分层：draft review只能检查草稿字段是否提供；source provenance与freshness由
原authority和MAIN审核；新模型原生任务/对照需自己的完整执行和消费回执；科学
结论最后按原门审议。任一层的成功不自动提升其他层状态。

## 5. 外部交付与当前状态

待原保管者提供原调用的完整attempt/reset记录或可证加载闭包；MAIN提供当前范围
证据和独立coverage批准、P5事前协议选择与经认证的候选身份；模型发布方/权利人
提供不可变来源、normalizer与训练成本及许可证明 [B,E]。若这些材料不存在或不能
审查，应保留unknown/blocked，不以协议草稿代替。本文没有调用GPU、模型、物理、
freshness resolver或fullstrict，也没有创建确认队列或许可。
