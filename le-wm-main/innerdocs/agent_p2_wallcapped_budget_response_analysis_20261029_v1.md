# P2 wall-capped：真实预算响应的结果盲分析字段声明

## 状态与适用范围

本文件只声明后续描述性分析字段，不修改已冻结的runtime三件套或v2执行协议，不改变450s、资源上限、算法、根、Q、normalizer、停止/导出规则或科学gate。作者未读取新run/partial/report/receipt、物理结果或收益，也没有启动NN/物理。MAIN已通知prepare/运行正在接续；因此这是一份**源码冻结后、作者结果盲的分析声明**，不倒签为原科学预注册，不声称在实际dispatch之前完成。

冻结runtime source `5e733e75f56b3dd2f5598d6e9c4ae7878c1e3dd0d613c5b4777c1ecd891b9776`，tests `95138a646e8459aad810127c6ba191ffcfd7a117d44ead239ebb53406c171146`，innerdoc `2628a145a8b52332e319775baba61bc70a3e6a5a4bec30f28b35a844363a5ac1`。实际新prepare/report/strict SHA均待MAIN提供；本文件不假设未来SHA、完成状态或结果。

唯一问题：在同一预声明wall/resource ceilings、原两个known roots和固定候选规则下，两策略分别消耗了多少预算、执行到哪一步、产生了什么可验证的model-only incumbent/export？这是wall-capped comparison，**不是equalFLOPs、等实耗、等最终质量或真实物理优势证明**。

## Authority与完整分母

后续分析只接受MAIN实际外部report SHA及其完整strict receipt；须记录receipt SHA和其绑定的prepare/source/model/全部artifact身份。strict未完成或失败时，结果为 `analysis_not_admitted_integrity_unresolved`，保留缺失清单，不以producer自报状态代替验证，也不读取partial来调参数重跑。

固定4行顺序为 `(root0,joint),(root0,composition),(root1,composition),(root1,joint)`，2个配对known roots。任务/goal/context/action bits由原slot和receipt绑定。任一失败、超时、无incumbent、无export都保留该行；不能补根、删根、换臂、只展示双方成功子集，也不能把候选或AD row当独立样本。

## Panel A：逐行预算与生命周期

| 字段 | 来源/固定定义 | 缺失与解释 |
| --- | --- | --- |
| root、arm、task_id/task_sha256、model hash | prepare slot、严格消费plan及bindings | 不以行号代替任务身份；plan为None时仍从原slot取任务 |
| serial_position、source/report/receipt/prepare SHA | 固定执行顺序和实际authority | 不猜SHA，不借其它run补齐 |
| strict_status、producer_status、stop | 原样分栏 | 工程完整性不等于科学成功 |
| errors、active_unit、pending、finished_units/units_completed | 完整account、原journal/row | 停止主因和所有错误同时保留，不能只选一个有利解释 |
| elapsed_seconds、wall_cap_seconds=450 | sealed同步搜索窗口 | 只比较响应；不要求两臂耗时相等 |
| wall_overrun、overrun_seconds | 原flag；`max(0,elapsed_seconds-450)` | 超限如实留行，不加容差改称within-cap，不截成450隐藏 |
| attempted / completed，每个资源项 | 原完整向量 | 失败intent不退款，不以completed覆盖attempted |
| failed_or_unreturned_requested，各项 | `attempted-completed` | 只叫请求/闭合差额，不冒充实际未做的FLOPs |
| resource utilization，各项cap>0 | `attempted/cap` | cap=0时保留原计数并标not_applicable，不能除零或补0比率 |
| peak_allocated_bytes、peak_reserved_bytes | 原实际allocator sample峰值 | null不当0；不是GPU全设备显存或精确kernel时长 |
| nominal/bank/seed/local/replay完成情况 | 原finished_units分类 | 未完成阶段保留missing；不得把局部执行子集称全部42完成 |
| export_complete/export_missing、plan_presence | 原account及strict plan是否None | 模型incumbent存在不等于双replay导出通过 |

资源必须完整展示predict、action_encoder、predictor、pred_proj、jacobian、mathematical_VJP、jacobian_elements、AD_rollout_steps、root_SVD、root_ridge、encoder、projector、physics、training。VJP/J元素/AD horizon在intent时计requested工作，不等于实际engine backward、实际完成矩阵或FLOPs。J primal的predict/模块已独立计入，不再次加到总forward。禁止跨类资源加权求“总计算”或用predict折算AD。

预留6次导出forward是admission预留，不是已经消费；只有原intent进入后才能记为实际attempt。unused capacity不是“应补齐”预算，不padding、不退款、不借给另一臂。自然完成的joint1902/composition1062是计划总量，不替代实际回执；失败/超时不能用计划量填实际列。

若wall停止后unwind产生非成功闭合，`stop=wall_cap`和`errors!=[]`可以同时成立；应并列呈现预算触发与清理/闭合记录，不能自动解释为模型/实现根因，也不能抹去错误。未知total_FLOPs与engine_internal_backward_calls始终保留null；不以unknown阻断本次共同cap描述，也不据此称equalFLOPs。

## Panel B：anytime model-only响应

每行原样报告有无完整nominal fallback、incumbent的tag/完成时刻/原mean192 Q、全部commit的tag/cost/elapsed/accepted/late。完整阶梯轨迹使用**所有**有效commit的原时间戳，不按改善最大、特定seed、特定horizon或结果选取时间截面；不插入新forward、拟合或平滑。

有效incumbent是严格已复算的完整full15 witness、原goal/metric、未晚于450s的accepted commit。平局保持较早者，不能分析时重新挑另一命令。deadline后的late commit保留原cost和late标识，但不得进入capped最终值或回填export。没有baseline/完整witness则Q为missing，不当0、无穷或失败惩罚值。

派生字段仅为：

- `Q_nominal`：该行完整nominal commit的原Q；没有则null。
- `Q_capped_incumbent`：原strict anytime incumbent Q；没有则null。
- `model_Q_gain = Q_nominal - Q_capped_incumbent`：正为原模型指标下降；任一缺失则null。不更换Q、不归一化成物理成功率。
- `incumbent_elapsed`及完成阶段：报告该候选何时、在哪个固定unit完成，不把它解释成独立“到达该质量的最快算法”证明。
- `Q_export`及`export_exact_incumbent_match`：只有完整双replay计划存在才有导出Q；无export保留null，不用incumbent假造可部署计划。

仅有2 roots，主报告逐根展示，禁止候选bootstrap、把轨迹时间点当样本、显著性宣称或参数检索。可以描述两根的better/tie/worse/missing数，ties使用原float标量精确比较，无新epsilon或按结果调阈值。

## Panel C：固定两根配对差异

统一方向：`delta = composition - joint`。逐root列双方原值和delta，绝不只给平均改善：

- elapsed及各资源attempted/completed delta：负值仅表示该根该策略实际消耗更少；不能直接称等质量speedup或更低FLOPs。
- `delta_Q_capped`：仅双方完整incumbent存在时计算；负值表示composition的原模型Q更小，不是物理任务收益。
- `delta_Q_export`：仅双方都有完整export时计算，必须另列其配对可用分母 **0..2 / 固定2**。这只是条件性描述，不覆盖包含missing的主表，也不能据此丢弃缺导出根。
- seed/local完成数与stop类型并列，说明两臂分别走到了哪里；不能通过补跑、删失败或事后给一臂加预算使其“可比”。

有任何一臂overrun时，仍列全部原记录，并把该root标记cap deviation；不得默称满足共同cap。其结果不用于宣称“共同cap内优势”，也不删除后只用剩余根作主结论。对所有根均报告deviation/missing分母；本文件不新增自动science gate。

## 不属于本次分析的内容

本次不读物理state/native success/真实RGB outcome来挑根、动作、质量指标或停止点；未来物理/终态RGB评价必须先由MAIN封存所有实际选择和严格回执，再独立授权、固定完整任务与分母。model Q下降不能替代任务XY/angle/success，也不能推出P2 mechanism或P1成立。

旧三程序报告只作为已有规则/输入authority，不把旧winner或旧预测免费计入新搜索，也不把旧运行墙时混入本次逐臂响应。原继承编码和加载成本单独注记，不摊给某一臂或抵销AD。不会在看见当前partial后更改450s、资源上限、候选顺序、alpha、重复次数或导出规则。

本文件不提供新runner/分析consumer，不改封存三件套。后续若实现saved分析consumer，应只产生NEW derivative文件，绑定实际report/receipt与本文件SHA，逐项实现这些字段，保留所有unknown/missing/deviation，并明确这是结果盲声明后的描述性分析，而非追认原科学预注册。
