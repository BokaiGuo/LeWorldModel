# 固定rank队列的独立真实物理消费者预声明

只消费本chat CUDA实际report `6df856e36b145162e6a8d5ce5f141880a49afe88f43d3cc88f8b98ad790b6d08` 的ownconsume。
不使用旧scheduler命令/终态，不重新选择winner，不训练、不调用神经。先封存本源码/测试/本文与全source闭包，后运行。

固定root-major (0,U/D_rank/S_rank/N),(1,U/D_rank/S_rank/N)，全部8分支，命令相同也独立执行，不去重退款。
所有root+goal、task/blind/selection身份及literal raw10严格绑定本新producer；原moving warmup raw10/RGB/state/body64bits逐位复现。
原frozen构造器检查、native physics/reset/goal/set_state/render hooks、capture guard与close原语只读复用，不修改旧模块globals或调用旧run。
相同PushT-v1无target/action marker RGB224、action_scale100、原native eval/reward/termination；不换目标、不使用物理反馈重规划。

上限600s、16MiB output/512KiB final reserve/200MiBfloor；prepare<=256KiB。
完整8分支上限construct/close8、reset16、set_state32、goal_render16、goal_bind8、public_step160(80warmup+80selected)、physics_step1632、explicit_render/PNG176。
逐操作attempt在dispatch前持久化，returned后completed持久化；partial仍保留八分母、不pad、不补根、不重试。
半途constructor失败内部工作unknown，不以hook零数冒充总工作零；close在journal失败时仍尝试。

主物理比较D-U；次D-S/D-N；分别逐根报告native_distance、native_success、wrapped_allxy_success、object_position_error和wrapped angle差。
不合并量纲；仅全部full10且无truncation时计算完整配对差，提前终止另列，不虚构full-H2终态。
same-Q真实RGB编码不在此consumer中；不拿pose或模型Q冒充actual latent Q。
完整saved consumer核external report SHA、prepare/模型/输入闭包、hashchain、每分支命令/goal、真实transition/metric、每帧PNG/guard与actual operation账。

当前已知CUDA winner同root四臂相同，预期可能真实差为0，仍按固定八分支实际执行并保留。
native证据仅这两个known roots工程对照；P1/freshness/P3及compute/wall优势始终false。
历史编码/anchors/本轮520predict+2J/384AD/2SVD成本另继承，不包含在新物理操作数，更非等FLOPs声明。
