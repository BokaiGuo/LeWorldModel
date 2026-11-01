# 并行推进记录：P2 六槽与 P3 C3

本记录是工程证据与执行顺序，不改变科学 gate。

- P2 physical v1 继承了四槽分母 4，但实际 order 与 plans 均为 6，禁止运行该版本。v2 修正为 6，修正旧 arm 和 prefix 标签；物理预算与科学阈值不变。相关联合回归 73 项通过。
- v2 真实 prepare `le-wm-main/results/agent_p2_intersection_physical_20261029_v2_prepare_MAIN01.json` SHA256 `619a3202a9e14c6d164375a46ee8e51e7e9fe217678e6fb6f3c2725e5219341c`；合同六槽、两个已知根、四个检查点，P1/P2 admission 均为 false。
- 在该 v2 物理执行之前，固定 P1 goal 语义 spec `le-wm-main/results/agent_p1_intersection_goal_adapter_20261029_v2_spec_MAIN01.json` SHA256 `8fd89e22374c61df38de33362338b3634bb73f1c07b58c1f6a8befa217e6ce3e`，重算 verify 一致；语义 prepare SHA256 `f77f4323d5f1c47a6ece122924df4cac060456a9250826f8303072b8e69fdc7c`。43 项接口回归通过。规格不含 RGB 配对，也不构成新根或 P1 准入；哈希本身不能证明外部时间权威。
- C3 真实主机 CUDA 已返回 report SHA256 `88e7ac468dbcda7dddd07ba72e3c42793ff0a3d10f61e2e0c338ff1e85521dfe`，manifest `daa2e3f60a74e8c82c44f3f219696edca6f8f7866dde86ff4ad044dc93cda0d4`，completion `a487accf30ca95e0bb972f34f700b2f9fbd3143d7aa6f715a738e82bfcdcd681`。报告中 K=1 六槽达到资源上限，K=17 六槽树完成；独立 saved consume 尚未完成，不能作为科学结论。

C3 完整 saved consume 随后通过，receipt `le-wm-main/results/agent_p3_c3_compact_host_cuda_20261029_v1_consume_MAIN01.json` SHA256 `cdd61fc20ff4b819e64092f226cbe84ce3777edfc2a2c8472322a01ec3a0601f`；仍是 known2 的工程验证，不代表完整 C3 或等实际 FLOPs。源码预算推导显示固定 width4 常规路径第三队列可能因 compile 轴 8+4>10 而被整队拒绝；各槽实际触发轴还须依据保存账本核实，不能把这一预算后果直接称为代码错误。

保存账本后续核对：K1 六槽中五槽由 compile `8+4>10` 拒绝，root0 adaptive 申请 R8/H3 时同时触发 predict、Jacobian、VJP、SVD、solve、compile 和 metric 多轴；replay `0+18+2=20` 恰等上限，不是拒绝轴。这是已注册的原子预检结果，六槽没有发布 model plan，不是理论不可能证明。

12槽/1728请求身份与27资源轴的保存分析 `le-wm-main/results/agent_p3_c3_budget_schedule_saved_20261029_v1_WORKER01.json` SHA256 `93a3b3ea7e1e2d0c8db8206ad42ae82fffc910400fce62a743b1735a56036a12` 通过 MAIN 重建 verify 和8项回归。拒绝队列请求与未分配 job 身份存在重叠，不相加；该分析继承 strict receipt、未重跑模型及上游 fullstrict，未知 FLOPs 保留 null。
- E1 六槽物理代码与存储分区独立测试 16 项通过，真实 prepare SHA256 `8c6426b249fadc3ad730f1ca3bad9aa3b805badbe5538ef11e5132be021dc922`，待真实 verify 与主机执行。模型内 common controller 不是官方 Hi-LeWM 复现。

E1 后续真实 verify 已通过；主机物理 report SHA256 `77cd42de14e1c7d75f8e7b5b4b3d9ad58fb87e94a77c209333ec8a5fc739c5ce`，完整 saved receipt `le-wm-main/results/agent_e1_six_slot_physical_20261029_v1_consume_MAIN01.json` SHA256 `6644156f48a79a30f5118d9c56c3c3a06c3bc818e44edb55e6fe8bf4c29d38d4`。六槽均完整10步、native success 0/6，120 public step、1224 physics step、132 PNG 均完成且无错误；并不证明不同中间目标间的质量可直接排名，也不构成新根或机制验证。

E1 固定18位置和18同根同时间配对的独立算术分析已全量 verify，`le-wm-main/results/agent_e1_six_slot_physical_summary_20261029_v1_MAIN01.json` SHA256 `64796746b7956fcade47e5919597887ff3b3fc72c1a1fb28a2a6e1e5c7a74a11`，10 项测试通过。第10步 dynamics witness 相对 learned prior 的对象 XY 距离在 root0/1 分别较小约 3.792/0.536，但 native7 距离分别更大约 33.943/104.999；是目标空间与七维距离不同的描述性观测，非优势证明。

时序限制：P1 spec 虽在本次 v2 run 之前生成，physical `run_started` 未绑定 spec SHA；这里的工作过程记录不能替代可信外部时间见证，故不宣称该 spec 的前瞻性已被独立认证。

P2 六槽 v2 主机物理已运行，report SHA256 `aba81af75d6de452f517f8b28a8d60a4baa7599877e96376f25d04c884e56a88`；报告为五槽完整15步、root1 joint 第14步 native stopped 且成功，149 public、1514 physics、483 render，errors=[]。独立 saved consume 正在重建，不能提前视为已验证结果。

上述物理 saved consume 随后完成：`le-wm-main/results/agent_p2_intersection_physical_20261029_v2_consume_MAIN01.json` SHA256 `e96462847db675dc0b0c5376067df3654edd40805d6e374ffa3f8f1d5224e6c4`。固定六槽/18同时间对照分析 `le-wm-main/results/agent_p2_intersection_physical_analysis_20261029_v2_MAIN01.json` SHA256 `1c94fcc8d79b5a6dcadebad0b740615831134d4a9a79ed09139ebdc4cbfe5766` 已 verify，14项测试通过。root1 joint 第14步成功不能与其他第15步结果混排；对应15步配对缺值不填补。共同资源上限并不代表相等实际成本，两个已知根不足以支持泛化或 P2 机制成立。

对应27轴的六个同根资源差已完整重建、10项测试通过：`le-wm-main/results/agent_p2_intersection_resource_deltas_20261029_v1_MAIN01.json` SHA256 `070827c5f871a7a72fbdcd337dd97126b6a9d0808d9dac1b09dd081011529ddb`。两根各自 joint/composition/GRASP 分别为 1902/1062/1902 predict；joint 和 composition 均为85 Jacobian、GRASP 为632 scalar backward。AD primal、数学 VJP 与 scalar backward 不可直接兑换；共同 cap 不等于等实际成本或等 FLOPs，未知项保留 null。

GRASP 原路径与 HOST v2 CUDA 对照报告 SHA256 `491367c06c8665c76c4aa3051e4f22db2554d0e44ded068ddb7b3d00d0448f87`，报告 errors=[]，但 saved consumer 在初始插值的 numpy 精确相等检查拒绝。正在检查 CUDA `torch.linspace` 与 numpy 算术及输入身份，原产物保留；在完成定位和严格修复前，不宣称两条路径等价。

MAIN 对保存的 host 原始 384 个插值分量做只读复算：原 numpy `float32(1/3,2/3)` 重建与记录相差 136 项，全部位于第二点，最大绝对误差 `2.384185791015625e-07`。另用主机 CUDA、无神经调用，实测 `torch.linspace(0,1,4)` 的 float32 bits 为 `[0,1051372203,1059760810,1065353216]`；numpy 直接转换分数的第二内部系数为 `1059760811`，相差 1 ULP。使用 CUDA 实测系数再按原算式复算，384 项全部精确匹配。这定位了该检查的一处验证器实现差异，不等于其他 saved 检查均通过，也不能代替完整重建；修复不得使用笼统误差容差。

报告专用 saved repair 已完整重建及 MAIN 独立 verify，`le-wm-main/results/agent_p2_grasp_interleave_saved_repair_20261029_v1_consume_MAIN02.json` SHA256 `7ea6ee8008bc78cc97368515c84898790dde0b20231a7648886f4f7fa7d8e152`；修复及 v2 联合测试16项通过。固定输入的三 packet、输入、loss 与记录的 x/v 梯度逐字节一致，但 producer 未记录初始化系数，实测不能证明当时未记录的系数；未独立重算梯度、未记录模块内部 IO stride，因此不宣称所有环境或完整优化轨迹等价。下一版须在执行时持久化 fraction。

此后 v3 已改为实际记录两侧 CUDA fraction 的 dtype/shape/stride/bytes，并严格使用封存 fraction 重建 v，无容差；14项测试及真实 prepare verify 通过，主机 CUDA report SHA256 `1b71c342cf92ec40311c073da95dac214382b707aaeb0af9006bc92d66aae1b5`，完整 saved consume 正在运行。

v3 saved consume 随后通过：`le-wm-main/results/agent_p2_grasp_interleave_cuda_20261029_v3_consume_MAIN01.json` SHA256 `5040e084b549212941623b25afb3d1e0b0de37e489c5a473514c3b5a701bbe26`，实际6 predict/2 backward，双方 fraction、三packet、输入与记录的 loss/梯度均逐位一致。仍非独立梯度重算、模块内部stride或完整优化轨迹等价，更非科学准入。

P1 目标语义从 physical v2 strict receipt 提取后的 6×4 分母已重新构造并 verify，分析 SHA256 `11ee4358d1b7beaba66a15403adf5fe16471c69927be8404419617bffa02d654`；23 位置有值、root1 joint 的 cp15 缺失，cp0 标记为原绑定状态推导而非 native return。8 项回归通过。该分析复用原语义算术，不重跑上游 strict，也没有外部可信时序或 RGB 速度碰撞证据。

对这23个已有状态的描述性计数，14处原生七维距离的平方项中速度误差占比超过一半；它们不是独立样本，且此计数不证明排序反转的唯一原因或支持改原指标。

P4 三反馈8臂保存分析 `le-wm-main/results/agent_p4_saved_pair_budget_persist_20261029_v1_MAIN01.json` SHA256 `aeaff75972afbb8400f8b3c458cacd6c45d143b6890f8449693f58e08e99916a` 已经 MAIN 全量重算验证、24项联合测试通过。保留四对、24个编译槽、已知两根；e01_R 第11控制步成功早停，step15/20与对应差值均为 null，另七臂走满20。既有全局计数 115 predict、23 Jacobian；每臂总成本、FLOPs、设备时间和继承费用未知。相同时间的 I/R 是不同轨迹的后验描述，不证明反馈因果收益或科学准入。

GRASP 原实现对照仍需补足 v2 来源绑定与失败前缀记账；E3 同环境连续闭环尚未完成。所有根均为已知根，不宣称等 FLOPs、P1/P2 机制成立或 C3 真实收益。
