# E1 同目标四槽主机 CUDA 结果

两已知根分别比较 zero-extra 和动作见证 extra；四槽的目标、原 72 候选库和顺序依固定合同绑定。主机 CUDA 完成全部四槽，外部 SHA 绑定的保存消费重建、两组候选库逐字节一致检查和完整 controller/choice/replay 账本通过。每槽各模型模块轴为 144 次原库、2 次 extra 和 4 次重放，共每轴 600 次；不同模块轴互相嵌套，不能求和当独立 forward 或 FLOPs。无新物理、编码、AD 或训练。

两根的两臂分别都选中原库候选 48、55：每根两臂保存的 `selected_raw` 内容哈希相同，extra 候选均未取代原赢家。故本次没有提供新动作的物理对照，也不证明动作见证子目标优劣或最终成功率。equal model-call count 不代表额外候选生成成本、信息或 FLOPs 相等；已知根也不是独立验证。原 P1 和科学门不变。

报告 SHA-256 `92a8c6d5c9a6fa60bb1738cfd5ed9abd70d5cc889ada04d62886b14c1ca5f3b7`；完整消费 `le-wm-main/results/agent_e1_same_target_host_cuda_20261029_v1_consume_MAIN01.json` SHA-256 `f385e1de76ca391a7ef4d03c198211cd303eb2e4d2bc3bdd69537146e8330eba`。该保存消费继承原来源认证，不声称独立重跑了上游生产。

## 强制候选对照的准备

后续探索固定保留每根的原库赢家、全零 extra、见证 prefix 加零后缀三种动作，合计六槽，计划分别做已付模型整段与分段重放、再认证真实同时间物理/RGB。原库/零/见证在两根的模型 Q 分别为 `0.318943/0.834809/0.536856` 和 `0.202344/0.814494/0.641814`：它们解释了本轮为什么没有选中 extra，不能证明其真实可执行性。

主线程审阅来源合同并独立运行 3 项测试，实际 prepare 和完整重建 verify 通过，合同 SHA-256 `494f806e5bc5d46221b28ce2ad7f288bee4b46d0b3eb3020706f4a07ecf2ed37`，路径 `le-wm-main/results/agent_e1_forced_candidate_contract_20261029_v1_prepare_MAIN01.json`。该设计是在看到前述模型结果后提出，尽管在任何新物理观察前固定，不是独立新根或无条件前瞻确认；合同不授权物理执行。模型 replay 宿主与物理来源、动作映射、资源和 early-stop 守卫尚需完成。

后续主线程完成六槽主机 CUDA 重放并通过完整保存消费：每模型轴支付 24 次 packet 操作；六槽、两个检查点的 whole/segmented latent 与 action history 均逐位相同。报告 SHA `ba2df80d87c825452cdbf3050813e6720f074cd69ba70aea4f716b9007d5717e`，消费 `le-wm-main/results/agent_e1_forced_candidate_replay_cuda_20261029_v1_consume_MAIN01.json` SHA `0605d651ffcb7b4edafed61c12604a89453594e7df5b573b3a944a048c548dc5`。这排除了这批同动作的整段/分段模型重放失配，不证明模型预测符合真实物理。固定六槽的原生 physical adapter、RGB 和早停守卫仍待完成，未做物理比较。
# 追加：同目标固定六候选真实物理执行

主线程审阅 E1 forced-candidate 适配器并复跑 7 项回归；完整上游保存 receipt 重建后，prepare SHA 为 `7d7400cbc3c74756c984bc1d5beaf3dd76f54119b12700ef0716015edc867b5b`。主机原生 PushT 六槽执行完成，报告 SHA `31736c26f5242dc4be38104ad5f51ef61ee7e3cfb88a96b85e7c359614ea69de`；完整 saved consume 通过，receipt SHA `519431b8db8541626ec5bd8e933280b053752f01ef955ed4c021d35575464da3`。消费验证实际动作、根/目标、原生状态、RGB、操作与来源；没有新神经推理。

两根各执行 `bank_winner`、`zero_extra`、`witness_extra`，固定六槽均执行完整 10 步，原生成功 0/6，工程失败 0；实际 120 public step、1224 physics step、132 PNG。以同根终态物体位置误差观察，三臂分别为根 0：25.32、70.79、26.85；根 1：57.57、147.51、140.83。角度与 native7/原生距离是不同指标，不应据此拼成单一总排名。

这是两根已知校准、固定开环动作的真实记录，不是独立确认、闭环成功率或 P1 能力改变证明，也不声称计算成本相等。模型端 bank winner 的选择和物理结果须分开；完整 paired 对照及早停规则仍须由固定分析消费者验证。

固定同根、同目标、同时间 paired 分析现已完成并验证，分析 SHA `9d15df249e17ecc35a99ad57db38ef6fc581ca0f1f7e5e7a40bdf406a6431efd`；18 个 checkpoint 格和 18 个 pair 均保留，消费者的 8 项 fixture 回归通过。第 10 步 witness 相对 bank 的物体位置误差差值为根 0 `+1.53`、根 1 `+83.26`；相应 wrapped 角误差差值分别为 `-0.0914`、`+0.2511`（均为 witness 减 bank），因此不能把单一指标优势表述为全面优势。分析继承已独立验证的物理 receipt，并非再次运行物理。

