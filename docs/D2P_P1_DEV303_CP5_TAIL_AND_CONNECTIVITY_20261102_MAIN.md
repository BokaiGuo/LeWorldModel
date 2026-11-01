# dev303 反馈尾部和边界连通性开发结果

这是一批已使用的开发根上的诊断，不是独立确认或 P1 通过。

## 反馈尾部真实执行

- 固定前缀后的真实 RGB 编码及相同预算的模型尾部规划已完成；
  两臂各有 10 次预测调用、3200 个预测样本。
- 事前物理合同 `p1_dev303_cp5_tail_physical_20261102_exclusive_PREP_DEV01.json`
  SHA-256 为 `1ee6d170a4f00d64a861d04b42b5b6003c5f0f8da450341ae713ac4456c59c50`。
- 实际物理执行报告
  `p1_dev303_cp5_tail_physical_20261102_exclusive_RUN_DEV01/report.json`
  SHA-256 为 `f9f367dff0913ab85287461178d089ad0aa214af28beb6af0d0fc75491ae838a`。
  两臂均通过前缀和 cp5 回放检查，各执行完整十步尾部，无原生提前成功。
- 独立保存核验器对上述合同、报告、动作日志和帧链返回
  `saved_chain_verified`，两臂各返回 10 个尾部动作。
  核验的第一次调用因人工传入错误报告 SHA 被拒绝；纠正输入后通过，
  不是物理执行失败。
- 两臂尾部每一时间步的物体位置与角度完全相同，仅有微小共同漂移。
  因而这批执行没有展示重配置导致的可区分物体响应或任务成功。
  差异化推杆状态和 reward 不能代替该机制证据。
- 保存帧中推杆和物体可见颜色掩膜之间的最小像素中心距离，
  RGB 接近臂从 cp5 的 43.68 像素降至十步中的最小 13.42 像素；
  zero 臂从 64.85 像素降至最小 42.76 像素。不能凭这些离散帧排除
  帧间短暂接触。逐帧来源与指标在
  `results/p1_dev303_cp5_tail_response_analysis_20261102_exclusive_ANALYSIS_DEV01.json`，
  SHA-256 为 `37b5e033e4cbca6fb9da7a2c1d2f696f4ab73daaf21c79cbf9858ad05bf0f071`。
- 代码中 V13 的几何条件是前三个动作在固定 cp5 接近方向的投影和
  不小于 0.15，不是逐步接近或物理接触约束，剩余七步不包含在该条件中。
- 这里只比较 RGB 接近前缀和 zero，不是 selected、zero、
  `task_greedy_direct`、random 四臂，也不是完整预算或强基线比较。

## 近边界候选的连通定义

新边界生成器第一版和 v2 使用四邻域连通。真实开发帧的推杆精确颜色掩码
含 38 像素：四邻域 6 个分量；既有 RGB 接口使用的八邻域为 1 个分量。
两种统计已对同一保存帧独立复算。旧拒绝记录保留，不改变旧 guard。

独立 v3 使用与既有接口一致的八邻域，同时保留推杆分量数量和包围盒、
物体完整性及边界检查。在 dev303 的事前保存图像上它生成 4 个有界的
RGB 几何接近候选；它们尚未完成冻结世界模型选择或真实执行，也不证明接触、
物理安全或动态谱程序机制。参数与非接触运动模型仍是开发配置。

交叉运行 handoff、物理执行和 v3 几何测试共 27 项通过。
P1 的机制、独立来源、同预算对照及阶段准入均未完成。

## v3 候选主机 CUDA 评分

在开发性事前合同
`p1_dev303_v3_cuda_executor_20261102_exclusive_PREP_02.json`
（SHA-256 `213345b6680e7cd139768b50857180864a0d512213ba0da535ce77f5c3dfde35`）
下，冻结模型逐一评估 4 个 RGB 几何候选及 zero、`task_greedy_direct`
和随机对照。报告
`p1_dev303_v3_cuda_executor_20261102_exclusive_RUN_DEV01/report.json`
的 SHA-256 为 `e9bb449bcb44aff75df4061b867f6dacd9ebcb17202dea0c4b7fa0078fb6f15c`。
四个几何候选中按预定 latent 评分选中 index 3；七项调用各计一次动作编码
与一次预测，模型参数和 buffer 哈希未变化。历史图像编码及完整总成本
未计入，报告自身仍标识 `ledger_verified=false` 和
`score_provenance_verified=false`，因此不能当作已认证的公平预算或物理选择。
几何候选不是由动作 Jacobian 编译的谱程序，也不证明 P1。

## v3 四臂前缀真实执行

固定合同 `p1_dev303_v3_four_arm_physical_20261102_exclusive_PREP_02.json`
的 SHA-256 为 `8e2692bf54cff2ddf1a7abcb4d447861ef25cc31836f64770f230d1358bed3d7`。
物理报告 `p1_dev303_v3_four_arm_physical_20261102_exclusive_RUN_DEV01/report.json`
的 SHA-256 为 `c74363f2c5a0f34866e9becdc0b9891f83c48c67872f88e69d14fff94c13ed93`。
selected、zero、task_greedy_direct、random 各完整执行五步，未出现原生
提前成功，独立保存核验返回四臂 `prefix_returned=5`。
四臂逐步物体位置、角度一致，终点推杆位置不同。共同后缀和各臂自行
重规划后缀尚未执行；这不是总任务指标上的非即时进展认定、接触证明、
公平性能比较或独立确认。

## 同开发根完整时域 CEM

合同 `dev303_cem_launcher_20261102_exclusive_PREP_CODEXCPU20261102A.json`
SHA-256 `9a4d3b33c2dc9fd113d183808e2169cce13947425e707db483289f9a63f8bc87`；
报告 `dev303_cem_launcher_20261102_exclusive_RUN_CODEXCPU20261102A/report.json`
SHA-256 `e2ec547d23cf62c2771f726f164a840a0d5c0ee7edc1538296bdd96c47b4fd71`。
冻结模型从真实 warmup 观测规划 15 步，执行 5 步后以真实 RGB 和实际动作
重新规划后十步；15 步均执行，原生成功为 false。保存链核验通过：
记录 325 次动作编码、325 次预测、15 个控制动作和一次 CP5 重编码；
初始化编码和 warmup 另计，模型内部计算和硬件身份未认证。
终态物体位置误差约 236.638，周期角误差约 3.051 rad。
这是已使用开发根的本地探索性 CEM，并非等预算或独立确认。

共同后缀开发合同
`p1_dev303_v3_cross_common_tail_20261102_exclusive_PREP_01.json`
SHA-256 `b93d5490192830c9f31abde30d7eb2b7577d89006da056e3b48f4755a1ebe88a`；
报告 `p1_dev303_v3_cross_common_tail_20261102_exclusive_RUN_DEV01/report.json`
SHA-256 `51f783344256f5edad4523602621f759c0fac1839c971dedd86abcdceb67471a`。
四前缀分别接四个几何后缀和一个 zero 后缀，20 个分支均执行完十步、
无删失、无原生成功，保存链核验通过。该后缀族是在已观察前缀后制定的
开发设计；有物体终态差异，但有利性需要以目标位置、周期角度及每个
前缀的 zero 后缀分别配对分析，不能以不同状态向量或 reward 替代。
配对分析现已完成，保存结果为
`results/p1_dev303_v3_cross_tail_object_analysis_20261102_exclusive_ANALYSIS_01.json`，
SHA-256 `cedddc7f6c98f304e8efc653d7be28ca190fceabff6cbbf1c0e481fe4e4d2c3e`。
相对各自的 zero 后缀，所有非零共同后缀都使物体位置误差增加、
2π 周期角误差减少；这是位置与角度方向不一致的开发结果，
不是任务成功或净收益证明。20 个分支均未观察到原生成功。
分析测试 5 项通过。

## 逐步真实反馈重规划

开发合同 `p1_dev303_feedback_live_20261102_codex_exclusive_PREP_DEV01.json`
SHA-256 `74b31b22db7a4bfb803264aa45889876fe935d361567949f8aa41e93a054d686`。
执行报告 `p1_dev303_feedback_live_20261102_codex_exclusive_RUN_DEV01/report.json`
SHA-256 `23e66529af689ed6a13e19c9b3d6f5ecc85813c5b1b05a3ec22ac31c0f5f78ee`。
从已封存的 RGB 接近及 zero 两条前缀出发，逐步采集真实 RGB 和实际动作，
两臂各重规划和执行五次，完成保存账本核验；冻结模型输出及计数
没有经另一模型运行独立复现，完整总成本未认证。

两臂五步均未成功，逐步物体姿态相同，仅有微小漂移。10 次选择中，
无一次选中每轮保留的 RGB 直接接近种子；这提示需要用独立固定的动作
和同反馈对照区分评分偏好与候选效用，不能只凭分数解释根因。该两臂
不是上述 v3 四臂的各自后缀，也不是公平强基线或 P1 机制验证。

后续新增两种前缀 × 两种选择方法的开发对照：两种方法每轮均完整
模型评分，一种执行模型选择，一种执行同候选库中的直接接近种子。
合同 `p1_dev303_feedback_choice_contrast_20261102_codex_exclusive_PREP_DEV01.json`
SHA-256 `8d052d1dcaec34586a73163872e0197a7c963a75adb1d8c321a507c65e0c7ba7`；
报告 `p1_dev303_feedback_choice_contrast_20261102_codex_exclusive_RUN_DEV01/report.json`
SHA-256 `f5001edf3752a436768589c7056d11994a65e7ea21b6e6909886e29b832c0431`。
四臂各完成五轮，保存核验通过，均未原生成功；四臂物体终态相同，
推杆位置不同。模型输出没有独立重算、完整成本尚未认证。逐帧位置和
动作成本仍需配对分析，不能将该开发对照视为 P1 准入。
