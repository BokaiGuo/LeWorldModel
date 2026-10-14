# Decision Sufficiency 研究方案：从预测质量到可靠规划

## 核心命题

文献和当前实验共同指向一个更窄、也更可验证的问题：

> frozen latent 只有在有限时域内同时满足 predictive sufficiency、action
> identifiability 和 decision sufficiency，并且部署动作落在校准支持集内时，才可被允许用于规划。

统一对象仍是 \(G_h^\Phi=D_\Phi A_\Phi^{h-1}B_\Phi\)，但每个候选动作必须经过真实 successor、
动作效果和独立任务后果三层核验。prediction MSE、Fourier、mode accuracy、alignment
或 commutator 单独都不再是规划证书。

## 三层证据链

| 层次 | 要回答的问题 | 最低证据 | 不能替代的证据 |
|---|---|---|---|
| Predictive sufficiency | latent/history 是否能预测声明 horizon 的 successor/effect？ | root-disjoint multi-horizon error、closure、预测不确定性 | action ranking、规划收益 |
| Action identifiability | 候选动作的真实 effect 是否可区分、可排序？ | matched `+a/-a/zero`、signed effect、inverse risk、support coverage | 单步 MSE、mode accuracy |
| Decision sufficiency | 该排序是否导致独立规划后果改善？ | held-out ranking regret、real/predicted mismatch、闭环 risk/success | 离线 decoder、privileged successor |

三层按顺序报告；后层不通过时，前层的改善只能作为机制结果或失败边界。

## 文献 insight 到方法调整

1. PlaNet/Dreamer 的历史聚合隐状态说明部分可观测可以通过 recurrent state 表示，
   但不保证接触模式或动作效果被保留。因此 history lifting 必须以 action-effect 和
   held-out consequence 验收。
2. Visual Foresight/MPC 说明 receding horizon 能减少长 rollout drift，不能修复短时
   action-ranking inversion。因此在线策略必须短 pulse、重新观测，并保留 action-support gate。
3. ContactNets、contact-mode planning 和混合动力学工作说明 sticking/sliding 等 mode
   不应只靠连续回归器隐式恢复。本项目将 mode observability 与 within-mode action
   identifiability 分开统计；前者改善而姿态不闭合时必须判为边界。
4. S4/Mamba/HiP-RSSM 说明更长记忆不等于决策充分。正式策略继续采用
   horizon-conditioned carrier/residual；fit-only selector 只作为 instability diagnostic。
5. D-JEPA/TD-JEPA/ATM 和 GDD-WM 方向说明决策偏好可以作为候选训练信号，但 teacher
   preference 必须 execution-grounded、candidate-set-independent，并通过独立规划验证。
   GDD-WM 保持 `Proposed / Not Yet Validated`，不把 frozen-model 负结果改写成蒸馏成功。
6. ATM 的真实域/预测域 inverse matrix 现在作为 baseline diagnostic，但本项目额外报告
   `state-only` action-prior shortcut、effect-only intervention probe 和有限时域
   candidate ranking inversion；低 inverse risk 不再被解释成 task-effect correctness。
7. `action_information_gain = R_prior - R_successor` 只表示平方损失下后继信息带来的
   风险下降，不等同于 Shannon mutual information；只有在平衡干预和真实 task contrast
   同时成立时，才可作为 action-identifiability 的辅助证据。

## 统一实验协议

在同一 frozen backbone、同一 root-disjoint split、同一 action bank、同一 horizon 下，
对每个方法保存：

```text
predictive:
  baseline_error, method_error, horizon, closure_error, uncertainty
action:
  baseline_ranking_regret, method_ranking_regret,
  signed_effect_accuracy, inverse_risk, mode_accuracy, support_coverage
decision:
  baseline_planning_risk, method_planning_risk,
  real_predicted_successor_error, pose_success, contact_retention
  action_ranking_inversion_rate, minimum_decision_margin
provenance:
  architecture, seed, environment, training_stage, root_ids, action_bank_hash
```

主要估计量是

\[
\Delta R\mid \mathrm{MSE},\mathrm{spectrum},\mathrm{stage},\mathrm{action\ risk},
\]

但只有 action ranking 与独立 planning consequence 同时改善，才允许解释为
decision-transfer evidence。方法选择只发生在 train/calibration；outcome/test 只用于锁定
后的评估。

## GPU-first 与 artifact 规则

- learned-model 入口先执行真实 CUDA tensor allocation；CUDA 失败时状态为
  `environment_blocked`，不静默切换 CPU。
- 每个阶段写原子状态、输入 fingerprint、schema、split 和 checksum；中断目录保留，
  不能当作科学负结果。
- 当前 host-permission CUDA 长实验继续复用，不重复启动同一个 writer。
- smoke、synthetic、privileged upper bound、offline decoder 和 RGB successor audit
  分开标注，均不能替代独立视觉闭环规划。

## 硬否决规则

代码接口
`le-wm-main/experiments/empirical_spectral_audit/decision_sufficiency.py`
固定如下规则：

- prediction improvement without action-ranking improvement → `boundary_or_negative`；
- mode-accuracy improvement without pose closure → 不得标记 pose success；
- 缺少独立 planning consequence → `incomplete_fail_closed`；
- panel 中存在任何未通过记录 → 不得用平均值覆盖失败记录。
- ATM diagonal 或 state-only inverse risk 低 → 仍需 effect-only intervention 与 task ranking；
  inverse consistency 不得单独授权规划。

ATM-style 四格矩阵的离线入口为
`experiments/empirical_spectral_audit/run_atm_cross_domain_audit.py`。它可以消费现有
`latents.npz` 的 source/true/predicted dynamics，但当导出不含 root/candidate/task-cost
字段时会显式输出 `unavailable_missing_candidate_root_identifiers`，不能把模型级 inverse
诊断误写成 candidate-wise ranking 证据。

PDTC 的代码入口为
`experiments/decision_benchmark/task_contrast.py`，包含：

- root 内 candidate pair 的 `contrast_error` 和 ranking inversion；
- calibration roots 上的最大 pairwise error 分位数校准；
- 依据 `predicted margin > calibrated bound` 的 `permit`；
- 支持不足或界缺失时的 `abstain / probe`。

该接口只提供 fixed-bank、exchangeable-root 条件下的边际候选集合校准；不宣称任意
在线 MPC 分布偏移下的条件覆盖。

当前 PointMaze shared bank 的真实离线运行已写入其
`task_contrast_audit.json`：8 个 outcome roots、960 个 candidate pairs，contrast
inversion rate 为 `0.4078`，独立 8 个 calibration roots 得到 root-level bound
`0.4283`。由于该 bank 没有 RGB realized latent 和独立 support calibration，
`certificate_preview` 强制为 `abstain_support_gate_unavailable`，释放数为 0；这是真实
协议边界，不是将 offline candidate 结果包装成在线证书。

当前负结果的候选原因因此被分解为可检验项：action support 不足、within-mode effect 不可
辨识、历史表示与 horizon 错配、RGB successor 与 privileged state 不一致，以及小样本
adapter 过拟合。下一轮修复必须改变对应协议或数据支持，而不是只调 selector 权重。

## 当前证据状态

- horizon-conditioned memory：`h=1` 固定 Fourier；`h=4` 冻结 Fourier+phase；selector
  只做稳定性诊断。
- cross-environment matched transfer：点估计为正，但预注册 sign-flip gate 尚未通过。
- PushT：mode/contact 与 signed action ranking 有局部改善，严格姿态闭合仍未建立。
- residual predictor repair：小样本 CUDA smoke 中 held-out multi-step error 恶化，不能部署。
- frozen RGB online planning、universal world model、universal commutator certificate：
  仍未建立。

### RGB successor 到 task decision 的新增门

RGB successor verification 不能直接进入规划许可。固定流程为：

```text
predicted RGB successor
    -> train-only task readout
    -> independent calibration-root contrast envelope
    -> train-only state/action support gate
    -> permit / probe / abstain
    -> held-out physical task consequence
```

该流程由
`experiments/decision_benchmark/audit_rgb_task_contrast.py` 实现。它只允许
在候选 bank、calibration roots、readout 和 support gate 均预先冻结时计算
selective risk--coverage--regret；不能将 RGB successor RMSE 的改善替换成
candidate ranking 或闭环规划收益。

当前 2026-10-14 artifact 显示：640/640 RGB successor rows 可对齐，8/8
outcome roots 通过 train-only support gate，但 realized-RGB task contrast
inversion 为 `0.4258`，physical task contrast inversion 为 `0.4235`，独立
calibration 得到 contrast bound `6.4728`，因此 `released_root_count=0/8`。
该结果明确定位了负结果的传播断点：预测改善没有传递为 task ranking 改善。
正式策略必须在此处 `probe/abstain`，不能通过缩小 bound 或读取 outcome
结果来制造 permit。

## 下一阶段门禁顺序

1. 完成长实验的 planning-alignment artifact，并检查是否有独立 geometry 增量；
2. 在 fresh root-disjoint 数据上运行 base vs calibrated residual，多 seed、GPU-only；
3. 将每个结果转成三层 schema，先排除“预测改善但排序不改善”的伪正向；
4. 只在 action ranking 通过后运行独立闭环 planning；
5. 若严格 matched benchmark 或跨环境 sign-flip 仍失败，保留为不可识别边界，不再扩大
   alignment 或 universal claim。

## ATM → HTSC 的阶段门

| Gate | 可运行对象 | 通过条件 | 失败处理 |
|---|---|---|---|
| G0 | root/action provenance | root-disjoint、平衡干预、真实/预测 successor 对齐 | 阻断所有下游 claim |
| G1 | ATM insufficiency | 合成反例成立；learned panel 中报告匹配失败与覆盖范围 | 只保留理论边界 |
| G2 | task contrast | 独立 roots 上 contrast error 与 ranking inversion 可估计 | 不升级 geometry 增量 |
| G3 | PDTC calibration | calibration roots 上 root-level bound 非空且 coverage 可审计 | 只输出 probe/abstain |
| G4 | RGB-only MPC | 相同放行率下 false-release 或 regret 降低 | Goal 4 保持 blocked |
| G5 | GDD-WM | realized/predicted preference 均可表示且 teacher-free planning 改善 | 不进入 joint training |

候选 pair 不是独立实验重复；统计推断单位优先使用 root、model seed 和 environment，
不能通过增加同一 root 内的 candidate pair 制造有效样本量。
