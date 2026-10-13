# 复现材料说明

## 目录

- `code/`：理论扫描、I-JEPA 测量、JEPA-WM/DINO-WM temporal evaluator。
- `results/`：与论文数字对应的 JSON/CSV 结果。
- `provenance/`：split、checkpoint、审计和修订信息。
- `data/`：预留给合作者放原始下载数据；本包不复制数十 GB 原始 archive。

新增的 `results/alignment_interventions_20261003/` 和
`code/alignment_interventions/` 记录 alignment intervention 的完整可读
结果与源代码快照。该目录专门检验 alignment-sensitive diagnostic 是否能
推出预测质量改善；它不承诺端到端 LeWorldModel 重训收益、真实 multi-step
rollout、online control 或 planning success。

其中 `results/pixel_world_alignment_20261003/` 和
`code/alignment_interventions/run_pixel_world_model_synthetic.py` 是一个
额外的 learned pixel-to-latent 合成基准：它联合训练小型卷积编码器和
action-conditioned predictor，并重复检查 alignment intervention 是否会
改善 one-step、组合和 rollout。该结果只支持 synthetic pixel-world 的
边界结论，不是 nonlinear LeWorldModel、PushT、planning、controllability
或 GPU 结果。

新增的 `results/predictor_equivariance_synthetic_20261003/` 和
`code/equivariance_interventions/` 是一个受控的 CPU-only 合成向量世界
消融，分别训练和测量 latent equivariance、predictor equivariance 以及
两者同时启用的情况。它使用三组 seed、相同数据划分和验证集 one-step
MSE 选 checkpoint，并报告 rollout、两动作组合、等变误差和 stable rank。
该实验用于区分机制，不是 LeWorldModel、PushT、planning、causal
controllability 或 GPU 性能结果；当前复现实验环境中的 PyTorch 无法初始化
CUDA/NVML（`torch.cuda.is_available()` 为 `False`，设备数为 0），因此结果明确
不包含 CUDA 训练或 GPU 效率结论。

新增的 `results/equivariance_functional_synthetic_20261003/` 是与上述机制
消融配套的功能性复核：在相同合成向量世界和验证集 checkpoint 规则下，另行
评估 action-magnitude OOD、初始状态偏移、h=8 rollout，以及受控的
candidate-set latent-planning proxy。48 条逐 seed 记录和 Table 24 的聚合结果
显示，几何干预对 OOD 与 planning proxy 的影响是非单调的，并伴随 one-step
与长程预测的 trade-off；这些 planning 列不是 online task success，也不构成
几何因素的纯因果识别。该结果同样是 CPU-only、synthetic-only；当前复现实验
环境中的 PyTorch 无法初始化 CUDA/NVML，因此不含 CUDA 训练或 GPU 效率结论。

新增的 `results/alignment_functional_synthetic_20261003/` 和
`code/alignment_interventions/run_alignment_functional_synthetic.py` 是对
alignment intervention 的功能性复核：在相同的 learned synthetic benchmark
上评估随机 action-sequence 的 $h=8$ rollout、ordered composition，以及以独立
真实目标轨迹为目标的 candidate-set planning proxy。该实验包含 33 条逐 seed
记录、11 个 mode/weight 汇总，表内报告均值，逐 seed 标准差保存在 JSON 中。
它显示 global alignment 降低 pooled commutator 时可能提高 conditional mismatch
并损害 composition、长程 rollout 和 planning regret；因此不支持“commutator 越低
越好”。该结果仍是 CPU-only、synthetic-only；planning 列不是 online task success
或 causal control；当前复现实验环境中的 PyTorch 无法初始化 CUDA/NVML，因此不含
CUDA 或 GPU 效率结论。

新增的 `results/sigreg_alignment_phase_20261003/` 与
`code/alignment_interventions/run_sigreg_alignment_phase_synthetic.py` 是
SIGReg × alignment 的交叉机制检查。它完整交叉三个 SIGReg 权重、无/全局/条件
alignment 以及三个 alignment 权重，共 63 条逐 seed 记录；stable rank 用于识别
covariance flattening 的替代解释。该结果只支持“commutator 下降不等于功能改善”
的 synthetic 边界，不是 nonlinear LeWorldModel、PushT、online planning、causal
control 或 GPU 结果。

新增的 `results/action_sequence_composition_20261003/` 与
`code/alignment_interventions/run_action_sequence_composition_synthetic.py` 用同一
learned synthetic action-conditioned benchmark 枚举全部二元 action sequence，长度
$h=2,4,8$ 的序列数分别为 4、16、256，并逐记录对齐 one-step、ID/OOD 随机序列
rollout 和 exact operator-product error。33 条逐 seed 记录通过 exact-repeat、有限性
和跨 artifact 数值一致性检查；h=8 composition--rollout 相关性是描述性诊断。它是
CPU-only、synthetic-only，不支持 nonlinear LeWM、PushT、online planning、causal
control 或 GPU 效率结论。

## 建议工作顺序

1. 先阅读根目录的 `README_中文.md` 和本目录的 `data_dictionary_中文.md`。
2. 先用 `results/` 中已有结果检查论文数字，不需要下载大模型。
3. 若要重跑理论结果，只需 Python 环境和 `code/` 中的 theory scripts。
4. 若要重跑 learned-model temporal 结果，再按 `provenance/source_manifest_zh.md` 准备 archive、checkpoint 和官方外部仓库。

若只复现新增的 alignment intervention，可使用包内
`code/alignment_interventions/` 的五个源文件，并对照
`results/alignment_interventions_20261003/README.md` 和
`results/pixel_world_alignment_20261003/README.md` 中的协议和结果。
功能性 alignment 复核可直接运行包内
`code/alignment_interventions/run_alignment_functional_synthetic.py`，输出到
`results/alignment_functional_synthetic_20261003/`。

新增的 conditioned-readout 复核位于
`results/conditioned_controllability_pilot_20261003/`。它使用三个 seed、每个
20,000 条冻结 PushT latent 行、八个 state cells、within-cell half/half
held-out split 和 action-row permutation null。三个 seed 的 held-out cell-level
$R^2$ 均为正，而 permutation null 均为负；该结果是 frozen-latent offline
detectability 诊断，不是 causal controllability、online planning、goal
reachability 或 multi-step rollout 结果。

新增的 `results/external_rank_target_specificity_20261004/` 和
`code/external_frozen_dynamics_audit/run_external_rank_target_specificity_audit.py`
是 frozen PushT 的 rank-by-target specificity audit。它在 21 个外部 episode、
留一 episode 外推、$k=1,2,4,8,16,32$ 和 $h=1,4,8$ 下，分别测试
dynamics-band latent increment、residual-band latent increment、完整 latent
increment 和 physical displacement。dynamics coordinates 只对 dynamics-band
target 给出稳定正向增量，residual coordinates 只对 residual-band target 给出
正向增量；physical target 没有超出 observed-action summary 的正向增量。该
实验是冻结表示的 target-specific predictive diagnostic，不是 physical causal
effect、online planning、task success 或 retraining gain。

本轮新增的 compositional predictor repair 与 candidate-ranking audit 位于
`results/repaired_predictor_mpc_compositional_96x24_20261005/`、
`results/repaired_predictor_candidate_rank_96x24_20261005/` 和
`results/repaired_predictor_candidate_rank_512_20261005/`，对应代码位于
`code/contact_predictor_repair/`。修复训练覆盖恒定、反向、分段扰动和随机
action sequence，并在同一 held-out simulator state 上重放相同候选集合。
该审计显示 predictor ranking 明显改善，但 oracle candidate 仍不能覆盖成功
的 agent--block--orientation 联合动作，因此 live MPC 仍为零 task success；这
是“预测器接口修复”与“任务规划成功”之间的明确证据边界。

本轮新增的 teacher-forced contact-predictor 深度诊断位于
`results/contact_predictor_deep_20261005_seed8421/`、
`results/contact_predictor_deep_20261005_seed9271/` 和两个 summary 目录。
它额外保存 `+a/-a/zero` 三臂、observed-latent upper bound、horizon profile、
calibration-size sweep，以及 epoch/anchor/training-case 消融。两个 seed 的
$h=4$ 三臂 point gain 均为正，但 observed-latent ceiling 仍明显较低；因此
该结果只支持“针对单一发布 checkpoint 的 predictor/readout mismatch 部分修复”，
不支持原始 frozen checkpoint 已具有 learned physical causality、准确
counterfactual、online planning 或 task success。

本轮进一步加入稳定接触 bootstrap 与 staged candidate coverage 审计，位于
`results/staged_candidate_coverage_20261005/` 和
`results/staged_candidate_coverage_h32_20261005/`，对应代码为
`code/contact_predictor_repair/run_pusht_staged_candidate_coverage.py`。首次
检测到 block 位移不再直接视为稳定接触；新协议要求近接触、低速和刹车后的
低 block 位移，并将 approach--brake--push 以及角度扇区加入候选。16 个
held-out 状态的 h=8 审计中，staged 候选的最佳 block progress 为 `60.231`
（legacy guided 为 `49.217`），最佳 task cost 为 `264.421`（legacy 为
`271.865`），但 full-task success 仍为 0；h=32 时 block-pose success 仍为
0，包括 simulator oracle。新增 live 结果位于
`results/repaired_predictor_mpc_stable_staged_20261005/` 和 visual 对照目录；
privileged repaired run 与 visual run 都为 0 task success，因此只支持局部
endpoint/interface 改善，不支持 causal physical model、online planning gain
或 task success。

## 证据边界

结果级 JSON 是论文数字的来源记录。它们支持有限维识别定理、冻结表示测量和明确 held-out temporal protocol；不自动等同于 task success、online control 或任意 learned model 的普遍定理。
