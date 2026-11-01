# P2 resume：真实 native 动作见证 → prospective queue 接缝

个人 bounded 交付：新 adapter、RAM/真实 saved 布局变异测试、实际 saved prepare/verify/consume 和本说明。只完成个人工程组件，不代表研究 goal 或 P2 科学通过。不派生 agent、不改旧源码/results/共享总账、不运行 GPU/physics、不训练或拟合模型/Q/readout/normalizer。

使用 experiment-code 技能组织固定配置、日志和回归；用户的 saved-only 限制优先，不进行训练、按结果优化或额外神经实验。祖先及项目树未发现 AGENTS.md。

## 为什么补这个组件

已有 `chat_p2_20261027_lifecycle` 实现 generic saved graph prepare/verify/run/consume，`chat_p2_20261027_producer_bundle` 能调用 native full exporter，但指向的 frontier CUDA v2 prepared01 没有 report，旧 saved bundle 如实 blocked。已有 generic graph 要求完整 base/edge/cap 资源，不能靠把未知 FLOPs/backward/cuda 时间补零来制造可执行图。已有 queue 能执行科学门，不是 planner。

本次不再造搜索框架，也不改变旧 loader。具体补齐 **paired-v2 已保存的全部候选动作图与 standalone SRA-v2 同库对照到现有 prospective queue 的身份接缝**。选择 paired v2 是因为其原先已有真实 all-candidate 一/两段 witness，不是因其胜负或效果；保留全部四个 known calibration roots、每 arm 40 候选，不读取 physical outcomes。

真实输入（相对 inner repo）：

| producer | report.json SHA256 |
| --- | --- |
| `results/p2_composition_paired_20261027_v2_prepared01` | `0c2a92842fa993cdc3dc566bab86bf124db15a6a0d5b054cbfef5e2d75f86148` |
| `results/p2_no_composition_sra_20261027_v2_prepared01` | `522afa5a68b23a6ff829c5c6d820d858953dccc0f778548e747ba594ff73937a` |

输入 SHA 绑定本机实际保存字节；不是本 adapter 独立认证执行真实性。先调用两 producer 原 `load_prepared`/`selected_actions`，完整检查旧 source/input seals、原 Q/goal/history、bank 与全部候选、保存 solve、稳定 winner/replay。保存 NumPy 重算费用原 exporter 明确单列，不是新的模型推理、训练或拟合 Q。

## 实际接口与固定分母

源码 `experiments/spectral_reachability/agent_p2_resume_20261027_witness_queue.py` 提供：

```text
prepare({composition:{directory,report_sha256}, no_composition_sra:{directory,report_sha256}})
verify(prepared, canonical_prepare_sha256)
consume(prepared, canonical_prepare_sha256, confirmatory=False)
```

CLI `prepare/verify/run-saved/consume`：run-saved 与 consume 仅重验并导出全部 saved actions；没有神经/物理 dispatch，不计算新 winner。prepare 写无换行 canonical JSON，因此文件 SHA 与 canonical prepare SHA 一致。所有输出必须位于 inner results、以 `agent_p2_resume_20261027_` 开头、文件不存在；exclusive+fsync，不覆盖旧结果。

完整四根顺序来自 sealed protocol，报告按 root SHA 显式 join，拒绝重复/缺失，不靠报告数组位置配对。paired 两 arm 顺序保持原 `joint_h2_no_composition`、`conditional_suffix_composition`：每根各 40 槽，共 320 候选、480 原生边。所有候选均保留，重复动作也不去重。失败/不足 40 时整次准备拒绝，native 原账本不退款、不筛掉失败候选再导出赢家。

standalone SRA 的全部 40 raw candidates 必须与 paired joint 的相应 40 项精确一致；不把 SRA winner 的单条 witness 冒充所有 SRA 候选轨迹。拓扑中的一段 joint witness 来自 paired producer 的真实记录，standalone SRA 是独立 full exporter + raw bank 一致性对照。不同运行的 source/protocol/bank provenance 不合并成虚构的共同 SHA。

每条 graph record 绑定 root/goal RGB、goal vector、metric/projection、完整 between-state history（三 latent/two action tokens）、normalizer/action domain、原始 checkpoint/encoder/source/protocol/bank provenance、native receipt digest、原 report SHA/根/arm/候选索引。每条边保存原 actions 及精确 little-endian float32 bytes/SHA；若需转换数值才能对应则拒绝，不做 tanh、clip、归一化、padding 或 suffix 重拟合。相邻边必须完整 context 精确闭合，suffix compilation 的 input-context SHA 也须匹配。不创建 kNN/近似桥、任意 latent waypoint、逆边或新路径。

图节点明确是 `saved_predicted` 的 **native predictor full context**，不是物理7维状态，也不把 latent 贴成 observed pose。原 context 字节语义及完整 state/history 一起保留。程序类型 progress 不表示实际物理进展。输入全部边先经 native receipt/compose 核对；这些保存记录不构成独立神经 replay。

资源：复用原 candidate census，保留 dispatch predict_calls/predict_samples/action_encoder_calls、失败尝试；另复用 entry metadata，将原 actual predict/AE 的 API calls、batch samples、context tokens、failed entries 分开列出，并保存 paired 原 resource_receipt。原不同运行成本不伪称同一次共享成本。本 adapter 的 saved 验证 SVD/solve 额外列在 `additional_saved_verification_work`；没有新 neural/physics。hash/JSON/host I/O 成本不冒充旧实验预算，未测项不填零。

## 工程 ready ≠ strict graph ready ≠ 科学 pass

`program_graph` 是原 native witness 的封存拓扑投影，不冒称旧 `moving_p2_saved_program_graph` 的完整 resource graph；严格 graph 所需 base、逐边 increment、未来 execution、总 cap 的各轴归属没有真实合同，故 `strict_resource_graph_ready=false`。未给出 graph search 目标/成本策略，也不会按模型或真实 outcome 选择新 target。此缺口不能由加 synthetic 预算解决。

adapter 将原 root/history/goal-vector 身份投影进 **已有 queue.build_draft / build_queue**，保留四 slot。原 native schema 没有 prospective blind packet/task authority，`blind_sha256/task_id=null`；不把 root RGB 文件 hash 换名成 blind SHA，也不自签新 task。P1/freshroot/MAIN authority、prospective seal 时间、完整资产/规则与各阶段成本保持缺失；queue 必须 blocked。`engineering_ready=true` 仅指此 saved witness 接缝可复算，`scientific_pass=false`、`execution_authorized=false`、P2=false。

确认性 consume/CLI `--confirmatory` 在任何读取/导出前直接 fail closed，没有 caller boolean、queue 自述 passed 或复用 known roots 的解锁口。未来真正确认性 executor 仍需 MAIN 提供：独立 P1（全部机制和任务收益门）、fresh source/全历史 exclusions、真实 blind/task identity、原目标/成功/控制器规则、完整 full-resource base/edge/cap 分配及 prospective spec 外部批准。现有接口不替 MAIN 签署 authority。

MAIN 最新 `docs/D2P_MOVING_CAPABILITY_PROBE_MAIN_20261027.md` 作为 source binding：40真实探针/32矩阵只是两 known roots 的机制定位，不能当 P1 admission。本任务不重复其方向预测、不改或重跑旧 probe。

## 实际 saved 命令（无 GPU / physics）

从 `/home/gbk/code/LeWorldModel/le-wm-main`：

```bash
OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m experiments.spectral_reachability.agent_p2_resume_20261027_witness_queue prepare --composition-output results/p2_composition_paired_20261027_v2_prepared01 --composition-sha 0c2a92842fa993cdc3dc566bab86bf124db15a6a0d5b054cbfef5e2d75f86148 --no-composition-sra-output results/p2_no_composition_sra_20261027_v2_prepared01 --no-composition-sra-sha 522afa5a68b23a6ff829c5c6d820d858953dccc0f778548e747ba594ff73937a --output results/agent_p2_resume_20261027_prepared_v1.json
OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m experiments.spectral_reachability.agent_p2_resume_20261027_witness_queue verify --prepared results/agent_p2_resume_20261027_prepared_v1.json --prepare-sha PREPARE_SHA
OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m experiments.spectral_reachability.agent_p2_resume_20261027_witness_queue run-saved --prepared results/agent_p2_resume_20261027_prepared_v1.json --prepare-sha PREPARE_SHA --output results/agent_p2_resume_20261027_saved_v1.json
OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m experiments.spectral_reachability.agent_p2_resume_20261027_witness_queue consume --prepared results/agent_p2_resume_20261027_prepared_v1.json --prepare-sha PREPARE_SHA --output results/agent_p2_resume_20261027_consumed_v1.json
```

PREPARE_SHA 取 prepare 回传值。本机已存在的结果禁止重跑覆盖；源码/doc 变化后旧 prepare 应按 hash 拒绝，不能重封旧 seal 绕过。

## 测试

```bash
OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m pytest -q -p no:cacheprovider tests/agent_p2_resume_20261027_test_witness_queue.py
```

RAM synthetic 完整 4×2×40 图控制流，外加从真实 sealed paired report 截取的 192维/3帧/2动作历史布局及其边界、goal、raw action、checkpoint、suffix-context 变异。没有模型/模拟器实例，没有 CPU neural 或物理 forward。测试与 saved 重算只是工程一致性，不能证明真实因果、独立性、资源公平或 P2 收益。
