# P4/P5 前置闭环：已有资产接点与最小缺项

日期：2026-10-29。独占新增文档；不修改 canonical、旧源码、封存结果或原 missing 槽。

## 1. 证据与执行边界

本清单只读已有 resolver/closure/audit、相关接口和指定实际产物元数据；没有重新扫描 fresh 历史，没有运行 resolver、NN、physics、编码、优化或 fullstrict。下列实际 receipt 的 SHA/元数据是已有验收权威的引用，不是本文独立重做 native 验收。原封存文档中引出的历史证据另行标明。

P1 未通过，不据此签 P2/P3/P4/P5 科学成立。known2、四条件、六臂、164 分支和80正负对均不增加独立根数。MAIN 已完成 arbitrary 两目标交付；Pauli 正在推进六个 tracking job，本文不承接、重复或预认证它们。

下文 `I=/home/gbk/code/LeWorldModel/le-wm-main`，`O=/home/gbk/code/LeWorldModel`。所有 `results/`、`experiments/`、`innerdocs/` 路径相对 I；`O/docs/` 指外层文档。`null` 表示本次未获真实输入，绝不是建议造出的文件名。

## 2. 当前 fresh 权威与旧 closure 的时间边界

| 实际保存输入 | SHA256 | 本次可读出的结论 |
| --- | --- | --- |
| `results/agent_p5_fresh_confirmation_preconditions_20261029_v1_MAIN01.json` | `d453e4cf70518839844916566fa274959c515d05310ae090fb5699e589cf3133` | `blocked_no_certified_fresh_cohort`；certified_fresh_count=0；P5_confirmation_denominator=null |
| `results/agent_p4_fresh_certificate_audit_20261028_v2_report_MAIN01.json` | `15e3ece87e36eba469a0cc2188efbb7edaa94fa00f1562f9e6adc2052e4e4ae0` | `/freshness/certified_fresh_roots=[]`；history_certificate=null、coverage_complete=false、original_execution_bounds_certified=0、unresolved_count=975 |
| `results/agent_p4_fresh_panel_manifest_20261029_v1_manifest_LOCAL01.json` | `29c919fe1365e8f7422951026d6e8f8c90b293061981086f175096ab647c848d` | `blocked_planning_artifact`；metadata_panel_certified=false；四个关键 request 附件为 null |
| `results/p5_closure_readiness_20261027_v1.json` | `07cd48053f31457ec390f99b03a728ece2c32376cd770c25886161b4b8381980` | 20261027 快照，`saved_evidence_valid_P5_blocked`；launch_command=null，execution_authorized=false |

audit 的 certificate_count=2 是构造资产计数，不能替换 certified fresh 数。975 是该 audit 的 unresolved 项数，不是975个新根，也不证明跨所有机器无限追溯才能解决。

旧 closure 的九个 unbound 科学槽仍须按其合同闭合，但不能据该快照断言后来 GRASP、反馈 GD、JEPA164/H2、latest492 或 arbitrary 交付代码/产物不存在。工程完成与科学槽闭合是两层证据。

## 3. 可立即执行的保存资产接点

本节“可立即”指已有真实输入足以继续 saved 接线或协议核对；不是本文授权生产，也不自动获得 fresh 资格。新增输出由 MAIN 选择独占路径，本文不写结果。

### A. 强基线 → 同任务资源向量，而非按 predict 数宣布等算力

现有输入：

- `results/agent_p2_common_resource_grasp_cuda_20261029_v1_consume_MAIN01.json`，SHA `d3ee465f2fa6d31345f369f9514165c4817c51a29c3d00ffdd2d681e5829cc1f`，status=`all6_common_cap_records_verified_not_admitted`。
- 对应 `results/agent_p2_common_resource_grasp_cuda_20261029_v1_cuda_MAIN01/report.json`；封存资源文档给出的 report SHA 为 `d781b493ecd624db846e3225ed0177bd3fe250d6ba9213d925b2a57956dd2242`。
- `results/agent_p3_feedback_gradient_mpc20_20261028_v1_consume_MAIN01.json`，SHA `09b3a3a5479a21e16959d94989f4da5fd36c97597e89d930d2591fbe8a22620d`，complete=true，`real_feedback_gradient_verified_NOT_admitted`。

直接消费者是 `experiments/spectral_reachability/agent_downstream_resource_contract_20261029_v1.py`：`p2_saved(report,receipt)`、`p3_saved(report,receipt)`、`p3_ledger(receipt)`，以及 `validate_shared_panel(rows,expected_slots,compared_axes,caps,wall_scope=None,wall_cap_seconds=None)`。既有完整 saved 输入 JSON 构造在同 stem innerdoc，不需要重新 fullstrict。新 panel 必须精确绑定 model/task/goal/normalizer/history/action_domain 六身份，先保留不同 horizon、feedback 和动作支持，再决定哪些轴可比较。

应输出 encode/projector、predict samples/frames、J/VJP/backward、solve/SVD、物理反馈和 wall 边界各轴；缺失的 scalar backward/AD primal 为 null/unknown，不补0，不累加嵌套 inclusive wall。未知算子 FLOPs 保持 unknown。P2 共同 cap 是1902 API/samples、450秒 searchwall，不代表各臂实际花费相同：composition 自然1062；GRASP variant 532+100更新/632 scalar backward，不冒称官方750配置或 equalFLOPs。

最小缺项：若要形成 P4 强对照结论，仍需同根同目标、同反馈/动作支持的真实 native 成功/成本配对和预注册统计；当前文件不能补出独立 fresh 根。反馈 GD 的四 episode 含重复 condition，仅两 known roots。

### B. arbitrary 目标交付 → tracking 输出的身份连接

MAIN 已确认实际输入绝对路径 `/home/gbk/code/LeWorldModel/le-wm-main/results/agent_p2_arbitrary_latent_delivery_20261029_v1_consume_MAIN01.json`，receipt SHA `1e47155d480c23a0bda9cae50e633f0fe7ef56c992d3f489a31e255e43286ffc`，schema=`agent_p2_arbitrary_latent_delivery_20261029_v1.receipt`，status=`all2_arbitrary_FIRST_numerically_verified_NOT_tracked`。这是已完成两目标交付的实际权威，不是 six tracking 已完成的凭据。

可接动作：将完整 all2 target、原 task/goal/history/Q/normalizer/source 身份作为 Pauli 六新 job 的只读交付输入，逐槽 exact join，不将目标向量改成程序 endpoint。保留64步完整数值轨迹与已付成本，tracking 新成本另栏。

最小缺项：六 job 的实际 report path/SHA、strict path/SHA 和 producer 最终 schema 当前未交付，均为 null；待冻结后读其真实 ABI，不能预编输出槽、声称已跟踪、补未来 SHA 或抢跑六 job。

### C. 同物理面板 LeWM / JEPA → 跨模型响应分解

现有真实 receipt：

| 输入 | SHA256 |
| --- | --- |
| `results/agent_p5_raw164_jepa_cuda_20261029_v1_consume_MAIN01.json` | `4f6bbed041f077818627e68157b0cfab99ddba2e38100827db10f46c4aa1b759` |
| `results/agent_p5_raw164_jepa_h2_cuda_20261029_v2_consume_MAIN01.json` | `783b65f67899791b5697bfe0819ccfb3231d57212b9e3e85aefc02419459c0ea` |
| `results/agent_p5_latest_slot_factorial_cuda_20261029_v1_consume_MAIN01.json` | `c636f8d9c6624317d4da4f3ddb127b4f20710f9a5c02c212872d0d70c9cab642` |

对应 H1 report SHA `e7ac4ef9eb7491f619de25b04c55f5e56560e2833a840dd5305f3db6207b3a5d`；H2 `e5a561215bfe7b6f1c1c224e7661211e3f237fb16458a6a409f5a1da0fc08579`；latest492 `a9479f81d87b1b70ebd1f64db0289d9d2c6a5ae57932cd021f85cc5ecfd47316`。report 路径分别为同 stem `_cuda_MAIN01/report.json`。H2 prepare SHA `09070780c3ce782a71ae01492e0c4b801de976e9a36cdf9ee1bf1a53be98e924`。

可接动作：从各自已验收 request/pair 标识，以 root/condition/h/coord/sign、原 raw bits、物理/RGB来源精确连全164/80，比较共同物理标签与各模型内部 Q/response；不能因为都80或都24便认集合相同。latest 的 AA/AP/PA/PP 交叉只在已有15个交叉资格上计算，missing 原样保留，不扩交集。

既有 `agent_p5_raw164_jepa_input_bridge_20261029_v1.py` 已提供实际输入桥；旧 handoff 的“桥未齐”不是当前状态。host_inputs 白名单是 history RGB/proprio/packets，labels_after_seal 隔离。JEPA 接入真实 agent `[x,y,vx,vy]` 而 LeWM 是 RGB-only，必须披露额外信息；JEPA 256×384 与 LeWM192 不跨维度直接比较 MSE 大小。

H1 实际5/10/15→20，H2实际0/5/10→pred15→pred20，不同历史/滚动时距分栏。旧 whole/latest 某些输入实际相同，不能只据名字宣称新的 history 干预。旧固定 full3 缺负时间帧也不能静默用短历史补齐。short-start full-roll 另有协议，656calls/1476frames 并非本文授权的新预算。

最小缺项：独立架构的正式 P0/task可达性、同一冻结 core 的新根机制/任务增益和匹配权限/成本协议，不由保存响应分析替代；当前对接无需新增编码/物理。

### D. 跨环境 → 精确任务/模型接口表，而非重跑旧 evaluator

现有 `results/p5_architecture_provenance_20261027_v1.json`，SHA `6c169f0685bb227913ae6deb821d74093e15eed29f063584e919bae892cde7b0`，status=`local_architecture_contract_verified_with_explicit_unknowns`。可直接读取 checkpoints/config_contracts/source_identity/normalization/observation_history/learning_cost/licenses/unknowns，连接：

- PointMaze：`results/enriched_jepa_pointmaze_observed_cuda_fixed_20261011/metrics.json`，SHA `4cf4a352b2e41a53dbbf65cbeecf9c085f3614313aa56bd554f4c7b309fce0af`。
- Wall：`results/theory_operator_alignment_iclr_oral_20260829/wall_temporal_gate/jepa_wm_wall_observed_metrics.json`，SHA `5c2af67a2e8e690af6f1bde012d4bc32d19640bf9de7c8247328affd8a3b5314`。

可完成只读 mapping：每环境 checkpoint/source/license、RGB/proprio 权限、history时序、动作维度/尺度/边界、native success/cost、学习/拟合成本及历史根来源。旧 temporal/operator-fit 指标不是当前冻结控制策略的任务改善。不得为做此表启动 `run_jepa_wm_pointmaze_temporal_eval.py` 或 `run_jepa_wm_wall_temporal_eval.py`，它们涉及拟合且可能 CPU fallback。

最小缺项：当前 P4 PointMaze/Reacher/PushT 的新鲜 paired native episode/task权威及全 multiround RGB反馈结果，actual path=null。三个 task checkpoint 属同一 JEPA 架构，不是三个独立架构；同模型多任务还须固定一个 checkpoint 并声明 goal/horizon/action restriction matrix，不能混用三个 task checkpoint 冒充该项。

## 4. fresh0 的最小输入缺口：只向原保管者取附件

以下均是既有工具合同，不新增科学门，也不要求全盘重扫。精确来源见 `agent_p4_identity_recovery_handoff_20261029_v1.md` 与 `agent_freshroot_authority_20261028_v1.py`。

### 4.1 历史执行封套或 exhaustive reset ledger

真实 response path/SHA 当前均为 null。既有接受 schema：`agent_freshroot_authority_20261028_v1.response`，字段为 `request_sha256,sources,historical_scope`；sources 行为 `{source_id,report_sha256,runs:[{assertion,origin_citation}],unresolved_reason}`。

assertion 必含 `call_id,report_sha256,route,attempts,attempts_complete,restarts_complete,reset_evidence,loaded`。两条路线择一：

- `exhaustive_ledger`：citation `{path,sha256,pointer}` 指向 `{call_id,attempts:[{index,reset_kind,disposition,identity_sha256}]}`，覆盖失败、未选及重启；仅失败允许 identity=null。
- `loaded_closure_and_exhaustive_envelope`：citation 指向 `{call_id,upper_bound,derivation}`，并附真实 loaded `{role,path,sha256}`，含 producer/factory1..4/environment/wrappers，相关 P0 dependency 与 shell launcher。

historical_scope 为 null 或 `{machines,worktrees,time_interval,unsaved_failures,origin_citation}`。必须与请求的原 source/call/hash 对齐，不伪造 episode_id、尝试计数、原始 loaded manifest。既有身份 handoff 显示原选中报告本身未保存 state7/snapshot/episode/attempt 映射，重复解同报告不能造回这些证据；原保管者须提供原附件或明确不可恢复。

### 4.2 四个实际缺失 panel 附件

在 `...fresh_panel_manifest...manifest_LOCAL01.json` 中四个精确缺口：

| JSON pointer | actual path/SHA | 所需既有 schema / 最小内容 |
| --- | --- | --- |
| `/request/history_request` | null/null | 原 `moving_root_exclusion_manifest_20261027_v1.request`，不能另造新请求冒充历史 |
| `/request/exposure_inventory` | null/null | `moving_root_exclusion_manifest_20261027_v1.identity_metadata`；schema/cutoff_utc/project_root/includes/complete/unknowns/records/reserved_reset_spans |
| `/request/actual_roots` | null/null | `agent_p4_fresh_panel_manifest_20261029_v1.actual_roots`；slots 四个 record 或 null |
| `/request/panel_approval` | null/null | `agent_p4_fresh_panel_manifest_20261029_v1.MAIN_panel_approval`；authority=MAIN、approved=true、panel_content_sha256/cutoff_utc/outcome_blind_selection |

descriptor 是真实绝对 canonical path+sha256，不接受符号链接替代。exposure includes 必含 failed/unselected/reserved/restarted/all_seen_outcomes/all_source_episodes。record 为 `record_id,root_seed,reset_seed,root_goal_sha256,root_state,root_snapshot_sha256,history_pixels_sha256,raw_history_sha256,source_episode,evidence`；source_episode 是 `{source:{path,sha256},episode_id,pointer}`，不能杜撰 ep0。

原四个保留槽/种子/单次不替补规则仍按原协议；不得自动迁移给 P5，不得因不足而更换根或依据成功筛选。metadata panel 通过也不是 source-origin真实性审查或科学 fresh批准的替代。

### 4.3 P5 独立确认及跨任务协议仍未冻结的输入

actual protocol path/SHA=null/null；机器 schema=not_yet_frozen，不能虚构一个可运行 JSON。MAIN 最少需固定 N/顺序/种子分布、source/episode split与排除、模型/task/obs/action/history/labels、arms/逐轴预算/caps、失败/停机、训练成本及统计承诺。P5 resolver request semantic SHA 为 `9b304bc853f678046cae8dabd7adeda1866ada6cbe8933b7cd6539b16ff699a0`，不是新 cohort 证书。

旧 closure 的 core freeze、conditional roots、local strong controls、architecture contract、native P0/task、formal P1、used components、independent replication、same-model multitask 九项应逐项对接后来的真实 receipt；一个新接口或 metadata report 不可批量签掉九项。

## 5. 严格 fresh0 下禁止启动或宣称的事项

- 不能启动原保留 panel 的独立确认、P4 fresh跨环境/多轮反馈确认或 P5 independent replication；不存在可替换 null 的 certified denominator。
- 不能拿 known2 的新 arm、额外 goal、164 replay、历史三个 seed 或 arbitrary 两目标充当新根；不替补失败，不据 outcome 改顺序/阈值/超参。
- 不能启动未授权的 root采样、补编码、模型搜索、physics 或 CPU神经 fallback 来“修好”缺来源；无原证据就保持 blocked。
- 不能把真实反馈 GD、GRASP变体、三程序组合、JEPA响应或 shared-panel 比较升格为 P1通过、P2/P3机制成立、P4控制增益或 P5跨任务泛化。

仍可立即做的是第3节的 saved 精确 join、资源向量、真实schema映射和协议补充；后续 known-root工程诊断也须 MAIN 另行授权，不是 fresh0 对所有 RAM/metadata工作的一刀切禁止。

## 6. MAIN 可执行接点与封存顺序

只读核当前状态（不重建 resolver，不运行 fullstrict）：

```bash
cd /home/gbk/code/LeWorldModel/le-wm-main
jq '{schema,status,certified_fresh_count,P5_confirmation_denominator}' results/agent_p5_fresh_confirmation_preconditions_20261029_v1_MAIN01.json
jq '{schema,status,freshness,history}' results/agent_p4_fresh_certificate_audit_20261028_v2_report_MAIN01.json
jq '.request | {history_request,exposure_inventory,actual_roots,panel_approval}' results/agent_p4_fresh_panel_manifest_20261029_v1_manifest_LOCAL01.json
```

资源 saved 接点：先按冻结 `innerdocs/agent_downstream_resource_contract_20261029_v1.md` 完整 JSON 构造自己的输入，显式放入实际 report/receipt path+SHA；再由 MAIN 调用以下模板，不把占位符当真实文件：

```bash
.venv/bin/python -B -m experiments.spectral_reachability.agent_downstream_resource_contract_20261029_v1 saved --input ACTUAL_INPUT_JSON --input-sha ACTUAL_SHA --output NEW_OWN_OUTPUT
```

仅在真实保管者 response 到位后，既有 source authority 的 `validate --response REAL_ABSOLUTE_PATH --response-sha256 ACTUAL_SHA` 可做 bounded syntax/bytes 验证；exit0仍需 origin审阅，不自行批准 fresh。当前不重复 request或validate。

封存顺序：已有实际资产 path/SHA核对 → 缺附件由原保管者交付 → 保存只读 join/资源输出并注明继承证据 → MAIN冻结新的 cohort/跨任务/预算协议 → 按原权威验证资格 → MAIN另行授权主机实验。本文停在接口清单，不生成 missing附件、不修旧产物、不启动后续实验。
