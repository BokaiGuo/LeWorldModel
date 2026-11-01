# World-model prediction budget 合同与 callback meter

本版是对比较准入协议的有界补充，不修改旧 all-axes 合同、冻结 Pareto、sealed 配置、
共享进度文档或任何神经/物理 runner。交付仅本 stem 源文件、测试和本文。
API 设计采用 contract-first：外部声明、dispatch token、终态与失败语义明确分开，
新增接口不改变旧消费者。

## 主张纠正与原提案边界

原研究计划 `docs/SPECTRAL_REACHABILITY_ATLAS_RESEARCH_PLAN_20261023.md` 的
Planning Compiler 比较表要求与匹配预算的 CEM/GD/GN/GRASP 比较真实执行结果；
Evaluation 行将 model calls、backward calls、wall time 分开记录。
本次按用户明确的“同 world-model 调用预算”实现下面两种互不等价的预算口径，
不把原提案含混的“预算”自动解释为任意一种模式。

旧 `moving_equal_budget_prospective_queue_20261027_v1` 的 all-axes 等价合同要求
包括导数、CPU、墙钟等成本及完整支持域逐项相等；冻结 Pareto 的完整轴判定也另有目的。
它们不能支持本次 proposal-specific 的 matched prediction-budget 主张，
也不能作为所有比较的唯一准入条件。旧版“不满足所有成本轴相等/已知”
不等于“无法在同预测调用预算下比较不同算法”。旧产物保留原含义，不追改结果。

本版绝不声称 `equal_total_compute`、equal FLOPs、等墙钟、全局等预算胜出，
不会把 backward/J/VJP/SVD 或模块调用折算、相加成一个总成本。

## 两种 MAIN 显式选择的 mode

| mode | 唯一主预算单位 | 原子扣费点 | 一个 API 内一次 batch B predictor forward |
|---|---|---|---|
| predict_API_calls_matched | model.predict API dispatch 次数 | predict_API 的 before_dispatch | API=1；samples=B 是另列诊断 |
| predictor_forward_samples_matched | predictor.forward 的 batch items | predictor_forward 的 before_dispatch | forward calls=1；samples=B，按 B 扣费 |

一次 API 若内部有多个 predictor dispatch，API 仍只算一次，samples 对实际各子 dispatch
的 batch items 分别累计；不拿 API 的输入 batch 猜测内部总 samples。
AE/pred_proj 自己的 calls/samples/context rows 单独记录，不能再加进 predictor/API 预算。
ADforward 是预测工作的标签子集，不是额外 forward。scalar reverse 不是 output-VJP。

MAIN 必须预声明 batch 限制：允许的 sizes、每次 API 最多 predictor dispatch 次数（失败也占次数），
以及 `API_predictor_batch_relation`：

- `same_logical_batch`：API 和子模块的逻辑 batch 必须一致。
- `bounded_child_batches`：显式允许内部拆批，各 child batch 仍必须属于 allowed sizes，
  受 fanout 限制；按实际 child batch 计 samples，不推断 API 到 predictor 的一一关系。

`rows_per_sample` 只用于 AE/projector 的 context/token rows，不能用来放大或缩小预测 sample 口径。
API/predictor 的该值必须为 1。例如 AE 处理 B×3 rows，不意味着 3B 次 world-model API 或 sample。

## 外部 MAIN 声明（模块不代签）

`PredictionBudgetMeter` 初始化和可选 CLI 声明均要求外部 canonical declaration SHA
及 `trusted_authority='MAIN'`。SHA 算法是 UTF-8、sorted keys、ensure_ascii=False、
allow_nan=False、indent=2，加末尾换行。外部传入的信任锚仅是调用方 authority 边界，
不是模块签名或密码学身份认证，不能把本地自算 SHA 当成独立 MAIN 审核。

声明 schema 为本 stem + `.MAIN_declaration`，严格字段见源码 `declaration()`，包括：

- `authority`、`declaration_id`、`issued_at_ns`、固定 `matrix_sha256`；
- `population='existing_calibration_two_roots'`、`allocation='per_root_per_arm'`；
- 固定六方法×两根的有序 `groups`，每组绑定 arm/slot/task SHA/sealed configuration SHA；
- 一个明确 mode、统一的每 root/arm positive integer attempted cap；
- `scope.prepare/search/ADforward/replay` 的显式布尔值；search 必须计入；
- `scope.failed_attempts='charged_no_refund'`、`scope.count_on='before_dispatch'`；
- 上面的 batch 限制；`no_outcome_based_configuration_change=true`。

scope 可由 MAIN 明确排除 prepare、ADforward 或 replay，但这些工作仍进入 observed 账，
included 账只执行声明的范围。这样的报告只能称声明范围内 prediction budget，
不能省略范围修饰而声称全 pipeline 公平。模块不根据 outcome 调整 cap、采样数、
迭代、bank 或旧算法参数，也不通过补跑/空转把旧不等预算结果“补齐”为新比较。

没有 MAIN 声明时不猜 mode/cap，不允许构造 meter；只可阅读 pinned saved calibration 报告。
报告的 `meter_configuration_ready=false` 与 `report_ready=true` 是不同工程状态；
即使有效 MAIN 声明使前者为 true，也不意味着可以启动神经实验。
本版不提供 declaration writer、新根证书、freshness signer、P1 authority 或模型 runner。

## 执行接口与失败语义

`PredictionBudgetMeter.before_dispatch(kind, ...)` 必须由实际 API/module hook 在真实
dispatch 之前调用；`after_return(ticket)` 在该同步调用返回后记录 completed；
`after_failure(ticket, error_type)` 只记录失败，不扣回 attempted。

`invoke(kind, callback, ...)` 是标准库 callback seam：callback 收到发出的 token，
返回值原样透传；异常原样抛出并尝试登记失败。它不隐式调用任何模型，也不读取输出内容。
典型集成形状如下，`meter` 必须已由外部声明初始化；此处不是模型 runner：

```python
def api_body(api_ticket):
    return meter.invoke(
        'predictor_forward', predictor_callback,
        parent=api_ticket, batch_size=actual_predictor_batch,
    )

result = meter.invoke(
    'predict_API', api_body,
    group_id=declared_group_id, stage='search', ad_forward=False,
    batch_size=actual_API_logical_batch,
)
```

predict_API 是 group/stage/AD 标签的拥有者；模块必须引用同 meter 的活跃 API ticket 并继承标签，
不能将 AD/replay 偷改成 excluded work。所有 ticket 是发出的对象身份，不能 clone、跨 meter
使用、重复完成、失败后补完成；父 API 不能先于活跃 children 关闭。

cap 使用 RLock 原子预留 attempted，callbacks 在锁外执行，支持并发竞争同一个 cap。
超 cap 在目标 dispatch 前抛 `BudgetExceeded`，不会调用该 callback；denied 自身未 dispatch，
不额外计 attempt。已接受后 callback 失败、取消或 journal sink 失败均不退款。
sample mode 在 predictor hook 拦截，因此 API wrapper 可能已进入，甚至 AE 等其他工作已发生；
不会虚称其他资源也被该 sample cap 限制。

sink 可选，用于把完整事件交给外部持久化设施；sink 失败或重入使 meter fault、后续 dispatch 关闭，
已有计数不回退，未完成 ticket 保留。snapshot 返回深拷贝，不会因外部改字典而修改账本。
completed 仅表示同步 callback 正常返回，不是 CUDA synchronize、数值正确或预测成功证明。
整个 batch 返回计 B completed samples；抛异常时不凭空认定部分 samples 已完成。

## 真实成本保留与工程 ready

CLI 只读真实 `budget_matrix.json`，固定 SHA
`721d978d29516f8ecf1d1056e159736ef231723e6849490a83678bd76baa1ebc`。
保留全部逐根/合计原成本、已知值/null、来源、合同及共享 model/encoding/physics/RGB cohorts。
不要求不同算法 backward/J/VJP/SVD 相等，也不要求墙钟相等。共享账目仍只计一次，
standalone/逐根是有 overlap 的视图，不重复收费或退款。

真实两根合计 predict/forward_samples 为 272/268/144/168/3604/176；
API 只有 gradient=272 和 GRASP=3604 有独立实际计量，其余 API=null。
不能把这些方法的 samples、predict 或 AE 次数抄成 API 次数。
即使指定 mode，旧矩阵仍不完整支持该声明的 prepare/AD/replay/batch/失败 scope，
且旧六配置计数不等，因此 `prediction_budget_match='unestablished'`，不回溯认证匹配。

真实 pinned 校准资源报告可标 `engineering_status='calibration_report_ready'`、
`report_ready=true`，只表示已有校准账目可阅读；缺少 freshness/P1 时仍严格
`scientific_admission=false`、`launch_authorized=false`。纯 RAM `calibration_report()`
默认 `saved_input_verified=false`，只有 CLI 通过原文件 SHA gate 后才标 true。

meter snapshot 无论是否填满统一 cap，始终标
`evidence_kind='unverified_callback_observations_not_real_model_receipts'`、
`actual_model_execution_verified=false`、`scientific_admission=false`。
`callback_attempt_allocation_matched` 只是所有声明组的 callback included attempted 均达到 cap，
且无 pending/fault；不是实际模型预算匹配、completed 全相等或科学证据。
meter 不观测的其他资源明确保留 null，而不是把旧 reference 成本冒充当前运行费用。
真实 hook 覆盖、设备、输入 batch、durable receipts 仍须 MAIN/HOST 独立核验；禁止用 fakecounter 代替。

## 命令与验收范围

在 `/home/gbk/code/LeWorldModel/le-wm-main`：

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q -p no:cacheprovider tests/test_moving_prediction_budget_contract_20261027_v1.py
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m experiments.spectral_reachability.moving_prediction_budget_contract_20261027_v1
```

若 MAIN 实际提供预声明文件，再由调用者显式附加
`--declaration FILE --declaration-sha EXTERNAL_CANONICAL_SHA --trusted-authority MAIN`。
没有提供时报告 mode=null，不生成假 MAIN 声明。

所有测试仅 RAM fixtures/callbacks：覆盖两 mode、单样本与批量、内部拆批、AE/projector
口径、AD 子集、scope 排除但 observed 保留、失败不退款、超 cap 前拒绝、并发 cap、
ticket/parent/fanout、sink 故障、非法声明及 SHA、fakecounter 不升级证据和 ready/admission 分离。
CLI 验证只读取已保存资源 metadata，stdout JSON，不落盘、不发起 HOST 作业或神经调用。
测试不是科学证据；本版也没有新根/真实预测或物理结果。

2026-10-27 验收：本文件 58 项 RAM 测试通过；加上冻结 Pareto、旧 prospective queue、
budget matrix 的有界回归合计 247 passed。真实 pinned matrix CLI 返回 report_ready=true、
saved_input_verified=true、mode=null、prediction_budget_match=unestablished、scientific_admission=false。
没有收到或生成 MAIN 声明，也没有启动模型 callback。旧 matrix 与冻结 Pareto saved02 的
JSON/Markdown SHA 与上一交付一致；本轮只新增三个独占文件。
