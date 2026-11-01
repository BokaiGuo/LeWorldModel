# 保存资源—实测结果逐根 join 与完整成本 Pareto

这是已保存数据的确定性分析，不是新模型/物理实验，不是前瞻等预算实验。
三个新独占实现文件使用 stem `moving_baseline_resource_outcome_pareto_20261027_v1`；
不修改原封存、前瞻队列或共享 progress/dispatch/closure 文档。只导入 stdlib 与既有
stdlib 资源矩阵模块，不导入模型、torch、physics 或作者 reported 表格。

## 数据与 SHA authority

固定主输入：

- `results/p1_moving_baseline_budget_matrix_20261027_v1_main01/budget_matrix.json`，SHA256
  `721d978d29516f8ecf1d1056e159736ef231723e6849490a83678bd76baa1ebc`。
- `results/p1_moving_baseline_comparison_20261027_v3_main01/comparison.json`，SHA256
  `96cf27b143bb1a839edf7bc1e4de02f4a97e433c6475c4c731c78af76300d32e`。

首先对 comparison 做完整资源重投影，必须与保存 budget_matrix 的整个结构一致，
而不是只对比 predict 数字或按行序 zip。随后只读取由这两个 pinned 主文件明确绑定的
两个 gzip task export、五份 model consume、三份 physical report、三份 RGB report。
所有文件必须位于本项目 results、是 nonsymlink regular file，具有明确 SHA 且不超过 8MiB；
gzip 解压也限 8MiB。拒绝重复 JSON key、NaN、冲突 SHA、路径逸出。
发布新结果前和写完后重新核验这些文件。不会遍历 comparison 的全部 527 项 bindings、
打开 checkpoint、重跑 ownconsumer 或模型 producer。

逐法逐根 join 绑定：

- task export 文件 SHA、task payload SHA（原 task_id）、task-contract SHA、blind SHA；
- 完整 root-context 与 goal 的派生 SHA、physical report 的保存 root-state 派生 SHA；
- arm、model source ID、完整 sealed 合同 SHA、commands SHA、own-plan SHA；
- physical selection SHA、RGB endpoint latent 文件 SHA，以及 compiler report SHA。

同根六算法必须具有相同 physical root-state；RGB provenance 必须精确匹配 task 的
root latent、goal 和同一所选计划。物理/RGB 每个 cohort 的完整行集合及资源计数必须
与 budget_matrix 一致；拒绝遗漏、额外 arm、重复 plan/root/endpoint 或混绑。
Kino consume 的封存格式没有 task_contract_sha256；本版不伪造该 plan 字段，
而由完整 task payload/file SHA 加 own-plan/commands/RGB join 绑定。

注意：真实 task export 的 `native_root_sha256` 为 null。输出明确保留为 unknown，
`physical_root_state_sha256` 和 `root_context_sha256` 明确是保存数据的派生身份，
不是补造的上游 native-root authority，也不证明独立 freshness。
上游 task-contract/own-plan 使用默认空格、无换行 JSON 哈希；commands 使用紧凑 JSON
加换行哈希。实现分别匹配原格式，不重新定义封存 SHA。

## 每根输出内容

`pareto.json.rows` 固定六方法 × 两根，共 12 条。包含：

- 原 model Q（诊断）、model root Q、实测 realQ root/prefix5/terminal10；
- literal native success、full horizon、实际返回步数；
- native distance、signed native angle term、object position error、wrapped angle error；
- 完整保存 native initial/final metrics，包括原本的 null，不用别的指标填补；
- 矩阵逐根原始成本向量、原始 source-resource profile、原 sealed CPU/搜索配置和共享账目引用。

只投影之前已验证的保存数值，并逐项检查 consume/physical/RGB/comparison 相等。
不重新编码图片、不读 NPY、不重新计算模型/实测 Q，也不把 model Q 当实测 Q。
不读取或混入作者 reported 值，不给两根描述性数据增加统计显著性/总体泛化结论。

## Pareto 合同

在每个同根的 15 个无序方法对上判断，共 30 对；不跨根平均后宣布 winner。
目标固定为：实测 `realQ_terminal10`、`native_distance`、signed `native_angle_term` 越小越好，
literal `native_success` 越大越好。角度项保留原生有符号值，不改成 wrapped angle；
wrapped angle 等其余指标另列。model Q 不参与实测 Pareto 目标。

完整成本轴包括：predict、forward samples、predict API、action encoder、prediction projector、
AD forward 子集、scalar reverse、backward/grad/J/J-API/output-VJP、SVD/solve，
image encoder/projector、physics/public steps、reset/set_state/render/goal bind、environment construct/close，
完整 CPU operation profile、native-source profile 和完整 pipeline walltime。
每轴具有 unit/scope；计数同时比较 attempted/completed，未完成的尝试不当作低成本优势。
CPU profile 是逐 operation 的 attempted/completed 映射，绝不把 SVD、score、solve 求和成一个 CPU 数字。

必须所有成本轴、目标均已知，unit/scope/derivative semantics、任务/根/目标/时域一致，
且物理 horizon 完整，才能判断：

- `a_dominates_b` / `b_dominates_a`：每个成本与目标维度均不差，至少一维严格更好；
- `tie`：完整向量完全相同；
- `nondominated_pair`：完整向量可比，但存在 tradeoff；
- `unknown`：任一必需成本/目标缺失或不可比，即使已知若干轴看似优胜也不删轴判断。

`frontier` 遇到任何相关 unknown 比较，明确输出 unknown；不把“尚未证明被支配”冒充
“已证明 nondominated”。不同搜索支持域、replay 与配置全部保留；即使将来完整向量可判定，
也仅是历史两根局部 Pareto，不能称全局等预算胜出、官方复现或独立确认。

## Shared ledger 与 overlap

17 个唯一账目条目：共享继承编码 1、model invocations 5、consumer CPU 5、physics cohort 3、RGB cohort 3。
CEM/MPPI 共用一次 model invocation，按同轴跨不交叠 root/arm 计数，例如 predict=412；
不是把每法整次 sampling run 重收两次。consumer CPU 缺完整独立计量则记录 unknown，
保存的验证 CPU 合同另留在 accounting_contract，不能误当生产者免费或已收过的同一次工作。

继承 encoder8/projector8 只记一次；逐根、two-root、standalone 是同一账目引用视图。
physics/RGB cohort 整次记录一次，不按方法/根均分；per-root 成本无法归属时为 unknown，
也不将整个 cohort 复制成每根费用。ledger 无退款、净额、减除或跨单位总成本。

明确保留 overlap：AD forward 是 predict 子集；module/API/sample 可能描述同次 forward；
backward/grad/J/VJP 描述可能重叠的导数工作；scalar loss sweeps 不等于输出 VJP 方向；
public/physics step 是不同嵌套单位；native-source profile 是原计数的别名视图而非新增费用。

## 命令、产物与限制

在 `/home/gbk/code/LeWorldModel/le-wm-main`：

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q -p no:cacheprovider tests/test_moving_baseline_resource_outcome_pareto_20261027_v1.py
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m experiments.spectral_reachability.moving_baseline_resource_outcome_pareto_20261027_v1 --output results/moving_baseline_resource_outcome_pareto_20261027_v1_saved02
```

运行器只允许 results 下本 stem 前缀的新独占目录，拒绝覆盖，要求 200MiB 余量、最多 2MiB 输出。
生成 `pareto.json` 与 `pareto.md`，stdout 返回每文件 SHA。重跑须改用未存在目录名；
不删除、不覆盖首轮 saved01。saved01 是本轮早期 join 结果，最终交付以 saved02 为准，
后者补齐完整 physical axes、原生 resource profile 和 CPU profile 类型检查。

RAM 测试覆盖完整向量支配/反向/平局/tradeoff、任何成本缺失均 unknown、
CPU 单位集合不匹配、未完成尝试、native bool 类型、各种 task/root/command/selection/RGB 混绑、
重复 cohort/额外行/计数不一致、reported poison 不进入数值、缺失不填零、
SHA/路径/JSON/gzip 防护、独占输出/空间余量和输入重检。RAM tests 不是科学证据。

本轮实读显示全部 30 对成本向量不完整，故均为 unknown；Q/native 数值和资源记录仍完整输出。
这是 saved-data 交付，不是新的预测/采集或真实实验结果；无训练、adapter、Q 拟合、模型/physics 调用，
无删除、发布、原封存/共享文档变更。未进行全库测试或 HOST 作业。

2026-10-27 验收：独占 RAM 测试 78 passed；与预算矩阵及前瞻候选队列合计 189 passed。
saved02 实际读取 15 个 SHA 绑定输入，输出 12 行、29 个必需成本轴、30 个 unknown 方法对、
17 个唯一账目条目，文件合计目录占用约 228KiB。两个主输入 SHA 与开始时一致。
`pareto.json` SHA256 为 `838b1916a7da99240a4593d994f4db6edd88779cbf8567190b20236273456e6e`；
`pareto.md` SHA256 为 `9443599050c0c6635742a297b3cae08a70c1e3122a81ab3ea3500998a65ea123`。
