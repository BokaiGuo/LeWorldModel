# P2 多程序组合前瞻队列合同 v1

日期：2026-10-27。交付状态：实现及 RAM synthetic tests 完成；没有实际独立根队列、
P1 通过证据或 P2 科学准入。本模块不是专职历史结果审计器，也不是 planner/executor。

依据外层 `docs/D2P_RESEARCH_CLOSURE_AUDIT_20261025.md` 的 P2、C2 与实验 2 要求，
及原方案附件 `861f6764-1b25-4227-bf15-f1a237ba7d8f/pasted-text-1.txt` 的 4.3、实验 2、
P1 kill experiment：比较组合与无组合 SRA，所有中间状态必须有生成动作解释，
P1 独立非贪心、真实能力改变及任务收益缺一不可。原两段 pilot、模型 Q 更低、
预测样本相同均不能代替这些要求。本次未读取或分析真实模型/物理 outcome。

适用 AGENTS.md 检查：祖先、外层及 le-wm-main 子树未发现文件。使用
api-and-interface-design 技能的 contract-first、严格边界校验与一致错误语义；保留旧版，
只新增本 stem 的源码、tests、本文三文件。不修改共享 progress/dispatch/closure 文档。

## 接口与使用

工作目录 `/home/gbk/code/LeWorldModel/le-wm-main`。源码位于
`experiments/spectral_reachability/moving_p2_composition_prospective_queue_20261027_v1.py`。
纯标准库，无 torch、模型加载、numpy 拟合、AD、physics、训练或网络依赖。

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m experiments.spectral_reachability.moving_p2_composition_prospective_queue_20261027_v1 draft --slots slot0 slot1

PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m experiments.spectral_reachability.moving_p2_composition_prospective_queue_20261027_v1 validate --manifest /ABS/manifest.json --input-root /ABS/evidence
```

以上 slots 只是语法示例，不指定科学样本数。`draft` 打印明确含 null 的 blocked skeleton，
不输出任何 P1/freshroot 证书。两条命令均只向 stdout 打印 JSON，不创建或覆盖结果文件。
`draft` 的退出 0 仅表示骨架构建成功。`validate` 的退出 2 表示缺失/拒绝；退出 0
仅表示 `contract_valid_pending_MAIN`，不是 P1/P2 或运行授权。

仅在 MAIN 实际提供外部批准文件和**独立渠道取得的真实 SHA**后，才可给 validate 增加：

```text
--main-authority /ABS/main_authority.json --main-authority-sha256 REAL_EXTERNAL_SHA256
```

不提供真实示例证书，不替 MAIN 签署。library 接口为 `build_draft(slot_ids)` 与
`build_queue(manifest, load_bytes, trusted=None)`；`load_bytes(relative_path)` 只能返回 bytes。
`trusted` 不可从候选 manifest 的自述复制；CLI 需要显式外部 SHA，不能自行算一个候选
文件的 SHA 就把它称为 MAIN 授权。调用方对该信任根负责，本程序不是签名/身份服务。

## 严格 schema

下列 STEM 为完整 `moving_p2_composition_prospective_queue_20261027_v1`。
所有列出的对象均要求精确字段；未知字段、重复 JSON key、NaN/Inf、指数溢出、
bool 冒充 count、负成本、缺字段均拒绝。SHA 为 lowercase hex64。

```text
manifest = {schema:STEM+'.manifest', sealed_at_ns, denominator,
            contract, p1:ref|null, freshroot:ref|null, slots:[slot,...]}
ref = {file:相对input-root路径, sha256:文件字节SHA}
contract = {
 assets:{model,encoder,normalizer,q,source_rule,task_scope,action_domain,
         goal_rule,success_rule,controller,observation_rule: 各为ref},
 horizon_steps:正整数,
 p1_required_metrics:[MAIN预声明的指标名,...], preregistration:ref,
 prohibited:{training,adapters,q_fitting,model_updates,cpu_neural,
             oracle_selection,outcome_filtering,root_replacement: 全为true},
 execution_backend:'HOST_CUDA_AFTER_MAIN_REVIEW'
}
preregistration = {schema:STEM+'.preregistration', p1_required_metrics,
                  assets_sha256:canonical assets JSON的SHA, horizon_steps}
slot = {slot_id, identity, shared_cost_upper:vector, matched_total_cap:vector,
        arms:{composition:arm, no_composition_sra:arm}}
identity = {root_sha256,goal_sha256,history_sha256,blind_sha256,task_id}
arm = {shared_sha256,composition_enabled,plan_rule:ref,cost_upper:{
       prefix_search:vector,suffix_search:vector,derivatives:vector,
       replay:vector,execution:vector}}
plan_rule = {schema:STEM+'.plan_rule',arm,program_count,
             action_witness_required:true,intermediate_states_from_actions_only:true,
             search_rule:ref}
```

canonical JSON 使用 sort_keys、紧凑 separators、UTF-8 默认 JSON ASCII 转义及禁止 NaN。
`shared_sha(contract, identity)` 绑定全任务合同与同一根/目标/真实 history/blind；两臂不能
各填独立动作域、goal 或训练政策。composition 的 program_count ≥2，无组合 SRA =1；
这是组合/消融语义，不是科学性能门槛。plan_rule/search_rule 是将来 runner 的封存规则，
不是已生成的动作见证。本模块不宣称执行过任意一个动作或核实过动作可达性。
模型、Q、normalizer、原始动作范围和物理成功规则都由真实绑定资产确定，不发明常数、
新 Q 或 threshold。将来 runner 还须审查这些规则的具体含义并逐候选保存真实生成动作。

## 成本：计划上界，不是测量

每个 vector 必须完整具有以下轴，值为非负整数或显式 null：

```text
predict_calls, predict_samples, action_encoder_calls, encoder_calls,
jacobian_calls, jvp_directions, vjp_directions, backward_calls,
svd_calls, solve_calls, physics_steps, host_wall_ns, cuda_ns, flops
```

每臂 standalone 计划总成本 = shared_cost_upper + prefix_search + suffix_search +
derivatives + replay + execution，逐轴求和；任何项 null 则该轴未知，不能补 0。
未知总量或未知 cap 阻断。shared_cost_upper 要包含 source construction/replay、
warmup/goal/history/encoder 等共同前置成本，在两臂 standalone 比较中分别全额计入；
它不是双臂联合运行实际成本或已支付缓存的退款规则。禁止只报 suffix 或 terminal
候选成本、只报 predict 次数而省略前缀/JVP/VJP/backward。

每个阶段的数值是**MAIN 审核的完整上界**，各操作须按不重叠的归属分配一次；
派生 Jacobian/VJP 指标可同时存在，但不得把不同轴相加充当 FLOPs。所有重复/失败
attempt、replay、执行后编码都必须在实际 runner 重新计账。flops 也不得用 predict
样本猜算；无可审核来源即 null/blocked。声明明确没有发生的操作可以写真实的 0，
缺测不可以。上界全部不超过同一 slot 的 matched_total_cap 只是共享资源封顶；
两臂实际消耗可不同，不保证等 FLOPs、等墙钟或公平预算。报告不提供实际成本估计，
`actual_experiment_cost=null`。本版没有运行实验。

## P1 与 freshroot 外部授权

外部 MAIN 文件（不是 manifest 内自述）：

```text
{schema:STEM+'.main_authority',reviewer:'MAIN',contract_sha256,
 spec_sha256,p1_sha256,freshroot_sha256}
```

`spec_sha(manifest)` 对 schema/sealed_at_ns/denominator/contract/slots 的完整 canonical
JSON 取 SHA，排除随后提供的 p1/freshroot refs，避免循环哈希。该绑定保护槽位顺序、
全部计划规则、预算和目标身份；批准后改变 budget/候选规则也必须重新 MAIN 批准，
不能沿用旧 authority。P1 证据必须先于 P2 seal；freshness authority 在 seal 后签发。
时间和 no-outcome 声明仍依赖 MAIN 的外部封存流程，不是本程序可证明的时间戳。

```text
p1 = {schema:STEM+'.p1',contract_sha256,certified_at_ns,
 gates:{independent_calibration,independent_outcome,non_greedy,
        real_capability_change,real_task_benefit,integrity: 全true},
 roots:[真实P1 root SHA,...],ci:{预声明指标:[lower,upper],...},
 preregistration_sha256,evidence:[ref,...]}
freshroot = {schema:STEM+'.freshroot',issued_at_ns,contract_sha256,
 identities:[与slots顺序完全一致的identity,...],excluded_roots:[SHA,...],
 coverage_complete:true,independent:true,not_used_for_design:true,
 no_outcome_selection:true,exclusion_manifest:ref}
exclusion_projection = {schema:STEM+'.exclusion_projection',excluded_roots,
                       unresolved_sources:[],evidence:[ref,...]}
```

P1 gate 必须全部明确 true，非空证据的实际字节 SHA 必须一致；指标集必须与已封存
preregistration 完全相同，每个有序、finite CI 的 upper 必须严格 <0，触零拒绝。
这是 closure 已有 P1→P2 约束，不新增效果量、显著性水平、置信度、最低根数或算法阈值。
本版不计算 CI、不重审 evidence 中的统计推导；它是 MAIN 真实独立审查结果的严格适配
接口，历史报告不能仅改 schema 或补 true 后使用。

新 slots 的 root/history/blind/task_id 必须队内唯一，root 不得在 P1 roots 或历史
excluded_roots 中。exclusion_projection 是 MAIN 外部批准的新 metadata 投影接口，
不是已有 history resolver schema；未解决来源不能置空。本程序不自行搜索/认证全部历史，
也不通过 root-goal 换目标认证新物理 root。physical pose/角度别名、snapshot、lineage
与 reset/seed 全历史排除仍须按 `docs/D2P_MOVING_FRESH_ROOT_PROTOCOL_20261027.md`
由 source/MAIN 审查；SHA 不同不证明物理或统计独立。

全部 refs 校验文件字节；CLI 限制相对路径、拒绝 ..、绝对路径和逃逸 symlink。
默认读取的 model 等 ref 应指向 MAIN 生成的小型封存 metadata，不能指向权重并把
读取权重当作已验证冻结。资产字节和机制证据的语义真实性仍由 MAIN 负责。

## 输出与当前缺口

有效 envelope 中所有 slots 保留原索引/分母，不筛选、补根、排序或挑胜者。
任何全局或行级错误阻断整个队列，其余已校验行的计划数值仍只是诊断。
输出始终 `execution_authorized=false`、`P2_admission=false`、
`scientific_evidence_generated=false`，并明确外部声明未被本程序独立证明。
没有 dispatch/released 列表或 executor。不能只看 CLI 退出 0 就启动真实实验。

目前尚缺的真实合同材料：

- 当前任务/模型合同下独立 P1 真实非贪心、能力改变与任务收益的完整 evidence、
  预声明指标及有效 CI、MAIN 实际审核；不能用原失败/未通过报告填充。
- MAIN 封存的具体组合/无组合搜索规则、候选动作见证协议、共享动作/目标/控制器
  与 success 合同，以及完整各阶段各轴资源上界，尤其 derivative/FLOPs/墙钟成本。
- 全历史 exclusions（含失败/预留/派生根和物理别名）的真实覆盖证明、未解决项处理、
  fresh source producer、真实盲 history/goal/model metadata 及 MAIN freshness authority。
- MAIN 提供的外部 authority 文件/独立 SHA 与完整 prospective spec 批准。上述新 schema
  当前没有被本任务认证的真实 producer，不能冒称现有报告可直接消费。
- 将来 MAIN 审核后 HOST CUDA 才可进行真实模型/physics、逐候选和逐操作实际计数、
  动作封存执行与独立真实组合增量评价。本次不实现、不启动这些实验。

## 测试

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m pytest -q -p no:cacheprovider tests/test_moving_p2_composition_prospective_queue_20261027_v1.py
```

测试全部 RAM synthetic JSON/bytes，CLI Path 读取和 symlink resolve 使用内存 mock。
无 fixture 文件、模型、真实 evidence、physics 或神经实验；不导入新模型运行时。
覆盖 5 阶段×14成本轴缺失、完整成本累加、共享资源 cap、MAIN/spec SHA、两臂根目标
合同、动作范围覆写、P1 全 gates/CI/根复用、freshness/历史未知、时间、固定分母、
非法 JSON/路径、help 以外的 draft/validate CLI 流程及旧 authority 不可重用。
最终实测 **123 passed in 0.27s**。仅证明这些软件合同反例和正例的行为，
不是独立性、P1 机制、P2 收益、预算公平或任何科学结果。
