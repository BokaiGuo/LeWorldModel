# P2 保存动作程序图：精确身份闭合搜索核心 v1

日期：2026-10-27。状态：独占代码及 RAM synthetic tests 交付；没有提供或认证
本 schema 的真实 graph，未生成 research 报告，未启动模型或 physics。

依据原 proposal 附件 `861f6764-1b25-4227-bf15-f1a237ba7d8f/pasted-text-1.txt`
4.1–4.3：组合对象是携带具体生成动作、预测后继和状态转移身份的程序，不是任意
latent waypoint；动作见证不保证真实可达。外层
`docs/D2P_RESEARCH_CLOSURE_AUDIT_20261025.md` 的 P2 条目及 P1 依赖继续有效。
P1 缺失不能因离线搜索找到路径而绕过执行门。

适用 AGENTS.md：检查祖先、外层及 le-wm-main 子树，未发现。使用
api-and-interface-design 技能定义严格边界、固定合同、显式失败语义。
只新增本 stem 的源码、tests、本文；旧 queue/共享 progress/dispatch/closure 不修改。
不是专职历史审计器，不遍历历史结果，不修补已有证书。

## 实现与命令

源码：`experiments/spectral_reachability/moving_p2_saved_program_graph_20261027_v1.py`。
只使用标准库；不导入旧 queue、torch、模型、模拟器、numpy 或任何拟合代码。
library 接口：

```text
search(graph, load_bytes, *, algorithm, max_expansions, beam_width=None)
validate_graph(graph, load_bytes)
endpoint(node, contract_sha256)
```

所有图和外部字节先快照并完整校验，再搜索；错误返回 `invalid_saved_graph`，
`validate_graph` 底层接口则抛出异常。load_bytes 仅消费显式相对路径的 bytes。
CLI 只读输入、向 stdout 打印软件搜索结果，不写 report/graph/动作文件：

```bash
# cwd=/home/gbk/code/LeWorldModel/le-wm-main；以下是真实资产到位后的调用形式，不是本次实验命令
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m experiments.spectral_reachability.moving_p2_saved_program_graph_20261027_v1 --graph /ABS/graph.json --input-root /ABS/saved --algorithm shortest --max-expansions MAIN_APPROVED_INTEGER

PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m experiments.spectral_reachability.moving_p2_saved_program_graph_20261027_v1 --graph /ABS/graph.json --input-root /ABS/saved --algorithm beam --max-expansions MAIN_APPROVED_INTEGER --beam-width MAIN_APPROVED_INTEGER
```

placeholder 不是可执行参数；不提供默认根数、宽度、成本或 scientific threshold。
MAIN 应预先批准具体搜索参数、图、任务及成本合同，不能看 outcome 后更换。
退出 0 仅表示 `saved_path_found`，2 表示非法/无路径/截断；均非执行授权。

## 保存图合同

下面 STEM 为 `moving_p2_saved_program_graph_20261027_v1`。所有 schema 对象精确
字段白名单，禁止额外 `passed/P1_passed/physical_success` 等伪准入字段。JSON 重复
key、非有限值、指数溢出、bool 冒充整数、空字节和 SHA 错配均拒绝。

```text
graph = {schema:STEM+'.graph',contract,base_resources:vector,
         nodes:[node,...],edges:[edge,...],start_node:node_id,target_nodes:[node_id,...]}
contract = {
 assets:{model,encoder,normalizer,q,goal,task,controller,action_domain: 各为ref},
 action_dtype:'float32-le-raw'|'float64-le-raw',action_dim:正整数,
 lower:[每维raw下界],upper:[每维raw上界],
 max_steps,max_l1,max_l2_squared,min_programs,max_programs,
 resource_caps:vector,
 prohibited:{training,adapters,q_fitting,model_updates,cpu_neural,
             oracle_selection,outcome_filtering: 全true},
 objective:'sum_saved_nonnegative_cost'
}
ref = {file:相对input-root文件名,sha256:真实文件字节SHA256}
node = {node_id,state_kind:'saved_predicted'|'saved_observed',
        latent:ref,state:ref,context:ref}
endpoint = {node_id,contract_sha256,state_kind,
            latent_sha256,state_sha256,context_sha256}
edge = {edge_id,program_type:'progress'|'reconfiguration',
        parent:endpoint,child:endpoint,contract_sha256,
        steps:正整数,commands:ref,resource_delta:vector,
        cost:保存的非负有限scalar,receipt:ref}
receipt = {schema:STEM+'.saved_transition',edge_id,parent,child,contract_sha256,
           commands_sha256,steps,resource_delta,cost,program_type,
           backend:'HOST_CUDA',status:'completed',producer_source_sha256,invocation:ref}
```

contract_sha256 是 contract canonical JSON 的 SHA：sort_keys、紧凑 separators、
默认 ASCII 转义和禁止 NaN。每条 edge 及其 receipt 必须精确绑定同一合同，包括
goal、模型、Q、normalizer、动作范围、控制器、资源 cap 和训练禁令。不能跨合同
拼图，不允许换目标、模型或 raw 动作空间再沿用边。receipt 的字段比较还检查
canonical 类型，避免 `true==1` 等隐式数值等价。

assets 应引用 MAIN 封存的小型身份 metadata，不应指向权重要求本模块读取模型。
latent/state/context 只校验保存字节 SHA，不解码为新模型状态，不做距离比较；context
必须是原 producer 保存的完整、可复用 predictor history/时间与动作历史身份。
state_kind 必须反映原资产的状态域，不能把预测 state 重新标成 observed state。
缺真实状态身份时拒绝构图，不能填占位 SHA 或用 latent hash 冒充 state 身份。

node_id 全局唯一，所有边的 parent/child 必须与对应节点的完整 endpoint 字典相等。
一条路径的相邻边 child/parent 因而在 latent/state/context/domain/contract/node_id
上完全闭合。完全相同的 endpoint 内容若使用多个 node_id 也拒绝，不擅自合并别名。
latent 相同或接近但 state/context 不同是不同节点，**不会产生可执行连接**。
不存在 kNN、epsilon 桥、逆边补全或 transitive edge 生成。

invocation 是原生成过程的保存 provenance；producer_source_sha256 绑定原 producer。
receipt 说明**已有**动作/端点生成记录，不是本模块执行过模型的收据。
本 schema 无物理 success、真实未来 cost 或“已执行整路径”的字段。
即使某单边具有 saved_observed 身份，也不能从单边记录推断组合已真实执行。
SHA/receipt 绑定证明字节自洽，不独立证明真实性：producer/MAIN 必须核实生成动作
确实导致这些保存后继。本模块不能通过不调用模型的方式证明原推理的因果真实性。

## 累积动作和资源

commands 保存原始 little-endian float32/float64 **raw 执行动作**二进制，
shape=[steps,action_dim]，严格 time-major；字节长度、finite 和每维范围均校验。
不做 tanh、clip、归一化、插值、精度转换、padding 或后缀重拟合。
返回原 witness refs、逐边 receipt refs、完整原 saved edges 和逐步 commands；
float32 解码到 Python float 只是精确数值表示，原 raw bits 仍由 refs 保留。
包括 signed zero 在内，按原 dtype 重编码须得到原始字节。

整条路径累计 steps、sum(abs(raw_action))、sum(raw_action²)，分别不超过
max_steps/max_l1/max_l2_squared；不是仅逐边检查。使用 Fraction 对**实际保存的二进制
浮点值**精确求和/平方，无容差或 epsilon。cap 也按保存 JSON 数值的精确二进制值
解释；例如 `.1+.2` 的精确 binary sum 可能严格大于 JSON `.3`，不悄悄放宽。
累计 cost/L1/L2² 输出 numerator/denominator，避免序列化时溢出或引入舍入误判。

每个 vector 完整包含：

```text
predict_calls,predict_samples,action_encoder_calls,encoder_calls,
jacobian_calls,jvp_directions,vjp_directions,backward_calls,
svd_calls,solve_calls,physics_steps,host_wall_ns,cuda_ns,flops
```

每轴必须是显式非负整数；null、缺失、float count 均阻断，不补 0。
路径资源 = base_resources + 所有路径边 resource_delta，逐轴比较同一 resource_caps。
base_resources 必须全额包含已保存整图生成/前缀搜索/后缀/AD及重放等已支付共同开销，
包括最终不入选的候选；不得只把 winner 的生成成本充当整图成本。resource_delta 是
原 producer 在相同计账合同下封存、每次路径使用所需的增量，不得与 base 重复记账，
也不得把已消耗的资源从失败分支退款。真实未来执行若成本尚不能封存，不能填 0。
consumer 不自动推导 FLOPs、预测样本等价或物理成本；未知即不可消费。

这些是**输入保存/声明资源的算术**，不构成本次新测量或等总计算预算的结论。
该 CPU 纯图搜索的 host 开销没有测量或混入旧 generation 收据，结果明确
`consumer_host_wall_ns=null`；expanded/generated labels 单独报告。MAIN 后续需要
重新封存实际端到端预算，并核对这部分开销。本模块不宣称已完成全链公平预算。

## 搜索、边界与失败原因

shortest 是非负 saved cost 的 uniform-cost 搜索。每个标签保留 node、完整 visited
node/path、累计动作及全资源向量；**不使用 best_cost[node] 全局剪枝**。因此能够保留
更贵但剩余资源更多的可行前缀。只搜索无重复节点的 simple paths，并受固定
min_programs/max_programs、动作/资源上界和 max_expansions 限制。
循环路径即使有真实保存边也不在本版搜索域中，拒绝原因明确，不暗称全图不可达。

shortest 第一个合法 target 被取出时，`optimality_proven=true` 只指上述固定保存图、
非负标量 cost 和 simple-path 约束内的最小 cost；不是全动作空间最优、真实任务
success 或实证性能证明。完整标签最坏指数多，堆队列为 O(L log L)，扩展另含候选
动作/向量处理；L 是受显式 expansion cap 约束的已生成路径标签数，不声称普通
单标签 Dijkstra 的 O(E log V) 复杂度。

beam 按每层累计 saved cost、edge_id tuple 排序，固定 beam_width 截断；不使用
几何距离或新模型评分。它可以返回真实 saved path，但从不声明最优。剪枝后找不到
路径是 `beam_exhausted_no_path`，不是证明不存在可行 saved path。
任一算法到达 expansion cap 返回 `search_limit`，不冒称不可达或最优；beam 如已
发现 incumbent 则保留完整见证，但该路径仍未经执行。

输入列表顺序不作为策略，outgoing edge_id 排序及路径 tuple 确保确定性 tie-break。
每个失败扩展记录 prefix_edge_ids/edge_id/reason，并汇总 rejection_counts：
max_programs、无保存 outgoing、cycle、累计步数/动作、逐资源轴超 cap。任意保存
边失配会使**整图** invalid，而不是删掉坏边再挑有利结果。无合法路径返回 path=null；
不补 bridge、不回退为零动作、不生成候选神经结果。

## 执行门与当前真实资产缺口

输出始终 `execution_authorized=false`、`P2_admission=false`、
`scientific_evidence_generated=false`、`physical_outcome=null`。
本模块没有“P1_passed=true”解锁口；不能凭 caller 布尔值或旧 queue 的软件有效标志
开放 execution。之后仍需 MAIN 审核的独立 P1、freshroot、实际 HOST CUDA 执行器、
输入/源/预算封存，以及真实逐边/整路径执行证据。不得从本模块绕开旧 queue 门。

本次没有提供或认证可消费的真实 graph：缺保存的完整 parent-child latent/state/
predictor-context、真实 raw action witnesses、绑定 invocation/producer 的完成收据、
全图 base 与逐边增量资源合同、固定目标和搜索 policy 的 MAIN 审批。未对仓库做
全量资产缺失证明，也不把已有两段原型自动转换成本 schema 的真实图。
不生成 synthetic research graph、P1 通过证书或 outcome 报告文件。

## 验证

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m pytest -q -p no:cacheprovider tests/test_moving_p2_saved_program_graph_20261027_v1.py
```

实际 **136 passed in 0.23s**。所有 tests 使用 RAM synthetic graph/二进制动作/receipt；
CLI 文件读取及 symlink 检查均为内存 mock，HOST_CUDA 文字仅是 mock 收据字段，不曾
启动 CUDA。20 个固定种子的 RAM 小 DAG 与独立穷举 oracle 比较验证资源约束最短路径；
其余覆盖完整动作、资源瓶颈多标签、精确端点、无近邻/逆边、beam剪枝/截断、循环、
类型/字节/路径/收据、缺失资源轴和 P1 不可绕过。

并行执行旧 queue 的独立 RAM 回归：**123 passed in 0.27s**，未改旧文件。
这些测试证明限定的软件行为，不证明生成记录真实性、可达性、P1 机制或 P2 收益。
