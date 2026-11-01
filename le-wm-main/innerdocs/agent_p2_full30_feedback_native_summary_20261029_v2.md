# Full30 O/I/F 独立原生轴分解 v2

独占新 source/tests/doc stem `agent_p2_full30_feedback_native_summary_20261029_v2`。
仅 saved 数学，stdlib、math.fsum；不 import producer/NumPy/torch/环境，不NN/physics/fullstrict。
按 experiment-code 技能组织来源和测试，不训练、不择优、不追加实验或调门。

## 固定输入与信任边界

根目录 `/home/gbk/code/LeWorldModel/le-wm-main`。只读以下五份 MAIN 完整 own-native receipts，
不递归消费历史producer、不重做journal/PNG/source全闭包fullstrict；这五份完整外部SHA是信任起点。
严格检查schema/status/own_native_saved_reconstructed_here、全部rows/plans、job/base/policy、
O ancestry SHA、相同root/task/goal/原fullhistory、原raw15与实际前缀checkpoint state、科学flags false。
v1完全冻结保留；MAIN发现其跨policy selection_digest强等错误，未写实际输出，不是科学负结果。
v2保留全部指标/顺序/分母，仅修身份域并补足局部执行与模型见证认证。

|receipt（相对根目录）|完整SHA256|
|---|---|
|results/agent_p2_two_program_full30_physical_20261029_v1_consume_MAIN01.json|aabc5cce3855021eba0fe5751c0ae87809cfefb4c2ed54b0d2899048fd6b3cd8|
|results/agent_p2_full30_control15_feedback_physical_20261029_v1_consume_J0_MAIN01.json|565c432e08a7965079a10c709d90773edab4a967004ec07e43202b88a6c96d6a|
|results/agent_p2_full30_control15_feedback_physical_20261029_v1_consume_J1_MAIN01.json|fc08ef91da4f158218d78a17da640a2664a8d1e63ebdf6e3b6fe6b6df5c2ee5e|
|results/agent_p2_full30_control15_feedback_physical_20261029_v1_consume_J2_MAIN01.json|94157f79fd3ca2ea28e825914e56a9f364ba4f87e7871b6d09d9c9eeaa7b9c38|
|results/agent_p2_full30_control15_feedback_physical_20261029_v1_consume_J3_MAIN01.json|8a66e6bb41c716c4d0b01b405745a7d356fcd4cfc7b6e17178e4084c27199f2d|

CLI defaults是这五份actual SHA/path；也接受显式五path/SHA，替代路径属于另一显式authority，不自动认证新实验。
joined结果2842de482d30ff5898dc52915f58ddca292d0c4958f84198377d8c852c9ae101不作为算术来源，避免复述聚合器。
所有校验/指标均不是对未保存原事件的独立重认证。

## 全分母和独立算术

base0/1=root0×two_program_transfer/joint_two_round；base2/3=root1同顺序。
每base O/I/F固定12槽；cp0/5/10/15/20/25/30，共84位置，absolute step=10+cp。
每base O-I/O-F/I-F各7，共84配对；delta方向明确为后者减前者。
成功早停后未到达的checkpoint state/metrics/delta=null，仍保留所有行，不用终态代替cp30、不填零。
破损receipt、缺已到达checkpoint或身份冲突拒绝，不将工程坏记录当科学零效果。

直接解checkpoint.state7和row.goal7，核snapshot、float64bits、对应原responses state。
七轴为agent x/y、object x/y、object raw angle、agent vx/vy。
每位置保存原state7/goal7、逐轴残差与平方，fsum计算agentXY/objectXY/jointXY/velocity/native7平方及根号距离。
每配对保存七轴state差、七轴平方贡献差、贡献fsum及其与两total相减的浮点残差，
并逐位置判object pose exact（含角度）、原生成功差、jointXY/native7的正/负/零/unknown。
不挑base0、不丢其他ties，不以jointXY下降冒充物块移动或能力提升。

原环境 `PushT.eval_state` 位于 `.venv/lib/python3.12/site-packages/stable_worldmodel/envs/pusht/env.py:347–355`，
源码SHA c6d2104adc368ff1c352edc0e35ac74da9f07fda3a748ddbef0ba6b7a2d1daa2；
五receipt均须绑定此源，工具只读取hash不import。原success为jointXY<20且
min(abs(angle_delta),2π-abs(angle_delta))<π/9，保留原式可能负数的angle term；
wrapped atan2只作独立诊断，不改原门。native7 reward=-sqrt(fsum七轴平方)。
fsum与旧NumPy norm可能末位不同，不据旧metric重写独立数值、不使用容差；
独立success和已有transition success并列并记录agreement，cp0没有return标签则null。
本工具不使用final_metrics/native_metrics/metric_breakdown/reward数值计算结果。

资源分别列O和四新job I/F、各native轴attempted/completed、PNG与job inclusive wall，
没有缺轴补0或equalFLOPs推论；sum不包含job间idle。
作者报告值不进入本地表；P1/P2/机制/科学放行/独立确认/物理release保持false。
这是已知探索性结果后的完整分解，不是预注册或物理因果证据，也不能推出训练有效/无效。

## MAIN CLI

```bash
cd /home/gbk/code/LeWorldModel/le-wm-main
.venv/bin/python -m pytest -q tests/test_agent_p2_full30_feedback_native_summary_20261029_v2.py
M=experiments.spectral_reachability.agent_p2_full30_feedback_native_summary_20261029_v2
S="$PWD/results/agent_p2_full30_feedback_native_summary_20261029_v2_MAIN01.json"
.venv/bin/python -m "$M" analyze --output "$S"
H=$(sha256sum "$S" | cut -d' ' -f1)
.venv/bin/python -m "$M" verify --summary "$S" --summary-sha "$H"
```

analyze只持久新文件（xb/fsync/dir fsync），stdout compact path/SHA。
verify从保存arguments重新读取五源、源码语义hash、全部84/84算术和自身三件套SHA，整份逐字节一致；
不重跑上游fullstrict。输入/输出各硬限16MiB，无递归目录扫描；source/preflight成本是本地saved读取。
worker仅运行synthetic RAM/临时磁盘build→write→wholeverify/CLI回归；实际分析由MAIN运行。
tests明确篡改身份/时间/历史/原前缀/资源/missing/source/输出均拒绝，坏producer metric不影响独立结果。

## 修复身份域：不是删除一个条件

O physical assemble第163行与I/F physical assemble第148行分别重新生成自己的execution plan。
它们均通过 native.core.digest → agent_allmode_physical.core →
p1_conditional_task_admission_20261027_v1.canonical/digest（第31–32行）：
`json.dumps(value,sort_keys=True,allow_nan=False).encode()`，默认separators含空格、默认ensure_ascii=True、**无LF**。
本地execution_digest仅排除selection_digest自身。分别核row.selection_digest==row.plan.selection_digest==重算值；
不要求O/I/F的不同执行计划digest相同。schema/index/policy/base/commands/new_model_plan变化使其本来就不同。

父planner_plan.witness_sha256与new_model_plan.new_witness_sha256则由postcontext kernel raw/digest
第23–28行定义：sort_keys=True、separators=(',',':')、allow_nan=False、默认ensure_ascii=True，**单LF**。
新feedback v2通过feedback v1 kernel继承此ABI。v2独立重算这两种模型witness（只排除各自witness字段），
检查新plan的parent_witness/task_sha/model_hash/goal与父plan、literal raw30与parent bits及prefix15。
raw30仅允许lossless float32、[-1,1]；不clip/修复，不从logits重建动作。
每条执行raw SHA与模型raw bits独立核验。O执行commands必须是其父模型commands。

跨policy共同祖先：planner_plan整份exact、row task/root/goal/goal pixels/root state、
原prefix15 exact和同时间前缀state；执行计划除原producer明确更新的
schema/index/base_index/policy/commands/new_model_plan/raw SHA/selection_digest外，其他字段必须与O exact。
不能通过同时重封父witness、新witness和execution digest将另一个父plan嫁接到原O。
本地见证自洽不是历史真实性的独立证明；外部五receipt仍是完整执行认证的信任起点。

source.ABI_OLD/ABI_NEW列出8个准确源码SHA，build核receipt.bindings与当前source bytes（只读不import），
summary.digest_ABI持久完整paths/SHA及两种序列化域。原环境success语义源绑定不变。
新增fixtures具有O/I/F三个不同且正确的execution digest；磁盘回归使用完整receipt文件及SHA，
包含全build/write/wholeverify、重封bad digest、换父plan全层重封拒绝。
不是用compact摘要冒充spaced execution digest，也没有借producer数学计算任何指标。

