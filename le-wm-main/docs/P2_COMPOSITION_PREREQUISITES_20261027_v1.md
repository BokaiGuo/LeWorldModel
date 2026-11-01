# P2 完整组合主张：可运行前置条件与代码合同审计

实现补记（同日、独立新资产）：授权实现范围内已新增
`experiments/spectral_reachability/p2_composition_paired_20261027_v2.py`、对应测试及
`innerdocs/p2_composition_paired_20261027_v2.md`。它将共享联合谱bank的40个固定
prefix分别接入无组合H2与本状态full10 suffix重编译，保存全部候选动作/双edge
合同并提供source-bound导出；填补下文A项的最小可运行接口缺口，而不是仅新增gate。
此补记不改变下面审计快照中的旧资产事实，也不声称补齐P1、严格总算力配对或
独立科学P2。CUDA由main评审后另行决定，本实现线程只做测试和prepare。

日期：2026-10-27。范围：只读检查本项目已存在的明确资产，新增本文及一个纯数据
inventory/arithmetic模块和测试。**没有重做Poincare的效果比较，没有P3分析，没有
神经forward、物理执行、下载、调参、筛根、改gate或改任何旧source/report/seal。**
本审计不宣告“组合算法不存在”，也不把 `P2_admission=false` 当作缺口解释。

## 结论：已有实现部件，尚缺统一且可执行的组合实验合同

现有代码足以演示固定动作库上的组合搜索、两段局部谱编译，以及真正联合H2无组合
SRA的CUDA/物理执行。尚不能据此建立“重配置使后续可达性改变，组合编译因此相对
无组合谱规划和强优化器带来独立真实收益”的完整D2P主张。

关键不是再补一个布尔标志，而是补齐：**原P1机制证据、连接局部编译与组合搜索的
runner、可归因的共同动作/目标合同、全候选动作见证与物理覆盖、真实来源兼容的
严格数据适配器，以及新独立cohort的预声明执行/统计合同。** 后文逐项给出资产形态。
这些是现有科学/数据约束的落地，不在本文新增成功阈值或重新选择方法参数。

依据：外层研究计划的“Planning Compiler / Runtime”和“核心机制”，
[严格P2校验器](/home/gbk/code/LeWorldModel/le-wm-main/experiments/spectral_reachability/d2p_p2_comparison_contract.py:51)、
[原P1→P2依赖](/home/gbk/code/LeWorldModel/le-wm-main/experiments/spectral_reachability/d2p_stage_admission.py:164)。

## 1. 实际资产，不混淆其完成层级

下表均为本轮读取到的保存状态；不比较任务分数、效应大小或方法胜负。

| 资产 | 已有能力/实际记录 | 不能替代的缺口 |
| --- | --- | --- |
| `p2_matched_composition_20261027.py`；`results/pusht_p2_matched_composition4_v1_20261027_cuda/p2_matched_report.json` | 四根CUDA完成；16条五步raw primitive，beam4；composition赢家两条edge，flat赢家一条edge；两臂每根80 planning+4 replay | 固定bank不是在搜索节点重新谱编译；64与40个H2终点不相等；仅赢家有完整witness |
| `results/pusht_p2_matched_physical4_v1_20261027_cuda/report.json` | 八个已选分支全部保留；真实动作与封存selection相同；只有5个完整H2，其余3个为`observation_unavailable` | 不是全部候选物理覆盖；不能把中止endpoint当作十步H2或删掉这些根 |
| `pusht_full_packet_compiler_20261027.py`；`results/pusht_full_packet_compiler_20261027_cuda_v1/report.json` | 真正谱prefix生成及每个prefix后本状态suffix Jacobian/ridge已实现；四根×constant/full×10＝80候选；`plans_sealed`；572 predict | 这是两阶段谱编译/动作chart消融，不是已接入严格P2的“组合vs无组合”runner；选择目标和约束也不同 |
| `results/pusht_full_packet_physical_20261027_cuda_v1/report.json` | 全80候选执行记录已存在，77 completed、3 observation_unavailable；原始物理产物不可丢弃 | 汇总为`integrity_fail`，`encodings_complete=false`；不能写成该谱编译器的完整可比物理编码已通过 |
| `p2_no_composition_sra_20261027_v2.py`；`results/p2_no_composition_sra_20261027_v2_prepared01/report.json` | 主线程在授权实现范围内预声明的联合192×20 H2 Jacobian、全20维ridge/谱bank；四根352 predict、4 Jacobian、8 SVD、768标量VJP方向；40候选/根 | 已补齐可运行无组合SRA，不再列为“代码不存在”；只有赢家有完整部署witness；不是严格总资源matched |
| `results/p2_no_composition_sra_physical_20261027_v2_cuda_main_v1/report.json` | Poincare消费者：四根各10步，全部completed；`calibration_recorded`，完整性gates全true | 已补齐已选SRA路径的真实物理执行；不替代组成算法、全槽位物理记录、独立cohort或组合增量机制 |

另有 `d2p_kinodynamic_program_search.search_programs` 和
`run_pusht_program_search_cuda.py` 的固定bank树搜索。其选择规则是跨depth的
best nonempty visited，允许H1/H2；不能直接当作严格固定H2候选。需要固定终点
长度时复用现有matched版本的语义，而不是给旧短轨迹补零/补包。

旧 `p2_composition_admission_20261027_v1` 只接受历史composition/flat/CEM/random
schema；新SRA的完成不会自动更新该旧审计器或其封存报告。保留旧记录，无须为
消除过期的“缺SRA”文字而回写它；新适配器应绑定新证据并准确报告其范围。

## 2. 哪种“组合”已有，哪种还缺

固定bank runner是真实的前向组合搜索：先评估16个prefix，再对保留的4个父节点
展开16个suffix；完整latent/action context沿边传递。其成功证明搜索和动作见证
可连接，**不证明首段通过改变后续局部动力学而获得非贪心组合收益**。
其edge在实现中均标记`progress`；witness允许`reconfiguration`字段，但改标签
不是重配置机制证据。

full-packet `compile_arm`则确实已经在各prefix之后重新计算suffix Jacobian，
并保存prefix/suffix参数、raw命令、局部Jacobian、预测prefix/terminal及ridge值。
不能重复实现这个原语后声称新贡献。缺的是把这些原语与组合节点/搜索/审计接口
接起来，并按目标主张冻结停止规则和消融。**至少两条真实程序edge已经可表达H2
组合；本审计不发明“必须H3以上”或必须在线反馈的新门槛。** 更长组合或闭环
重编译只有在另行主张时，才需对应预声明深度及观察/重编译记录。

### 必需执行资产 A：节点编译—搜索接口与端到端runner

可复用：full-packet局部编译、`RootAnchoredCudaReplay`、beam搜索和witness v2。
缺少一个封存的新runner，将“完整父context→本节点局部response/程序→实际raw
edge→完整子context”串为同一条可追溯数据链；不能只把root静态bank换名。

该runner至少需输出：父/子节点身份及context摘要、局部operator/chart/nominal、
Q/goal与action尺度、进展/重配置依据、每条动作生成见证、累计资源、所有展开/
剪枝/失败/停止原因、固定长度候选集合与stable selector；通过全串及分段独立
重放核对winner。若复用缓存，缓存key必须包含完整context与生成合同，缓存命中
和复用成本显式记录，不能把额外编译藏到bank预处理之外。

完成判据是这些数据可经新prepare绑定、只读加载、host-CUDA执行和schema消费，
而不是import成功、合成测试通过或仅保存胜者。是否授权实现/运行由main另定。

## 3. 原P1是具体的机制/物理缺口，不只是未开放的flag

已读模型校准：`pusht_d2p_p1_root_anchored_v5_calibration_20261026_cuda/report.json`
为`pass`，scope明确“model-side，physical candidates not executed”，不能替代P1
独立物理outcome。已存在的历史
`pusht_d2p_p1_outcome_physical_20261024_cuda/report.json` 为`fail`，
`all_selected_prefixes_physically_no_more_goal_greedy_than_raw_zero=false`。
这定位的是原先非贪心物理机制条件没有通过，不是“没有写任何P1程序”。

### 必需执行资产 B：原机制合同下的独立P1证据consumer

复用原P1物理calibration/outcome入口及`d2p_stage_admission.p2_admission`，
需要完整绑定上游P0、真实执行根、参数不变、合法动作、原模型eligibility、实际
非贪心/未来值改善证据和预声明控制。原必需controls为
`zero_prefix_two_stage_d2p`、`greedy_two_stage_d2p`、`cem_full_horizon`，
并保留原已声明的额外controls；各独立paired CI须严格有利，触零不通过。
原机制gate的选择、最小eligible根数及原stage-specific步数由原协议决定，本文
不替换成SRA校准的H2或四根门槛。不能将模型`pass`或新SRA物理完成冒充它。

新机制/任务合同若由main明确另行授权，可以新版本实现并独立验证，但必须保留
旧fail及原claim边界，不追认旧P1。本文不打开任何新保留outcome。

## 4. 缺可归因的动作支持与选择目标消融合同

### 必需执行资产 C：共同候选资格/目标的预声明配置及消融driver

必须把组合干预与以下差异拆开，而不是仅共享raw box与goal即认为公平：

- bank composition只用16个单个raw二维命令重复5次的primitive；full20 CEM/random
  和SRA允许包内时变20维动作。需要预声明共同支持或明确的因子化消融，不把
  静态bank限制的变化解释为组合收益。
- full-packet两阶段编译器选择`stage_cost + own_reach_value`，其中含有界线性ridge
  惩罚；SRA v2用完整H2非线性原Q终点成本选40候选。缺“只改组合/重编译、其余
  固定”的主效应消融。可以保留原方法目标并用单独目标消融归因，不能事后把
  nonlinear diagnostic换成正式selector来追认旧结果。
- native每packet球cap=`sqrt(5)*.75`；SRA v2是联合球cap=`sqrt(10)*.75`。
  两个per-packet球的乘积虽与联合球有相同最大全轨迹L2范数，却不是同一可行集：
  联合球允许能量集中在单包，native per-packet约束不允许。需要明确记录这项
  资格差异，不称proposal分布完全相同。

共同配置必须固定dtype/time-major mapping、原normalizer、root/goal/context、
Q投影（不能将保存的Q误开方一次）、候选ID/顺序、动作步数、种子、停止规则。
不重新挑已用四根或事后选择最有利的原Q/pose/official指标。

## 5. 缺完整候选/来源适配器，旧严格schema不能直接接现有report

### 必需执行资产 D：全候选receipt writer及真实来源保留的adapter

直接可复现的三处不兼容：

1. **候选覆盖不相同。** bank composition的80展开＝16个H1＋64个H2终点；
   flat的40完整H2也恰好80predict。原严格校验器要求每root/arm恰好相同slot数，
   `candidate_evaluations`等于slots总数。H1只能作为中间展开账本，不能充当H2
   slot。不能删24个已评估终点、补24个未评估候选或把winner当唯一评估来过合同。
2. **不是每个候选都有receipt。** bank attempts只含`completed/cost/path`，
   每arm只保存一个winner witness；SRA40候选含命令和成本，但也只有winner的
   完整context/witness。它们不是`rows[].candidates[]`所需的全槽位receipt/edges。
   缺非winner中间/终点context不能仅根据path/cost离线伪造，需要新运行时保存或
   被明确计费的新验证执行；已有winner收据保持原样。
3. **原strict schema使用单一全局provenance。** `_check_candidate`要求每个case/
   arm的六项provenance与同一`manifest.provenance`完全一致；但实际四根有四个
   metric SHA，跨方法的source/protocol/bank SHA也各自真实不同。即使按root拆
   manifest，跨arm来源仍不等。不能篡改真实receipt的这些SHA使之看起来共同来源。

故需要新外层comparison-run manifest/adapter：共同模型、每case原metric/root与
每arm实际source/protocol/bank分别绑定，保留原receipt及其摘要；跨arm相同科学
输入由显式映射验证，而不是改写成相同六项SHA。新的schema应保留原严格合同的
完整覆盖、无结果筛选和全预算要求，新增错接根/指标/来源、漏槽/重复槽、假共源
及winner-only反例测试。**本文没有实现该新科学协议或放宽旧校验器。**

现有admission审计也不支持新SRA schema。这是一个具体的consumer缺口，不是
要求把旧审计器中的`P2_admission`改true。

## 6. 总资源和全槽位物理覆盖的可执行缺口

### 必需执行资产 E：组合/控制共有的资源合同与真实instrumentation

本节仅核算P2执行前提，不计算质量—成本frontier或做P3工作。

- 旧bank每arm/root为80+4。新SRA为nominal2＋Jacforward2＋candidate80＋replay4
  ＝88/root，四根352；不能把4次overhead忽略后称84 matched。
- native编译器每根共享23predict：nominal1＋root/full Jacobian forward1＋constant
  chain-rule Jacobian forward1＋FD20；两arm各10候选，每候选planning4＋replay2。
  合计`4*(23+2*10*6)=572`，与实际phase records一致。不是84或88/root。
- native源码每根2个root/chain Jacobian＋20个suffix Jacobian，四根88；旧report
  字段`jacobian_backwards=88`不是内部backward API/engine次数。数学输出方向为
  16896。源码推算显式SVD96，`matrix_rank`诊断另引入8个NumPy SVD，合计104；
  bounded ridge88。这是源码算术，不是已有instrumentation实测的kernel/FLOPs。

需要main冻结的新总资源分配及logger覆盖所有编译、验证、encoding、Jacobian、
VJP方向、SVD（含rank诊断）、solve、候选/selector、失败/blocked调用、upstream
构建、物理步数和共同wall cap。每种资源的actual/requested/completed要分清，
未知成本不能填零。不能靠空跑/伪账目“补齐”方法成本。原strict budget键集是
精确集合，直接增加`svd_calls`等字段同样会被拒绝；额外instrumentation需在新
协议/sidecar中明确消费并保留旧合同，不静默删去超额开销。

### 必需执行资产 F：按冻结slot执行/编码的物理consumer

原strict合同一旦提供physical observations，就要求覆盖每个case/arm/slot，
绑定exact raw动作、root/goal、receipt SHA、metric SHA及真实执行产物SHA。
Poincare现有SRA consumer已经完成四条winner，不是160候选全执行；旧bank
consumer也是已选八条。native有全80候选物理原型，不应重写，却尚有3条观察
中止及`encodings_complete=false`，不能当作全slot可比编码完成。

需要独立的新全slot执行manifest和durable消费层，完整保留失败/中止，按预声明
规则标记不可用端点而不是回填terminal或换根。若所有slot为40个H2、四根两arm，
全部完整执行的条件算术是`40*10*4*2=3200`物理步，不是两个winner的80步。
这只是成本量级例子，不是本审计授权的新运行、规定40个slots或新门槛。
失败处置若原strict schema无法表达，应先另行批准科学合同，不改旧gate掩盖缺失。

## 7. 独立组合机制/收益验证还缺哪些封存输入

### 必需执行资产 G：预outcome独立cohort与机制/分析manifest

四个历史根是实现/calibration队列。新SRA方法在其新outcome前冻结不等于这些根
重新成为独立outcome。需要外部保留的pre-outcome digest，绑定新的root/goal/
context身份、方法/全部slots、原任务metric、stopping/失败处置、控制集合及
root配对不确定性/多重比较分析。候选不能作为独立样本；按稳定root身份配对，
不能按报告数组顺序或只保留双方完成的有利子集作事后推断。

机制记录至少应区分真实非贪心首段、随状态变化的局部可达性预测和后续实际执行
收益，并给出不重编译/同bank greedy/无组合谱等预声明消融的对应执行绑定。
单条多edge、模型预测Q下降、交换顺序有差异或SRA物理成功，都不能单独证明
组合重配置的独立贡献。本文只列需要冻结的证据接口；数值比较由Poincare负责，
不重做其分析，也不选择新有利阈值。

## 8. 可复验审计入口与边界

cwd `/home/gbk/code/LeWorldModel/le-wm-main`：

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m experiments.spectral_reachability.p2_composition_prerequisites_20261027_v1
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m pytest -q -p no:cacheprovider tests/test_p2_composition_prerequisites_20261027_v1.py
```

新模块只读明确allowlist，输出stdout，不新建results，不执行任何旧代码或模型。
记录报告schema/status/gates、coverage、schedule算术和输入/当前源码SHA；缺文件
记missing。退出0表示inventory生成，不是P2 pass。它**不是全库不存在性证明，
也不独立认证历史执行或重新验证所有source seals**。源码推算与报告元数据分开
命名，防止将静态算术变成实测算力。

本轮4项测试通过（0.22秒）：bank终点覆盖算术、native编译含共享FD/replay的
算术、SRA overhead/全slot物理量级，以及禁止导入neural runtime的实际资产
inventory。没有新真实模型或物理实验；没有写入旧资产。

本轮关键报告字节摘要，便于main/Poincare引用同一证据而不重跑比较：

| 报告 | SHA256 |
| --- | --- |
| bank composition planning | `75dc6df37b5c0d64ec74655969da0d5bd35da769469fc21719fb85c44b9e6ac8` |
| bank selected physical | `9c0693a1f58a77eaa281fe155515802ef6b7eb93e38986ef71a9d3e4afcf35fd` |
| native full-packet planning | `05eb37c9b1fc3fc4976da702239b13dce018eb9bb4067660687ada4e6d3988dc` |
| native full-packet physical | `0e308b3f64c01de4ac0a6b1937a1ab293ec28c7810320bd200f637bb450a7a31` |
| SRA v2 planning | `522afa5a68b23a6ff829c5c6d820d858953dccc0f778548e747ba594ff73937a` |
| SRA v2 selected physical | `20554d967c49eec3e93e7ee846772d48be4967002d0d609cd70a8479678bcfd5` |
| P1 model calibration v5 | `99cbb6b6dc4653eb629427353f6f6d05022c219333f3f7362cf4ca43a3076bc2` |
| historical P1 physical outcome | `f67d22b56e8f8bc3f41f4890ccea40ddf0c24f8735f1ea7713dd6d6d5badfe5e` |

这些是本轮观察到的bytes，不证明其冻结时间。后续新资产必须使用新路径和明确
新来源，不能覆盖这些report、重新seal旧prepared或用新runner追认旧gate。
