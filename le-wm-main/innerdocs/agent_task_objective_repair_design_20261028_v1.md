# 下一任务目标修复：同物理目标的双 renderer 原型与完整终态优化

日期：2026-10-28。状态：**可实施设计，未实现新 runner、未运行实验、未验证效果**。本轮仅新增本文；control RGB 和其它 source/tests/doc/results/gate 全部保持冻结。本文不是原 P1 的补签、不是唯一根因结论，也不是训练/拟合 Q 的计划。

## 1. 决策与边界

最小建议不是再拟合 latent→物理 readout，也不是把 P5 的 patch 权重塞进 P1 的 CLS 预测器，而是复用已有有限 goal-set 距离：**只用已认证同一完整物理目标的 off/on 两份 RGB 编码，构造两个原生192维目标原型；用到集合的最小完整终态 MSE 接现有动作优化器。** 不移动 agent、不改目标 pose、不加/删速度坐标、不编辑候选图像、不重编码当前目标。

这是一项新的目标表示/损失假设，不是新的基础模型。两份原型都来自原生同 state 渲染，并非几何等价的推断。它比移动 goal-agent 的目标集合更保留原任务语义，但依然改变了原合同的“固定 condition 对应单一 goal192”的 planner loss，**须 MAIN 另立新协议和独立输出，不能沿用旧算法身份，不能据新 loss 放宽旧 P1 gate**。原 native success、native7、jointXY、wrapped angle 全部继续原样评价。

问题不是已经确定“renderer 造成全部错配”。这一步检验的是一个窄假设：在当前同任务固定库内，允许目标显示模式的不确定性，是否改变选择，并使预测目标进展与实际任务进展更一致。它不能修复 encoder 丢失信息、未观测速度、动作条件预测误差、接触/长时误差，也不保证角度收益。

## 2. 已有路径核对：哪些复用，哪些不能搬用

下列路径相对 `/home/gbk/code/LeWorldModel/le-wm-main`，本轮只读源码/文档/已有 JSON；未调用其实验入口。

| 已有路径 | 已有能力与本设计处理 |
| --- | --- |
| `experiments/spectral_reachability/pusht_goal_set_metric.py:distances` | 已有 `min_g mean((z-g)^2)` 和稳定 catalog 顺序 tie；复用其算术语义，不宣称新算法。其 `catalog_images` 只搬纯色 agent，不能用于本设计造目标。 |
| `pusht_clean_goal_sprite_catalog.py:build`、`run_pusht_clean_goal_catalog_cuda.py` | 已有完整 sprite/白背景 eligibility 与 goal-set 编码；历史 report 已有 construction_completed、916 encoder samples/60 calls、selection_changed=false。不能重复编码后宣称首次提出，也不能把历史四根目标移植到本批四槽。移动 agent 会改变本任务 agent 目标，故不作为本设计生产目标。 |
| `p3p4_goal_patch_metric_20261027_v1.py`、`p3_goal_patch_consistent_compiler_20261027_v2.py` | 已有 goal-cell 消融→native192 span projector，以及 Q 一致 residual/J/ridge/terminal 编译。不是未实现的“修复空白”；本设计不重拟合/重编这些 Q，不用其旧任务目标代替当前目标。 |
| `pusht_goal_object_tangent_metric.py`、`pusht_scene_nuisance_metric.py`、`p1_goal_orientation_contrast_20261027_v1.py` | 有固定颜色位移/旋转/场景 nuisance 与完整 raster orientation 准备。颜色子集不是完整物体，旋转采样不是物理旋转认证；不把已有诊断重命名为新控制算法。 |
| `agent_p5_taskmetric_patch_bridge_20261028_v1.py`、`agent_p5_spatial_patch_attribution_20261028_v1.py`、`main_goal_area_weights_20261028_v1.py` | P5 有全256 patch诊断/goal-only面积权重；P5 feature为256×384，P1真实未来预测为projected CLS192。原图encoder的完整257×192 hidden不意味着predictor能预测这些patch。禁止 patch均值对比预测CLS、把CLS projector逐patch用、无训练逆解/伪造decoder。 |
| `agent_p1_renderer_matched_goal_20261028_v1.py` | 已验证当前两根同目标 off/on RGB、原encoder输入/257hidden/CLS/projector/192桥接与off/on/off同state恢复；诊断为actual20→25固定80对，不是旧teacher15的76/4。这些目标原型可以复用；本设计不是重复该固定动作renderer×goal事后分解，而是给新选择/动作优化 loss。 |
| `agent_p1_terminal_witness_selector_20261028_v1.py`、`agent_mixedmode_prefix_20261028_v1.py` | 全 allmode42×4 + mixed222×4 已有完整suffix预测、原commands/context/replay；无需新预测即可先比较新loss选择。mixed每prefix的原nominal/J还可给第二目标分支产生真正新suffix提案。 |

历史 clean catalog report 的当前文件 SHA：`c77b6bed64d7633faf6c2035ee0e93fc4c51792a11218c7124489f69dfaf6b20`。只据读取的元数据确认已有路径，不声称本轮重建了该历史完整认证链。

## 3. 当前依据及其有限含义

本轮核文件 SHA：

- control RGB report：`results/agent_terminal_witness_control_rgb_20261028_v1_MAIN01/report.json`，`67904a04ae37c7bb3e9f1216c193b47cb7f93161dbffe23cf0ca9e3ecaa8e9b4`。
- control full strict：`results/agent_terminal_witness_control_rgb_20261028_v1_consume_MAIN01.json`，`4f40159b636dc5252ce2c24a6be29f2374c7850b3723f81e8b062f745b8ebec8`。注意4f40159b是receipt，不是CUDA report SHA。
- 原NPY独立配对：`results/main_terminal_witness_rgb_contrast_20261028_v1_MAIN01.json`，`b740fe3e086d8e49c6bb4b0c7dbd3016d1f5761194060bba8991384bb022c67d`。

该配对保留全部4槽、2known、0fresh；三个changed动作的predicted与actual原Q均下降，jointXY两损害、wrapped angle三损害；相同动作一槽predicted/actual latent与state exact相等。其含义是：**这个固定对比中的物理不一致不能全部归为预测Q排序反转**。不推出普遍encoder物理不对齐，不推出agent/renderer为唯一原因，也不证明新目标集合会有效。该独立配对标明未自行重建全部上游；完整消费证据由MAIN各自strict提供，本设计不重新签发它们。

## 4. 数据前置：无需新目标RGB，但必须做本批精确join

目标authority直接取已有renderer-matched链，不能只按root_slot猜配对：

1. `results/p1_moving_target_overlay_source_20261027_v1_main01/report.json`，SHA `85ca1df3de910628bfd6c5bd1a0a482f61880a2ee76bfd404732a77dd8c7666a`，对应 `s{0,1}_{off,on}_blind_inputs.npz` 的goal_rgb。
2. `results/p1_moving_overlay_recompile_cuda_20261027_v1_cuda_MAIN01/report.json`，SHA `2b2e5064f74f8aca712ae0d0415fcbfb64a2c8e6325bdffe24ae4fdc3b373381`，image索引3/7/11/15为root0 off/on、root1 off/on原目标编码。
3. 复用 `agent_p1_renderer_matched_goal_20261028_v1.py` 的 `validate_goal_triplet` 及目标图像/数组验证逻辑，**只复用目标authority验证，不伪造其40-action/80-pair schema**。off/on/off前后完整snapshot/state bits相同、off恢复、像素→原preprocessing→257hidden→CLS→projector→192全部闭合才成立。
4. 对每个当前allmode/mixed slot，要求其原condition对应goal192与该来源编码原始float32 bits相同，task/blind/context/renderer/goal physical fingerprint均能逐项绑定。不同condition的context不互相替换；只能为同一根的目标原型建立集合。normalizer、model weights/tensor/config、投影坐标必须同源。

两种pose不可混淆：constructed physical goal是source终态完整state7；on图保留的是原环境默认outline pose，并非自动等于constructed goal pose。上述同state证明来自native journal，不从outline反推goal。该链不能满足任一当前slot时，**该槽blocked且仍保留4分母，禁止借旧四根catalog或补画/相近goal继续**。本设计查到已有充分目标资产，不新增目标编码请求；完整生产join验证仍是未来实现/prepare职责。

## 5. 精确目标ABI与可证明性质

以下是待实现接口规格，不是当前已安装的CLI/API：

```python
GoalSet192 = {
    'schema': 'task_objective_renderer_goalset192.v1',
    'ids': ['off', 'on'],                 # 永远此顺序，重复也不去重
    'vectors': float32[2,192],           # 原编码NPY bits，不从JSON低精度重建
    'task_sha256': str,
    'physical_goal_fingerprint': str,    # authority/provenance，不进模型
    'rgb_refs': [Artifact, Artifact],
    'latent_refs': [Artifact, Artifact],
    'same_state_authority': Artifact,
    'model_coordinate_sha256': str,
}
score_np(z: float32[...,192], goalset) -> {
    'branch_Q': float64[...,2], 'value': float64[...],
    'active_branch': int[...], 'original_condition_Q': float64[...]
}
loss_cuda(z: CUDA_float32[...,192], goals: CUDA_float32[2,192]) -> tensor[...]
```

固定 `D_r(z)=sum_d (float64(z_d)-float64(g_rd))**2/192`，`L(z)=min(D_off(z),D_on(z))`。`score_np`沿192维用原float64 reduction，除192一次；严格tie取off，无epsilon/温度/softmin/beta/标准化/权重/拟合。`loss_cuda`在同CUDA设备上显式转float64再算相同公式，模型forward仍float32；不得detach z，goal常量detach。branch选择也采用off优先的显式比较；不把torch默认tie梯度平分混入协议。跨backend末位reduction误差单列，不改科学比较阈值或隐去sign差；封存选择以规定的saved `score_np` 为准。

在唯一active分支，`grad_z L=2(z-g_active)/192`，通过原冻结predictor的动作计算图可以回传；tie处不可微，选off分支只是一项规定的分支导数，不声称一定是下降方向。更稳的实现是分别优化两个光滑 `D_r`，把所有分支结果以完整非线性L重评。无须patch predictor、decoder、新训练head、物理state或可学习Q。

若动作域A与目标集合无关，则 `min_{a∈A} min_r D_r(fhat(a)) = min_r min_{a∈A} D_r(fhat(a))`（将有限r与a的联合下确界交换；存在极小值时取min）。这是目标分支可枚举的代数事实，**不是说局部优化器找到global optimum**。若保留全部旧动作，则新有限库的最小L不会高于旧库同L最小值；也不保证原单goalQ或真实收益单调。

仅对两个编码原型本身可保证L(g_off)=L(g_on)=0；不能据此称“renderer不变表示”。一般同物理state两张终态图的编码z_off/z_on到集合距离仍可能不同；encoder碰撞或目标附近非任务方向也可使错误物理状态得低L。这些是本试验可失败的条件，不是后验追加修复理由。

## 6. 最小落地分两步：先无需CUDA的选择，再真正suffix优化

### A. 首次实现只需saved selector接缝

读原selector已绑定的allmode42×4、mixed222×4全1056候选，每条取同raw10完整suffix终态192；计算两个branch_Q和L。原stage+ridge winner、原single-goal nonlinear winner、新goalset terminal winner三栏严格分开。allmode/mixed独立panel，不混成新union挑winner。每panel按 `(L, candidate_index)`；active goal按off/on顺序，不能用物理标签破tie。前缀的“model uphill”同时报告原single-goal定义和新L定义，但两者均不是物理非贪心证据。

输出每个panel的完整candidate_count、task/context/raw/goalset SHA、两个branch_Q、原分数、新分数和选定raw10。缺预测必须列literal forward请求而非ridge代用；本批已知完整预测，不应新增神经调用。动作选定/原始输出seal完成前不加载任何native/RGB endpoint评价字段。上游混合report若含native metadata只供prepare认证，不将其传给selector纯函数。

这是**目标loss重评分、不是重优化库生成**；原库由旧目标生成，可能根本没有新目标的有益动作。A的负结果不能证明B没有可能，A的正结果也不能倒签新目标协议。首次交付A即可实际跑，不必等待新CUDA optimizer。

### B. 最小真正action优化：两目标分支的有界suffix提案

不另造优化器大工程：复用 `agent_mixedmode_prefix_20261028_v1.compile_arm` 已封存222个prefix及每个自己的 nominal/J，与 `pusht_full_packet_compiler_20261027.bounded_ridge`。本设计仅给新模块接口，禁止修改这些冻结文件或替换globals。

```python
propose_suffix(after_context, nominal192, J192x10, target192) -> {
    'params64': float64[10], 'suffix_raw': float32[5,2],
    'linear_diagnostics': dict, 'context_sha256': str,
    'target_id': str, 'normalizer_sha256': str,
}
select_terminal(programs_with_actual_model_endpoints, goalset) -> sealed_raw10
```

每槽固定222 prefix，每prefix保留两条完整10动作：condition原goal分支保留旧suffix原字节及旧终态见证；alternate-goal分支按 `bounded_ridge(J, g_alt-nominal, damping=.002, cap=.75*sqrt(5))` 求提案。Q仍I192；原ridge未除192的优化约定保持。params只做一次NumPy float32 cast+tanh，固定time-major5×2；prefix不变，公共normalization不改、无额外/100。相同raw提案仍各自保留计费，不按收益去重。

J必须来自**该prefix、该原context、该zero suffix logit nominal**的认证见证；不能借另一个condition/teacher20/J。复用J用于第二目标是同一点线性化的合法数组操作，不是新AD已完成的说法。模型输入仅原完整latent/action histories、RGB编码、raw actions及公开goal；完整state与physical结果不进提案。

新增suffix从已封存完整after_context真实预测，再从同原initial context独立完整10动作replay，要求完整3帧latent/两段action history及终态exact相等。**linear_terminal_cost/own_reach_value仅诊断，不用于选择**。每槽444完整program，合计1776；用真实非线性预测L排序，保留原888程序保证有限库包含关系。stable key `(L,prefix_index,branch_order)`，branch_order固定original_condition在先、alternate在后；它与目标active tie顺序是两回事。winner seal先于物理，随后whole+segmented独立double replay；不相等即blocked，不按效果换winner。

这是固定prefix-bank的suffix优化消融，不是完整新metric-consistent spectral prefix生成，不宣称穷尽non-greedy机制。若MAIN现有非线性动作优化器接 `loss_cuda`，应在新协议里另冻结初始controls/迭代/步长/调用上限，分别保留两分支和incumbent；本文**不授权新增调参/多起点搜索**，也不把本次ridge提案称精确非线性最优。

## 7. 数据与资源合同（未来执行上限，不是本轮消耗）

本轮：只读核对与写本文，新增neural/physics/AD/SVD/训练/拟合均0。

- 目标资产：复用4张原goal RGB/4个原生编码；新encoder/projector=0。不重复历史916样本，不把重用记成免费历史研究。
- A：1056 candidate slots，2112个192维goal分支MSE；全四槽分别保留allmode/mixed分母。新predict/AD/physics=0；原consume的saved-array/SVD/hash/I/O另计，不宣称零总CPU。
- B仅mixed：888个alternate suffix ridge，各1 saved-array SVD；原branch不重算SVD以免改变字节。888个新suffix各1次packet forward + 2次独立whole replay packet forward＝2664次；4个winner whole+segmented各4次packet forward＝16次；**总2680次新predict、action_encoder、predictor、pred_proj各自attempt cap**，batch1/每次3个context tokens。AD/JVP/VJP/new Jacobian/encoder/projector/physics=0。backend若一次packet有更多内部调用，prepare必须按实际ABI分列修正账目并在启动前重新冻结，不能运行后隐去超支。
- B全部goal MSE评估的正式计数：1776终态×2＋4winner重复验证×2＝3560；prefix诊断额外4×222×2＝1776；initial/zero/greedy若重用已算值须列cache计账，不称FLOPs匹配。
- 保守新B输出cap 512MiB（完整predict histories/J引用/raw/独立replay/事件）；final reserve16MiB，disk floor1GiB，wall7200s。A输出cap32MiB、wall300s、floor200MiB。都是资源上限而非科学阈值；prepare先核序列化估计，不足则停止，不删候选/降精度。目标与预测数组保留原NPY、compressed bytes和array bytes双SHA；model前后weights/tensor/config/normalizer身份一致。
- durable ledger按attempt→returned落盘，记录image/predict/actionencoder/predproj/arraySVD实际次数、batch/context tokens、CPU读写字节及CUDA时长/peak memory；失败不退款、不自动重跑。FLOPs未知。借历史root forward/J/物理的资源全部单列，不把predict API计数当FLOPs或收益。

模型阶段无新physics。未来物理评价最多4个新winner full10加原frozen reset/warmup；8段context warmup等细节必须由MAIN根据真实prepare upfront计账，不能仅写40 public steps冒充总physics。已有相同task/context/raw10物理可按原whole-row认证继承，非相同动作必须新执行；不拼接旧prefix和新suffix，不重标fresh。

## 8. 可测试假设、拒绝路径与决定规则

先固定算法再看新分数，不扫goal集合、beta、角度/颜色阈值，不按当前损害槽挑根。已看过的四槽只算calibration，不能声称prospective efficacy。以下检验全部保留分母、ties和unknown，不以某项改善作为新science gate。

1. **ABI/实现**：singleton goalset精确退化原single-goal MSE；重复两个goal不改分数；已认证相同raw/context的control终态exact一致；未知目标/错坐标/dtype/NaN/时间/context/raw bits篡改必须拒绝。CPU/CUDA reduction与tie分支单独测试，不通过就停止，不解释成科学失败。
2. **目标来源**：off/on/off state/snapshot/PNG/preprocess/latent和当前slot goal字节闭合。错root、on图换另root、默认outline冒充constructed goal、损坏NPY都必须拒绝。这是实现完整性前置，不是疗效证据。
3. **选择干预是否发生**：A逐槽报告三种winner及active branch、原/new loss和model uphill。全不变是“固定库本干预无选择first-stage”，不是新算法成功；仍输出全1056，不加第三goal挽救。
4. **真实动作优化是否发生**：B全部888个新suffix保留，报告raw_changed、所有完整终态L与incumbent差，不以只保留最好若干提案省账。全部未改善是已固定局部提案预算的负结果，不是全局不可能。exact replay失败属于接线/执行阻塞，不与loss失败混为一谈。
5. **目标表示 vs 预测误差**：选定后对同动作actual RGB用原encoder算actual192，按同两个目标复算actual L；同时保留两种原single-goalQ。若预测L下降、actual L不降，提示动作条件预测/协议问题仍在；若actual L下降而jointXY/angle/native success不改善，直接否定“这项新目标在该对比足以带来任务收益”，不能再只怪模型预测；若两层都改善，也仅是2known上的有限结果。
6. **任务评价**：保留原native7含目标非零velocity、agent/object XY各轴、jointXY、wrapped angle、原success/reward/termination。新目标不获得velocity oracle，不把native7改善替代success，也不把object-only改善称全任务收益。原图观测不能唯一识别的目标动态分量仍是明确未知量。

finite-set的模型优势没有到物理优势的定理：两个物理不同状态可能编码碰撞；min目标会吸引到不应得低分的分支；当前context只按一种renderer训练/预测，跨renderer原型可能分布失配。另有contact、bounded动作、长时累积等独立因素。出现这些结果不在本协议内追加mask/Q/颜色校正，记录后另提假设。

## 9. 交付接缝与下一最小授权

本交付只有设计文档，没有伪造可执行新CLI。下一source实现只需A的pure `score_np/select_terminal`、原goal authority importer、prepare/analyze/verify，复用真实selector schema入口与公开task数据；不需要新GPU，更不需要先做新物理或全项目认证。A代码审读通过即可实际执行并冻结选择。B是在同一goalset协议下独立run/consume扩展，复用真实model forward/normalizer/低层Store和strict event验证；禁止改旧producer常量或globals来模拟新schema。

MAIN若接受此路径，应先批准“固定单goal→同state双renderer goalset”的新算法协议、全四槽和固定分母、本文A/B资源上限，再实现/运行；**本次用户只要求文档，不构成该实验执行授权**。既有P1 freshness、真实物理non-greedy、强基线统计门全部保留；若未来新目标有效，也要另行满足原研究目标而非重命名gate。fresh reserved本阶段完全不触碰。

本设计受experiment-design技能的最小阶段化原则影响：A先验证目标loss与选择接口，B再付出真正动作优化预算；不执行该技能中不适用于本任务的调参/训练/按最佳结果扩展建议。本文不是审计报告，不替MAIN优化器和完整审计工作。
