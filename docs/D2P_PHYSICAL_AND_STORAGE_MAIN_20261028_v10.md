# D2P 全40真实响应与无损存储修复（MAIN，2026-10-28）

本轮继续原始C1–C3、三组实验及强基线/独立确认，不将实现和测试改称研究闭环。不训练、不拟合新Q/adapter、不改原成功阈值；神经只用HOST CUDA。旧v1失败与所有旧结果保留。

## 1. 共同动作库全40支已真实执行，但完整RGB准入仍关闭

MAIN完整审读647行物理adapter及文档，独立RAM回归49 passed / 49.07秒。首轮3个失败均为fixture错误：zero→zero没有实际替换；capture层先拒绝时后层不得虚构abstain；FakeIO缺少fileno。只修测试，生产动作校验、fsync、守卫和预算不改。

新prepare MAIN02 SHA `b125d28c4d931dca7661233d5a81e0da890ef53e2a55ceef55172134d0b08f5a`，旧MAIN01保留不用。实际HOST physics 29.06519秒：40construct/close、80reset、800public control、8160内部physics、160set_state、120goal_setter、80goal_render、40goal_bind，2638显式render。前15步身份全核；全40真实执行到suffix5，没有换动作、补步或重放退款。

报告 `le-wm-main/results/agent_p1_teacher_cross_library_physical_20261028_v1_MAIN02/report.json` SHA `464656755bb83cec14b7acdd457641592b11be6d335f285166687df15368bc01`。39支horizon_completed；index12（root0/on、Z、-uZ）最后off帧触发原 `fresh RGB/canvas guard`，所以全批partial。该支agent中心为 `[388.785817139554,514.7195833901377]`，y超出512画布。保存PNG仍有8个canonical蓝色边缘像素；**不是颜色完全消失**，不能将肉眼初看“无推杆”当像素事实。不得放宽canvas守卫、补图或拿state冒充缺失on图。

saved strict已真实核完整文件/事件/动作/PNG/NPZ并保存partial回执，SHA `44acd71c60c8edc7fbfa196c2f46f7d8692ef487ec0c3a01b6b8f7e5071e9212`，状态 `partial_teacher_cross_library40_preserved_not_admitted`；CLI exit1是partial状态，不是回执缺失。全40 RGB release关闭，failed endpoint20仍null；实际last_return单独保留。

## 2. MAIN独立原生状态、接触标签与同后缀响应复算

新独立审计不导入producer/env/model；重核报告/partial回执、完整journal hashchain、800raw returns及全部literal raw10，按原native成功公式复算。结果 `le-wm-main/results/audit_teacher_cross_physical_MAIN_20261028_v1_MAIN02.json` SHA `3dfc301c40471e0620caf8660baf4d8ad6534f25055e546aadc7466f61e79b78`。

- 40支实际last_return均为suffix5，原native成功0/40。这里是已返回真实状态的诊断，不补造失败支的完整RGB终态。
- 31支在root之后的某控制步改变了物体pose；800返回控制步中161个 `info.n_contacts` 标签非零。标签不是实际callback数，不与P5的96callback混用。
- 因此不能把该批失败解释为“全部没有接触或物体不动”；也不能把响应存在解释为非贪心任务成功。

另对完整五项共同库逐字节核Z/S相同suffix，分别减去各role同elapsed zero末态，再比较响应；不挑赢家。`le-wm-main/results/compare_teacher_cross_responses_MAIN_20261028_v1_MAIN02.json` SHA `006838fadb0cae0347c461890e9d6e574ff2d43a77c264828d0f2316602f5960`。例如task0的同uS相对zero物体XY响应范数Z为4.82493、S为38.30617；task1为0、34.01603。所有五项/四task保留，含负结果/观测失败。不隔离state/velocity/contact中介，S仍stage-greedy、仅两known根，故不能升格为C2非贪心机制或泛化。

## 3. P3 v2：只修无损存储，真实CUDA重跑已启动

MAIN全读681行v2及文档。单gzip member、逐事件sync-flush/fsync，原canonical JSON事件/原精度数组不变；strict流式解压，一次核一个child全部trace，原SVD/choose/queue/paid ledger检查不删。总512MiB、final reserve256MiB、900s、原46 acquisitions/48 method分母不变。

正确经验journal预测为 `2*(46*564813+23*1200584+1123)+16MiB = 123969122 bytes`，report231664974，合计355634096；只是两倍观测最大值加余量，不是熵上界，实际cap仍failclosed。MAIN一度手算误将测试改成123970322，发现后恢复正确测试，未更改源码公式/预算以迎合错误期望；保留旧prepare MAIN01不用，重新prepare MAIN02。

最终MAIN独立RAM回归67 passed / 34.93秒；source/test/doc SHA为 `1a7ea154b09c0caf59b5bc4c4d8b4b333493aaac4b6dc1a1afe1363f9ecc2670`、`9814e24cafd2be4ccf60d39d7b4721fee2f1225c228d004d8e6ec57676634487`、`fcb962c30b9e26bcb41588668113003c8c942cd0c719a6f7fe984f7116f05786`。实际prepare `le-wm-main/results/agent_p3_child_signal_cuda_20261028_v2_prepare_MAIN02.json` SHA `242dc62b1817bc3925cd836740cc9eaf60c8f94644e327b494fb834eb98ec298`。

HOST CUDA作业实际启动，session54058/PID759666，输出 `le-wm-main/results/agent_p3_child_signal_cuda_20261028_v2_cuda_MAIN02`。截至本段记录已核进程存活，gzip33MiB；**未以运行中视为完成或科学通过**。终态及完整strict仍待本页追加，不重启超时观察中的同一作业。v1完整失败报告与267806913-byte旧journal封存。

**后续实际终态覆盖上述运行中快照**：该CUDA作业已exit0、236.18337秒，全部46 acquisitions成功，368predict/92J/17664 scalar VJP、grad及raw callbacks/92array SVD/46schedule；errors=[]、cleanup_ok/sources_verified=true、模型未变。报告SHA `cccb96d7d57dcf041021f8c43ab1f027a1012732ec56068ef1fb85348829bf5a`。原始输出约136MiB（gzip46MiB+report90MiB），原512MiB预算未提高。

首次完整strict已返回到CLI写出阶段，但通用 `write_new` 的64MiB metadata cap不能容纳约90MiB全rows严格回执，故未落盘；这是第二个明确的**消费端工程容量问题**，不是信号或科学失败。新MAIN薄wrapper调用原consume全量校验、不改validator/rows，使用独立saved-receipt 256MiB上限及200MiB floor持久化完整返回，原CUDA输出cap不变。wrapper的完整保全/不覆盖/cap拒绝/symlink拒绝3项RAM测试通过（.19秒）；当前重新全量strict消费session54884，回执未落盘前不宣称完成。

**完整strict及落盘现已完成**：上述session54884 exit0，完整94,023,655-byte回执 `le-wm-main/results/consume_child_signal_MAIN_20261028_v1_MAIN02.json` SHA `efccb0ad57ff76e126e7027e9518b1c24b092c806a71a0ed55dcd96fa7ac90b5`，状态 `paid_child_arrays_verified_not_admitted`。原native数组/192方向/J行、全部队列/choose/SVD、事件链和全48方法分母均经原validator核验；不是RAM或部分panel通过。两种存储工程缺陷均已取得真实终态修复证据，仍不构成物理收益/C3成立。真实回执已交给独占child compiler实现。

实际report固定决策为：fixed 23项H1×4 pending；adaptive 18项H3×4 pending、5项H3×8因48>24整节点拒绝；另1child/2method not_reached。这说明真实调度确已产生不同深度，但不是收益证明。compiler必须保41pending/5refused/2unknown全48；fixed后续实际8search预测与adaptive24必须分列，不能将相同上限说成相同实际总算力。

## 4. P2继续区分“预测到真实Q”与“真实Q到物理目标”

上一轮全8替换actual z5的CUDA/strict/独立NPY审计已完成（v9）。新330行lineage分析与文档已由MAIN全读，saved-only独立测试与实际落盘进行中，不重复跑神经。

新实现按身份而非列表位置连接旧plans；明确older2来自初始context slots1/2（真实warmup10/15的编码路径），不是两个imagined rollout状态。逐帧原native encoder回执不足仍标unknown，不能由源码推导“表征已正确”。原动作history/goal/Q保持不变。全4根actual terminal Q与jointXY偏好均不同的描述待MAIN真实分析输出确认；无训练或事后Q选择。

**MAIN实际验证已完成**：41项独立测试通过（5.62秒），saved分析输出 `le-wm-main/results/agent_p2_midpoint_history_lineage_20261028_v1_MAIN01.json` SHA `c8d136c8c9125bbfb665f1ddf3da6ef40384c16c14c94a2d58a3b4043036fb10`。确认全4根 actual terminal Q 与 jointXY 偏好不同；teacher-suffix同Q预测一致仍为2/4。新增神经/physics均0。后续精确真实RGB triplet资产inventory已派发，不能凭旧context来源推导逐帧编码执行已独立证明，也不能默认older2 imagined而错误归因。

## 4.1 P5公开原生context四程序已接上主机CUDA

MAIN全读465行新source与文档，55项RAM测试通过（1.11秒），source/test/doc SHA为 `bd5b85b308fcb621adda1f4452911f58e233b9ba0812e8d569d0dbd540940930`、`bf11d51657c1a80413b5cd842c010e955195dacc65b942a662be692d64b5965d`、`38734c5b0cfd01ecf78b066f65cbf9bd30260ab3d0ab8c2f1247a14eb8f0a523`。仅public七NPY及投影模型authority，原生proprio归一化/原双递推不变；四word固定XY/YX/X−X/zero，不读新source private pose/contact/suffix。

实际prepare SHA `572e2fded1c8927302c4ffe66921ce05e58f5954b43a9bc44bc022c19c6bbfb6`。HOST CUDA session84517已启动，目标1 official encoder API/3samples、1 proprio encoder/1sample、8predictor/8samples；无AD/physics/拟合。真实终态及strict未完成前不将四程序预测差异解释为物体控制能力、非交换性或P5准入。

**CUDA及strict现均完成**：实际6.13599秒，三资源轴全部attempt/return匹配、errors与finalization_errors=[]、模型保持。report SHA `4fd032a4e5fb365f5a11eecbaba416c1ace7352e7afd092aa0adf6b972bded7f`。全部43NPY、action/history归一化、visual和proprio双反馈、事件链严格消费通过；回执 `le-wm-main/results/agent_p5_contact_action_response_20261028_v1_consume_MAIN01.json` SHA `247a8a6328053f34b8367f39dca63f81fb009ca2a41734db0f618d1ea6eab2c3`，`all4_saved_public_action_words_verified`。XY/YX patch MSE=.09773935485017947、X−X/zero=.22437807454348624；仅模型feature差异，physical_release/action_witness/scientific_admission仍false、task_success=null。同public context全四word的真实物理适配器已继续独占派发，不用已有两参考臂冒充新四程序结果。

## 5. 未完成范围

P1仍缺非贪心真实能力及任务收益；新39/40 RGB编码须保完整分母，缺帧不得补造。P2仍缺组合相对强对照的实际闭环收益，public/manual调用协议核查准备中。P3真实信号及strict已完整，但仍待child compiler/等预算物理验证。P4历史fresh认证未补齐，四预留根保持封存；新文档已精确列A–H原run的8来源/16refs/10dispatch及L/C两条合法补证路线，当前SHA和新执行均不能证明丢失的历史事实。P5冻结JEPA context/action-response已真实CUDA及strict通过，待新全四word物理。共同controller对照的两臂校准源码/测试/文档已交付冻结，worker39项RAM通过；MAIN尚未全审和执行，不偷换成完整Experiment1，learned prior仍unknown。

原目标保持active；任何当前否定结果都只在已验证的实现/协议/任务范围解释，不预设理论成立，也不预设理论失败。
