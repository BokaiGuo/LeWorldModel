# P2 known2/off/raw15：GRASP结果盲接入准备

本stem source/tests/本文独占新增，不改旧资产/结果/共享文档。MAIN审读并决定生产；worker不执行神经、AD、物理、下载、训练/拟合或结果选择。experiment-code仅用于先固定协议、原源码接缝与RAM回归，不执行通用训练/强制正结果/绘图模板。

## 交付结论

**有真实本地LeWM输入与官方LeWM-GRASP源码，但不能在“官方实现不变、P2同输入/同动作域/同预算”三者同时成立下直接运行。** 本版实现可运行的结果盲prepare/verify、真实任务输入提取、raw15坐标/归一化/首调用ABI和具体source-line差异，不是CUDA runner。无法公平接入时停止在可复核的差项；不造future报告、原process证书或NN结果。`official_execution_request`无论caller伪造任何ready标志均拒绝，CLI无run。

## 原P2合同与已有实物复用

固定来源 `results/agent_p2_three_program_20261028_v1_prepare_MAIN01.json`，实际SHA `86af2d24f73e769fb3b91fa8a571f8ca265df95c93af780d3924ea644585d461`；producer source SHA `3111312477ff3cd9a764dc7d063861c1becaaf58885dc0fb25c2f3ec73d15fd5`。只读prepare和两份原task gzip、源码/config，不读planner或physical报告、selection、score、teacher后续轨迹、blind NPZ、PNG、权重或保留fresh根。

原source两known roots，off，root10 observed latent历史对应warmup0/5/10，action_history为warmup1..5与6..10的两包真实动作；原raw10前5[float32 .1,0]、后5[0,float32 .1]。latent_history[3,192]、action_history[2,10]、goal192、identity192 terminal MSE、原checkpoint/config、raw-normalizer/source/blind/goal像素身份全继承。计划未来3包×5=raw15；原encoded任务本身沿封存horizon2字段不修改，本版另给target_future_packets=3，不能把原H2标签改成H3来伪造新任务身份。

两份真实task为 `results/p1_moving_overlay_recompile_cuda_20261027_v1_cuda_MAIN01/task_00.json.gz` SHA `80f0f4ed14498509e596322b830f0f8d1bedc4d38f9d3b7d64b57da375f9a132`、`task_02.json.gz` SHA `009ed5769fb55513100c1223c5659bd5b34911b5cfe13e9b121117c14a0db2d7`。使用producer自己的`core.task_id/core.digest`，不拿本版compact JSON跨schema重算。解gzip有界2MiB、必须EOF无trailing/第二流、拒重复key/NaN。

模型声明权威沿原prepare：weights SHA `48938400ae3464c9680731287f583a9cb516f55a8ec64ea13a91be47fb15b607`、config SHA `2564086e961e7b5c7c04dffc451091115b389a590645ff19653c64fd0bc16e09`、tensor SHA `4b12c01f5859f4b46bea478f3a8d5e2d3f516df456a622a3feec31bde7e33546`。本版只重核小config的224/patch14/257token→CLS192 ABI；**不打开权重，不声称本prepare重新认证tensor值或完整原encoding consumer**。原像素/NPZ/SHA是来源pointer，不伪装新编码。bind完整源码/两task/normalizer源码/config及自己三件套，读取前后hash核等。

本地mean/std保原float32约[-.0087,.0068]/[.2019,.2002]，只有一次(raw-mean)/std，raw已经是action_scale/100后相对量，不能再/100。`training_provenance_verified=false`保留。原process缺失阻止官方表级复现，但不抹掉这份可用的本地固定坐标，也不要求现场fit才能做明确适配。

## 具体官方差异与最小实施停点

官方可读文本来自 `results/p2_grasp_official_contract_20261027_v1/sources/`，复用原`read_source/anchor`的固定SHA和行号摘录；不import/exec外来源码。官方LeWM适配repo `michael-psenka/le-wm-grasp` commit `94be3c42aa2f0867c6cb1e38bda498b293b6921e`，`grasp_solver.py` SHA `53e7761019e3eed0d776df46aa6e0f8e84c6a084d9588802b2528b006be690df`。DINO原repo `michael-psenka/grasp` commit `64f988aebf04ad9f674f9821bc6d2b9e22879105`另保留出处/源码，不以DINO patch输入冒充LeWM192。此commit与论文实验表的逐run配对unknown。

|项目|官方LeWM适配实际行为|P2要求 / 最小差项|
|---|---|---|
|历史输入|`_jepa_encode`取最后frame；lifted `_jepa_predict_step`单state；sync `_jepa_full_rollout`从单s0起步|P2全3latent/两between-actions不可丢。必须新history-aware collocation及serial sync实现；这是明确算法扩展，不是官方H1同输入|
|动作坐标/support|直接优化无界normalized动作，未做tanh或joint球限制|P2 joint30 logit L2 `.75sqrt15`，NumPy tanh(float64)→float32 raw→normalize；要新bounded-coordinate扩展，不能照抄旧per-packet球或clip后称官方|
|zero初始化|normalized zero逆变换是raw mean|P2 logit zero是raw zero，normalized不为0。本版真实计算两者字面数组，不能静默视同|
|优化与评价|soft-dynamics+dense-goal loss，state输入detach、virtual目标侧梯度、virtual噪声、serial GD sync|算法loss不同本身不禁止比较；末端都能用原goal/Q评价，但须保真接线/原生数组与逐调用账，不能替成普通GD|
|750步资源|见下表；官方seed绑定`${seed}`，`randn_like`噪声用global RNG而非constructor generator|P2原joint/composition账不同；需MAIN事前给比较资源/停止规则和RNG规则，不能以缩减步数或padding凑等算力|
|原process/论文身份|原论文checkpoint/data/process/seed配对unknown|官方复现仍需原frozenprocess及配对；局部工程适配可使用原本地固定normalizer但必须正确标注|

可复用：已完成`p1_moving_grasp_baseline_cuda_20261027_v1`是完整real-H3/bounded/raw10工程适配，core `p1_constructive_grasp_baseline_20261027_v1`确有soft dynamics、virtual noise和sync；不是普通GD改名。但它写死未来H2、一个virtual、逐包logit球及float32 chart；**不能只改horizon标签就变成P2 joint30/raw15**。`p2_grasp_reference_core_20261027_v1`支持未来T3但H1-reset且无bounded raw域，不能解决同输入问题。不重写/重跑这些旧模块。

最小下一实现（需MAIN明确选择而非本版假放行）：新独占H3/joint30 raw15 bounded-GRASP变体，在lifted的两virtual节点构造正确滑动3帧及live action历史，sync始终从原H3真实context serial三包；固定完整global30域/原chart，逐阶段保存原生数组/grad/noise/optimizer reset与模块attempt-return，独立whole15及三segment重放6predict；继承完整目标/动作/normalizer，不读未来outcome。若坚持源码不变，则所有对照需另建H1/无界normalized协议，不能和这轮P2直接配对。缺口首先是**真实适配代码及前瞻资源选择**，不是还缺下载一个Hi/PlanWAM包。

## 可执行输入ABI与预算

`make_request(prepared,exports)`完整验证两原任务并提取白名单model_input（root_context/goal/q/normalization）；注入native state/reward/teacher/selection等字段拒绝。`raw15_inputs(model_input,logits30)`严格核原joint球，返回literal raw15、三个normalized outgoing packets、首模型latent3/输入action3；不接受backend。未来context明确null，绝不生成预测或动作物理结果。`initial_coordinates`给rawzero与normalizedzero对照；这是算术，不是新计划或神经实验。

750 joint、interval50、sync10来自官方YAML与constructor默认gd/schedule_decay=false的AST，只读取literal值不执行。以下全部是**前瞻调度账，不是实际returns**：

|每根batch1资源|数量|边界|
|---|---:|---|
|lifted transitions|2250|750×3|
|serial sync transitions|450|15×10×3|
|官方solver合计predict|2700|原solve没有额外终点验证|
|scalar backward API|900|750+150，不乘192当数学VJP|
|若新增P2 whole15+segmented15核验|6|显式额外，不伪称官方solve已有|
|未来带双重放总predict|2706/root，5412/two roots|AE/predictor/pred_proj各一套嵌套账，不相加冒充API|
|直接官方pixels solve|2encoder+2projector/root|本地cached latent入口可省新增调用但属于显式API适配，继承编码不是免费|
|旧P2 joint / composed standalone（仅旧prepare协议）|138 / 936 predict/root|不是本轮common-cap对照；其AD/J/SVD向量不能与900 backward互换|

MAIN最新通报本轮common-cap实际joint1902/composition1062 predict、各85J，共同predict cap1902/wall450秒。本版字段明确标记这是MAIN消息来源，未重新consume其实际receipt，不能作为本版自行认证的saved authority；绝不能以旧138/936声称当前预算不等。官方原配置完整2700predict已超过1902，仍原样保留，不能为追求结果改750或截断后冒称官方配置。

**具体选择建议：** 若目标是当前P2同输入/同动作域，下一项应独占实现明确命名的history-aware joint30 bounded GRASP变体；不选官方last-state当作同合同。MAIN需事前确定：完整750步另列非等资源对照，或另行固定1902cap下的变体停止规则（包括一次完整sync/最终双重放的预留与失败分母，不能运行后择优截断）；同时固定global RNG及原rawzero初始化。另需绑定最新common-cap真实prepare/receipt作为资源authority，本版不猜路径/SHA。若选择官方源码不变，则应另立last-state输入协议、所有对照重配，不能回答当前同合同问题。本版准备工作到此封口，不继续扩展refused wrapper。

FLOPs、实际wall/显存/engine backward开销、attempt/return未执行，均null。不同算法允许比较，不要求family/algorithm字符串相同；但没有事前共同资源规则就不能称公平/等API/等FLOPs。本版不设新科学阈值、不给P1或freshness签名，physical_release/production_dispatch始终false。

## MAIN可运行CLI

cwd `/home/gbk/code/LeWorldModel/le-wm-main`。prepare只写NEW本stem文件，<=2MiB、floor200MiB、xb/fsync文件和目录；verify按外部SHA全量重建本请求，不消费旧模型/物理结果。`inspect`只stdout，不落receipt。

```bash
OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m pytest -q -p no:cacheprovider tests/test_agent_p2_grasp_raw15_prepare_20261029_v1.py
.venv/bin/python -B -m experiments.spectral_reachability.agent_p2_grasp_raw15_prepare_20261029_v1 inspect
.venv/bin/python -B -m experiments.spectral_reachability.agent_p2_grasp_raw15_prepare_20261029_v1 prepare --output results/agent_p2_grasp_raw15_prepare_20261029_v1_prepare_MAIN01.json
.venv/bin/python -B -m experiments.spectral_reachability.agent_p2_grasp_raw15_prepare_20261029_v1 verify --prepared results/agent_p2_grasp_raw15_prepare_20261029_v1_prepare_MAIN01.json --prepare-sha ACTUAL_PREPARE_SHA
```

exit0只表示结果盲输入准备/差异重建成功，输出status明确`inputs_prepared_official_same_contract_refused`；不是CUDA授权。没有run或physical命令。RAM测试覆盖真实gzip回归、rawbits/一遍归一化、完整历史与不等zero、域越界拒绝不修复、两根分母、元数据变异、成本不padding、读集不含outcomes/NPZ/weights/reserved、cold import精准拒NN/网络与伪ready仍拒绝。测试结果和三SHA在最终冻结交付中给出；worker未生产prepare、不运行NN/physics，完成个人实现goal不等于研究goal完成。

## 实际回归与冻结范围

上述测试命令实际 **46 passed in 0.88s**，含真实已封存prepare/task gzip的只读build及临时RAM元数据写入/重建，未生产prepare。首轮45pass/1fail来自测试把`socket.socket`替成函数，破坏标准库SSL继承；改用独立子进程audit hook拒socket创建/连接/解析。中间一次修正脚本缩进仍45pass/1fail，最终已修正，不删除失败记录、不改变production校验。这些是测试接线错误，不是科学负结果。

本版仅冻结本stem source/tests/本文。无CUDA/AD/物理、无新增模型输出或科学证据；后续变动必须新版本。MAIN最小可执行入口是上列prepare/verify；真正H3/joint30 bounded-GRASP runner和事前资源/RNG选择仍未实现，不能用本版请求直接dispatch官方solver。
