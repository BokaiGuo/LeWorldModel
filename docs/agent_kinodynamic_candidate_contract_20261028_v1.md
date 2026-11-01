# Kinodynamic候选接口：现有可调用实现交接，不重复实现RRT

2026-10-28，Jason独占新文档。MAIN负责代码审查和实际HOST CUDA/physics。本轮只读定位既有实现、明确路径的小型saved metadata与SHA，运行纯fixture测试；未改任何原源码/结果/共享主表，未派生agent、下载、训练、拟合、运行神经或物理。

## 结论：本地动作见证树搜索与完整实际管线都已存在

不是只有方法名：

1. 纯回调内核 [d2p_kinodynamic_program_search.py](/home/gbk/code/LeWorldModel/le-wm-main/experiments/spectral_reachability/d2p_kinodynamic_program_search.py) 的 `search_programs`，逐边从父完整context执行raw primitive，节点保留动作序列与primitive_indices。
2. 固定moving-H2 runner [p1_moving_kinodynamic_baseline_cuda_20261027_v1.py](/home/gbk/code/LeWorldModel/le-wm-main/experiments/spectral_reachability/p1_moving_kinodynamic_baseline_cuda_20261027_v1.py)，真正调用原冻结 `TorchBackend.step` / `CudaReplay`，不是latent最近邻路径；有prepare/verify/run/consume。
3. 该runner已有2-root实际模型report、own consume、独立物理cohort与RGB评价，均在六法comparison v3中登记。既有innerdoc末尾“没有实际prepare/模型/物理”是实现交付时的历史状态，不能覆盖后来MAIN产物。

按本次“已有则交接”的分支，只新增本文，不写重复planner，不给beam改名RRT，也不宣称覆盖所有kinodynamic search或任何具名论文官方实现。

## 1. 两个可调用接口，不能混用时域语义

### 通用纯动作程序搜索

模块 `experiments.spectral_reachability.d2p_kinodynamic_program_search`：

```python
search_programs(
    primitive_bank=bank,
    root_context=context,
    replay=replay_callback,
    target_metric=saved_goal_metric,
    max_depth=predeclared_depth,
    beam_width=predeclared_width,
    max_attempted_expansions=predeclared_cap,
)
```

这只是接口说明，不是可直接运行的缺参实验。真实模型回调仅准MAIN HOST CUDA；fixture回调可以纯数组。

- `primitive_bank`：事前固定非空数组，每primitive为完整5步包的倍数 `[T,2]`、有限raw、`abs(raw)<=1`；不补步、不裁剪、不改原dtype精度。
- `root_context`精确字段：`layout/context_length/latent_dim/action_dim/latent_history/action_history`；其他witness/native/evaluator字段拒绝。
- `replay(parent_context, next_primitive)`：只推进下一条动作，不从root重复整个prefix，不从别的latent节点拼接；返回相同layout的完整新context。父context与动作独立copy。
- 节点保留完整successor context、累计raw actions、primitive_indices、depth、cost；每次尝试先记账，包括异常。action witness沿parent path真实拼接，不是仅有latent waypoint。
- 以 `(cost,depth,index_sequence)`稳定排序；跨已访问深度选best。**它可能选择H1**，所以不能直接用它替代固定H2 runner，再给短赢家补零。
- expansion尝试数不等于predict samples：primitive可含多包，真实神经预算须在回调底层每次调用前计量。内核报告明确`external_PredictionBudget_not_expansion_counts`，不能用一个node冒充一次predict。
- 普通callback异常保留失败后继续剩余分支；预算异常立即停。该语义与下面固定runner的any-error-stop不同，不能不经预声明切换。

### 原moving两根的固定H2接口（实际已有）

模块 `experiments.spectral_reachability.p1_moving_kinodynamic_baseline_cuda_20261027_v1`：

```python
beam(task, primitive_bank, dispatch)
# dispatch(before_full_context, raw5_or_raw10, phase, index_path) -> after_full_context
```

`beam`纯排程，生产应经原 `Dispatch(task, backend, meter, sink)`，不能裸回调绕过计量。原构造器和底层模块hooks已经存在，无需新写适配器或monkeypatch globals。

事前固定：每slot16个raw5×2 constant-packet bank；全16首段按(H1 cost,path)保留4，再全4×16后段，仅从64个H2末端选赢家。包含zero primitive，不删负/零结果。未来执行10raw；不跨H1/H2选择、不补零、不按native outcome挑枝。

每条edge保存phase、path、before full context与literal raw，return保存完整after；path索引结合固定bank可逐边重建动作。输出literal `commands[10,2]`，不是让物理controller去追latent节点。fresh segmented与fresh whole都从原root重新推进，要求search/fresh-segmented/fresh-whole完整context exact。

## 2. 原任务/transition/动作域与计量

- 两个原moving task exports及其SHA、原goal、native192 mean-square Q、固定float32normalizer全部继承。真实非零history前5次[.1,0]、后5次[0,.1]；3个latent、2个between-state tokens，每边追加outgoing token后滚动。不是zero padding，也不读sourcepack witness或native outcome选计划。
- 与原规划器共享冻结LeWM及`TorchBackend.step` / whole `CudaReplay`，生产CUDA-only/no_grad、MATH/TF32off，模型前后hash。每个实际AE/predictor/pred_proj入口先计attempt，return后计completed；封存实际latent3/action3输入和native successor，禁止额外image encoder/projector。
- 每root：16+64=80 planning predict；fresh segmented2＋whole2，共84。83个高层dispatch中最后whole包含2次predict，不能以83报预算。两根168 predict/AE/pred_proj，各自是同工作不同轴，不相加成504次预测。失败attempt不退费，无retry；完整实际账才释放commands。
- 原bank文件SHA `38679b9a7339f822eb4d8cb9af2d5cf794c61badee6213021ac14434a0234720`，位置0/1事前映射opaque slot0/1，不用native root身份反查。原raw均界于[-1,1]。
- **重要公平限制**：constant-packet离散bank有部分项超出D2P `.75*sqrt(5)` 的packet logit L2球；D2P又可逐时刻改变动作，bank只能固定方向/幅值。两边域并不相等。不能为了本轮“公平”悄悄投影旧bank再沿用旧结果/封存；若MAIN需要严格同球RRT，属于新预声明实现任务，不是当前旧beam已有的科学覆盖。
- prepare≤256KiB，run≤8MiB、floor200MiB；共享原image encoding成本单列，不免费，也不重复每node编码。没有本次新训练/AD/J/SVD/physics。

## 3. 实际已有产物与本轮读取范围

下表所有SHA本轮仅按精确路径重新计算，读取顶层status/resources及身份字段；未重复完整journal/PNG/模型运行或独立效果审计。

|层|实际路径（inner results）|SHA256 / 已保存状态|
|---|---|---|
|prepare|`p1_moving_kinodynamic_baseline_cuda_20261027_v1_prepared01.json`|`df3727dea165a1114043e388b28c655cf01609e589f099c40f5a52f869db7935`|
|模型|`p1_moving_kinodynamic_baseline_cuda_20261027_v1_cuda_main01/report.json`|`8d2a8d6bc71eac53194c6001647c2d90736f292f9dfdcf5a2c838a58d36d2507`；`fixed_H2_beam_plans_sealed_not_admitted`、errors[]、两根commands各10、实际168/168三个模块资源|
|own consume|`p1_moving_kinodynamic_baseline_cuda_20261027_v1_consume_main01.json`|`e2efc1d1b456e8d167017423c4a15ccbb69ebdcd961041e7549b6e734083743b`；`fixed_H2_commands_verified_not_admitted`、2 plans|
|物理|`p1_moving_baseline_physical_20261027_v1_kino_main01/report.json`|`2db7370214bf62810c21ad8778c1d58bc397dbe8f08c9f5ef1db435468abc655`；`moving_baseline_physical_recorded`、errors[]|
|真实RGB评价|`p1_moving_baseline_rgb_cuda_20261027_v1_kino_cuda_main01/report.json`|`71dafeeebd1d2bd2567028e65800f4f27f479c72b066da3d2ace6a35ef8e7869`；`moving_baseline_rgb_encoded_not_admitted`、errors[]、4 encoder/projector|
|综合|`p1_moving_baseline_comparison_20261027_v3_main01/comparison.json`|既有固定SHA `96cf27b143bb1a839edf7bc1e4de02f4a97e433c6475c4c731c78af76300d32e`；kino单独cohort，非原三法共用物理账|

本轮不新增或择优汇报任何性能数值。已保存native/actualQ保留，不能因寻找“缺失实现”而丢弃负结果；两known roots不因此成为fresh独立样本。上述完整管线存在意味着不应为补表再次运行相同beam。

## 4. MAIN可直接执行的命令

工作目录 `/home/gbk/code/LeWorldModel/le-wm-main`，先saved，不重新运行模型/物理：

```bash
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -B -m pytest -q -p no:cacheprovider tests/test_d2p_kinodynamic_program_search.py tests/test_p1_moving_kinodynamic_baseline_cuda_20261027_v1.py
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m experiments.spectral_reachability.p1_moving_kinodynamic_baseline_cuda_20261027_v1 consume --output results/p1_moving_kinodynamic_baseline_cuda_20261027_v1_cuda_main01 --report-sha 8d2a8d6bc71eac53194c6001647c2d90736f292f9dfdcf5a2c838a58d36d2507
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m experiments.spectral_reachability.p1_moving_kinodynamic_baseline_cuda_20261027_v1 verify --prepared results/p1_moving_kinodynamic_baseline_cuda_20261027_v1_prepared01.json --prepare-sha df3727dea165a1114043e388b28c655cf01609e589f099c40f5a52f869db7935
```

`consume`输出精确schema `<runner>.consume`、status `fixed_H2_commands_verified_not_admitted`，计划字段名是 **plans**，不是rows。每plan有source_slot、task_export/task_sha256、task_id、blind_sha256、arm、literal commands、path、terminal_cost。物理consumer已白名单注册authority=`kinodynamic`并从plans取值；不能传一个自称相同arm的任意新producer。

原下游为 `p1_moving_baseline_physical_20261027_v1` 与 `p1_moving_baseline_rgb_cuda_20261027_v1`，原comparison/budget已接通，不另需新的动作→物理适配器。若源码/绑定漂移，verify/consume应拒绝，不改旧seal强行兼容。

若MAIN另授权新实验，原prepare/run仍按既有innerdoc，必须NEW output；本次没有提供自动重跑指令、没有创建prepare，也没有调用任何模型/物理。RRT新版本不是使用旧beam的必要条件。

## 5. 真正仍缺与不应声称的内容

1. 没有动作空间RRT/随机target扩展的正式新合同；**现有的是model-native beam**。RRT只是另一搜索族成员，不能等同所有kinodynamic方法。
2. 没有与D2P完全相同logit球/逐时刻动作域的该beam成绩；没有等predict或等总算力认证。改变bank/预算要NEW预声明版本，不能追认旧结果。
3. 没有任意新source、overlay/on或真实observed-prefix任务的该runner通用批准；当前fixed任务入口不能直接喂新context冒用旧schema。
4. 未建立具名外部kinodynamic论文原算法/代码/benchmark忠实性或普适优势；当前是本地离散模型搜索对照。
5. 未取得独立freshness/P1准入。增加节点、RAM测试或已有物理重放不替代独立任务前置。

## 6. 实际测试与冻结源定位

本轮执行上述第一条，**86 passed in7.18s**。包括原kernel纯图fixture、固定H2完整84预测的非神经数组fixture、动作见证、非零history滚动、三路径、保存数据callback、计量/损坏/失败分母测试；fake CUDA包装仅测试guard，不是CPU神经fallback。不新建test tmp/results，不运行神经/physics。完整consume/verify命令本轮未执行，留MAIN按需复核。

已重新核SHA：

|既有文件|SHA256|
|---|---|
|纯内核source|`c92a945ca884d503b918c7f7357be7ee55fa36377d9de60e2dddf68b42208b7d`|
|moving runner source|`66c090cb98ad8a954bae5c94e39a8cbb707b5a2e8dc82027e63fa00f2583ae6d`|
|纯内核tests|`61ec68176533d216ce14bedfbda4f4336495516d8c9664789151b2ddcc32a165`|
|moving runner tests|`ac095e37169b0f248d3a80216232c7d0713fac3e07ccf7346af6f608776ce563`|

只新增本交接文档；未扩大扫描、未写新算法、未修改任何现有文件。不把fixture、ownconsume或历史MAIN实跑说成本worker新物理验证。
