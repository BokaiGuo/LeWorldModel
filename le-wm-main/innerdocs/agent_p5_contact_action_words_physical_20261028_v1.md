# 同public context四word原生物理重放：MAIN待审读执行

独占新增source/tests/本文，所有native/numeric v1/v2及封存结果保持不变。使用experiment-code技能固定原计划、独立资源轴、partial/失败日志和RAM mutations；worker不prepare、不运行physics/CUDA/CPU神经、不联网、不训练/拟合/teacher/adapter/repair，不按预测或物理结果筛词。新结果只能来自本模块MAIN实际执行。

## 实际四词authority，不混两臂v2

唯一模型authority是MAIN已完成的**四词numeric v1**，不是两臂numeric v2，也不是旧source push_x/raw_zero。固定XY、YX、X−X、zero_zero及顺序；每词两包raw float64[2,5,2]，直接reshape成raw10执行，不逆normalization重建动作、不padding。

prepare必须MAIN实际提供native manifest、numeric manifest、CUDA report、strict receipt的path+FILE SHA：

|输入|实际SHA256|
|---|---|
|原native prepare MAIN01|`b728eb810a78f976bc413d891984766b7743e6ae55a13be8dd5a191531bc3b10`|
|numeric v1 prepare MAIN01|`572e2fded1c8927302c4ffe66921ce05e58f5954b43a9bc44bc022c19c6bbfb6`|
|numeric v1 CUDA MAIN01/report.json|`4fd032a4e5fb365f5a11eecbaba416c1ace7352e7afd092aa0adf6b972bded7f`|
|numeric v1 consume MAIN01.json|`247a8a6328053f34b8367f39dca63f81fb009ca2a41734db0f618d1ea6eab2c3`|

prepare绑定原冻结numeric源SHA `bd5b85b308fcb621adda1f4452911f58e233b9ba0812e8d569d0dbd540940930`，重算其saved-only consume并与MAIN持久回执逐字段精确相等；43NPY、全event链、8predictor、原模型closure以及四plan都纳入原consumer，再继承所有输入/报告/事件/数组SHA。不调用模型runner。原native源SHA `78700888a70d5caf263d04301ca8a62f907364ba4b38cf9d3bdaf689d115fecd`，native.verify重建recipe，无physics。

动作由实际numeric manifest四个plan.raw读取，要求与原word生成函数精确相等；不看patch MSE决定执行顺序或子集。旧push_x/raw_zero合同仅取原constructor/seed/prefix和goal身份；旧两条参考suffix绝不冒充新四词结果。物理prepare是特权重建域，不能传planner；模型输入仍只来自已发布七NPY。

## 精确重建与byte gate

原source recipe：seed2026102801，原支持的constructor reset_to_state `[180,271,256,256,0,0,0]`，其余native参数保持原合同（relative100、with_velocity/with_target、原T/阻尼/COG/原目标）。不在reset后调用setter，不restore(savedpoststate)，不替换旧prefix。它是刻意构造OOD校准场景，不算fresh。

四个独立live env分别constructor→显式seed→native reset一次→十步raw64零prefix。四个prefix全部完成才检查每臂每帧0..10的五原文件（RGB/state/proprio/transition/full_public_state）与**认证旧source prefix**字节精确相等：4×11×5=220文件比较；另检查每臂显式seed RNG与10command字节。原source report SHA `ecd6afea8bcc9ee7f6fe32234c91a4fb4be62ecdd6e6bbc78351ed8e20d8a4ba`只用于取prefix文件inventory认证；解析该旧JSON不是使用其参考suffix/outcome作新结果，也不根据其contact/pose选择动作。

同核public payload SHA `fdc063a6475577071fb74cf4ccd0853c4b2e4e5c061693d73efbc437686ffec3`：旧prefix RGB0/5/10的NPY SHA必须等于实际公开RGB，proprio0/5/10 stack必须精确等于公开raw agent xy/vxy，prefix raw history必须等于公开history。原numeric consumer另严格验证七NPY及原normalization，goalRGB只身份绑定。每帧原goal依旧[256,256,pi/4]；不新增goal场景/goal setter/render。

完整44frame gate写入持久journal，严格位于最后一个prefix step return与第一个suffix intent之间，然后依固定顺序给各自原live env执行十步raw suffix。四环境最终全部close、释放本模块owned profiler。任一失败不补步/换臂/恢复/退款；保留四分母、partial、原工程错误和close状态。constructor异常未能返回env时只能记录construction_attempted，不能保证关闭未知底层部分对象。

## 实际native观测与资源

复用封存原panel.capture、physical.Writer/Journal/validate_frame、原NativeCalls；本新source独立callback观察器，不修改旧globals/execute/env/_handle_collision/Space.step。使用Python sys.setprofile观察原callback call/return：真实arbiter agent/block/static body身份、contact_point_set坐标/法向/距离、post_solve impulse、计数before/after、Space.step上下文与substep。未启动physics前不能知道实际callback数；8192 callback/16points仅工程cap，不预填RAM150等值。

每臂frame0..20全部21帧、共84帧保留五文件和每帧callbacks，原生state7 `[agent_x,agent_y,block_x,block_y,block_angle_mod2pi,agent_vx,agent_vy]`；full_public_state保留block/agent位置、原angle、速度/角速度、接触计数、目标、RNG/shape/环境设置。每step保存actual raw argument、100倍位移、绝对target、前一public SHA；验actual latest_action。done恒False按原绑定native source核验，coverage/reward原样保留，coverage不是本研究task成功。

|资源轴|每词|四词总量|
|---|---:|---:|
|constructor / close|1 / 1|4 / 4|
|seed（含constructor内部一次）|2|8|
|reset|1|4|
|public control steps，prefix10+suffix10|20|80|
|Space.step，reset1+20×10|201|804|
|native render / get_obs|21 / 21|84 / 84|
|setup / set_state / Space constructor|1 / 1 / 1|4 / 4 / 4|
|native goal-body|41|164|
|Body constructor|44|176|

public durable intents/returns预期construct4/seed4/reset4/step80/close4，与内部native calls分开。Space统计是Python call entries+实际dt/context，不冒称C积分成功；step intent→return原日志另计，失败attempt不退款。各臂相同20public/201Space预算，但不称全算法等FLOPs/总算力，callback实测可能不同、wall独立。模型/AD/backward/JVP/VJP/SVD执行均0；原CUDA预算只并列绑定，不与physical轴相加。

复用封存Writer的128MiB accounted、96MiB payload、8MiB final reserve、2048nodes×8192B overhead、200MiB floor、900s wall+SIGALRM；写前/dispatch前guard，失败不退款。旧Writer常量不改。全callback计数扩大会增加journal成本；超cap必须partial停止，不能删bindings或回调、抽帧过门。C代码可能延迟alarm。final reserve用于close/terminal；不可持久化故障可能只留partial，不能strict通过。

strict saved消费重核原actual prepare、完整inventory/terminal/journal、全部操作顺序与actual counter连续性/804 dt、四env cleanup、220 byte门、公开身份和各帧native语义/command/callback链；输出40条private_suffix_frame_rows保留四词全部agent/block/coverage/contact。不设置contact必须发生/物体必须动的过门；负结果完整保留。partial返回private rows=null，不缺失填0。所有scientific_admission/action_witness_established/physical_release=false、task_success=null；仅四词物理重放，不证明物体非交换操作能力或D2P贡献，不能把agent惯性当block机制。

## MAIN调用

cwd `/home/gbk/code/LeWorldModel/le-wm-main`。worker只RAM，未执行这些真实命令。

```bash
OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m pytest -q -p no:cacheprovider tests/test_agent_p5_contact_action_words_physical_20261028_v1.py
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m experiments.spectral_reachability.agent_p5_contact_action_words_physical_20261028_v1 prepare --native results/agent_p5_contact_source_native_20261028_v1_prepare_MAIN01.json --native-sha256 b728eb810a78f976bc413d891984766b7743e6ae55a13be8dd5a191531bc3b10 --model results/agent_p5_contact_action_response_20261028_v1_prepare_MAIN01.json --model-sha256 572e2fded1c8927302c4ffe66921ce05e58f5954b43a9bc44bc022c19c6bbfb6 --report results/agent_p5_contact_action_response_20261028_v1_cuda_MAIN01/report.json --report-sha256 4fd032a4e5fb365f5a11eecbaba416c1ace7352e7afd092aa0adf6b972bded7f --receipt results/agent_p5_contact_action_response_20261028_v1_consume_MAIN01.json --receipt-sha256 247a8a6328053f34b8367f39dca63f81fb009ca2a41734db0f618d1ea6eab2c3 --manifest results/agent_p5_contact_action_words_physical_20261028_v1_prepare_MAIN01.json
sha256sum results/agent_p5_contact_action_words_physical_20261028_v1_prepare_MAIN01.json
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m experiments.spectral_reachability.agent_p5_contact_action_words_physical_20261028_v1 verify --manifest results/agent_p5_contact_action_words_physical_20261028_v1_prepare_MAIN01.json --manifest-sha256 <ACTUAL_PREPARE_SHA>
```

MAIN完整读码/测试/prepare后独立fresh进程HOST物理，无torch加载；所有输出exclusive create：

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 PYTHONPATH=/home/gbk/code/LeWorldModel/le-wm-main/.jepa-physics-site .venv/bin/python -B -m experiments.spectral_reachability.agent_p5_contact_action_words_physical_20261028_v1 run --manifest results/agent_p5_contact_action_words_physical_20261028_v1_prepare_MAIN01.json --manifest-sha256 <ACTUAL_PREPARE_SHA> --confirm-physics --output /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p5_contact_action_words_physical_20261028_v1_physics_MAIN01
sha256sum results/agent_p5_contact_action_words_physical_20261028_v1_physics_MAIN01/report.json
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m experiments.spectral_reachability.agent_p5_contact_action_words_physical_20261028_v1 consume --manifest results/agent_p5_contact_action_words_physical_20261028_v1_prepare_MAIN01.json --manifest-sha256 <ACTUAL_PREPARE_SHA> --report results/agent_p5_contact_action_words_physical_20261028_v1_physics_MAIN01/report.json --report-sha256 <ACTUAL_PHYSICAL_REPORT_SHA> --output results/agent_p5_contact_action_words_physical_20261028_v1_consume_MAIN01.json
```

RAM fakes是record-only token callbacks/数组，不做积分，不能作为contact或物理科学证据。覆盖全producer布局/四prefix门/actual ledger/early failure/cleanup/mutations及saved prepare authority、封hash篡改、四词raw身份；真实回归只核已存JSON/SHA，不跑prepare/模型。旧source参考push/zero只验证prefix身份，旧endpoint不回填任何新词。
