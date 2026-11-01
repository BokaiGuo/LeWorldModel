# P2后续 / E1：可执行子目标gap的具体准备合同

本独占source/tests/doc；保留全部旧结果/改动，不训练、改Q/normalizer/gate、挑根或按outcome调配置。仅stdlib metadata ABI与RAM fixture；无torch/NumPy/model/physics/full strict导入或生产调用，不下载、不创建下级agent。使用experiment-code技能落实固定合同/日志/回归，其训练、调参、收益门和图表要求不适用本限定任务。

## 原提案与不重复边界

原件 `/home/gbk/.codex/attachments/861f6764-1b25-4227-bf15-f1a237ba7d8f/pasted-text-1.txt` §7、527–570行：arbitrary直接在latent中优化目标；learned采用已有分层方法训练的subgoal/proposal；compiled从冻结动力学动作响应导出动作，再产生中间状态。原文要求低层能否实现、执行偏差与最终闭环任务成功，不是只比较latent距离。§4（416–424行）明确model action-witness并不保证真实物理可达。

已有 `agent_subgoal_gap_contract_20261028_v1` 已实现三臂saved评分，`agent_subgoal_gap_runner_20261028_v1`已有callback/meter骨架，`agent_subgoal_gap_assets_20261028_v1`已打包旧四根metadata；不重写它们。本模块复用contract.deviations算术，不调用其legacy scorer或任何真实consumer。旧runner要求P1/freshness authority的run接口原封不动；本任务只准备known-root calibration接口，不把科学P1=false变造成禁止今后MAIN授权工程校准的新理由，也不自行授权run。

现有P2三程序/H3 full15、physical与RGB已执行，不再重复这些程序或编码。两known off roots上的既有动作见证只作为接口资产，不换绑成新三臂、同预算或独立确认。固定root0/1×三arm=6分支，cp5/10/15=18槽；这和旧E1四根/H10打包器不同，不能拿旧prepared直接称本协议可跑。

## 当前真实资产与精确缺口

`inventory()`仅查源码内CATALOG的六个已明确路径，不递归扫描/加载checkpoint、不运行full strict。输出每文件实际bytes/SHA、schema/saved status，标记`metadata_read_NOT_full_strict`，不是worker独立认证MAIN执行。默认路径均在inner results：

| 类别 | 已有真实入口/本轮检查 | 未闭合、不能替代什么 |
|---|---|---|
| arbitrary latent | 原提案要求直接latent优化；旧E1已有arbitrary_rule接口 | 未提供当前P2 ABI上的真实直接latent optimizer及固定目标函数/初始化/步数/计量source；随机向量、final goal、旧action endpoint不是该臂 |
| learned prior | `agent_hilewm_rgb_high_low_20261028_v1_consume_MAIN01.json`和`agent_hilewm_low_cem_20261029_v1_consume_MAIN01.json`已存在；对应low CUDA report也已存在。安全NPY export metadata已有 | 不能仍称“Hi整包/可信loader完全没到手”。但现Hi使用owned不同权重/预处理、单帧z与zero a_hist，当前P2为原三帧causal history；没有到P2的tensor/encoder/preprocess/normalizer坐标等价证书；禁止训练mapper或按192维强转 |
| learned→raw | 冻结low CEM source `inverse_transform:228`显式拒绝全部authority；low输出是normalized actions，非真实raw动作 | 原StandardScaler mean_/scale_、flags/dtype/XY单位与checkpoint/eval数据出处未认证。已有 `agent_hilewm_normalizer_asset_route_20261029_v2`只找到小metadata，不能取别处统计或重fit |
| compiled witnessed | `agent_p2_three_program_20261028_v1_consume_MAIN01.json`已存在，完整4plan ABI；固定取其中root0/1的three_program slots `/plans/1`、`/plans/3`，不按Q/物理结果再选 | 保留原selected plan身份，只作已付接口示例。旧joint/compiled standalone调用138/936也不同；不能复用为本次同预算新生成。不重复旧物理/RGB作为新gap实验 |
| common controller | 旧E1 callback骨架、原P2 native recipe与CEM等底层代码可复用 | 未提供三臂统一真实target-tracking callable/config/source、无privileged action warmstart、同观察/预算和配对delivery receipt；有代码不等于接线已认证 |
| 任务/根 | 已有原两known roots/off、warm10、goal与source recipe；资产清单给task/source/context pointer | 不新增root，不把六分支当六独立样本；独立fresh authority与P1仍未开放。本任务也不发新根证书 |

inventory将compiled receipt与其report、Hi low receipt与其report的**实际文件SHA字段**相联，只检查保存ABI，未重建其中full strict、权重或全部依赖。任何ABI冲突直接失败；缺文件报告path/sha=null/bytes=null，不能记0或补造prior。完整生产认证仍由MAIN后续执行。

## 共同输入、预算与可执行性量

本版本route固定`current_P2_frozen192`；共享task/root、原full3 latent/两between-action tokens、goal192/Q、冻结model/encoder/preprocess、normalizer、raw15域与每5步RGB反馈。low controller对所有臂接同root/public context及**实际delivered target**，不得给compiled额外raw witness作为warm start；动作见证仅验证/诊断。如果另走Hi配对backbone，必须新版本让三臂全部重编/编码，不能只替换一臂或cast latent。

MAIN需在结果盲声明中选择一个budget mode：`predict_API_calls_matched`或`predictor_forward_samples_matched`，给共同cap、batch_limit、精确matched_model/callable SHA。scope固定包含context_prepare、proposal、controller、AD_primal、whole_replay、segmented_replay，失败attempt不退款；API与samples分轴，batch不能隐去。prepare**无默认数值cap**，不会从旧936/1902、已完成outcome或wall时间反推预算。相同cap只叫common boundary cap，不保证实际用尽或actual matched，不padding或返还旧成本。

Hi的high predictor不是同一low predictor调用，必须另列高层API/样本/token及其proposal开销；若只匹配共同低层boundary，就明确仅该boundary同预算，**不能称所有world-model总调用相等**。若MAIN要所有world-model调用匹配，需另审高/低boundary与各臂生成预算分配的新合同，不能将不同预测对象悄悄合并。所有AE/projector/encoder、AD primal/J/VJP/SVD/backward、replay、physics/render/CPUcapture/wall独立计账，未知null；FLOPs/internal engine backward仍null，不要求导数或wall逐轴相等，也不称equal_total_compute。

评估完整18槽：actual encoded checkpoint到本臂**delivered target**的L2/MSE；predicted-to-target偏差另列；轨迹只指已有cp，不伪称每个physics tick。realizability比例须另有MAIN事前固定的latent_l2_tolerance/criterion资产；缺时只报告连续偏差与ratio=null，不能事后按结果调门。native success由原任务evaluator和完整goal7独立评价，不能从tracking小或object-only误差推成功。earlystop/unknown prior/缺RGB保null与完整分母，不替终点、不伪编码。

## 可执行prepare/verify与外部声明ABI

CLI默认无外部声明时生成明确blocked的规划artifact：三臂全部not_dispatched；缺MAIN budget、arbitrary_optimizer、prior_pairing、common_controller、native_rebuild、tracking_criterion。它不表示这些能力理论不可能，也不是实验负结果。

可选 `--declaration ABS_JSON --declaration-sha ACTUAL_SHA`：

```text
{schema:NAME.declaration, route:current_P2_frozen192,
 budget:null | {mode,cap,matched_model_sha256,matched_callable_sha256,batch_limit,
               scope:[context_prepare,proposal,controller,AD_primal,whole_replay,segmented_replay],
               failed_attempts_refunded:false},
 assets:{arbitrary_optimizer:null|ref,prior_pairing:null|ref,common_controller:null|ref,
         native_rebuild:null|ref,tracking_criterion:null|ref}}
ref={path:实际绝对路径,sha256:实际字节SHA}
```

每份asset为`{schema:NAME.<role>,model_hash:原P2实际模型hash,bindings:{角色:ref,...},...}`；绑定仅小型metadata/source（JSON/PY/MD/TXT/H）原字节，不打开权重、pickle、NPY或archive。所有role必须包含`model/encoder/preprocess/normalizer/context/q/raw_domain/task_goal`，且这八个ref在各role间**精确相等**，不能仅有相同model hash。role具体必检字段见源码asset_check：arbitrary必须direct_latent_optimization、parameter_shape=[3,192]、uses_action_rollout_as_target=false；prior必须真实Hi/SAGE身份、fallback=false、实际raw_inverse及target_delivery/prior_checkpoint_inventory refs；controller须cp5/10/15、同callable、无privileged warmstart，另有predictor_callable source ref，其SHA精确匹配budget.matched_callable_sha256；native须原两根/warm10、禁止savedpoststate reset；criterion须事前正有限tolerance。

这些metadata字段和hash**不能自签配对真实性或MAIN批准**。即使所有ABI均齐，状态也仅`ABI_prepared_MAIN_full_strict_and_runtime_review_pending`，`production_ready/dispatch_allowed/full_strict_rebuilt=false`；没有run或动态加载入口。MAIN仍需完整strict/来源/许可/逐tensor坐标与真实loader审查后新runtime接线；本模块不是又一层可绕开strict的批准证书。verify完整重建prepare和所有本次绑定，小清单最终前后SHA复核；原脚本/封存结果从不覆盖。

```bash
cd /home/gbk/code/LeWorldModel/le-wm-main
OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m pytest -q -p no:cacheprovider tests/test_agent_p2_subgoal_gap_protocol_20261029_v1.py
.venv/bin/python -B -m experiments.spectral_reachability.agent_p2_subgoal_gap_protocol_20261029_v1 inventory
.venv/bin/python -B -m experiments.spectral_reachability.agent_p2_subgoal_gap_protocol_20261029_v1 prepare --output /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p2_subgoal_gap_protocol_20261029_v1_prepare_MAIN01.json
.venv/bin/python -B -m experiments.spectral_reachability.agent_p2_subgoal_gap_protocol_20261029_v1 verify --prepared /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p2_subgoal_gap_protocol_20261029_v1_prepare_MAIN01.json --prepare-sha ACTUAL_PREPARE_SHA
```

output排他本stem、≤16MiB/floor200MiB/fsync；没有输出时prepare仅stdout。未来SHA必须来自实际文件，不写“预计hash”。fixture meter仅演示predispatch拒绝、API/sample独立、batch约束、attempt/return/失败不退款与边界身份；其`real_meter_authenticated=false`永不作为生产实付证据。fixture_measure复用旧scorer偏差，learned缺失保unknown，不生成任何冒牌prior。测试不是真实三臂或科学证据。

## 下一可执行子目标与明确未完成项

先由MAIN决定/认证common-backbone路线；对当前P2路线最小外部输入是**真实prior到当前encoder/preprocess/history/normalizer坐标的配对证明**，或允许三臂统一迁到已有Hi backbone的新合同，并提供原冻结raw inverse统计。没有这些，不能合法跑真实三臂，更不能新训repair。独立可继续的代码slice是明确arbitrary direct-latent optimizer和统一target-tracking低层接口source-first准备，但预算/目标函数/初始化必须先声明，不能用已有model Q或物理结果调它。本版不假称已经实现真实优化器、full runtime/物理成功或独立确认；P1关闭，known-root工程准备与科学结论分开。

## 实际验证记录

指定inner venv命令实际 **39 passed in 0.43s**：固定catalog ABI/report-receipt字节join、四plan到两个compiled指针、order/model/source篡改、缺prior非零填充、实际saved-only prepare→verify及CLI roundtrip、外部声明与完整synthetic资产仍不自授权、共同encoder/context引用及predictor边界不匹配、raw inverse/任意目标冒牌/privileged warmstart/poststate/tolerance拒绝、API/sample/batch/失败不退款，以及cold subprocess禁止torch/NumPy/physics/生产runtime模块导入。fixture不认证模型计算或实际支出。

另实际调用prepare到stdout（未新写workspace结果），得到6分支、2known roots、`blocked_missing_assets`，六项blocker正是budget加五角色。只读本次metadata实测：compiled report `56e8b4f2df522f7a0c4cb9bc37ad74bcbc5d4f89e2c5cd601d42cd0eb071d8ef`、receipt `c491555fda5bb543af69f70987541edb0abb99fa32c559f3aa6190901393017f`；Hi low report `68d23f06b88114d3f3a3df227cefe7ca247cb399ba452defbf5456dbda9f32d0`、receipt `96be2b84aadabc4ca23a8e84f2f54f889b86367d41d4d344e0d7d56b46e3b43e`。这些是字节/ABI核对，不是本worker重跑full strict；原已有结果不当新研究产物。所有旧冻结三件套未修改；无生产NN/physics/full strict、无新根与新科学结论。
