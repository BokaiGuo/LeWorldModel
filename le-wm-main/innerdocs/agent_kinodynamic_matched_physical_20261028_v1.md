# Matched-domain beam：两个原 moving root 真实物理 adapter

只新增本stem source/tests/本文；producer三文件继续冻结。采用experiment-code的固定输入、预算与失败日志流程；用户仅授权实现/RAM测试，故不训练、画图、运行模型/物理或创建后代。旧个人blocked goal未覆盖或假complete。

## 权威与源链

只接受新producer `agent_kinodynamic_matched_domain_20261028_v1` 的实际report SHA `d4976aeca357b5324c74be8b2b7e3bfc254e82b834b3acfe1b0ab90a1ef4614c`、实际CUDAprepare SHA `de3db628c5bd15f62f700fa56c917582d5e9fa27cf3cb979abf25a181d5e8dc9`。source/tests/doc严格pin最终 `8e6d03cd… / e724f10a… / eca8217e…`，源码常量含全SHA。prepare调用producer自己的**纯saved consume**验证全部280nodes、whole/segmented重放、288API/samples/module完整账；不调用producer.run、不新调用模型。不是独立神经验证，不借旧rawbank成绩。

固定只有2plans：(source_slot0,matched_domain_H2_beam20x6)、(source_slot1,同arm)。不能混旧kinodynamic、CEM/MPPI、GN/D2P或删负结果。每条原plan完整selection digest保留；selected_path对应冻结bank精确拼接raw10，保bankSHA、task/blind/source/model/report/prepare/artifact闭包。

原moving task loader，不用旧zero-history padding loader。原raw历史float32前5[.1,0]后5[0,.1]；任务original192 goal/identityQ、原normalizer、原RGB3与sourcepack c2d8d573f94884ffc081163f25f2bfa269f98ffe8155887704109219910ffec6 完全不变。producer原plan没有logits/task_contract字段；task bridge仅在内存从该plan已认证path/bank和SHA认证export导出validation副本，不改原plan字节或伪造producer字段。使用原encoder.load_task验证source_slot/task_id/blind/history/goal/raw/tanh/logit球。实际hidden257 tokens与本无模型物理执行无关，不额外假设197。

## 原source重建与执行

不调用任何旧execute/material/源episode/witness controller，不猴补globals。复用封存 `moving_prediction_matched_sampling_physical` 的schema-independent Store、Ops、close_env（source SHA50552d50…）及封存moving kernel的native_hooks/branch。自己编排固定两个fresh env，每根1arm；source/witness只作root/goal evaluator恢复，不作候选选择。

原root seed reset → 原placement recipe/placement reset → 原nonzero warmup10。真实0/5/10 RGB/state/snapshot须与原source精确相同，goalRGB及goal同时验证；不能用saved poststate作reset替代。离线goal attach无physics，不改变goal_pose，with_target=False/render_action=False/RGB224。再执行该arm own raw10，不二次tanh、clip、/100、重选或重试。遵守原native terminal/truncate与agent+object成功定义；wrapped/object诊断不能替代native。每rawcommand fresh RGB guard，先保可用RGB/state/guard收据再停；step内部异常可能没有故障后新帧，不能把旧帧称真实故障现场。

## 固定分母与实际计量

科学分母始终2root×1arm，两个row执行前预置。完整时：2construct/2close、4reset、8setter、4goalrender、2goalbind、40public（20warmup+20own）、408space.step、44explicitrender/PNG（每分支warmup0..10与plan0..10各11）。这几轴重叠，不能相加为总成本。checkpoint计划0/5/10共6，复用实际capture，不额外render。earlystop降低实际counts但不降分母，缺checkpoint保留缺失、不补帧。

Ops记录pre-dispatch intent与真实callback entry/return/failed，失败不退账；hook计内部space.step，constructor在hooks前按已冻结native源码证明pristine；中途构造异常工作量未知，不填0。module neural调用0，不训练，不拟合Q。继承288predict/AE/predictor/pred_proj单列，不与物理费用混合，不等于等FLOPs。

output NEW32MiB硬write cap，512KiB final reserve、200MiB floor；prepare≤256KiB，不复制大闭包；wall≤600s从Store创建起，在dispatch和普通write检查，不能抢占阻塞native。cleanup/final尽力保存可越wall但不能越cap/floor。完整operation receipts、event hashchain、所有44PNG、固定2rows、primary/secondary、source闭包与report保留。坏journal/close/source/hash/report不放行成功。nativeearlystop可作为实际工程终态，非horizon全完成或科学成功。

## Selection manifest与CLI

MAIN独立NEW manifest（schema和authority不要冒用旧模块）：

```text
schema: agent_kinodynamic_matched_physical_20261028_v1.selection
sourcepack / sourcepack_sha256: 原moving sourcepack绝对路径 / c2d8d573f94884ffc081163f25f2bfa269f98ffe8155887704109219910ffec6
source_prepared / source_prepare_sha256: 原moving source实际prepare绝对路径 / 实际SHA
roots: [{root_slot:0,root_sha256:原第一根},{root_slot:1,root_sha256:原第二根}]
arms: [matched_domain_H2_beam20x6]
authorities: [{authority:agent_kinodynamic_matched_domain_20261028_v1,
  output:真实新CUDA结果绝对目录, report_sha256:上述d4976…完整SHA,
  source_sha256:最终8e6d…完整SHA, tests_sha256:最终e724…完整SHA, innerdoc_sha256:最终eca8…完整SHA}]
selection_fixed_before_physics: true
physical_outcome_selection: false
wall_seconds: 600
```

sourcepack/source_prepared/roots可从原moving来源manifest取**来源字段**，不能复用其schema、arms、authorities或selection digest；不能从物理outcome挑根。用manifest实际外部SHA封存。

在 `/home/gbk/code/LeWorldModel/le-wm-main`：

```bash
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -B -m pytest -q -p no:cacheprovider tests/test_agent_kinodynamic_matched_physical_20261028_v1.py
.venv/bin/python -B -m experiments.spectral_reachability.agent_kinodynamic_matched_physical_20261028_v1 prepare --manifest MAIN_NEW_MANIFEST_ABS --manifest-sha ACTUAL_MANIFEST_SHA --output MAIN_NEW_PREPARE_ABS
.venv/bin/python -B -m experiments.spectral_reachability.agent_kinodynamic_matched_physical_20261028_v1 verify --prepared MAIN_NEW_PREPARE_ABS --prepare-sha ACTUAL_PREPARE_SHA
# MAIN审读授权后独占执行，worker不运行：
.venv/bin/python -B -m experiments.spectral_reachability.agent_kinodynamic_matched_physical_20261028_v1 run --prepared MAIN_NEW_PREPARE_ABS --prepare-sha ACTUAL_PREPARE_SHA --output MAIN_NEW_PHYSICAL_DIR --authorize-physics
```

本版本未另建saved physical consumer：report保存prepared/prepare_sha、全部实际operation_receipts、journal hash/tail与checkpoint receipt（PNG/fileSHA/pixelSHA/state/snapshot/observation_state）。后续独立RGB/saved消费者须先verify本prepare，核原journal/44PNG/固定2分母，不得套旧4行assembler。本次不生成prepare/manifest或actual物理证据；MAIN负责审读/调度。known2非独立样本；P1与formal scientific admission恒false。
