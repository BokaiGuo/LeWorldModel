# Moving三反馈边界真实driver v1（MAIN运行）

仅新增本stem source/tests/本文。使用experiment-code技能：先固定完整协议/输入/预算，后RAM
回归；本worker不运行模型、CUDA、物理、网络，不派生。旧源码/封存证据完全不改。

## 固定实验，不是旧single-feedback wrapper

复用原moving两个任务、GN/D2P两种封存prefix，四层各I/R两臂，共8episodes。相同原prefix5，
随后在真实控制step5、10、15分别编码并重新编译一个5步包，最终各执行20步，warmup另10步。
三个边界与8×3=24编译槽预固定，所有失败/未达槽保留，不重试/补根/挑结果。
两臂每边界都付相同编码+冻结H1 bounded-ridge计算；I滚自己的predicted successor，R使用
自己的新RGB latent。terminal20只编码/评价，不成为第四次控制决策。物理early stop原样保留，
不padding到20，不将截短臂计成完成或据此宣布equal actual compute。

这是新实际循环：native构造→精确reset/warmup→prefix5→observe/encode/compile/seal→step6–10
→自己的下一边界→step11–15→第三边界→step16–20→native/latent分栏终点评价。
run显式加载一次HOST CUDA冻结模型，实际调用原Handler visual、冻结evaluate及native环境；
不是外部driver待填的callback-only状态机，也不回调原single-feedback.run。

## 实际沿用与缺口适配

沿用原 `Handler`、`Store`（新子类只增PNG读回）、`Meter`（子类设置新固定caps）、`Sink`、
`native_factory/metered_class`、`capture`、`native_transition`、`DispatchGate`。
setup复现原driver同样的native两次reset与完整warmup0/5/10 RGB/state/snapshot比对，之后
连续执行，绝不restore到prefix或中途reset。原goal7/RGB绑定与goal_pose不变，with_target=false。
根/goal和冻结encoder/Q/normalizer保持原样，不生成新根或拟合latent mapper。

原encoder只接受step5/10和严格PNG名。每个实际边界建唯一`e.._b10`等namespace，把**当前同一
PNG字节**复制到legacy `_plan_05.png`名，先保存实际step/原PNG SHA→legacy名映射，再调用
原Handler.encode_observation。该`legacy_step=5`只是兼容单帧编码命名，不谎称物理step5；
reported actual_step仍5/10/15/20，原帧文件/像素/encoding SHA、观察/编码事件顺序全绑定。
不借旧prefix/terminal图，不新增render，不以重复命名增加独立根数。

旧suffix.validate_packet硬要求history首包=warmup后5且step=5，不能安全用于10/15。因此
实现RollingSuffix的严格新白名单，直接复用冻结 `moving_history_observed_suffix.evaluate`、
`Modules/SuffixAD/Meter`、原 `PredictAPI`、`TorchBackend`、`check_result/saved_arrays`，
不patch旧validator、不调用旧loader、不重写bounded ridge或Q定义。所有完整native模块数组、
J/SVD/AD及5次predict API和返回均由原hook保存/核对，失败不放行旧suffix。

每次raw10恰为最近两个**实际完整执行**5步包，public float32 subtract/divide为两个10维token；
无/100、clip、历史tanh、零填充。R缓存本episode历史编码并追加当前编码：step15恰为其
step5/10/15三个actual latent；I从原同prefix预测开始，后续只接本臂新动作的预测successor。
两臂都以本臂真实已执行动作更新history，不能跨臂借state/图/latent/actions。

compiler精确白名单只有mode/boundary/episode/layer/context/raw/原goal192/modelbinding及当前
观测哈希；无env、native state、snapshot、native success、物理goal7或未来图。hidden state
只用于拒绝/评价，不能选动作。诊断old_suffix固定为该层原suffix5，每次都预测但永不fallback。
编译器无候选outcome筛选：固定新bounded-ridge动作无条件执行，原代价仅诊断。

每边界完整request/数组/response/dispatch seal先durable落盘才发下一步；调用原DispatchGate
时6..10是**包内legacy ordinal**，journal另存真实global step6..20，不混淆后两轮时间。
编译前后snapshot/goal精确不变；pure verifier后不许额外predict。PNG/array/journal均用原Store
fsync、独占命名与配额。startup先落完整8episode/24slot分母，异常保持现场。

## 预算和未知

完整8臂：32encoder/projector各、120predict API/predictor/action_encoder/pred_proj各、
24AD-forward/J/SVD/jacobian API/内部grad API各、4608数学VJP方向；每臂4编码、15predict、
3J、20control+10warmup。共享来源/旧encoder成本仍unknown，不去重成本。
native caps：constructor8、reset24（含可能隐式reset）、setter48、goal_render24、goal_setter32、
goal_bind8、public240、physics2448、explicit_render256；每个attempt/return独立持久化。
600s含验证/模型加载，64MiB整output、200MiBfloor、256KiB final、3GiB allocated/4GiB reserved
沿原限制，不放宽。真实output或wall不足就fail closed，不按mock大小保证真实能跑完。
FLOPs/kernel次数/device时间/inherited compute未知null；同调度已知API/执行步数不等全算力。

native_success仍原eval_state，位置包含agent+object，角度语义不改。native_distance、object
position、wrapped angle、identity-Q各自报告；未达/缺测null或缺字段，不补零。
测试不等科学证据；P1/P4/freshness/independent confirmation始终false，只known-task工程校准。

## MAIN命令（worker未执行模型/物理）

```bash
cd /home/gbk/code/LeWorldModel/le-wm-main
OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m pytest -q -p no:cacheprovider tests/agent_p4_moving_multiboundary_20261028_v1_tests.py
.venv/bin/python -B -m experiments.spectral_reachability.agent_p4_moving_multiboundary_20261028_v1 prepare --output results/agent_p4_moving_multiboundary_20261028_v1_prepare_MAIN01.json
```

prepare输出真实prepare_sha256后，MAIN使用该SHA：
`verify --prepared ABS_PREPARE --prepare-sha ACTUAL_SHA`。
审读后MAIN才可：`run --prepared ABS_PREPARE --prepare-sha ACTUAL_SHA --output
results/agent_p4_moving_multiboundary_20261028_v1_MAIN01 --authorize-known-cuda-and-physics`。
本worker不创建未来SHA、真实授权或结果。不支持正式efficacy/confirmatory放行。
