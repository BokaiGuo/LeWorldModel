# P5冻结模型入口：有限资产接线，不训练、不运行

只新增本stem source/tests/本文，保留旧P5与baseline readiness封存；无子代、模型加载、神经/physics、网络/下载、pickle扫描或反序列化。本模块复用已冻结`agent_baseline_launch_readiness_20261028_v1`纯stdlib hash/命令接口并核其源SHA；不重复做全仓库发现。

## 已存在的真实入口，而非泛用manifest

|候选|真实冻结入口及保存事实|不是哪种证据/缺口|
|---|---|---|
|TwoRoom / LeWM CLS192|`p5_local_asset_binding_20261027_v1 --replay-cuda`；实际report SHA `8e035871f205c734d96936d5588ec902f8ef4a72a98555f6f2f6cf4a6ac72ada`，`replay_completed`，frozen/buffer/参数/input gates均true|同LeWM架构的新环境权重，不是独立架构；已做过，无需重跑来增加样本。原calibration fail保留，不按结果换根|
|PushT / JEPA-WM AdaLN patch384+proprio|`p5_jepa_fresh_fidelity_main_20261027_v12 cuda`；实际report SHA `d8667a4e8537a5e005d13507d4d05b8d7907996549889decc8a00e1db873eda6`，`fresh_native_panel_fidelity_recorded`，predictor/proprio strict exact，参数/源不变|真实独立动力学**候选族**，不是只有encoder；已有native factual诊断不是P5机制+真实任务收益复制，不重跑已知根充新样本|
|Cube / LeWM CLS192|本地官方cache config/weights/model card在原41项manifest内；weights SHA `2839a907362f403f9136383016e91774373a295d958ae75121791f22a9fddf89`，config SHA `4d446944fe28922cc2c5763f43d4ef9132a457bd89e9a0ce5dbceac183994999`|action_encoder input25＝5维raw×5，不能套TwoRoom10维模板。已知`run_live_cube_action_adapter_planning.py`要求`--adapter-artifact`，本轮排除；未认证source-native冻结5维normalizer/新runner闭包，actual command=null|

TwoRoom、Cube、PushT、Reacher官方cache均`stable_worldmodel.wm.lewm.LeWM`，是同架构跨环境/不同权重，不能数作四架构；JEPA的pusht/pointmaze/wall三checkpoint同AdaLN家族，也不能数三架构。DINOv2视觉encoder缓存不是第三个动作动力学模型。源码修改/输入proprio信息边界、已有预训练成本须披露；本工具不把dirty native源码或本地acquisition hash当远端官方真实性证明。

## 精确有限输入与前置

源码SPECS固定7份small JSON的实际file SHA：原41项asset manifest、TwoRoom实跑、architecture v2、native v12 prepare/实跑、P5 closure、native C1 outcome汇总。只读取已知状态/身份，不读取outcome选root；不自动扫描latest、不跟目录时间选最优报告。

- TwoRoom沿原41项清单逐个核hash/原缺失存在性，保留已知缺DINO-WM dynamics；严格核原8个calibration seed202610360–367，原runner只取预声明首根202610360。HDF5只stat已知`/home/gbk/.stable_worldmodel/datasets/tworoom.h5`的12,775,849,984 bytes，不读/全hash它；实际训练packet pool身份由原runner在MAIN执行时核，不现场fit normalizer。旧41项中额外JEPA/cache是原verify_manifest闭包依赖，不把它们当TwoRoom额外训练。
- native沿已认证prepare的988项绝对路径绑定核hash，不import native模型代码；精确核三根2026102702/03/04、training=false、CPU fallback=false、root_selection_changed=false、P5=false及strict loads。`report.manifest_sha256`是compact sorted JSON canonical摘要，**不是manifest文件SHA**；源v11的digest口径已只读核实，分别验证，不报假漂移。prepare文件SHA `41ef9bf8f5c5ae5c4ff8890a25e6a83137b7e637058a8033281a5eabc9cea07f`，报告canonical摘要为`9283f25481795ecee6c0f8bdfbf5ee9f83bee96b945bd4ca80e071332dd83aa1`。
- 文件读取不递归，只沿固定已hash清单；共享文件hash缓存一次。最多1100个唯一hash、1.5GiB总字节；缺失/漂移列具体path/expected/actual。二进制只是raw hash，不解pickle。dataset全SHA、远端revision与license的独立认证仍unknown；本地MIT模型卡不升级为法律/远端真实性证明。
- native原988项包含`/tmp/lewm-dinov2-source-DSuJiP7Y/official.tar.gz`，预期SHA `c27dcdaf50e9fb5bbdf2bb529da357716372e19c6afab17d5350f3f0094aed4b`。仅此精确路径被纳入只读allowlist，不扫描/tmp、不解tar、不换镜像或下载；文件丢失/漂移必须阻断，不把其存在追认为远端认证。
- 最新P5 closure仍`saved_evidence_valid_P5_blocked`；native C1已执行事实保留，不用其值补P5。不训练repair/teacher/adapter/readout来凑独立模型。

## 实际可复制命令（本agent只运行纯离线检测和RAMtests）

cwd `/home/gbk/code/LeWorldModel/le-wm-main`：

```bash
OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m pytest -q -p no:cacheprovider tests/test_agent_p5_frozen_replication_assets_20261028_v1.py
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m experiments.spectral_reachability.agent_p5_frozen_replication_assets_20261028_v1
```

默认stdout、不写results；exit1表示至少一个候选blocked（Cube当前预期），exit2工具错误。ready只叫`ready_for_MAIN_frozen_runtime_preflight`，不等于launch授权或科学准入。输出可复制的原`p5_native_stage_admission_20261027_v1 --require-admission`saved-only命令（预期exit3），交MAIN复核，不在本模块执行该审计。

若MAIN另有**新诊断目的**，可只生成一次真实NEW命令，不执行：

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m experiments.spectral_reachability.agent_p5_frozen_replication_assets_20261028_v1 \
 --new-run-candidate native_jepa --new-run-output "$MAIN_NEW_ABSOLUTE_RESULTS_CHILD" \
 --main-purpose "$MAIN_PREDECLARED_NEW_PURPOSE"
```

native实际argv为原v12 `cuda --manifest <实际manifest_frozen.json> --output <实际NEW目录>`，只用已保存factual输入，**0新physics**。TwoRoom选项`--new-run-candidate tworoom`生成原`--replay-cuda --binding-manifest <实际41项manifest> --output <实际NEW>`，输出名须原runner要求的`p5_local_asset_binding_20261027_v1`前缀；它会做**15个原warmup物理动作**，因此另需MAIN physics批准，不能宣传成纯CUDA离线回放。二者均要求NEW直属results目录、MAIN新目的及200MiB floor，另由MAIN批准实际output/wall/显存预算/设备/可信加载/许可；本模块不修改原caps或原统计。仅hash不能证明硬件runtime ready。

## 未完成

实际RAM测试：本stem **15项通过**，与baseline readiness联合 **28 passed（0.43s）**。真实只读检测共1031个唯一hash、1,043,030,411 bytes；TwoRoom原41项中38个存在资产全匹配、3个原缺DINO路径仍缺，native988项全匹配。二者为ready_for_MAIN_frozen_runtime_preflight；Cube保持blocked。没有执行原模型/consumer/physics或输出新实验结果。

未认证独立许可/远端revision；Cube没有本轮允许的native冻结normalizer+无adapter runner；未建立P1/core/适用组件冻结、本模型P0/任务前提、独立根预声明、机制及真实任务收益复制/强对照与E3矩阵。`independent_architecture_replications_established=0`、P5/P1不开放。既有已完成factual或warmup诊断不替代这些义务。RAMtests不算复制成功。freeze SHA在交付回复中列出。
