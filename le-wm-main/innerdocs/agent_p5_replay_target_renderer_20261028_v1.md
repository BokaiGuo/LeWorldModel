# 原生 replay target renderer v1（MAIN 执行，非科学准入）

本版仅新增 source/tests/本文。替代静态 public hydrate 的诊断路径，不修改 v3、原物理、Pauli encoder。静态 v3 的接触红点差异不是允许容差的理由：arbiter cache 未公开，不能从 public snapshot 重建。本实现真正重跑既定四词，保存五个 live space，禁止静态 hydrate/reindex、补画像素、隐藏碰撞点、改 goal_pose、挑词/根、重试、神经/拟合。

## 固定输入与 prepare

`build()` 绑定已审 `agent_p5_renderer_factorial_20261028_v3.py` SHA `b2b41d1aee81d37a1569fe4943fa1b31cb8c7dbee30790fa4c07ee7133a97fcd` 与真实 prepare `results/agent_p5_renderer_factorial_20261028_v3_prepare_MAIN01.json` SHA `fdf76d7f0d404065bdee5abd404d4f39e0f68877263cc4862ce2dc84fe20109a`。调用 v3.verify（纯 saved）后，从 native_manifest.authority.manifest 的 attribution 路径读取原 physical report，再用 report.manifest_sha256 绑定原 physical prepare；manifest_digest 同时匹配。plans 只能来自该原 prepare 的 source_contract.generator_private.plans，不根据效果选择。

固定 state 顺序 `xy,yx,x_minus_x,zero_zero,goal`。每词原 constructor/seed、prefix_raw_u10、reference_suffix_raw_u10 逐项保留，包括 float64 的负零。四词均先 reset/prefix10，全部44 prefix frame 字节门，再各 suffix10，与原 pipeline 相同；第五个 goal 用其封存 scene report.protocol 的 constructor/seed/reset，加原 geometry_witness 的一次额外 goal body。goal scene 源 SHA `ea2e77e43a6e46f9f9c8aafd6c728eeb999cbfefd0284656f5492178c8ece16d`。不拿 v3 为 static hydration 制备的 reset_to_state=None 作 replay constructor。

prepare 完整保存原 states 的 source_state、native_path/native_sha、rgb.path/sha256/bytes，全部85原 observation 的五文件 SHA/bytes 与闭包 bindings（含本三件套），无未来运行回执、无伪成功。`verify(path, external_sha)` 重新 build 全量相等。真实 metadata build 可以运行，但本交付不生成 MAIN prepare/result。

prepare 文件自身上限2MiB、写前至少保留200MiB；不沿用其它实验的256KiB cap。本版原始全精度 metadata 约272KB，不压缩/截断 bindings。metadata build 实际通过，最终字节数及冻结SHA见交接消息。

## 事前资源与原生执行

5 constructor；内含 seed5，再 explicit seed5，共 seed10；reset5、setup5、set_state5、space_construct5；80 public controls，每次10 substep，加5 reset 内部各1步，共 **805 Space.step，dt=.01**。85 getobs，reset5+control80 共85原生 render，再10颜色 render，共95 render/debug_draw。goal body176（四词164，goal reset1+geometry1，颜色10），body_construct191（setup15+goalbody176），close5。public intent 分别 construct5/seed5/reset5/step80/geometry1/color10/close5。不能称 physics0，也不能将 Python return-event 视为异常路径的成功。

run 直接调用 **v3.load_runtime(原完整v3 prepare)**，沿原 environment/pymunk/pygame 来源；不猴补 globals/方法/碰撞回调，不调用旧 execute。sys.setprofile 只观察原函数，逐 entry/return-event、Space.step dt/context 写 fsync hashchain；observer 错误被记录并在下一公开操作封闭，不在 profiler 中抛出以免 Python 自动移除 observer。实际 public intent 在调用前持久化，returned/failed 独立；关闭在 finally 尽力执行。一次异步900s alarm 中断长工作，逐 dispatch/write 同时核 wall；cleanup/final reserve 允许尽力落盘，可能使失败路径尾部超过900s，完整路径 consume 要求报告 wall<=900s。加载含在 execute 的计时内，外部 verify 不含。

32MiB 全输出 cap，工作 payload 留2MiB final reserve，至少200MiB free floor；开工要求 free>=32+200MiB。每次写核实际剩余空间/字节。原85 observation 保存85 PNG/NPY，十颜色再10，共 **95PNG、95RGB NPY**，还保存实际 native state/proprio、transition、公有 snapshot、80原始命令、before/after/restored、runtime probe、journal/report。RGB NPY 约14.3MiB，PNG/公有状态/日志在余量内；超 cap 必须 partial，不缩分母/精度。

先存实际返回图/state/transition/public snapshot，再检查原五文件逐字节一致；五原 baseline（含完整公有状态的 current_time_step=.01）全部通过才允许首个颜色 render。各 baseline 的 before/after 来自 profile 对 **原 native render call/return** 的真实快照，不将事后状态伪称 render 前状态。颜色 render 仅临时设 pygame.Color('LightGreen') 或 pygame.Color('White')；图和 after 先保存，再核 public_state/RNG/topology 不变；finally 恢复原 goal_color 对象并保存恢复收据。颜色不在 public_state，故单列 RGBA/对象身份恢复。LightGreen 输出亦必须与原 baseline 全像素等同，包括原生接触红点。

原 step/render 内抛异常时不能保证有故障后图，绝不把旧 last observation 冒充现场；若 render 已返回但 invariance 失败，真实失败图/前后状态在 stop 前已落盘。原 done 返回必须立即 honor，不补足动作。5状态/10颜色科学分母一直保留；缺槽、异常、计账不完整仅工程 partial，不释放给 encoder，不是任务负结论。final journal/hash/report 分别尽力保存，底层磁盘/IO 连失败时不能保证 durable report，CLI仍返回2并输出错误。

## 新 encoder 接口（不是旧 static schema）

prepare schema `agent_p5_replay_target_renderer_20261028_v1.prepare`；report schema同前缀 `.report`，成功状态 `replay_targets_recorded`，否则 `partial_engineering_failure`。顶层有 prepare_sha256/prepare_digest/protocol、固定 rows5、all_baselines_passed、errors、meter、完整 artifacts、journal_terminal_sha256/journal_events/wall_seconds、scientific_admission=false、task_success=null。

每 row：state_id/status/records/baseline/cells/**shots**。records 为四词0..20，goal仅0；baseline 引用该行最后真实 record。每 record.files 有 rgb.npy/state.npy/proprio.npy/transition.json/full_public_state.json；image 有 npy/png 文件收据及 pixel_sha256。文件收据统一 `{path:相对output,sha256,bytes}`。所有 PNG 是原 uint8 RGB224 的无损保存，无重绘/缩放/插值。

**shots固定顺序 baseline,visible,white**，每槽 `{shot,status,files}`，files 含 **rgb（NPY收据）/png/before/after**；颜色槽再含 restored。baseline 复用 frame_20（goal frame_00）图，before/after 在 `<state>/baseline_before.json` / `_after.json`；颜色图 `<state>/visible.npy/.png` 或 white，前后 `<state>/<color>_before.json/_after.json`。颜色 restored JSON含 color/same_color_object/public_state/topology_unchanged，颜色 cells另含 requested_rgba/original_color/status/image/before/after/restored。失败时固定槽仍存在，但可无文件、status=not_run，不补图。

encoder v2 应要求本版 **strict_saved_complete** 回执及外部真实 report/prepare SHA，精确核本版源码/依赖；从 rows.shots 的 visible/white 读取10图，baseline仅作原图证据，不重复计编码。状态从 prepare.states按state_id闭合，实际 before/after 与 source_state 全字典一致（不删 dt、不 hydrate）。这与旧 static 的 render-view dt0、不推进物理合同不兼容，不可冒充旧 consumer。当前无实际成功声明。

`consume(prepare_path, prepare_external_sha, report_path, report_external_sha)` 不加载环境/模型：重验 closure、全目录文件清单及SHA/bytes、全journal hashchain/native805计数和dt/public intent-return及阶段门、85全帧原始五文件字节、80原动作/绝对目标/前快照、95PNG解码=NPY、5baseline与10颜色前后状态及颜色恢复、visible contact pixels原图全等。缺项 fail closed。返回 schema `.receipt`, status `strict_saved_complete`, states5/cells10/png95/native_entries，scientific_admission=false。

consumer还逐公开 dispatch 重算 native entry/return-event delta，核精确 constructor/explicit seed/原始角色、帧、颜色顺序，以及原 goal geometry 文件字节。public_operations 在回执中为由 journal 实算的111次公开操作。它不消费旧 static receipt，不将静态失败记作本次 replay 成功。

## CLI（MAIN 审读后独占执行）

在 le-wm-main 下，使用原 native 隔离环境（不是普通 `.venv` physics）：

```bash
.venv/bin/python -B -m experiments.spectral_reachability.agent_p5_replay_target_renderer_20261028_v1 prepare --output results/agent_p5_replay_target_renderer_20261028_v1_prepare_MAIN01.json
.venv/bin/python -B -m experiments.spectral_reachability.agent_p5_replay_target_renderer_20261028_v1 verify --prepare PREP --prepare-sha PREP_SHA
PYTHONPATH=/home/gbk/code/LeWorldModel/le-wm-main/.jepa-physics-site PYTHONNOUSERSITE=1 .venv/bin/python -B -m experiments.spectral_reachability.agent_p5_replay_target_renderer_20261028_v1 run --prepare PREP --prepare-sha PREP_SHA --output results/agent_p5_replay_target_renderer_20261028_v1_render_MAIN01 --main-authorized
.venv/bin/python -B -m experiments.spectral_reachability.agent_p5_replay_target_renderer_20261028_v1 consume --prepare PREP --prepare-sha PREP_SHA --report REPORT --report-sha ACTUAL_REPORT_SHA --output results/agent_p5_replay_target_renderer_20261028_v1_consume_MAIN01.json
OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m pytest -q -p no:cacheprovider tests/test_agent_p5_replay_target_renderer_20261028_v1.py
```

PREP/PREP_SHA/REPORT/ACTUAL_REPORT_SHA 是真实执行后提供的参数，非未来常量。prepare和consume输出排他创建，run仅允许本stem的results新子目录。RAM tests只证明控制流/接触cache seam反例/缺槽拒绝/颜色恢复/失败图保留等，不是物理复现成功或P5准入。worker不运行 physics/GPU/network；MAIN负责审读与实际权限。

验证范围：33项 RAM tests 包含完整 execute→strict consume 的无求解器 Python fixture、baseline/缺槽/PNG mutation、接触cache三红点反例、错误baseline保存后停止且10格分母不变、load失败全槽保留。fixture为纯内存文件系统，无环境/模型/磁盘输出；绝非假造实际receipt。真实 source SHA闭包的 metadata build 另已实际检查；未 prepare、未 run、未采集真实图、未宣布原生重放成功。
