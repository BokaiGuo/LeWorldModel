# P5 原生target visible/white静态渲染诊断 v2

独占本stem source/tests/doc，v1及封存producer保持不动。此版本落实MAIN补充授权：可以原生constructor→`_setup`，无reset/`_set_state`/Space.step地复原公开静态render-view属性；不是只能写blocked metadata。完整agent×target factorial的agent-hidden仍拒绝，不删词/挑根；实际支持四固定终态+goal五状态×target visible/white十条件，并额外生成五张原色baseline。没有新模型/encoder/控制/物理推进，也不把渲染差异当planner新分数、C2或任务收益。

## 更正原色来源

v1 `original_target='white'`是错误标注，不能作为证据。四词原physical prepare中的constructor.with_target=true，四封存full_public_state中的settings.with_target=true；goal同为true。`external/jepa-wms/evals/simu_env_planning/envs/pusht_env/pusht_env.py:704–707`：true分支在705给goal_color赋`pygame.Color("LightGreen")`，else才在707赋White。原contact source contract `constructor`（83–86）只改reset_to_state，不改with_target；`agent_p5_action_words_physical_20261028_v1.py:108`声明with_target=True。四词原physical execute调用同constructor+reset；未事后改goal_color，原生_render_frame在约592行使用该字段绘polygon。v2将所有原色标为source-inferred visible，**goal_color未被旧public state序列化**，因此仍须逐状态baseline真实RGB字节复现才承认新toggle结果，不拿推断替代图像验收。

## 静态重建与严格边界

读取实际v5 prepare/report/receipt，原full consume返回值精确匹配；复用v1只读来源解析，复制出新对象，不改旧globals/source。五源full public state/RGB、goal generation/report/geometry/journal均保留绑定；source/model/旧结果不改。原goal object_pose=[256,256,pi/4]，agent=[64,448]只是渲染nuisance，不是coverage目标。

新env constructor settings来自各源公开settings（包括原legacy、shape、CoG、damping、with_target），reset_to_state=None因无reset永不消费该字段。constructor原本只声明/seed，不创建Space；随后显式原seed、原`_setup`一次。_setup产生的local shape/material/filter/color inventory须逐项精确等于源，不伪造T或改geometry。若某源有非默认shape属性，这一版本明确拒绝而非猜配。只通过公开body/Space属性setter复原body mass/moment（动态体）、CoG→angle→position、velocity/angular_velocity/force/torque；非动态mass/moment和body_type必须本来一致，sleeping源拒绝。各body调用原生`Space.reindex_shapes_for_body`更新debug_draw需要的变换缓存，三次/状态；这不是Space.step。

latest_action/coverage/contact计数/teleop与两个RNG恢复自原源；goal、settings、controller参数等未改的字段仍纳入全量精确核验。不调用_set_state、不把native float32 state7冒充double snapshot。不对浮点roundtrip放宽容差，任一公开字段不匹配或原色RGB不等即保留失败。没有step因此不生成新接触/coverage结果；复制原contact/history不是新观测。

唯一显式source差异为`space.current_time_step`：保存值0.01，而新Space零步值0.0。绑定pymunk/space.py:359–369只有getter，禁止写私有C结构或用step伪造0.01。`view_state(source)`仅为**比较投影**将此单项期望置0；不改变源文件或native getter。完整源公开state不声称全部相等，报告`source_full_public_equal=false`和该精确差异。其他所有公开body/shape/Space/env/RNG字段必须完全相等；每张图渲染前后实际完整公开state（包括实际0 timestep）精确一致。private contact cache未恢复，不认证动力学重启、fresh或环境新实验。

五状态先依次construct/seed/setup/hydrate并真实原色baseline capture，baseline新NPY必须逐字节等于原sealed NPY；五张全部通过后才开启十条件。visible/white只变原生goal_color，with_target不变；每状态finally恢复原颜色对象并重新核全量公开state。native renderer必换screen surface，这是显示cache，不是物理state；原render_action=false，offscreen；无像素涂抹、自定义DrawOptions、shape removal或agent移动。visible重复图也必须与baseline byte相同，不复用旧RGB充新结果。

## 固定预算、清理与saved严格验收

完整运行：constructor5、seed10（原constructor内5+显式5）、_setup5、Space构造5、公开view_restore5、body reindex15、native render15、debug_draw15、goal_body15、Body构造30（_setup的static/agent/block共15+render的detached goal15）、close5。reset/_set_state/control/Space.step/额外_get_obs全0。profile在native Python入口计量并于禁止/超cap入口执行体之前抛错；实际stage另有持久intent/returned，失败不退款。Python return事件不等价C solver成功；本模块不运行solver。五env全close，任何cleanup/资源/source漂移使strict拒绝。失败完整十cell分母保留not_run/attempted/completed状态，不补图、不重试。

32MiB总输出含1MiB尾部reserve，200MiB free floor，600s从授权run的runtime加载前设alarm，writer再计量实际phase；C调用可能推迟Python超时，consumer严格wall<600。源码最终saved核验在计时内；输出exclusive目录，文件fsync+目录fsync。40个固定operation各intent/return共80条链，另terminal共81条；consumer逐操作重放精确native计数增量，核五baseline在所有toggles之前、固定十cell、15张NPY SHA/bytes/shape、每次before/after、readonly差异、颜色/恢复/close、目录完整inventory及report/terminal身份。任何partial或新增未知字段不能签成功；不修改原日志reader或门槛。

运行runtime probe复用原`.jepa-physics-site`和原项目venv，不加载神经。原public reader和native setter源码、Chipmunk二进制SHA进入新prepare闭包。新的静态setter复原实现必须MAIN完整读码及实际baseline验证；RAM通过不是native静态复原成功证据。

## 精确MAIN命令

cwd `/home/gbk/code/LeWorldModel/le-wm-main`。worker只执行tests及saved临时prepare/verify，不运行以下native run。

```bash
OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m pytest -q -p no:cacheprovider tests/test_agent_p5_renderer_factorial_20261028_v2.py

OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m experiments.spectral_reachability.agent_p5_renderer_factorial_20261028_v2 prepare \
 --manifest results/agent_p5_renderer_factorial_20261028_v2_prepare_MAIN01.json \
 --authority results/agent_p5_terminal_goal_attribution_20261028_v5_prepare_MAIN01.json \
 results/agent_p5_terminal_goal_attribution_20261028_v5_cuda_MAIN01/report.json \
 results/agent_p5_terminal_goal_attribution_20261028_v5_consume_MAIN01.json
sha256sum results/agent_p5_renderer_factorial_20261028_v2_prepare_MAIN01.json

OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m experiments.spectral_reachability.agent_p5_renderer_factorial_20261028_v2 verify \
 --manifest results/agent_p5_renderer_factorial_20261028_v2_prepare_MAIN01.json --manifest-sha256 ACTUAL_PREPARE_SHA

PYTHONPATH=/home/gbk/code/LeWorldModel/le-wm-main/.jepa-physics-site PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 \
 .venv/bin/python -B -m experiments.spectral_reachability.agent_p5_renderer_factorial_20261028_v2 run \
 --manifest results/agent_p5_renderer_factorial_20261028_v2_prepare_MAIN01.json --manifest-sha256 ACTUAL_PREPARE_SHA \
 --main-authorized --output results/agent_p5_renderer_factorial_20261028_v2_render_MAIN01
sha256sum results/agent_p5_renderer_factorial_20261028_v2_render_MAIN01/report.json

OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m experiments.spectral_reachability.agent_p5_renderer_factorial_20261028_v2 consume \
 --manifest results/agent_p5_renderer_factorial_20261028_v2_prepare_MAIN01.json --manifest-sha256 ACTUAL_PREPARE_SHA \
 --report results/agent_p5_renderer_factorial_20261028_v2_render_MAIN01/report.json --report-sha256 ACTUAL_REPORT_SHA \
 --output results/agent_p5_renderer_factorial_20261028_v2_consume_MAIN01.json
```

actualprepare/report SHA必须MAIN实测填写，不预造。测试使用Python metadata/hook fakes（含原public_state读取、真实hydrate函数和真实sys.setprofile）验证成功路径、全baseline-first、禁step执行体、失败持久化/close、saved篡改拒绝；fake图像只是固定返回，不是原生绘制或科学证据。下一状态为待MAIN代码审阅及实际静态render/strict；不宣称已隔离渲染因子成功。
