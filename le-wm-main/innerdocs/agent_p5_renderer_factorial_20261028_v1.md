# 原生固定状态renderer factorial：完整设计阻塞，禁止伪造toggle

仅本stem新source/tests/doc。未运行任何物理/render/神经/网络，未修改原环境或封存图像。此版本诚实交付阻塞诊断与可复用原生target-color live kernel，不宣称20图已可执行。prepare允许保存blocked planning artifact，不开放physical release。

## 原生接口核对与确切阻塞

绑定 `external/jepa-wms/evals/simu_env_planning/envs/pusht_env/pusht_env.py` SHA `6ba34790b6c2ce017a76322c2f1c205ef97d4d3b08a620f844a57075c608a723`。`_render_frame`每次先把白背景填满，再逐个block polygon以`self.goal_color`绘goal，随后`self.space.debug_draw(DrawOptions(canvas))`。`with_target`只在`_setup`决定LightGreen/White；事后改with_target不改变goal绘制。合法target诊断用同一native字段暂设LightGreen/White并finally恢复原对象；white不是不调用goal polygon，而是原生白色目标样式。

绑定的是 `.jepa-physics-site/pymunk` 而不是项目venv较新pymunk。Shape、Space.debug_draw、SpaceDebugDrawOptions和CFFI绘制callbacks的当前实际源码都没有`ignore_draw`读取/属性合同；仅examples/platformer.py、spiderweb.py的动态赋值不能证明其有效。`Space.debug_draw`的C路径经color_for_shape只读取color/体类型，Python路径只逐shape调用draw_shape；没有已认证的单agent visibility filter。**本模块拒绝agent hidden因子，不通过设置任意动态属性制造通过。** 此处是缺认证，不冒称已实际render证明ignore_draw无效；任何属性/调用路径变化要求新版本审查。不能改DrawOptions、移除shape、改agent颜色成白色/透明、挪agent、涂抹像素来代替原生隐藏。

第二硬阻塞：既有4word env及goal env已经close，保存的是完整公开数值/RNG/shape记录，不是存活handle或隐藏solver缓存完整序列化。原`_set_state`必调用一次Space.step；`reset`调用setup及_set_state。禁止本模块用它们恢复终态，也不逐setter拼装近似live状态或重新物理重放。解除此阻塞需要MAIN另行授权保留原生live env的上游捕获安排及合法agent-hide API认证；仅补文件hash不能解除，不提供未来SHA。任何要求改native renderer的方案都超出本版本“仅原生开关”权限，必须先获新协议授权。

## 固定设计与计量

四词xy/yx/x_minus_x/zero_zero终态各4条件=16，另独立goal状态4条件，共**20**固定分母。固定顺序每state为agent visible×target visible、visible×white、hidden×visible、hidden×white。goal object=[256,256,pi/4]；goal agent=[64,448]仅渲染nuisance，不是coverage目标。四词原target=white，goal原target=visible；均绑定实际v5 full strict及原goal报告/generation/geometry/journal/完整frame0文件，原goal RGB逐字节等于public goal。

全factorial `run`在任何native import/constructor之前拒绝，输出20个not_run和两个硬阻塞，实际所有计数0、assets=[]。consume只可认证“阻塞记录正确，NOT rendered”，不能发20图验收证书，不能把这些行算科学负结果。没有重试/填0图像/旧RGB作为新输出。

`render_live_target_pair(env, expected_full_public_state, sealed_rgb, main_authorized=True)`是独立受支持的1×2核，不是full factorial CLI放行。必须MAIN已持有原生且精确匹配sealed full-public-state的live env，原renderer文件和physics overlay来源验证；无torch进程。先原色baseline渲染并与sealed RGB原像素逐字节门，再visible/white两条件；共3次native render/Space.debug_draw/_get_goal_pose_body/独立Body构造，均实际call/return计量，失败attempt不退款；不把原goal pose body的detached构造当作env推进。

每次前后保存完整公开body/shape/Space/RNG/env合同，且Space/agent/block/shapes/bodies对象身份不变；goal_color finally恢复原对象。screen是原renderer必更新的纯显示cache，明确排除于物理state，window/clock须None，render_action须False。guard拒绝reset/_setup/_set_state/step/Space.step/seed/constructor/close在执行体前进入，不覆盖native函数。每次render要求精确1render/1debug_draw/1goal_body/1detached Body call/return。完整私有缓存未被序列化；不把公开state证书称hidden solver精确恢复。env属MAIN，kernel不关闭或替换；失败保留异常，不能假装恢复被非法修改的物理状态。

若未来五个live state齐备，只做target pair核也需15renders（5 baseline+10新条件），而不是20全设计已完成。当前CLI不提供用saved快照启用它的入口。full20预算不可预测为已执行；无model评分、encoder或planner，P5/scientific_admission始终false。不能从RGB差异推出物体机制或C2收益。

## 测试与实际prepare命令

cwd `/home/gbk/code/LeWorldModel/le-wm-main`。prepare/verify/consume纯saved；tests只RAM记录与实际saved验证，无pymunk/env实例化。

```bash
OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m pytest -q -p no:cacheprovider tests/test_agent_p5_renderer_factorial_20261028_v1.py

OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m experiments.spectral_reachability.agent_p5_renderer_factorial_20261028_v1 prepare \
 --manifest results/agent_p5_renderer_factorial_20261028_v1_prepare_MAIN01.json \
 --authority results/agent_p5_terminal_goal_attribution_20261028_v5_prepare_MAIN01.json \
 results/agent_p5_terminal_goal_attribution_20261028_v5_cuda_MAIN01/report.json \
 results/agent_p5_terminal_goal_attribution_20261028_v5_consume_MAIN01.json
sha256sum results/agent_p5_renderer_factorial_20261028_v1_prepare_MAIN01.json

OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m experiments.spectral_reachability.agent_p5_renderer_factorial_20261028_v1 verify \
 --manifest results/agent_p5_renderer_factorial_20261028_v1_prepare_MAIN01.json --manifest-sha256 ACTUAL_PREPARE_SHA
```

可选MAIN写阻塞执行记录（明确exit2，无native运行），不是建议重跑物理：

```bash
OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m experiments.spectral_reachability.agent_p5_renderer_factorial_20261028_v1 run \
 --manifest results/agent_p5_renderer_factorial_20261028_v1_prepare_MAIN01.json --manifest-sha256 ACTUAL_PREPARE_SHA \
 --main-authorized --output results/agent_p5_renderer_factorial_20261028_v1_blocked_MAIN01.json

OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m experiments.spectral_reachability.agent_p5_renderer_factorial_20261028_v1 consume \
 --manifest results/agent_p5_renderer_factorial_20261028_v1_prepare_MAIN01.json --manifest-sha256 ACTUAL_PREPARE_SHA \
 --report results/agent_p5_renderer_factorial_20261028_v1_blocked_MAIN01.json --report-sha256 ACTUAL_BLOCKED_REPORT_SHA \
 --output results/agent_p5_renderer_factorial_20261028_v1_consume_MAIN01.json
```

所有写入exclusive不覆盖失败/封存结果。protocol列未来工程上限600s/8MiB/200MiBfloor，但当前没有获准native job，不能把此声明当实测资源cap认证；live target kernel只是调用级原生核，不具备独立生产job的wall/disk持久化budget，不能单靠它签新数据集release。缺合法agent-hide接口与live-state来源认证前，不提供可绕过这两项的physical命令。
