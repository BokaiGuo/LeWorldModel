# 原生几何返回观察器 v1（MAIN 执行，非 renderer 修复）

独占 source/tests/innerdoc stem 为 `agent_p5_native_geometry_observer_20261028_v1`。
本协议唯一用途是获得新一次执行中实际原生 world vertices、shape 遍历顺序、GEOS WKB 与 coverage/reward 算术证据。历史 cached transform、历史遍历顺序和历史 matrix 仍 unknown；不回证历史，不 release 旧 renderer，不作 P1 或任务成功结论。

## 固定输入与执行顺序

读取并完整 verify 原 replay prepare `results/agent_p5_replay_target_renderer_20261028_v1_prepare_MAIN01.json`，SHA `c2ecf01381cf339bf8f7e89fa428f4f62ac46075b10458ee7cfc8deac661941d`；绑定失败 report SHA `1b02ed4a2ca136b4dc7f8549073ff030620df8587d654eb2a07ca86036cad0c6`。新 prepare 包含旧完整 bindings 与自身三文件。runtime 直接调用封存 v3 `load_runtime`，不 monkeypatch。

事前固定 xy/yx/x_minus_x/zero_zero 四环境原 constructor、显式 seed、reset、prefix10；44 个 prefix frame 的五文件逐 byte 相等才允许 suffix。保留四个 live 环境，随后仅 xy 的原 suffix 前三动作，止于 frame13；最后 close 四环境。无重试、无新目标、无新颜色渲染。不得把未执行余下 suffix 当通过。

预算：4 reset ×1 +43 public controls ×10 =434 Space.step，dt 全为 .01。实际 native entry/return-event 预算如下；return-event 不是独立成功证据，另存 public intent/returned/failed。

| 项目 | 次数 |
|---|---:|
| construct/setup/set_state/Space construct/reset/close | 各4 |
| seed（含 ctor 内部 seed） |8|
| public controls |43|
| Space.step |434|
| get_obs/render/debug_draw |各47|
| goal body / Body construct |90 /102|
| pymunk_to_shapely |86|

输出 cap32MiB，其中2MiB终结保留，free floor200MiB，wall900s（含 runtime load；工作与清理受 one-shot alarm）。无神经、模型、训练、额外数学 Body/Space。旧 Store 的每次 durable write/dispatch 检查空间和墙钟。系统失败可保留 partial，不保证内核/磁盘故障后一定能写最终 report。

## 观察方式与证据边界

`sys.setprofile` 只旁观原函数。只在 xy/frame13 的两次转换 return 保存几何细节，其余调用仅计账。对原 `Poly.get_vertices` 的真实 return 捕获 shape Python identity、实际所属 body identity 与 local vertices 顺序；对原 `Body.local_to_world` 真实 return 捕获每个 local/world pair。不会再次调用转换以制造 world vertices。原函数 return 的 `arg` 直接提供 MultiPolygon WKB、各 component WKB、exterior coords；顺序来自真实子调用而非重读 `Body.shapes` set。两次调用独立记录顺序，不能假定它们一致。

每次 return 持久化 `geometry/return_0.json`、`return_1.json`，然后才检查 trace；字段：ordinal/context/body_id/env_block_id/body_role/body_public/shapes/shape_body_ids/local_vertices/world_calls/world_vertices/wkb_hex/component_wkb_hex/public_state。identity 仅限本进程；不是跨运行 root 身份。goal 身份由先后调用及非 block identity 判据验证，shape ownership 必须都是当前 block。public_state 是转换 return 时点，coverage 尚未 append，不冒称 post-step。profile 内异常收集，不注入原函数；下次调度拒绝继续。观察器的只读查询/IO会扰动时间与分配，不声称对地址/遍历顺序无影响。

47 原生 frame 均保存 rgb.npy/png、state.npy、proprio.npy、transition.json、full_public_state.json，先持久化再 guard。frame11..13 的 RGB/state/proprio 仍逐 byte 门；transition/full-public 全量逐 byte 比较并原样报告差异，但这个有界诊断可继续到13，不把差异改成 pass。旧 v1 gate 完全未修改，`renderer_release=false` 永远固定。

所有 native 返回保存后，从两个实际 WKB 的 component 构造四个 MultiPolygon 排列（goal2×block2），不 union/buffer/repair，不 sin/cos 重建。按原 native `intersection.area / goal.area` 与 `np.clip(coverage/threshold,0,1)` 得到数值及 IEEE754 binary64 bits。`observed_permutations.json` 记录每个排列是否与本次实际 final_coverage/reward 精确匹配；actual 顺序须 bit-exact 才能 strict consume。历史 membership 永远 null，不能拿本次排列认证旧 run。

## CLI 与交接

在 le-wm-main 根目录，用 `.venv/bin/python -B -m experiments.spectral_reachability.agent_p5_native_geometry_observer_20261028_v1`（下记 `$CMD`）：

```bash
$CMD prepare --output results/agent_p5_native_geometry_observer_20261028_v1_prepare_MAIN01.json
$CMD verify --prepare PREP --prepare-sha PREP_SHA
PYTHONNOUSERSITE=1 PYTHONPATH=.jepa-physics-site $CMD run --prepare PREP --prepare-sha PREP_SHA --output results/agent_p5_native_geometry_observer_20261028_v1_diag_MAIN01 --authorize-main-physics
$CMD consume --prepare PREP --prepare-sha PREP_SHA --report OUT/report.json --report-sha REPORT_SHA --output results/agent_p5_native_geometry_observer_20261028_v1_consume_MAIN01.json
```

`$CMD` 是上述完整命令的文档缩写，不是脚本自动定义变量。所有输出排他创建。MAIN 先审代码/RAM tests 后才授权 run；worker 没有执行 run/prepare 或物理。

prepare/verify 仅 saved metadata；run 输出 report schema `<stem>.report`，rows 固定四臂、47计划 frames，即使 partial 也不删行。artifact inventory 包含 journal/runtime_probe/execution_binding、47帧的六文件（含PNG）、43 commands、两个 geometry returns、排列报告。consume 输入四个显式 prepare/report path+SHA，独立核 hashchain/实际 counters、dispatch 次序和逐操作 native delta、原始 raw action、PNG/NPY/字节比较、trace 世界坐标顺序及 WKB 排列算术；不调用任何物理或 renderer execute。receipt schema `<stem>.consume`，historical_transform_certification=null、renderer_release=false。

RAM tests 用纯 Python lookup body/shape 和 Shapely fixture，不加载 native env/Pymunk Body/Space，不把 fixture 当科研结果。覆盖真实 profile return、shape order、WKB/coords、坏 trace 先保存、observer IO失败、partial return、one-ULP不接受、缺 slot、原helper预算不改。实际命令：

```bash
OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m pytest -q -p no:cacheprovider tests/test_agent_p5_native_geometry_observer_20261028_v1.py
```
