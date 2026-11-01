# P1 fixed80：原始形状绘图输入与整数 raster token 分解

## 范围与可执行路径

独占 `agent_raw_raster_factor_20261028_v1` 三件套。只消费已完成的164 native jobs（4 condition panels、2 known roots；各41 jobs），保留全部80 minus/plus pairs，h=.01/.001各40，不按RGB、native收益或模型误差筛对。不是新selector、原P1合同替换或renderer修复；P1/freshness/gate/deployment均不签发。

无需新环境、物理step、神经、GPU、训练或拟合。source冷导入只有stdlib/NumPy；读取PNG时用Pillow。`analyze` 中由MAIN明确授权才导入 installed Pymunk，**只使用未加入任何Space的 Body/Poly/Circle 做几何计算**；既不导入 `stable_worldmodel`/gym/pygame/torch，也不构造env/Space、不调用step/debug_draw/render。原几何构造函数中的 `space.add` 只进入局部Python ownership collector。这不是物理实验，也没有新rasterization。

直接提取原AST函数体，而不是假设T形或抄一套近似renderer。原 `add_tee` 会覆盖reset的scale=40为30；其他形状走其各自原函数。Pymunk detached Poly 用实际 native hull/顶点顺序；world转换使用snapshot导出的未取模angle和local0 world position，不拿native7 wrapped angle替换。独立zero-CoG pose body只还原导出的坐标变换，避免把原body的CoG位移重复算入；原geometry CoG另存。若local0不能逐float64字节复现，拒绝，不以容差通过。

## 实际输入authority

固定输入，不接受另选根或任意report替换：

| 产物（`le-wm-main/results/` 下） | SHA256 |
| --- | --- |
| `agent_observed_raw_coordinate_physical_20261028_v1_prepare_MAIN01.json` | `47ee177e40cb1247a73e7d08bfe5f31ce5d50884a09a5e0bce909027a1b63bd1` |
| `agent_observed_raw_coordinate_physical_20261028_v1_MAIN01/report.json` | `c4610c7f495137cbda7d70b147c74670db6f85339561da611ce5bee152d7bb10` |
| `agent_observed_raw_coordinate_physical_20261028_v1_consume_MAIN01.json` | `8df105747cf6835056d41ca1b1fd80f7f5b4170f7f527ec538213addc4c9d0bb` |
| `main_raw_observability_audit_20261028_v1_MAIN01.json` | `1411a87e95d36ae84971b27e2bbaaed848fbc5c277e9675d591d374125614d3f` |

已存在的完整strict receipt是本次authority，不伪造另一个strict receipt、也不重跑旧物理。新工具核对其完整rows与report相同、10824张PNG的既有strict声明、真实prepare/report SHA，以及全164 job/plan/task/raw10/终态身份。本次独立重新读验cp10的全部492张off/on/off原PNG（包括164张实际条件终态），而不是把“已全验10824张”冒称为本次新工作量。

原installed `envs/pusht/env.py` SHA `c6d2104adc368ff1c352edc0e35ac74da9f07fda3a748ddbef0ba6b7a2d1daa2`、`envs/utils.py` SHA `2e5d98a5c13227cba6f5574bf72b6eab149ee5b3783df0481b8d240f34047f56` 必须与物理prepare bindings相同。输出进一步绑定当前Pymunk Body/Shape/callback/debug_draw/pygame_util源码与单一installed native binary、以及新三件套SHA。当前native binary的绑定不凭空认证旧运行未封存的历史binary身份。

读取完整原journal并核对其report封存SHA，逐一取branch/plan/step10的off/on/off事件，要求pre==post、snapshot/state/goal/goal_pose/renderer/reset variations/selection_digest与真实分支一致。`row.trajectory`每步只有选定condition；完整triplet来源是journal，不能把trajectory末尾三项误作triplet。

## 输出ABI与解释

`NAME.analysis`：`jobs[164]`保存原index/panel/local/root/condition/task、producer job digest/raw10 digest、cp10 PNG/file/pixel/snapshot绑定，及：

- 实际shape索引/名字、请求scale、原CoG；每个circle offset/center/radius、每个polygon全部local/world vertices。
- 从封存 `DrawOptions` 原函数提取的所有pygame primitive参数：circle两层fill；polygon light fill；每条edge的line、quad、两端cap。不只比较agent center，也不只比较polygon填充顶点；保留颜色、半径、宽度、闭合参数。
- 整数转换使用原 `to_pygame` 的Python `round`，包含half-even行为；polygon原函数内部radius=2覆盖也原样保留。
- scene constants保存renderer/goal_pose/目标开关与真实goal polygon或circle命令；不改goal physical pose。

`rows[80]`固定顺序 `panel,h,coordinate`，minus/plus索引为 `41*panel+1+20*hi+2*coord` 及下一项。输出原observability标签，但标签不影响计算或选择。重新比较实际PNG数组，必须与原80分类中的不可见身份相符。每对给出agent/block各自浮点geometry变化、整数primitive变化、整数center/vertex变化数。

互斥分类：

1. `same_draw_inputs_sameRGB`：重建的shape primitives及scene constants相同，原RGB相同。
2. `changed_draw_inputs_sameRGB`：重建shape primitives变化，原RGB仍相同；遮挡、512 raster覆盖与resize尚未隔离，不能唯一归因。
3. `changedRGB`：真实原RGB变化；同时记录是否反常出现重建shape输入相同，绝不丢掉此种数据。
4. `unknown_draw_inputs_sameRGB` / `unknown_draw_inputs_changedRGB`：源dispatch没有对应几何等明确缺失；保留pair与原因，不伪造形状或整数token。

`rounding_erased_shape_change`严格表示“浮点shape几何变化、但全部重建整数primitive相同”；velocity-only或原浮点几何本来相同不计入。它不等于“已证明24对都由rounding单因导致”。agent center相同而block vertices变化会落到后两种已知分类，而非误归rounding。

### 必须保留的证据缺口

**`same_draw_inputs` 是明确shape-scoped的名称，不是全场景callback认证。** 当前snapshot没有封存Chipmunk transformed-shape cache、实际debug iteration顺序和contact/constraint callbacks；原DrawOptions默认还包含collision-point debug绘制，不能凭snapshot自行宣称不存在。原512surface也未保存；只有resize后的224 PNG。故所有pair均设 `full_scene_sufficient_cause_certified=false`，`unsealed_inputs`逐项列缺失。

原snapshot给出了完整导出的body信息，但原snapshot函数自身说明不包含solver warm-start cache。新工具恢复可计算的shape参数，不能因此签发“实际debug callback必定等于重建参数”。若MAIN需要充分因果闭合，下一最小干预须新协议、同state捕获**真实原renderer callback流和原512 surface**，保持224输出、pose和动作不变；比较callback→512→224三层。那才可拆分rounding前信息丢失、遮挡/raster覆盖、resize消失；本工具不擅自执行、不提高分辨率冒充修复。

## MAIN CLI

```bash
cd /home/gbk/code/LeWorldModel/le-wm-main
.venv/bin/python -m pytest -q tests/test_agent_raw_raster_factor_20261028_v1.py

.venv/bin/python -m experiments.spectral_reachability.agent_raw_raster_factor_20261028_v1 analyze \
  --authorization MAIN_RAW_RASTER_DETACHED_GEOMETRY_ONLY \
  --output /home/gbk/code/LeWorldModel/le-wm-main/results/agent_raw_raster_factor_20261028_v1_MAIN01.json

.venv/bin/python -m experiments.spectral_reachability.agent_raw_raster_factor_20261028_v1 verify \
  --authorization MAIN_RAW_RASTER_DETACHED_GEOMETRY_ONLY \
  --analysis /home/gbk/code/LeWorldModel/le-wm-main/results/agent_raw_raster_factor_20261028_v1_MAIN01.json \
  --analysis-sha ACTUAL_NEW_ANALYSIS_SHA
```

只允许新stem输出，32MiB cap、200MiB disk floor，`xb`写入禁止覆盖。verify完整重算而非只检查report摘要；没有future SHA默认值。输出不保存新图片，不改原renderer、源码、gate、normalizer或Q。

## 测试与交付证据

TDD首次RED为新增模块不存在。首轮10项通过。实际saved ABI测试曾捕获新consumer错误地把trajectory末尾三项当完整triplet（1 failed/18 passed），改为从journal取得完整三帧并与唯一选定terminal行及cp10精确联接后24 passed in 5.94s。补齐同agent整数center但block顶点变化、unknown保留、velocity-only与精确origin恢复失败边界后，最终28 passed in 6.08s。

测试只使用NumPy RAM几何替身、封存源函数提取、真实metadata/原PNG/journal只读核验；未导入native Pymunk、没有新physics/neural/raster或生产analysis。RAM几何替身不是native hull/变换数值证据，其分类不得用作实测结论。MAIN仍须按CLI实际执行native detached几何重建后再报告分类数量。
