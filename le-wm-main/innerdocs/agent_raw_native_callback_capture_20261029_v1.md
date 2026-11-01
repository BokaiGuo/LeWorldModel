# Fixed164 native callback → canvas512 → resize224 capture

独占本stem source/tests/innerdoc。MAIN审读后主机执行；worker只做RAM协议替身测试和保存来源核验，不导入物理/pygame/cv2/神经运行时，不执行生产重放。不是新控制器、renderer修复或P1准入。

## 固定来源与分母

严格绑定 `agent_raw_raster_factor_20261028_v1.PINS` 中实际物理prepare/report/full strict receipt/80pair observability四项SHA；物理report `c4610c7f495137cbda7d70b147c74670db6f85339561da611ce5bee152d7bb10`。完整164原plans/jobs/task/goal/reset source均来自原prepare，不根据原结果挑动作或根。四panel，每panel baseline1 + 两个已有h(.01,.001)×10coord×两个sign，共164。pair索引沿原80 schedule，含所有零/非零/失败。

prepare校验已完成的全量strict receipt与完整report rows/counts、全部已绑定来源和PNG实际字节、原journal SHA，原schema的execution validator，及原source_material里的reset recipe/原始raw10历史/goal。不把本次检查称作重新执行旧consumer；旧strict receipt本身是固定authority。额外绑定当前callback/pygame/Chipmunk/cv2源码和二进制。新当前binary绑定不回证旧运行未封存的loaded bytes。

## 真实重放与tap

运行只复用已审 `mixed.episode` actor、原 `native_hooks` 和cp5 guard；不调用旧execute/run、不修改旧globals/合同。原fresh native env constructor→原两次reset/placement→同非零warmup raw10→原plans raw10，全动作逐float32 byte，原off控制/goal/pose/action_scale/renderer flags不变；原off/on/off观察顺序保留。不是reset到poststate，不hydrate，不重建碰撞缓存，不新造替代Body/Space来画图。

tap是有范围、可恢复的透明观察绑定：

1. `sys.setprofile`旁观原 `_render_frame` 和原DrawOptions/SpaceDebugDrawOptions的 `draw_circle/draw_segment/draw_fat_segment/draw_polygon/draw_dot/color_for_shape` 原函数入口/返回。记录实际收到的世界坐标、颜色、radius、flags、transform和完整嵌套顺序（包含collision-point `draw_dot`）。shape身份按原body role及实际local shape accessor记录，不用进程地址决定两pair相等。
2. 在此范围临时包装 `pygame.draw` 的9个原生primitive入口。保存**实际传入参数**和返回Rect，原callable以同args/kwargs恰好调用一次。不是AST重建primitive、不是再次执行callback。原模块文件不改，退出恢复原函数身份。该进程须单线程独占执行，已有profile时拒绝，wrapper/profile漂移记工程失败。
3. 透明包装原 `cv2.resize`：只读原surface像素与原resize输入，二者必须uint8 `(512,512,3)`逐byte相同；保存原512 PNG；以原参数恰好调用一次原resize；保存实际224返回PNG。不改变interpolation、不放大后替换原图、不额外render、不隐藏contacts。
4. 第一笔draw前读取真实canvas，要求原background fill后均匀；记录实际background颜色和全部512像素hash。保留所有实际pygame primitive参数，包括原Python round/int完成后的值。
5. 每个render保存完整前后native snapshot、goal/goalpose、goalRGB存在性、renderer/variations/flag、latest_action、env RNG和全局Python/NumPy RNG；必须完全相等。首次reset goal render在`_goal`赋值前，真实None原样保留，不补goal图。

CFFI可能吞掉Python callback/profiler异常，因此错误另有latch，resize/结束必须检查；不允许丢一条callback后仍签发成功。callback内部不重算sin/cos、不重新查询solver作“替代callback”。tap额外读只读snapshot/accessor，不增加physics/public action/render调用。

## 精确门与失败保留

每个原生explicit observation完整receipt/pre/post与旧journal对应项、PNG文件与像素hash均逐byte核对；tap224与原actor写出的PNG亦相同。原全部row、所有resource intent/return、实际raw/transition最终须与旧report/journal完全一致。任何ULP或PNG差异都失败，不容差、不挑重试。旧fullpublic失败不被本模块释放。

失败立即停止后续native dispatch，全部164 row和80pair仍保留，未开始不填结果；失败实际PNG/partial transcript不删除。消费失败产物只返回blocked，不宣称“充分原因已闭合”。仍保留pair missing/null及不完整artifact清单，不把少数完成分支当新分母。工程诊断不保证确定性重放必过。

## 预算

原164物理预算原样：constructor164/close164/reset328/public3280/physics33456/state_setter656/goal_setter492/goal_render328/goal_bind164/explicit_render10824。嵌套计数不相加冒充独立调用。另记录所有真实render=11152（10824 explicit +328 reset-goal），不是physics0。每render primitive≤1024、callback entry≤2048，越界拒绝不截断。

新输出4GiB含所有旧224 PNG、新512/224 PNG、lossless gzip完整JSON callback transcript、journal/report；256MiB终态reserve，初始需要4GiB+1GiB floor；持续floor1GiB，wall3600s含加载后的实际执行与终态source核验，无重试。原裸512 RGB上界11152×786432≈8.77GB；采用无损PNG，不降低精度/分母。若实际不可压缩导致4GiB不足，保留失败，不能静默省图。metadata≤256MiB（原prepare已114MiB），单shot JSON≤2MiB。无NN/encoder/AD/训练/下载。

## Output ABI / saved consume

`report.json`: schema=`NAME.report`, status=`raw164_callback_capture_exact`或`partial_or_failed`，`rows[164]`完整原物理plan身份，`captures`实际render按顺序，`actual_operation_receipts/counts/errors/artifacts/journal_chain_tail/full_replay_exact`。

每capture: `index,branch,operation_scope,operation_stack,file,sha256,transcript_digest,primitives,callbacks,surface512,rgb224`；后两者各含`file,sha256,shape,dtype,pixel_sha256`。真实explicit观察再附 `original_observation={phase,step,condition,overlay_index,png,file_sha256,pixel_sha256}`。reset-goal不伪造原explicit键。

`tap_XXXXX.json.gz`完整记录before/after、原生module/source行与每个callback参数/返回、primitive intent/return交错顺序、background及resize参数；对应`tap_XXXXX_512.png`、`tap_XXXXX_224.png`。original actor PNG仍用原名。`journal.jsonl`新独立完整hashchain，混合native资源/actor事件和tap intent/return、精确gate事件；不冒充原journal。

`consume(output, report_sha)`纯saved，不导入native runtime。重验来源、全artifact SHA、PNG像素、完整transcript调用闭合及计数、所有original observations/资源动作证据。固定全80 terminal pair输出callback equality、全部整数primitive equality、512 equality、224 equality；不把其自动视作唯一因果结论。分类明确标注相同primitive却不同512、相同512却不同224为unexplained。没有旧callback证据时 `historical_callback_membership=null`、`sufficient_original_cause_certified=false` 永远保留。即使新重放全门通过，论断也只限新实际捕获链与旧224/state/action精确对应，不称旧cache逐位相同或整个P1失败已解释。

## MAIN CLI（inner根目录）

```bash
.venv/bin/python -m pytest -q tests/test_agent_raw_native_callback_capture_20261029_v1.py
.venv/bin/python -m experiments.spectral_reachability.agent_raw_native_callback_capture_20261029_v1 prepare --output /home/gbk/code/LeWorldModel/le-wm-main/results/agent_raw_native_callback_capture_20261029_v1_prepare_MAIN01.json
.venv/bin/python -m experiments.spectral_reachability.agent_raw_native_callback_capture_20261029_v1 verify --prepared /home/gbk/code/LeWorldModel/le-wm-main/results/agent_raw_native_callback_capture_20261029_v1_prepare_MAIN01.json --prepare-sha ACTUAL_PREPARE_SHA
.venv/bin/python -m experiments.spectral_reachability.agent_raw_native_callback_capture_20261029_v1 run --prepared /home/gbk/code/LeWorldModel/le-wm-main/results/agent_raw_native_callback_capture_20261029_v1_prepare_MAIN01.json --prepare-sha ACTUAL_PREPARE_SHA --output /home/gbk/code/LeWorldModel/le-wm-main/results/agent_raw_native_callback_capture_20261029_v1_MAIN01 --authorization MAIN_RAW164_NATIVE_CALLBACK_REPLAY_ONLY
.venv/bin/python -m experiments.spectral_reachability.agent_raw_native_callback_capture_20261029_v1 consume --output /home/gbk/code/LeWorldModel/le-wm-main/results/agent_raw_native_callback_capture_20261029_v1_MAIN01 --report-sha ACTUAL_REPORT_SHA --receipt /home/gbk/code/LeWorldModel/le-wm-main/results/agent_raw_native_callback_capture_20261029_v1_consume_MAIN01.json
```

只用真实生成SHA；任何未来report/receipt不预填成功。RAM tests使用Python函数+NumPy surface协议替身，验证转发一次、调用顺序、状态/RNG变化拒绝、原函数恢复、full80缺项保留、计数/预算、lossless/hash/缺槽拒绝；不是实际Pymunk callback/碰撞cache或原cv2数值证据。生产物理仅MAIN在审读冻结后运行。

## 交付前实际验证范围

已实际只读 `source_material()` build smoke：164原plans/164原rows/2roots，20569条完整绑定，prepare JSON约3.94MiB，未写生产prepare。`torch/pygame/pymunk/gymnasium/cv2/stable_worldmodel` 均未导入；没有生产render/physics。第一轮RAM foreign-surface fixture缺`screen`属性引发AttributeError，已补足fixture，使其确实验证拒绝异体surface；未放宽生产检查。后续正式prepare必须绑定最终三文件SHA，不能复用修改前内存smoke的字节值。
