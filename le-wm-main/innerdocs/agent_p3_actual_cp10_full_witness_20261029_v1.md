# 五actual-cp10完整保存见证（仅saved）

固定顺序 `(r0,C),(r0,Z),(r0,P),(r1,Z),(r1,P)`，源slot分别C0/ZP0/ZP1/ZP2/ZP3。
使用experiment-code的自包含实现和失败回归约束；不采用训练、择优或运行实验建议。
旧阶段C schema不改。本件不调用任何上游consumer、不导入模型/physics，MAIN身份认证继承
已经完成的consume，而非重做生产fullstrict。库的临时fixture build会明确
`actual_MAIN_authority=false`；CLI只能接受以下四个实际MAIN pin，不能把fixture称ready。

## 权威和抽取范围

阶段A固定四SHA（prepare/report/completion/receipt）：

- f2fac71516c36bdb654a9a84bbc0115d6155c04ed10231bc6b87b99ed0bda846
- b5744f311d3f412cfc5016bf37c8854cc7b5272a6c37e62055c7f3aa68f2c657
- 37588f050ed9381b191673e27c0b6fcab3ddf0c726e63a54bb75d9579963facf
- af67caf8536a9637d709fcfb11c947d993e566bd4de5914725f4640a75de4d6f

从其source_authorities和bindings定位C/ZP原prepare/report/completion/receipt/source，
report.journal_sha256须同时被原receipt.bindings、阶段A bindings及原completion认证。
整条journal逐行按原spaced `json.dumps(sort_keys=True,allow_nan=False)`，无换行的event
digest，核连续index/previous/eventSHA和tail；重新连接完整嵌套intent/returned/failed账、
全counts和所有branch_closed（包括no_witness），不是只读取五个后缀。
不递归重验旧planner/model/weights，不冒独立全链认证。原PNG只核文件SHA，pixelSHA继承
原MAIN认证，不在本工具解码或重新渲染。所有实际读入文件在结尾再次核SHA。

每端点保留66个完整观察事件：warmup0..10、plan0..10各off/on/off；连全部20个public返回、
20条native transition、10条own response，完整原snapshot/pre/post/receipt不裁剪未知字段。
同时保存原source_row（含plan/goal/root）、source_root、source_slot_metadata、原placement/
goal事件、scoped_original_events、完整prefix_reference、stageA row/task和PNG源路径。
raw warm10/control10保存原值及20对little-endian float32位串；不clip/归一化/生成新动作。
actual_response10与cp10、StageA source/task/raw/history abs10/15/20精确连接。

## 51与66守卫边界

原producer `agent_observed_suffix_physical_20261028_v1.prefix_match`要求pre/post全字段等值，
receipt除运输png名称外physical/RGB字段等值。本新件更严格保留并比较所有receipt未知字段，
仅在**prefix_reference比较**中移除明确的 `/receipt/png`，每个51位置输出一条原值映射：
`/observations/{i}/receipt/png` ↔ `/prefix_reference/frames/{i}/receipt/png`。
没有其它忽略字段，没有按basename忽略任意path，完整原事件（含scope/hash/index）不变。
cp5 checkpoint也只允许png运输名称变化，单独保存checkpoint_transport_normalization的
`/source_row/checkpoints/1/png` ↔ `/prefix_reference/checkpoint/png`及两个原始值；
原durable cp5 accepted guard必须在全部51观察之后、
control6 intent之前。新66精确参考是输出的整个observations，不是靠忽略未知字段得到的投影。
将来新replay若必须改变branch/scope/index/hash链，只能另行枚举新adapter明确的运输映射；
本件没有授予全局scope忽略权。snapshot里实际记录state/bodies/state_float64_bits，
**没有**凭空添加rng/public_state。RNG/public_state/contact caches未记录的语义明确unknown，
source_slot reset/root seed保留不等同完整RNG状态。不能据此认证hydrate动力学重启。

## MAIN命令

在 `/home/gbk/code/LeWorldModel/le-wm-main`：

```bash
OPENBLAS_NUM_THREADS=1 .venv/bin/python -B -m pytest -q -p no:cacheprovider tests/test_agent_p3_actual_cp10_full_witness_20261029_v1.py
.venv/bin/python -B -m experiments.spectral_reachability.agent_p3_actual_cp10_full_witness_20261029_v1 build \
  --prepare /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p3_same_time_cp10_rgb_bridge_20261029_v1_prepare_MAIN01.json \
  --report /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p3_same_time_cp10_rgb_bridge_20261029_v1_cuda_MAIN01/report.json \
  --completion /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p3_same_time_cp10_rgb_bridge_20261029_v1_cuda_MAIN01/completion.json \
  --receipt /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p3_same_time_cp10_rgb_bridge_20261029_v1_consume_MAIN01.json \
  --output /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p3_actual_cp10_full_witness_20261029_v1_MAIN01.json
.venv/bin/python -B -m experiments.spectral_reachability.agent_p3_actual_cp10_full_witness_20261029_v1 verify \
  --witness /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p3_actual_cp10_full_witness_20261029_v1_MAIN01.json \
  --sha256 ACTUAL_BUILD_RETURNED_SHA
```

verify从原四权威重新构建**整份输出**比较（含全部未知字段、所有bindings），不只验status。
排他xb+文件/父目录fsync，不覆盖失败记录；单读入文件与输出各128MiB，journal单行16MiB。
本件是saved输入材料，不执行StageC每job64MiB预算；后续adapter必须独立检查其81PNG等预算。

整数预算原样保存为next_adapter_budget（不是执行授权）：10条独立job，每job constructor1、
close1、reset2、set_state4、goal_render2、goal_bind1、goal_setter3、public25、physics254、
render81，共374 operations、748 intent/terminal事件，再81 observation+2guard+3 lifecycle，
共834事件。全批250 public/2540 physics/810PNG；每job64MiB含8MiB reserve、200MiB floor、
wall900s，不因原snapshot/未知字段较大扩大上限。本提取器自身新PNG/NN/physics均0，
只核已有330张源PNG，保存330观察、255个prefix引用位置；后续预算需按真实序列化再核。

## 未完成/不声称

worker只做临时文件fixture build/wholeverify和mutation测试，不运行真实saved build/fullstrict。
MAIN实际build/wholeverify尚待执行。阶段B未来七signal/五U双重放计划未提供，本件保持
stageB_authority=null、stageB_ready/production_adapter_ready/physical_release/scientific_admission=false。
不能据五个已知端点声称fresh、C3、任务成功或原生重放已重新通过。
