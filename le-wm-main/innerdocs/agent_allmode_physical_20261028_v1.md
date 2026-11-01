# agent_allmode_physical_20261028_v1

仅新独占 source/tests/本文档。MAIN 审核和执行；本实现者不运行 prepare、真实 consume/reuse、模型或 physics。
RAM 测试不是实验或 P1 证据。旧源码、报告、共享文档不改。

## 首选 saved 复用，不强制本批重复 physics

`reuse` 必须先完整真实 `agent_allmode_prefix_20261028_v2.consume`，再完整真实
`capability_suffix_factorial_physical_20261027_v1.consume`。两个消费者的完整返回值都保存，
包括旧 24 行；旧报告除重复 rows/actual_operation_receipts 外的 metadata（含 errors/counts/绑定）也保留。
两边所有输入/任务/源码/prepare/report/PNG/journal 的传递 hash 合并，冲突或最终漂移立即拒绝。
两边任一消费者抛错直接传播；绝不凭 commands 相同绕过 consume。

固定四行 `(slot0,off),(slot0,on),(slot1,off),(slot1,on)`，原 anchor ordinal 固定 `1,3,5,7`，
旧 spectral24 行 index 固定 `4,10,16,22`（`teacher_full × full10_conditional_D2P`）。
不搜索 GN/open_full/teacher_small/observed4，不按结果选成功证据。
精确连接 task_id、blind_SHA、原 paired source SHA、完整 encoded task 文件 SHA（含 context/Q/goal/model）、
原 selection_digest 和 original_plan 的全部字段；源任务 native7 root/hash/goal/goal_pose/条件 goalRGB 另核。
seed/placement/warmup 来自同一封存 source ancestry，旧消费者完整校验原 51 帧身份桥和新 suffix 全轨迹。
新 raw10、原10候选 raw10、旧 teacher_full 执行 raw10 必须全部 float32 **逐位相等**才允许继承。
完整消费后，raw 不同或旧行失败/未开始保留该项 `unknown`、`evidence=null`、`native_success=null`，不补 0。
继承保存旧行全文（含真实 checkpoint/PNG/SHA/状态/earlystop/attempt-return），不补齐缺失步数。
早停是原 native 终止语义，不把 2 步成功补成 5/10 步。

**原 full8 physical 模块没有 consume API。** `moving_overlay_capability_probe_physical.material`
只覆盖 prefix51，不是 full10 消费，禁止用它单独认证原 full8 的后5步。
这里采用有完整 consume 的 spectral24 实际物理轨迹作为可选证据，**不声称其 commands 必定等于原10**。
MAIN 通知 allmode 四 winner 等原10，并不能保证 teacher_full 重编译 suffix 也相同；若不同就 unknown。
本版没有为缺完整消费者的原 full8 后缀制造授权出口，也不会自动触发新的 physics。
所有 inherited 证据均 `freshness=false`，`fresh_roots=0`，known roots=2，不是独立4根。
不同规划器预算不等价，`equal_total_compute/equal_FLOPs/P1/P3/scientific_admission=false`。
资源 nested 各轴保留而不相加。实际能继承几项必须 MAIN 严格 CLI 后才知道。

## MAIN 精确 saved-only CLI（未由实现者执行）

以下所有 seal 是已存在真实文件 SHA，非未来预造 SHA。根目录：

```bash
cd /home/gbk/code/LeWorldModel/le-wm-main
.venv/bin/python -B -m experiments.spectral_reachability.agent_allmode_physical_20261028_v1 reuse \
  --output /home/gbk/code/LeWorldModel/le-wm-main/results/agent_allmode_physical_20261028_v1_reuse_MAIN01.json \
  --producer-output /home/gbk/code/LeWorldModel/le-wm-main/results/agent_allmode_prefix_20261028_v2_cuda_MAIN01 \
  --producer-report-sha c3e38899a12cd163d75918da79ef029adb14b2479f74f15bdfac0b6ffc57242d \
  --producer-source-sha 2cc7201377c24533aaa8f4c2c968ae7959a3c55df97b40964116e26a944380f5 \
  --producer-prepared /home/gbk/code/LeWorldModel/le-wm-main/results/agent_allmode_prefix_20261028_v2_prepare_MAIN01.json \
  --producer-prepare-sha c11543eaa251903ff6d0429d2447f03c7427e1fe65be3d0aabfb0d9c4ef79d0d \
  --old-output /home/gbk/code/LeWorldModel/le-wm-main/results/capability_suffix_factorial_physical_20261027_v1_MAIN01 \
  --old-report-sha cc92465141b400659e024e052b05683153b4dc6c013b81bdf1e3e29355fd2663 \
  --old-source-sha 15125f4d86e5a6b3344f750205101a0e8a6f4a8dfba82ce46ce3d925f4ea8bd6
sha256sum results/agent_allmode_physical_20261028_v1_reuse_MAIN01.json
```

实际创建回执后，MAIN 将下面唯一占位替换成该真实 SHA，再重建两边消费者验证。
输出必须 NEW，不重入、不覆盖（验证前即拒绝），32MiB cap、200MiB floor、fsync。
verify-reuse 只读 saved 输入；不运行 planner/env，可能读取双方完整封存闭包。

```bash
.venv/bin/python -B -m experiments.spectral_reachability.agent_allmode_physical_20261028_v1 verify-reuse \
  --output /home/gbk/code/LeWorldModel/le-wm-main/results/agent_allmode_physical_20261028_v1_reuse_MAIN01.json \
  --report-sha ACTUAL_REUSE_SHA_FROM_PRECEDING_SHA256SUM
```

## 通用新物理 adapter（非本批强制运行）

`prepare` 接受上面同组五个 `--producer-*` 参数，`--output` 为绝对 NEW
`results/agent_allmode_physical_20261028_v1_prepare_MAIN01.json`。
精确 prepare 命令（本批优先 reuse，非要求执行此命令）：

```bash
.venv/bin/python -B -m experiments.spectral_reachability.agent_allmode_physical_20261028_v1 prepare \
  --output /home/gbk/code/LeWorldModel/le-wm-main/results/agent_allmode_physical_20261028_v1_prepare_MAIN01.json \
  --producer-output /home/gbk/code/LeWorldModel/le-wm-main/results/agent_allmode_prefix_20261028_v2_cuda_MAIN01 \
  --producer-report-sha c3e38899a12cd163d75918da79ef029adb14b2479f74f15bdfac0b6ffc57242d \
  --producer-source-sha 2cc7201377c24533aaa8f4c2c968ae7959a3c55df97b40964116e26a944380f5 \
  --producer-prepared /home/gbk/code/LeWorldModel/le-wm-main/results/agent_allmode_prefix_20261028_v2_prepare_MAIN01.json \
  --producer-prepare-sha c11543eaa251903ff6d0429d2447f03c7427e1fe65be3d0aabfb0d9c4ef79d0d
```

它保存完整 v2.consume、四 plan、原 source slots、17个原 warmup reference/root、原任务 metadata 和完整输入绑定。
`verify --prepared ABS_PREPARE --prepare-sha ACTUAL_PREPARE_SHA` 重建全部 saved 材料。
MAIN 另行明确批准 physics 才可使用：

```text
run --prepared ABS_PREPARE --prepare-sha ACTUAL_PREPARE_SHA --output ABS_NEW_OUTPUT_DIR --authorize-physics
consume --output ABS_NEW_OUTPUT_DIR --report-sha ACTUAL_NEW_PHYSICAL_REPORT_SHA
```

不得把 ACTUAL 占位当未来 seal；新 report 存在后才取得 hash。v1 失败 cap 报告永不释放子集。
新 schema/Store/Meter/execute/consume 均属本源；仅复用冻结原 place/bind_goal/capture/native_hooks 等函数，
不改旧 globals、不借旧 schema 越白名单，无新模型加载，无新控制律。
固定四项包含 unchanged/no-op plans；同根 seed/placement + 非零 float32 warmup（前5 `[.1,0]` 后5 `[0,.1]`）
逐位权威，选定 raw10 不 tanh、不 clip、不乘100（环境仍保持原 action_scale100）。
目标图使用源相应 off/on 真图；默认 goal_pose 不是构造 goal，不编辑。
完整日志保留实际 off/on/off PNG file/pixel SHA/native7 snapshot；prefix51 是新 warmup0..10+plan0..5 身份桥。
原 source 只有17个 warmup receipt/root，不虚构33个：中间8个 on 无旧参照，显式列 unknown；
off_after 中间步可绑定该步原 off。原 0/5/10 三帧条件 history 必须一致。
native 成功/截断即停止，不补 checkpoint；工程失败四分母仍在，缺失指标为 null。
dispatch intent 与真实 entry/return 分离、嵌套核对；失败不退款。未返回 constructor 的内部工作 unknown。
四 branch caps：constructor/close4，reset8，public80，physics816，render264；不把 nested 相加。
新物理目录32MiB/floor200MiB/final4MiB、wall900s，NEW独占；清理只允许原 close，不重新运行。

## 验证与限制

RAM 测试命令（array fake env、内存 PNG/journal、tmp_path 元数据；没有真实神经/physics）：

```bash
.venv/bin/python -B -m pytest -q -p no:cacheprovider tests/test_agent_allmode_physical_20261028_v1.py
```

实现者实际执行结果：**113 passed in 33.79s**。最后仅补本文档交付说明，无源码/测试再修改。

覆盖 full4/非零 warmup/zero不放宽、非零与零候选都保留、earlystop不padding、primary/secondary failure、
source on/off身份、renderer/goal绑定、实际 PNG/pixels/hashchain、篡改后重封链也拒绝、大小/floor/重入、
两真实 consume ABI 的 RAM 边界与错误传播、固定旧24行和四项 task/raw ancestry、未知不补0、saved回执重验证。
实现者只做 schema字段只读核对和 RAM测试，未实际创建 prepare/reuse/物理产物，未执行真实 consume。
MAIN 仍需审读/测试/实际 saved复用验证；原 full8 缺完整 consume 的后缀复用仍不支持。
没有把已知 negative/unknown 改成通过；任何实现完成或测试通过均不代表 P1通过。
