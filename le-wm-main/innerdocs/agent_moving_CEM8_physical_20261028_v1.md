# CEM8 physical：原51项精确重建后的新raw5

独占三文件实现；不改旧 CEM、observed4、fixed24、封存 results 或共享文档。按 experiment-code 技能采用明确合同、实际计量与 RAM 失败测试；用户范围优先，不新增模型、训练、physics 或图。本 worker 不执行 prepare/run，不产生物理结果。MAIN 审读并实际执行。

## 身份与实际释放 ABI

固定8 anchors：root_slot 0/1 × condition off/on × prefix_arm line_search_GN/full10_conditional_D2P，严格原序，不按新物理 outcome 选择。CEM solver arm 为 teacher_full_CEM64，与 prefix_arm 分开。仅2个已知根，fresh=independent=0，不是完整 CEM-MPC、独立确认或 P1/P3 准入。

实际 producer `agent_moving_CEM8_20261028_v1.py` SHA `6dcb425c13ea8cade14bf23c97ecfe87b004b7804d155c5ef06282045e99c251`。
MAIN CUDA report `results/agent_moving_CEM8_20261028_v1_cuda_MAIN01/report.json` SHA `660751b5784d25cce652d0ecb06ffb4cf3810a24ed135755606c97c3f957da0a`。
实际 saved release wrapper `agent_CEM8_release_20261028_v1.py` SHA `6bde2af9b57cdd037d74f0268f0dca4cdfd76d72db7145a7877ae905b95fad55`。
MAIN receipt `results/agent_CEM8_release_20261028_v1_MAIN01/receipt.json` SHA `878fb418d92fc8c8ec3fc3bf4f2bd0d7f40f5a50322280bcb7f91fd923c6f609`，input_count=9902。

`consumer_report` 参数接这个完整 envelope，不接裸 consume JSON。`consumer_source_sha256` 接 wrapper SHA，非 producer SHA。先验证外部实际 source/report/receipt SHA，再导入唯一冻结 wrapper；调用其 saved-only `verify`，严格重建 request、完整 input_sha256 inventory 与 envelope，重新调用真实 producer.consume 并逐字段对比完整 `producer_result`。adapter 仅从 `envelope.producer_result` 取8 plans，不把 envelope 误当原返回，不仅检查 commands_released/status。wrapper 没有可选替代 provider。此验证仅文件哈希与已保存数组算术，绝不调用神经或环境。

prepare 保留完整 release input_sha256、request/receipt/wrapper、own/inherited prepare bindings、实际 producer artifacts、parent artifacts、原8 prefix evidence、native实现与本模块三文件。按 canonical absolute path 合并，同路径不同 SHA 拒绝。缺任何实际文件、receipt、source 或完整检验即 failclosed，不制造未来 hash。每次 verify/consume 重新验证全部闭包；封存后改任一绑定源会拒绝，不能回填旧状态。

计划按 ordinal/root/source_slot/condition/prefix_arm/task/blind/original_selection/context SHA/full original PLAN_KEYS 绑定；校验完整原 prefix5 float32 bits、own winner64 selected_index/model_Q、独立 replay raw、selection seal 与全报告 SHA。新 execution_sha256 实际存在于 selected suffix item；它封存整个 item，不能与旧文档中不存在字段混用。原520 predict/AE/proj 各自 attempted=completed=520，是不同轴，不相加；物理阶段不追加预测。

## 精确恢复路径与输出

只复用固定 SHA `15125f4d86e5a6b3344f750205101a0e8a6f4a8dfba82ce46ce3d925f4ea8bd6` 的 `capability_suffix_factorial_physical_20261027_v1.episode` 内核及其原底层 helpers；不调用旧 execute/probe/witness，不改 globals。新 meter/store/executor 独立拥有8分母。

每 anchor 新建原生 target-free fresh env，原 reset/seed/placement recipe、原 warmup10、原 goal bind、原 prefix5，逐项重放 warmup0..10 与 plan0..5 的 off/on/off 共51项 RGB224、native7、完整 body snapshot、goal/renderer 不变量。必须全部与**原8 prefix report**精确相等才进入 suffix。原 report SHA `56962e9370fcf249a2ed88f004a400414b491f503a4c3a9db1bd951cf984a521`，不是 fixed24 新 suffix outcome。只读原证据，不重跑 probe40/原8。

不允许 reset(savedpoststate)、suffix前setter近似恢复、goalpose编辑、缓存 old endpoint 或仅比选中 PNG。suffix 是自己的 strict-consumed literal float32 raw5×2，原生 action_scale=100、relative=True，直接派发，无 tanh/clip/额外缩放。内部给冻结内核的 suffix 字典仅数据视图，不冒充旧 compiler 认证计划。所有控制 with_target=False；观察时临时 off/on/off，finally 恢复，无额外物理。

state 是实际7维 `[agent_x,agent_y,block_x,block_y,block_angle,agent_vx,agent_vy]`（只读核对 installed `.venv/lib/python3.12/site-packages/stable_worldmodel/envs/pusht/env.py` 的 `_get_obs`：末两项为 `self.agent.velocity`，不是block velocity）；保留原生 native_distance/native_success 及位置/角度等完整 metric 字典，不拿 model_Q 或作者值填充。未改state payload/metric/native函数。response.delta_native7 相对实际 prefix5；角度另存 wrapped delta。checkpoints0/5取本次 prefix 实际帧，10仅来自本次 suffix5。每 suffix 返回有实际 reward、terminated/truncated、state/snapshot 与 command bits。native stop 立即停止、不padding、不重试、不补缺 checkpoint；unknown=null。

完整成功每 anchor 66 PNG，共528。失败仍保留固定8 rows、first/secondary errors、warmup/prefix/suffix attempted/returned、earlystop、not_started。各分支 partial 留存，不复制旧端点。saved consume 核对 journal hashchain/嵌套dispatch-return、输入/计划/execution seals、全51恢复、literal命令、实际native response/终止/checkpoint joins、PNG文件与像素SHA、完整输出census。完整 partial 可保存检查，不等于科学通过；破损journal拒绝。

## 真实计量与工程边界

完整8分支固定流程预期上限，资源分别计量，不相加为总成本：

|资源|attempted/completed 各上限|
|---|---:|
|constructor / close|8 / 8|
|reset|16|
|public_step|160|
|native physics_step|1632|
|set_state|32|
|goal_setter|24|
|goal_render|16|
|goal_bind|8|
|explicit_render|528|

计量依据真实原生代码：两次 reset/分支（首次空间构建+原 placement）各2内部physics，20 public steps各10 physics，即每分支204。reset各2 set_state、1 goal_setter、1 goal_render，另1 goal_bind/goal_setter。constructor在安装hooks前，仅对冻结且检查为 pristine/无space/native状态的实现声称无内部physics；失败constructor内部工作为 unknown，不冒填0。close记录实际entry/return，不等同析构所有外部系统成本。

durable intent 在实际callback前写，meter在callback entry计attempted，成功return计completed；失败不退款。caps/floor/900s在dispatch前检查，故障后停止正常dispatch，但尽力close并恢复hooks/flags。900s不是无法中断native call的硬抢占承诺；超时保留partial并关闭工程成功。64MiB完整输出cap，4MiB final reserve，200MiB disk floor；每写/dispatch检查，最终fsync；拒绝覆盖、symlink、未计入文件。清理失败也留错误，无删除。

prepare 单独16MiB cap，不沿用 observed4 v1 的512KiB错误阈值；保留所有bindings，超限只失败不裁剪。RAM尺寸回归包括9000长绝对路径、全8×51 trace；实际release9902项也全部合并。真实只读 `material(arguments)` 完成：4912844 bytes、12433 bindings、8 anchors/8 suffix plans，低于16777216 cap，未写prepare或调用环境。此文档后续等长SHA绑定不改变serialized material大小。不把RAM fixture尺寸当真实产物大小。实际prepare仍由MAIN运行。saved validator、RAM测试或source SHA不构成物理/科学证据，不称equal FLOPs/total compute，不解锁freshness/P1。

## MAIN精确CLI

从 `/home/gbk/code/LeWorldModel/le-wm-main` 执行。只有MAIN审查且其独立release verify通过后才prepare。输出须NEW，失败目录/文件不覆盖。下面实际SHA均已由MAIN提供，不是未来值：

```bash
OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 \
.venv/bin/python -B -m experiments.spectral_reachability.agent_moving_CEM8_physical_20261028_v1 prepare \
  --compiler-output /home/gbk/code/LeWorldModel/le-wm-main/results/agent_moving_CEM8_20261028_v1_cuda_MAIN01 \
  --compiler-report-sha 660751b5784d25cce652d0ecb06ffb4cf3810a24ed135755606c97c3f957da0a \
  --producer-source-sha 6dcb425c13ea8cade14bf23c97ecfe87b004b7804d155c5ef06282045e99c251 \
  --consumer-source-sha 6bde2af9b57cdd037d74f0268f0dca4cdfd76d72db7145a7877ae905b95fad55 \
  --consumer-report /home/gbk/code/LeWorldModel/le-wm-main/results/agent_CEM8_release_20261028_v1_MAIN01/receipt.json \
  --consumer-report-sha 878fb418d92fc8c8ec3fc3bf4f2bd0d7f40f5a50322280bcb7f91fd923c6f609 \
  --physical-output /home/gbk/code/LeWorldModel/le-wm-main/results/p1_moving_overlay_recompile_physical_20261027_v1_MAIN01 \
  --physical-report-sha 56962e9370fcf249a2ed88f004a400414b491f503a4c3a9db1bd951cf984a521 \
  --output /home/gbk/code/LeWorldModel/le-wm-main/results/agent_moving_CEM8_physical_20261028_v1_prepare_MAIN01.json
```

以下 `CEM8_PHYSICAL_PREP_SHA`/`CEM8_PHYSICAL_REPORT_SHA` 必须由MAIN复算真实文件SHA后设置；不预填。run唯一CLI factory创建 `gym.make('swm/PushT-v1',render_mode='rgb_array',resolution=224,with_target=False)`，无模型runner：

```bash
OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m experiments.spectral_reachability.agent_moving_CEM8_physical_20261028_v1 verify \
  --prepared /home/gbk/code/LeWorldModel/le-wm-main/results/agent_moving_CEM8_physical_20261028_v1_prepare_MAIN01.json \
  --prepare-sha "${CEM8_PHYSICAL_PREP_SHA:?MAIN actual prepare SHA}"
OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m experiments.spectral_reachability.agent_moving_CEM8_physical_20261028_v1 run \
  --prepared /home/gbk/code/LeWorldModel/le-wm-main/results/agent_moving_CEM8_physical_20261028_v1_prepare_MAIN01.json \
  --prepare-sha "${CEM8_PHYSICAL_PREP_SHA:?MAIN actual prepare SHA}" \
  --output /home/gbk/code/LeWorldModel/le-wm-main/results/agent_moving_CEM8_physical_20261028_v1_MAIN01 --authorize-physics
OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m experiments.spectral_reachability.agent_moving_CEM8_physical_20261028_v1 consume \
  --output /home/gbk/code/LeWorldModel/le-wm-main/results/agent_moving_CEM8_physical_20261028_v1_MAIN01 \
  --report-sha "${CEM8_PHYSICAL_REPORT_SHA:?MAIN actual report SHA}"
```

Python接口 `prepare(output,arguments)` 的arguments键严格为 compiler_output/compiler_report_sha256/producer_source_sha256/consumer_source_sha256/consumer_report/consumer_report_sha256/physical_output/physical_report_sha256。`material(arguments)`只读返回 `(prepare_dict,data)`；`verify(path,sha)`重建完整material。`run(...,factory,authorization=AUTH)`是MAIN唯一物理授权口；RAM factory不是真实runner。

## RAM测试

```bash
OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m pytest -q -p no:cacheprovider tests/test_agent_moving_CEM8_physical_20261028_v1.py
.venv/bin/python -B -m experiments.spectral_reachability.agent_moving_CEM8_physical_20261028_v1 --help
sha256sum experiments/spectral_reachability/agent_moving_CEM8_physical_20261028_v1.py tests/test_agent_moving_CEM8_physical_20261028_v1.py innerdocs/agent_moving_CEM8_physical_20261028_v1.md
```

RAM fakeEnv/MemoryStore与实际producer JSON形状，只测协议与算术，不是真实神经/物理证据。覆盖8固定分母、全51漂移、raw ULP/来源/根/顺序/context/winner/replay变异、真实资源cap、earlystop/失败不退款、cleanup、未知constructor、重封journal变异、release envelope/调用/inventory/source封存、完整closure和尺寸回归。最终实际测试数与冻结SHA由交付消息给出；文档不自包含自身SHA。
