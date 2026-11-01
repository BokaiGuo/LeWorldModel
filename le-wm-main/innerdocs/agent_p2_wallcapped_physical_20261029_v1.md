# Wallcapped 四槽 raw15 原生物理交接 v1

仅新增本 stem source/tests/innerdoc。未修改 wallcapped runtime、旧四臂物理 adapter、任何封存结果或共享文档。worker 未生产 prepare、运行模型/CUDA/物理、下载或读取新物理结果。本版本待 MAIN 审读后执行，不签科学准入。

## 输入 authority（不得凭 status 绕过）

输入是 `agent_p2_wallcapped_cuda_20261029_v1.consume(output, report_sha)` 的实际完整返回，而不是旧 three_program receipt。prepare 必须提供实际 planner source/report/receipt SHA；先核 source SHA 再 import，调用实际 consume 全量重建，要求持久化 receipt 整体相等，再调用其 verify 重建原 prepare。原 report/prepare、所有输出文件、runtime bindings、source/task/model/source-warmup 以及本三件套全部进入 bindings。verify 再走相同 material 路径；运行前后与消费后重新检查 bindings。这里确实重复完整 upstream strict，不冒称仅复用 receipt。

不能在 runtime 尚未冻结或没有真实 strict receipt 时生产 prepare。CLI 中 ACTUAL_* 只代表 MAIN 后续取得的真实值，本实现没有预填未来 SHA。模型 receipt 的 `physical_release=false` 表示不签科学权威，不意味着可以跳过 strict；本 adapter 还要求明确 MAIN 工程物理授权。所有科学门保持 false。

## 固定槽 / 不替补

顺序为 `(root0,off,joint), (root0,off,composition), (root1,off,composition), (root1,off,joint)`，不是旧 adapter 的 root-major 同臂顺序。两根都是已见 known roots，不是四个独立根。plans 长度永远为 4；`None` 必须和同槽 account.export_missing 一致。

非空 plan 检查 runtime 自有 `.plan` schema、其 compact-JSON+LF selection digest、task/root context/goal192/model hash/warm_raw10、lossless float32 commands15x2 与原 raw bytes。它没有旧 compiler 的 edge/predicted_Q schema，不能伪装。完整 context/replay、Q 计算、归一化与搜索/导出预算验证由实际 upstream consume 重建；本地验证不是替代 upstream。

空槽保留 plan=None、task/root/arm/account、status=export_missing、所有 metrics=None 和全部 missing checkpoints，不调用 factory/reset/step/render/close，不重新选候选、不复制其它槽、不按 outcome 换根。全空也可验证“完整记录四个缺失槽”，但绝非四臂物理完成。receipt 的 `all_four_physically_executed` 仅在四槽都实际返回 15 步时为 true；原生早停仍保留，不补动作。

## 真执行 / native strict

执行循环从冻结 `agent_p2_three_program_physical_20261029_v1` 的四臂 full15 实现演化为本版独立 schema，复用未修改的 native.place/bind_goal/dispatch/native_hooks、moving、paired、原 Store.emit/put 和原生指标函数；不修改其它模块 globals，不伪造旧执行或 receipt。

每个非空槽独立 constructor/reset(seed)/原 placement/reset(state)/warm10，核原三帧 RGB history 和 native7 root/goal hash，再真实绑定原 task goal。随后执行原 commands15 的精确 float32 rawbits，不恢复 imagined state、不重规划；保留实际 public return、goal pose/RGB、原生 reward/success/终止、逐步 off/on/off RGB 与 cp0/5/10/15。warmup 的旧 goal 与 bind 后 task goal 分开验证。原 whole-body native success 不等同 object-only coverage。

严格消费重建：完整单 LF journal canonical/hashchain、所有已知事件与未知拒绝、nested intent/return/failure 和费用、每槽顺序闭合、原 reset/physics/public/render 关系、请求/返回 actual rawbits、task goal 绑定时序、全部 native7 reward/距离/终止、逐步 RGB/state/snapshot/pose、旧 warmup 真参考（缺历史 on 帧保持 unknown）、checkpoint 缺项、所有 PNG 原 bytes/pixels/guard、全部文件集合及前后 SHA。

真实 native earlystop 不 padding；工程错误保留 failed/partial rows、首错误、pending/failed 费用及完整四分母。损坏 raw/RGB/goal 或不闭合 journal 不因 report 写成 partial 就获认证。`RECORDED` 仅指非空槽完成或原生早停、空槽完整记录，不是 P2 成立。全局失败可能留下 not_started 槽；其缺项仍保留。未知指标不填零。

## 资源合同

最大四臂：constructor4 / close4 / reset8 / public100 (=4×(warm10+raw15)) / physics1016 / set_state16 / goal_setter12 / goal_render8 / goal_bind4 / explicit_render324。缺槽与早停只记真实支出、不退款失败尝试、不补齐预算，绝不声称等 FLOPs/等总计算。模型预测/J/训练新增均 0。

whole-stage wall900s（run 开始于完整 saved verify 前，因此这不是单纯物理窗口）、输出64MiB、final reserve8MiB、disk floor200MiB、metadata128MiB。Store/check/Meter 在 dispatch 前守门，close 可 cleanup，最终报告保留失败，不把超时写 complete。run 不 import torch；gym 注册与 factory 仅 MAIN 授权路径。

## 测试与职责

RAM 测试使用明确的 NumPy/模拟环境 fixture，不运行模型或原生物理。覆盖 full4×15 实际 adapter 回调路径、earlystop/失败、None 单/双/四槽不建环境、完整 native strict 重算、raw/context/task/goal/account/model 与事件/计量/PNG 篡改拒绝、源 SHA 先验拒绝、真实磁盘 Store.put/emit 两事件单 LF/hashchain roundtrip，以及原 runtime compact digest 与物理 codec 不同。测试通过不是实际物理证据。最终测试终态和三 SHA 见交付消息。

## MAIN CLI

工作目录 `/home/gbk/code/LeWorldModel/le-wm-main`。source/tests/doc 冻结后再 prepare；每个输出必须不存在。无需重新运行 CUDA planner。

```bash
.venv/bin/python -m pytest -q tests/test_agent_p2_wallcapped_physical_20261029_v1.py

.venv/bin/python -m experiments.spectral_reachability.agent_p2_wallcapped_physical_20261029_v1 prepare \
  --planner-output /ABS/ACTUAL_WALLCAPPED_CUDA_OUTPUT \
  --planner-report-sha ACTUAL_WALLCAPPED_REPORT_SHA \
  --planner-receipt /ABS/ACTUAL_WALLCAPPED_STRICT_RECEIPT.json \
  --planner-receipt-sha ACTUAL_WALLCAPPED_STRICT_RECEIPT_SHA \
  --planner-source-sha ACTUAL_FROZEN_WALLCAPPED_SOURCE_SHA \
  --output /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p2_wallcapped_physical_20261029_v1_prepare_MAIN01.json

.venv/bin/python -m experiments.spectral_reachability.agent_p2_wallcapped_physical_20261029_v1 verify \
  --prepared /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p2_wallcapped_physical_20261029_v1_prepare_MAIN01.json \
  --prepare-sha ACTUAL_PHYSICAL_PREPARE_SHA

.venv/bin/python -m experiments.spectral_reachability.agent_p2_wallcapped_physical_20261029_v1 run \
  --prepared /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p2_wallcapped_physical_20261029_v1_prepare_MAIN01.json \
  --prepare-sha ACTUAL_PHYSICAL_PREPARE_SHA \
  --output /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p2_wallcapped_physical_20261029_v1_MAIN01 \
  --authorization MAIN_WALLCAPPED4_FULL15_PHYSICS_ONLY

.venv/bin/python -m experiments.spectral_reachability.agent_p2_wallcapped_physical_20261029_v1 consume \
  --output /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p2_wallcapped_physical_20261029_v1_MAIN01 \
  --report-sha ACTUAL_PHYSICAL_REPORT_SHA \
  --receipt /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p2_wallcapped_physical_20261029_v1_consume_MAIN01.json
```

Python 接口：prepare(output, arguments)；verify(prepared, expected)→(p,data)；run(prepared,expected,output,factory,authorization=...)；consume(output,expected)→自有 receipt。receipt 返回 physical_report_sha256、plans[4]（含 None）、rows[4]（真实原生 rows）、missing_exports、counts、metric_breakdown、bindings，明确无科学放行。未运行项为 MAIN 生产 prepare/verify/native run/完整真实 consume；没有冒填结果或实际输入 SHA。
