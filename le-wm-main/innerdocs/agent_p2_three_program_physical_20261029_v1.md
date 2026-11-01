# P2 three-program full15 native adapter v1

本版是供 MAIN 执行的真实原生物理适配器，不是科学准入或新选择器。仅新增本 source/tests/innerdoc 三件套；不修改冻结 planner、旧 driver、结果或 gate。worker 只进行了 NumPy/RAM 测试，没有生产 prepare、神经计算或物理运行。

## 固定输入与执行

运行时先核 planner source SHA，再调用实际 planner `consume` 重建完整 receipt，并与传入的已保存 receipt 全量比较；随后重建 planner prepare 与原生任务来源。不是把新 planner 伪装成旧 schema。四分支顺序固定为 root0 joint、root0 three-program、root1 joint、root1 three-program，均为 off；2 known roots，非 4 independent/fresh roots。

每分支使用独立新环境，原 reset/place/warm10 后直接执行已封存 selected raw15，不恢复 imagined state、不反馈重规划、不读物理收益选动作。保留原完整 task、goal、warmup、renderer 和字节动作见证；cp0/5/10/15 均取真实观察。每个 warmup0..10、plan0..15 捕获 off/on/off 三次原生 RGB。复用冻结原生 placement/hooks/capture，full15 loop、schema、资源计数与 strict validator 为本版局部实现，不修改旧全局。

原生早停立即停止后续 dispatch，保留四分支分母与缺失终点；异常不退款，保留错误和清理错误。严格消费仍拒绝不合法 RGB、替换动作、日志或身份不一致，不能为了给失败 run 签 receipt 而跳过完整性检查。原生 full7、reward、termination、原目标指标逐返回复算；warmup 原 placement goal 和 bind 后 final task goal 分相验证。

## Authority 与 codec

MAIN 已提供实际 planner authority（CLI 显式传入，而非代码未来默认值）：

- source `3111312477ff3cd9a764dc7d063861c1becaaf58885dc0fb25c2f3ec73d15fd5`
- prepare `86af2d24f73e769fb3b91fa8a571f8ca265df95c93af780d3924ea644585d461`
- report `56e8b4f2df522f7a0c4cb9bc37ad74bcbc5d4f89e2c5cd601d42cd0eb071d8ef`
- receipt `c491555fda5bb543af69f70987541edb0abb99fa32c559f3aa6190901393017f`

Own schema digest 沿用 native spaced JSON；producer selection/context/goal/edge seals 使用 planner compact JSON 加换行。selection seal 排除 producer consume 后附加的 report/prepare SHA；两类 codec 不混用，测试明确证明普通对象的 hash 不同。prepare 绑定完整 producer/task/model/source closure 以及本三件套；run 不加载模型。

## 固定预算与 strict

完整成功预算：4 constructor/close、8 reset、100 public steps（40 warmup + 60 plan）、1016 physics steps、16 set_state、12 goal_setter、8 goal_render、4 goal_bind、324 explicit_render/PNG。wall 900s，输出总 cap64MiB（final reserve8MiB），磁盘 floor200MiB，prepare metadata cap128MiB。所有计数来自实际 callback/intent/return，不是仅声明预算；失败资源不退款。

消费验证全部事件种类、分支/操作嵌套时序、动作位、原生返回、phase-goal binding、所有 checkpoint 与轨迹、资源 census、PNG 文件及 pixel hash、RGB224/原 guard，以及精确文件集合；未知事件/额外文件拒绝。完整文件读取使用本版64MiB cap，不继承旧32MiB cap。消费结束再次核输入字节和 source closure。

P1/P2 admission、P2 mechanism、freshness、independent confirmation、equal FLOPs、physical release 均为 false；这里只交付工程可执行性，不能由 RAM 测试推出物理收益。

## 测试记录

最终 worker 命令：`.venv/bin/python -m pytest -q tests/test_agent_p2_three_program_physical_20261029_v1.py -x`，结果 **37 passed in 25.68s**。全部为 RAM/NumPy/mock 环境，不是原生物理证据。覆盖实际 producer `model_plan` schema、两类 digest、full4/full15 生命周期、324 PNG 完整 saved consume、cp15、所有字节动作、早停/异常、清理、事件/goal/binding/renderer/census 篡改和64MiB cap。

开发中保留的实际 RED：初始缺新模块；新增完整生命周期时缺 `validate_saved`；扩展测试时 1 failed / 21 passed，原因是 fixture 错误期待 invalid-render/替换动作事件可以获得 strict 通过。修正测试期待，未放松生产 validator。最终以上37项通过。测试导入 producer 构造 RAM plan，不调用 forward/AD/encoder；不将 fixture 结果当真实实验。

## MAIN CLI

从 `/home/gbk/code/LeWorldModel/le-wm-main` 执行；所有新输出须不存在。worker 未执行以下生产命令。

```bash
.venv/bin/python -m experiments.spectral_reachability.agent_p2_three_program_physical_20261029_v1 prepare \
  --output /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p2_three_program_physical_20261029_v1_prepare_MAIN01.json \
  --planner-output /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p2_three_program_20261028_v1_cuda_MAIN01 \
  --planner-report-sha 56e8b4f2df522f7a0c4cb9bc37ad74bcbc5d4f89e2c5cd601d42cd0eb071d8ef \
  --planner-receipt /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p2_three_program_20261028_v1_consume_MAIN01.json \
  --planner-receipt-sha c491555fda5bb543af69f70987541edb0abb99fa32c559f3aa6190901393017f \
  --planner-source-sha 3111312477ff3cd9a764dc7d063861c1becaaf58885dc0fb25c2f3ec73d15fd5

.venv/bin/python -m experiments.spectral_reachability.agent_p2_three_program_physical_20261029_v1 verify \
  --prepared /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p2_three_program_physical_20261029_v1_prepare_MAIN01.json \
  --prepare-sha ACTUAL_OWN_PREPARE_SHA

.venv/bin/python -m experiments.spectral_reachability.agent_p2_three_program_physical_20261029_v1 run \
  --prepared /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p2_three_program_physical_20261029_v1_prepare_MAIN01.json \
  --prepare-sha ACTUAL_OWN_PREPARE_SHA \
  --output /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p2_three_program_physical_20261029_v1_MAIN01 \
  --authorization MAIN_THREE_PROGRAM4_FULL15_PHYSICS_ONLY

.venv/bin/python -m experiments.spectral_reachability.agent_p2_three_program_physical_20261029_v1 consume \
  --output /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p2_three_program_physical_20261029_v1_MAIN01 \
  --report-sha ACTUAL_OWN_REPORT_SHA \
  --receipt /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p2_three_program_physical_20261029_v1_consume_MAIN01.json
```

Python 消费 ABI：`consume(output, expected_report_sha)` 返回本版 `.consume`，含固定顺序 `rows`、counts/errors、planner authority、bindings 和 metric_breakdown。每行 checkpoints 含实际 cp0/5/10/15 capture（提前停止可缺后续 cp），其 PNG/file/pixel/state/snapshot 来自完整日志并已验证。下游应读取本版真实 schema/receipt，不伪造旧 report。
