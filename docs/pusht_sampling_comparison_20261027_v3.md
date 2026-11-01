# V3 physics-only runner：V2 sealed plans 原样重放与可靠 finalize

V2 planning 已实际成功：8 plans，4112 planning + 16 verification predict samples。
V2 physical 8 个分支 JSON 已落盘，但最终 hash gate 把 str binding key 直接传给
只接受 Path 的 base.digest，导致报告未生成。不能把该目录描述成拥有完整通过的 report。
本次不修写 v1/v2 源码、测试、文档或历史输出；新建 v3 runner/test/doc。

V3 只使用 V2 `load_prepared`、`validate_plans`、`verify_rgb_context`、`guarded_branch`，
不调用 planning，不加载模型，不做 CUDA 推断、训练或优化。仍是四根原 calibration
cohort、8 分支、同样 root/goal/context/commands/liveRGBguard；所有中止、失败保留。
官方 labels 和 pose 沿用并与 installed evaluator 比对。

每个 binding hash 调用显式 `digest(Path(path))`，逐项保留失败，不依赖会抛出的
`all(generator)`。最终 sourceverify/hash 异常进入 failures 和 false gate；先写完整
report 再返回错误。环境关闭异常也写入 report。不可写/磁盘耗尽依然可能导致写出失败，
不承诺对存储故障也能落盘。

可选 `--compare-v2` 在物理重放前绑定旧 protocol、8 JSON 的 SHA，确保其绑定同一
V2 plans；比较每个分支整个 trace 精确相等（无数值容差）及整个 branch JSON 对象精确
相等。比对仅作重放审计，不能影响命令、重试、筛根或动作选择。任一不同会保留输出并
使最终 integrity gate 失败。V3 在任何比较/最终检查前先落盘每个物理分支。

## 主线程准确命令

```bash
cd /home/gbk/code/LeWorldModel/le-wm-main
.venv/bin/python -m experiments.spectral_reachability.run_pusht_sampling_comparison_20261027_v3 \
  --run results/pusht_sampling_comparison_20261027_v2_prepared \
  --compare-v2 results/pusht_sampling_comparison_20261027_v2_physical \
  --output results/pusht_sampling_comparison_20261027_v3_physical
```

无需重跑 prepare 或 CUDA plan。V3 在物理前创建 execution_source_seal，包含 v3
runner/tests/doc、所复用 v1/v2 helpers、visible executor/guard/labels、installed evaluator
及物理上下文/root collector；V2 prepared seal 也在前后核验。输出必须全新。

```bash
.venv/bin/python -m pytest -q -p no:cacheprovider \
  tests/test_pusht_sampling_comparison_20261027_v3.py \
  tests/test_pusht_sampling_comparison_20261027_v2.py \
  tests/test_pusht_sampling_comparison_20261027.py \
  tests/test_pusht_visible_command_executor.py
```

测试读取真实旧 V2 physical JSON 和 protocol 中的 str-key bindings，使用严格 Path
digest 完成 FINALIZE；另测 sourceverify 抛出、hash 抛出/不匹配、重复分支、trace 微小
篡改与 official label 篡改不能伪通过。测试不加载神经模型；真实物理重放由主线程执行。

预估 CPU 物理几十秒至几分钟，新增结果数 MiB，无 GPU 显存分配。
Q 的 agent-motion/static-object 假象未修复；P1/P2 不准入，不声称任务效果、官方
MPPI/GRASP 或等总算力。旧 V2 八份成功分支不等于旧 V2 最终收据成功；新 V3 报告
只声明本次物理重放和完整性审计状态。
