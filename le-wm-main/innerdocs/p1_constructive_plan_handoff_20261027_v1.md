# Explicit planner-consume handoff bridge

NEW独占source/test/doc；不修改冻结 planner/physical。此报告是 **adapter handoff**，不是伪造的原 planner report，
也不是独立 planner saved audit。物理 consumer 的 compiler_report 字段显式指向此 handoff，后者绑定真实 planner report 外部 SHA。

仅 CPU 已存档文件/数组检查；无模型执行、编码、训练或 physics。`planner.consume` 会哈希读取模型文件以校验已有闭包，
不会加载模型或执行神经计算。桥不读取 sourcepack/certificate/witness、native state、物理结果。
原 planner.consume 校验全局 release、prepare/source/model/artifact 闭包、所有 own selections；桥再以 task_export SHA 校验后
读取明确 task metadata，join task_id/source_slot/blind SHA。固定两根×原两 arms，不筛选、不重跑、不从任何 outcome 选计划。
source_slot 与 approved blind SHA 必须匹配冻结 encoder APPROVED_INPUTS。both arms 必须绑定同一 task；缺根/重复/交叉任务均拒绝。
按 source_slot/固定arms排序仅为 metadata 重排，保留 planner_order 和 selection SHA，不改变 rawcommands。
native roots 映射只在第二步 evaluator_map 提供；physical.prepare 最终验证真实 sourcepack/原 recipe/根/goal。

## MAIN命令

工作目录 `/home/gbk/code/LeWorldModel/le-wm-main`；以下均不运行模型/物理。

```bash
.venv/bin/python -B -m experiments.spectral_reachability.p1_constructive_plan_handoff_20261027_v1 handoff --planner-output /home/gbk/code/LeWorldModel/le-wm-main/results/p1_constructive_blind_planners_cuda_20261027_v1_cuda_main01 --planner-report-sha 4dd5823e9a273b820d6b7b3f6a36a4a5a19b193383eaa9cbf94442082e79d98d --output /ABS/NEW_handoff01.json
.venv/bin/python -B -m experiments.spectral_reachability.p1_constructive_plan_handoff_20261027_v1 verify --handoff /ABS/NEW_handoff01.json --handoff-sha HANDOFF_SHA
.venv/bin/python -B -m experiments.spectral_reachability.p1_constructive_plan_handoff_20261027_v1 selection --handoff /ABS/NEW_handoff01.json --handoff-sha HANDOFF_SHA --evaluator-map /ABS/evaluator_map.json --evaluator-map-sha EVALUATOR_MAP_SHA --output /ABS/NEW_selection01.json
.venv/bin/python -B -m experiments.spectral_reachability.p1_constructive_plan_physical_20261027_v1 prepare --manifest /ABS/NEW_selection01.json --manifest-sha SELECTION_SHA --output /ABS/NEW_physical_prepare01.json
```

每次写入返回实际文件 SHA；write 使用 xb+fsync，独占、单文件<=256KiB、200MiB floor。不写中间结果，不复制大资产。
prepare 后 MAIN 审核并另行运行物理；此桥不提供物理执行入口。

evaluator_map 必须恰有以下字段（roots 填真实 sourcepack identity，非 blind 算法输入）：

```json
{
  "schema": "p1_constructive_plan_handoff_20261027_v1.evaluator_map",
  "sourcepack": "/ABS/sourcepack.json",
  "sourcepack_sha256": "9ad3a414fd2148258524d4a7e9d4f070f59532ec2e5943cfa61f9145b3a41bd9",
  "source_prepared": "/ABS/source_prepare01.json",
  "source_prepare_sha256": "98ffb806018a270f5ef27ffa76be929393a74355c213bc274bf158c72d80b086",
  "roots": [
    {"root_slot": 0, "root_sha256": "f7aec52327fa6a6c8fb444919ba426a467ec2382b57da6d8a4f3701f82c65962"},
    {"root_slot": 1, "root_sha256": "f47e9958781f7eac4c810f9dfe24232b635f41d5b4c56423d8c2623f9fd5a328"}
  ],
  "wall_seconds": 300
}
```

handoff rows 为 `source_slot,arm,commands,blind_sha256,planner_order,task_export,task_sha256,task_id,planner_selection_sha256`。
selection 自动指向 `/rows/i/{commands,source_slot,blind_sha256,arm}`，digest 用冻结 physical 的默认JSON分隔符算法，
不误用 planner compact canonical。每行 compiler_schema/status 显式为本桥 schema/status；其 authority 字段声明复用 consume 的范围。
桥不判断物理可达性、plateau、P1准入或独立样本数；global freshness 保持 unknown calibration only。

## 测试

```bash
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -B -m pytest -q -p no:cacheprovider tests/test_p1_constructive_plan_handoff_20261027_v1.py
```

纯内存mock，无tmp目录；独立验证 metadata 错配/partial/duplicate/commands precision/字段与digest接线，
包括直接调用冻结 physical.selection 对桥产物的接口测试。真实 saved planner 接线只读结果另行报告，不能用 mock 冒充。
