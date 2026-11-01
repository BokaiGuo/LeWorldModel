# Common-resource RGB 独立标量复算 v1

仅新 saved-only consumer；不导入 producer 或其数学函数，不载入模型，不运行神经/物理/生产消费。读取显式外部 SHA 绑定的 Bohr common-resource RGB prepare/report/完整 strict receipt，直接解压原 float32 NPY，再以 float64 独立计算。不是重新颁发上游 native/preprocessing 认证：该认证由指定完整 strict receipt 提供，本工具检查其 SHA、全部来源文件和标量一致性。

## 固定接口与分母

实际 own schemas：`agent_p2_common_resource_rgb_20261029_v1.{prepare,report,consume}`，physical 为 `agent_p2_common_resource_physical_20261029_v1`，model 为 `agent_p2_common_resource_grasp_cuda_20261029_v1`。不冒用 wallcapped/GRASP 单独版 root/edge schema。模型上下文来自 physical prepare 中绑定的完整 `planner_plan.contexts[3]` 和 `model_witnesses`，对照 runtime strict 全六 plans；不是假定 `root0.json`。

六 plan 顺序为 root0 joint/composition/grasp、root1 grasp/composition/joint；各 cp5/10/15，固定18槽。root/time 分组固定6，每组三对依次 composition−joint、grasp−joint、grasp−composition，总18对。原生早停缺项保留 null，不替换终态、不填零。若 runtime export 为 null，三个 model/real 槽同样保留 null；无成功/根/时间筛选。通常17 real 的实例不是代码硬门。

逐槽核对 original task/context/goal192、identity Q、原 raw15 signed bits、warm10+prefix 的真实同时间 raw10、native goal7、PNG/state/snapshot/response。physical Q digest 保持 spaced JSON 无换行，与 runtime compact+LF 编码明确区分。真实 NPY 校验压缩文件 SHA、数组 SHA、float32 dtype/shape、完整单 gzip/NPY 无尾部数据，并与保存 projector 输出逐字节一致。绑定 prepare 全 bindings、全部 RGB artifacts/census/native seal、physical report/strict/prepare、model report/strict/prepare；结束再次校验所有来源未变。

## 数学与输出

令 p=pred192、a=RGB actual192、g=原 goal192。独立计算 model Q=mean((p−g)²)、real Q=mean((a−g)²)、prediction error=mean((p−a)²)，分母192（sum=192×mean）。

`model Q − real Q = 2 mean((a−g)(p−a)) + mean((p−a)²)`。

每对进一步核验 model margin−real margin = 两槽交叉项差 + 两槽平方项差。恒等式 residual 的界仅针对 float64 算术舍入：单槽 gamma(k), k=16×192+64；配对加两单槽界与32 eps×标量绝对值和。它不是科学容差/排序阈值，不用于把小差值置零。

native7 独立公式保留 velocity 和原 angle term `min(abs(delta),2π−abs(delta))`（可能负值），同时分开给 wrapped angle、agent/object XY、joint XY、raw-angle、pose5、velocity、native7 distance/reward、原 success。不得拿 wrapped diagnostic 替代旧 success 或 state distance。每对分栏 model/real 与各物理轴符号：negative/zero/positive/missing。关系表区分 strict_same/strict_reversal/both_tie/left_tie/right_tie/missing；reward 保留原始 delta/sign，比较偏好时显式取 loss=−reward。无跨轴合并判定、无唯一根因声称。

## MAIN CLI

在 `/home/gbk/code/LeWorldModel/le-wm-main`，以下路径全部传 canonical absolute，SHA 必须是实际终态值；当前不预填未来 report/receipt SHA。

```bash
.venv/bin/python -m experiments.spectral_reachability.agent_p2_common_resource_rgb_scalar_summary_20261029_v1 analyze \
  --prepared /ABS/RGB_prepare.json --prepare-sha ACTUAL_PREPARE_SHA \
  --report /ABS/RGB_output/report.json --report-sha ACTUAL_REPORT_SHA \
  --strict /ABS/RGB_consume.json --strict-sha ACTUAL_STRICT_SHA \
  --output /ABS/agent_p2_common_resource_rgb_scalar_summary_20261029_v1_MAIN01.json

.venv/bin/python -m experiments.spectral_reachability.agent_p2_common_resource_rgb_scalar_summary_20261029_v1 verify \
  --summary /ABS/agent_p2_common_resource_rgb_scalar_summary_20261029_v1_MAIN01.json \
  --summary-sha ACTUAL_SUMMARY_SHA

.venv/bin/python -m pytest -q tests/test_agent_p2_common_resource_rgb_scalar_summary_20261029_v1.py
```

仅新 ownstem 输出，排他创建≤16MiB；verify 重建全部汇总并精确对照，不覆盖任何封存。上游 JSON≤128MiB、RGB总文件≤64MiB、单 NPY压缩/解压≤2MiB。不是训练、不拟合 Q、不选择、不部署、不改 gate；P1/P2 admission、mechanism、equalFLOPs、physical benefit 均不签发。封存动作的后验诊断，不冒称预注册算法。

## 测试状态

首轮 NumPy/RAM 单元测试55 passed /11.86s（无失败）。最终完整65 passed /17.14s，包含真实临时文件写入→绑定读取→原 GZ NPY→独立18槽/18对复算，以及17-real、全缺失、null-export、来源/标量/配对篡改拒绝。fixture 是合成字节，PNG placeholder 不能作为真实渲染/strict证据。worker 未运行任何生产 analyze/consume、神经或物理。三件套随此次测试记录冻结，等待 MAIN 外部实际 SHA 执行。
