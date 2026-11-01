# P3 joint scheduler：独立 saved-outcome 算术审计 v1

## 输入与范围

固定真实 CUDA 报告 `results/p3_joint_horizon_goal_metric_scheduler_20261027_v1_cuda_main01/report.json`，SHA256 `23eae72d18fa02e9997859311bef282fc4f5e3478790d846524a3fa6f6b70950`；prepare SHA256 `7886406284604cf17ba97671030ef32fde9d32f6f4c18b8293db0c2129654a47`。仅审计这一份完整成功的 saved report，不是重跑实验或任意失败执行恢复工具。

审计器不导入 scheduler、compiler、consumer 或任何 model/physics runtime；只用 NumPy 与纯 IO/source-seal 工具。旧源码、prepare、实际结果全部只读。独占新目录输出小 JSON、输入哈希和新 source seal，不复制旧帧、权重或大型报告。每个证据文件最大 32 MB，拒绝路径穿越、符号链接、重复 JSON key、非有限值与哈希漂移，结束重查输入。

## 独立重建内容

1. 验证 prepare/protocol、source live/archive seal、所有 source/binding bytes；核对 root JSON 与总报告完全一致、journal 路径和 SHA。四根通过完整 root SHA join，不凭 report 数组位置识别身份；原 seed_index 与 balanced arm order 另按封存协议检查。
2. 核对原 packet hash、goal、旧 Q 和两个 active metric ID。原始 native 192 维 goal/context 不做 Q 空间伪装。每 Q/H 直接从保存的 J 和 nominal 重算 `K=QJ`、`r=Q(goal-nominal)`；独立数组 SVD 重算 rank/chi（相对 cutoff 1e-6）。同一 SVD 的谱另检查 ridge 谱，KKT 检查 damping .002 的 ball-ridge stationarity、feasibility 和 complementary slackness，以及各线性代价/半径/params/raw proposal。此过程是 saved-array 数学检验，不重新验证 AD 导数真实性。
3. uniform 固定 blocks 5/5；dynamic 固定 `gain=max(chi2-chi3,0)/max(chi2,1e-12)`、`blocks3=floor(10*(.25+.5*gain)+.5)`。不按 outcome 改规则。保存 chi 的精确值用于核对原 allocation，独立 SVD 重算 chi 也必须得到相同 block 分配，避免浮点边界无声换合同。
4. 独立 PCG64 重建所有 bank：zero index0、own-Q ridge index1，随机尾保持原 pool[k]，不左移、不去重。验证 H/index 完整覆盖、raw/logits 内容、每个候选和 winner 的 own-Q terminal cost `mean((Q@(z-goal))**2)`、稳定 `(cost,H,index)` argmin。全部 nominal/AD prefix/candidate/whole/segmented context 均用独立 float32 time-major token 算术验证 history 内容、layout、metadata 和仍保留的 latent 顺序，非仅 shape 校验。
5. 按真实顺序消费全部 journal 事件，包含每个 action encoder/predict 的 intent/completion 对、共享 nominal/AD 前向、J 数组、两 Q 响应/SVD、全部候选、whole 与每段 winner replay。拒绝缺失、重复、额外操作与错误 role，不只相信汇总。原始 failures/blocked/partial 若出现会拒绝完整成功结论，不静默删掉或记为零成本。

## 预算及分母

四个既有 calibration roots ×四臂 =16 arm records，两个 metric 各有四个 scheduler 配对，共8配对；不是16新根，新增独立根为零。当前成功记录：

- predictor 1066、action encoder 1066，分别独立计账，均低于1096各自总 cap。
- shared 40 = nominal20 + AD forward20；四臂搜索每臂60，共960；winner whole+segmented 验证共66 = `2*sum(Hwinner)`。三项相加1066；AD forward/verification 已含在1066，不额外相加。
- fresh Jacobian8（H2 `[192,20]` /H3 `[192,30]`），jacobian API8，数学 VJP directions1536；backward engine API 次数未知，不冒充1536次 API。
- execution SVD32；两次 source validation 另32 saved-array SVD。此次独立审计新增16次 saved-array SVD，单列，不混成神经成本；测试期间的算术调用不算新增科研样本。
- inherited metric/planner 工作完整原样单列：历史 metric80 encoder +80 existing projector +8 SVD；历史 planner552 predictor/action encoder、8 Jacobian/8 SVD等不是本次新增工作，也不是零成本。cached history取得成本未知。

各臂 search forward 数匹配，不等于各臂总实际成本或全 pipeline/FLOPs 匹配。Hwinner不同使verification成本及未来物理长度不同；无padding退款或把剩余预算转搜索。不同 joint H 参数维数、AD backward 和CPU求解开销也不能被样本数掩盖。

## 比较与候选覆盖

只比较同 root、同 source input/goal、同 active metric 的 dynamic − uniform；绝不计算 goal-patch-Q − old-Q。逐 Q 标签、逐 same-H /mixed-H 分层，逐根保留真实 own-Q 差值；不对根特异 Q 的差值幅度做 pooled 均值。每个 H 另列两边候选 indices、shared/only indices 和各自 best own-Q cost；shared candidate 的 action/context/cost 必须逐值相同，动态增删候选覆盖不能伪装成相同候选集。

实际结果：old-Q四配对均same-H，两根严格下降、两根完全tie。goal-patch-Q三same-H配对全部tie，另root1为 uniform H3 /dynamic H2 的mixed-H，dynamic own-Q cost高 `0.008208458768816457`；不与same-H混算。不采用容差把非零差值改成tie。精确逐根差值及所有候选代价保存在新报告。

这不是物理结果：无新 consumer/physical endpoint，不能称物理任务改善。全部根已暴露且只有四个，不能独立P3准入。模型 unchanged、输入训练normalizer来源及保存 latent/Jacobian 是否确由所述模型产生不由本审计独立再生成；source绑定与数组一致性不等于端到端科学有效性。

## 命令

在 `le-wm-main` 下，以下均不调用模型或物理：

```bash
OPENBLAS_NUM_THREADS=1 .venv/bin/python -m pytest -q tests/test_p3_joint_scheduler_saved_outcome_audit_20261027_v1.py
OPENBLAS_NUM_THREADS=1 .venv/bin/python -m experiments.spectral_reachability.p3_joint_scheduler_saved_outcome_audit_20261027_v1 --output results/p3_joint_scheduler_saved_outcome_audit_20261027_v1
```

输出路径必须不存在；已封存目录不覆盖。测试覆盖实际 saved census 与代价重算、候选与context/metric/own-Q KKT篡改、漏账/错role/额外journal、root/arm遗漏或重复、历史成本清零、source validation汇总篡改及跨Q比较拒绝。通过只表示这些非神经检验，不表示已重放真实模型或物理。
