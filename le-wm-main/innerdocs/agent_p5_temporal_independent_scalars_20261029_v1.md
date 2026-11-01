# Temporal saved 独立标量核验器

仅 saved 数组与 NumPy 独立算术；不 import producer、旧分析器或其数学函数，不导入神经/物理模块，不重建 fullstrict。MAIN 的完整持久 strict 是时间顺序、原生输入语义与加载闭包的信任起点。本工具不能再次证明这些边界。

固定164 branches、80 signed pairs、四 root×off/on strata（每组41/20），不按效果筛选。四 cell 为 T_AA、T_AR、T_RA、T_RR；字母分别表示时刻5、10的 actual/rolled 联合 visual+proprio；时刻15保持 causal pred15。混合 context 不是物理反事实。

## 固定真实 authority

- temporal report `results/agent_p5_temporal_history_factorial_20261029_v1_cuda_MAIN01/report.json`：`f046c617ba2eeb6fcbd79799238e8ccbd00ca7ecd6b5414f2021844550672747`
- temporal receipt `results/agent_p5_temporal_history_factorial_20261029_v1_consume_MAIN01.json`：`86149cf64a0e5c423ee99f23f98dde9ce06ccf8ced837f70c65648e3518e51ea`
- execution_binding 所指 prepare：`7fac018fdb7b84362c174f84345c0d0b10c0045e34bf675f8a912eb988d52bef`
- early report `results/agent_p5_early_history_factorial_20261029_v1_cuda_MAIN01/report.json`：`4fc86a7fdc69fbe95003e417ee38e3a4d430360a8eb5bd0c89f73ee845926d0f`
- early receipt `results/agent_p5_early_history_factorial_20261029_v1_consume_MAIN01.json`：`d9461f7a21fa13454eca1f4371488431cf833a08ee758990c2dcb999b342600e`

先验证两份完成回执和绑定，再解数组。AA 必须为旧 report 的 `/calls/{4*i}/outputs/$tuple/{0,2}/$array`，与旧 rows、当前 prepare AA_outputs、当前 inherited_T_AA 全等，不能猜路径。三新角按 branch-major 三 calls 逐一归属。全四 cell 的 full3 visual/proprio 都检查 dtype、shape、有限值、inventory、文件 SHA，无 NPY trailer 或别名；只末 visual slot 用于 MSE。当前与旧完整 batch2 labels 字节全等，评分用 `[1:2]`；不会用旧 H1 label 替代。

独立复算每 branch 的 float64 全256×384 MSE，以及 slot5、slot10、interaction、joint 的 MSE/feature effects。令 a,b,c,d 为四 cell：slot5=((c-a)+(d-b))/2；slot10=((b-a)+(d-c))/2；interaction=d-c-b+a；joint=d-a。独立重算全部80 pair 的 response L2、dot、cosine、error、中央差分及 response effects；真实/预测 norm 严格零则 cosine=null，无 epsilon 或阈值。输出整体及四层 summary、全部六种 cell pair 的16格符号转移。逐字段 exact 对照 producer 保存的标量，不使用 allclose。

资源分列本次492/1476及继承 early656/1968实际收据，不相加宣称独立实验或等 FLOPs。此工具新增 encoder/predict/AD/physics/training 均0；32MiB metadata、4MiB单NPY、900s、200MiB磁盘 floor，按41 branch组缓存末帧。输出独占前缀、exclusive create、fsync；verify 全量确定性重算本工具结果，不调用 fullstrict。所有已用数组和元数据结束再验SHA，输出绑定自身source/tests/doc。

## MAIN CLI（inner root）

```bash
cd /home/gbk/code/LeWorldModel/le-wm-main
.venv/bin/python -m pytest -q tests/test_agent_p5_temporal_independent_scalars_20261029_v1.py
.venv/bin/python -m experiments.spectral_reachability.agent_p5_temporal_independent_scalars_20261029_v1 analyze --output results/agent_p5_temporal_independent_scalars_20261029_v1_MAIN01.json
.venv/bin/python -m experiments.spectral_reachability.agent_p5_temporal_independent_scalars_20261029_v1 verify --analysis results/agent_p5_temporal_independent_scalars_20261029_v1_MAIN01.json --analysis-sha ACTUAL_ANALYSIS_SHA
```

CLI report/receipt 默认固定上述真实路径与SHA；显式替代SHA不匹配则拒绝。分析产物SHA只能在 MAIN 实际写出后取得，不预填。

## 证据边界

worker 仅运行 synthetic tempfile/RAM fixtures 与 CLI help，未执行 actual analyze/verify/fullstrict，未运行神经/物理。测试不是科学通过。该分析在既有 outcome 后实现，非预注册、非独立根确认；不会恢复旧未知历史或打开 P1/P5 gate。完整journal、输入因果语义、RR原生IO bitwise证明及模型真实加载字节依赖已绑定 MAIN strict，本工具不重复认证。缺authority、缺角、错误label、nonfinite、篡改或标量不等均拒绝，不降分母。
