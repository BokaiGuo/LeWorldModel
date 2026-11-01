# Line-search GN actual saved receipt 独立审计

仅 NumPy/stdlib，不导入 runner、screen、模型或物理代码，不重新执行任何 forward/AD。
输入 `results/p1_line_search_gn_cuda_20261027_v1_cuda_main01/report.json`，
SHA `5f9825f98ed40f637c9d48e49c7c53428d1380e4234e3a1e5503a76f74c4bf1e`。
原 prepare SHA `e1b6d9783da6a4e32d360326869fe1b64fef8c3f4f168095be4b7f8c215b26d9`。
旧三文件及所有实际结果保持封存不变。独立模块不是原 helper 的绿色结论转抄。

## 逐项重建

先验证原 report、exact五文件 census、四个 artifact SHA、prepare SHA、全部 source/input/model-file
SHA、两个原 task_id 的 JSON 原编码身份及模型/goal/Q/normalizer绑定。gzip 要完整EOF/CRC且无尾随垃圾，
单文件压缩和展开各上限32MiB。全部读取后再次复验来源SHA防 TOCTOU。

严格顺序 cursor 消费全部5660事件，按两根分别重建计数器，每个 attempt/return 必须唯一配对、
index/phase/evaluation_id/root一致。完整 J 是包裹 ADforward/VJP 的资源，不能把其内部 forward 再重算一遍；
每个 packet 强制 actionencoder→predict，各自 attempt/return→context。初始三路重放及四轮
各五尺度+三路重放都必须存在，不接受只保留 selected 或 skipped trial。重复候选照样保留计账。

全部140普通 packet +16 ADforward packet 验 context layout/完整latent chronology 与
float32原mean/std逐元素 normalized action history。raw命令严格是保存 logits 的 NumPy float32 tanh，
逐packet球约束原样，不clip；每次 terminal 按同一 task 的原192维 Q/goal exact重算 MSE。

8次J：每次192条顺序 raw-control derivative 行共1536条，独立拼接再乘
`1-commands**2` 一次，要求与 joint_logit_jacobian 和 derivative_complete 的192x20数组 exact相同。
ADforward两个context、logits、terminal必须是本轮 verified incumbent 的完整路径；后轮必须绑定
前轮新选优重放结果，不能沿用zero J或旧点。原Q gradient独立重算。

不调用原SVD程序：用 `solve(A.T A/192+.002 I,-A.T residual/192)` 独立重算 ridge delta，
记录与保存方向的最大绝对差和法方程残差。允许1e-8 relative/1e-10 absolute数值核验误差，
这不是后验科学passing阈值。用保存delta独立复算原packet球投影/float32向内舍入及全部五个
`[1,.5,.25,.125,.0625]` logit segment。所有五个fresh nonlinear cost加 incumbent稳定选优，
selected的nominal/whole/segmented三路与候选exact一致，最后两个root gzip对象和top摘要逐字段重建。

## 结果范围

每根78predict/78actionencoder、4J、768数学VJP；合156/156、8J/1536VJP，
另8SVD/solve/proposal/selection。普通fullH2 rollout70个，AD fullH2 forward8个。
其156个packet forward已含ADforward，禁止再加16。源image/acquisition及旧B8是inherited，不是新成本。
不把VJP方向数当FLOPs；不能与原B8称总计算匹配。

两根实际final cost分别1.1405115382133657、1.9203170978443598，每根尺度顺序
`[1,1,.25,.25]`。这是同原Q的模型内优化记录，不是物理改善、隐藏pose证据、新独立根或P1准入。
rawJ行与chain账一致不独立证明AD本身是正确导数；若要该主张仍需对应fresh-point FD等独立证据。
模型前后tensor hash来自CUDA producer记录；本审计只重新hash冻结文件和复核receipt，未重新load模型。

测试含actual saved重算、21种数组/chronology/预算/selection反例、来源SHA/TOCTOU、gzip故障与独占
小报告读回；mock/变异是审计器拒错覆盖，不是新的模型运行结果。输入有partial/错误时拒绝认证，
不改写、不清理原失败记录。

```bash
cd /home/gbk/code/LeWorldModel/le-wm-main
PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -B -m pytest -q -p no:cacheprovider tests/test_p1_line_search_gn_saved_audit_20261027_v1.py
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -B -m experiments.spectral_reachability.p1_line_search_gn_saved_audit_20261027_v1 --output results/p1_line_search_gn_saved_audit_20261027_v1_saved01.json
```

新审计输出<256KiB且保留200MiB floor，以xb独占创建/fsync，不复制原J到新报告，只记录核验摘要及J的SHA；
完整原始J仍在不可变输入gzip中，可按其SHA重建，不删减原证据。
