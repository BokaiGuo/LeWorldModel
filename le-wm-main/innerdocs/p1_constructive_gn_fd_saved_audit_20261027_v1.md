# Constructive GN FD saved独立审计 v1

2026-10-27。仅新增本stem源码、tests和本文；不改producer或其prepare/results。纯saved/NumPy，不import FD producer，不调用其collect/requests/analyze/compare，不运行模型、神经编码、AD、物理或训练。

## 外部批准输入与独立核验

MAIN批准实际report：`results/p1_constructive_gn_fd_cuda_20261027_v1_cuda_main01/report.json`，SHA256=`107b958988530df78592aa2a98701cc091ba9daaddbabb31aa984402e76cc23d`；prepare=`9485b6ddc08f246b27649744ac745f2bdecfb6bbc6dd809686cff4974351754a`。CLI必须显式给完整report SHA，不自行发现/猜测它。report绑定prepare SHA及binding，外部report授权链传递到prepare。

冻结producer源码SHA=`f6147161d839952f2671592b275bf382908a8c79b579ff8fabee435fae8b098a`。旧planner report=`4dd5823e9a273b820d6b7b3f6a36a4a5a19b193383eaa9cbf94442082e79d98d`、旧prepare=`30b58155a59f73fdc8cc132ff069e645ed6c2bbff9434f474b1f7b9c412f7eb4`。独立从旧planner raw journal取两个`iter_1_AD`零点J，检查192raw rows、joint J、AD forward和initial triple的两个完整context，重建saved Jv及点身份。批准task export只读、identity192与原normalizer/cap校验；不读sourcepack/blind NPZ/certificate/物理outcomes。

新FD journal严格逐事件消费：每H2 `evaluation_attempt`，两个packet各actionencoder attempt/complete、predict attempt/complete、native_packet，最后evaluation_complete；总456events、38H2、152module attempts+152completions，其中predict76/actionencoder76。禁止遗漏、多余事件、退款、错序、partial冒完整。用saved terminal、原3states2actions和float32原normalizer独立重建76contexts并hash；两个fresh nominal必须精确匹配旧AD的terminal和两context hashes。

独立生成全部固定输入：packet0e0、packet1e0、fixed20signunit；epsilon1e-3/3e-3/1e-2，minus/plus。独立复算18个central FD、Jv、有效float32 secant方向与J乘该方向、requested/effective误差、midpoint drift及邻eps差。复用旧纯saved auditor的通用JSON/gzip/metric/context数学工具，不复用其旧任务collector或FD producer计算。保存analysis逐字段重算校验的rtol/atol=1e-12仅为receipt算术一致性容差，不是导数精度或科学阈值。

## 输出与结论边界

独占新JSON≤100KiB，200MiB floor；报告含auditor source/tests/doc hash、输入report/prepare SHA、完整分母、各eps六个比较的min/median/max（requested/effective relativeL2、cosine、absoluteLinf）、root×eps与direction×eps分层、18比较标量及eps稳定性。不保存重复大数组，不丢零分母（null及undefined计数）。所有eps完整报告，不事后选eps或改阈值。

实际原GN156predict（含16ADforward）、8J/1536数学VJP，以及D2P132predict/22J/4224数学VJP等完整旧资源receipt单独保留；本审计新增神经/物理/AD均0，不称总资源公平。不把saved syscall receipt当作独立CUDA sync证明；历史磁盘free无法由终态重建，仅核声明floor/cap和实际artifact总大小。status `saved_fd_arithmetic_verified_not_admitted`仅指saved链/算术一致；derivative_verdict仍not_assigned，threshold=null，P1=false。partial/error输入严格拒绝但不删除或修补源证据。

## 命令

cwd=`/home/gbk/code/LeWorldModel/le-wm-main`。

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q -p no:cacheprovider tests/test_p1_constructive_gn_fd_saved_audit_20261027_v1.py
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m experiments.spectral_reachability.p1_constructive_gn_fd_saved_audit_20261027_v1 --report results/p1_constructive_gn_fd_cuda_20261027_v1_cuda_main01/report.json --report-sha 107b958988530df78592aa2a98701cc091ba9daaddbabb31aa984402e76cc23d --output results/p1_constructive_gn_fd_saved_audit_20261027_v1_actual01.json
```

测试全RAM，不创建tmp/prepare；覆盖完整event流、归一化/上下文、float32有效方向、negativeJ、分析篡改、缺失/额外/错序/refund/anchor/identity破坏、gzip完整性与外部SHA必填。实际分层数值以独立JSON为准，不把MAIN先报误差当作pass条件。
