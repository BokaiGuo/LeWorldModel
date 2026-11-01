# P1：目标代价方向相反，不等于 AD 接线错误

主线程新增独立复算并实际完成，未调用模型或物理环境，未改选择器：
`le-wm-main/experiments/spectral_reachability/main_observed_fd_scalar_cause_20261028_v1.py`。
结果为 `le-wm-main/results/main_observed_fd_scalar_cause_20261028_v1_MAIN01.json`。

## 范围和实际结果

固定旧 FD 的四个已知根，各自原 selected/zero 前缀，共八个上下文；两个
atanh packet 参数、两尺度(.01,.05)，共32个方向比较。**不是**最新两个根
十坐标实验，也不是原80/76对排序面板，不能混合计数。

- 32项中8项模型与真实编码的 scalar Q secant 符号严格相反。
- 25项的响应误差、基点偏差、交互三项存在异号相消。
- 一个条件的四项真实编码响应均为零，不能记成余弦0或可靠方向。
- 二次恒等式最大浮点残差6.25e-14。
- 10项测试通过，包含实际32项全量读取与合成代数反例；不是10个科学样本。

## 分解与解释

令 `Jm/Ja` 为模型/真实终态的中心 secant（不是假设物理可微），`rm/ra`
为各自正负终态中点减原目标，经同一原投影后得到的残差，则：

`gm - ga = 2 mean[(Jm-Ja)*ra] + 2 mean[Ja*(rm-ra)] + 2 mean[(Jm-Ja)*(rm-ra)]`。

分母采用原动作**参数**的实际float32跨度，不偷换成raw action单位。
8项反向中，响应项绝对值均大于另外两项之和的绝对值（描述性核对），
不支持把本批差异只解释成动作无关平移。与此同时，即使J完全一致，
基点偏差也能改变scalar目标梯度，已用解析反例加入回归。

所以，向量FD接近AD只能支持当前模型导数接线正确；不能据此推导
物理优化方向正确。三项代数删除不是干预，不能拿真实J或真实终态
进入选择器来“修复”模型，也不能据此排尽代码/协议因素。

## 本次实现接缝错误与修复

新分析第一轮实际测试9通过1失败：错误假设原observed报告只含两个
分支。真实原报告包含全部16分支，而FD预先固定只测原selected/zero。
已修为：FD的两个branch身份必须相同，且都是原16分支的子集；
绑定原plan SHA、root/candidate和完整原goal/Q不变。修复后10通过。
这是本次新增分析器的schema接缝，不是历史P1控制损害的已证根因。

## 当前改动与下一因果对照

新增scalar检查以避免“全向量余弦高就认证目标优化”的错误推论；未
根据结果改参数或改gate。运行前协议已固定在
`docs/D2P_RAW_COORDINATE_CAUSAL_PROTOCOL_MAIN_20261028_v1.md`，
下一步以当前实际cp5的原suffix为锚点，逐raw坐标对齐模型与真实响应。
此处旧报告由原神经报告绑定物理/observed SHA及plan SHA，**未在本工具
重新执行完整上游consumer或CUDA**，结果如实标记full_upstream_rebuilt_here=false。
研究尚未闭环，P1/P2/P3科学准入仍false。
