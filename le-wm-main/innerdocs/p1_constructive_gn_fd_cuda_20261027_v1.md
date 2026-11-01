# Constructive GN first-zero full-H2 finite differences v1

2026-10-27。独立新协议；不改旧封存。MAIN独占真实HOST CUDA执行。本文件先于新FD结果冻结；不取物理outcomes、不改Q、不选科学阈值、不追认P1通过。

## 数据与固定计算

唯一实际planner输入：`results/p1_constructive_blind_planners_cuda_20261027_v1_cuda_main01/report.json`，SHA256=`4dd5823e9a273b820d6b7b3f6a36a4a5a19b193383eaa9cbf94442082e79d98d`；其prepare SHA=`30b58155a59f73fdc8cc132ff069e645ed6c2bbff9434f474b1f7b9c412f7eb4`。原planner source=`e56bee610f4eb412cc03fbe49b3df91054d0defe23d24b6fd9c77e6a7f1e9f27`。

只用两个已批准blind latent task的GN `iter_1_AD`：均为2×10零logits。由原始192行raw-control J、joint logit J、AD-forward terminal、两packet contexts、initial三次路径互相绑定；零点tanh chain=1。稳定身份是有hash的opaque task_id及明确task export顺序，不伪造native root hash。不读sourcepack、certificate、blind NPZ或物理结果。会解码整个已封存planner journal以取首J，但后续迭代、D2P或选优损失不参与点/方向/eps选择；这些实际artifact仅hash绑定。

每点方向固定：packet0 e0（flat0），packet1 e0（flat10），固定20符号单位向量（源码`directions()`继承旧FD的明确符号表，非RNG/梯度/谱方向）。每方向epsilon固定`1e-3,3e-3,1e-2`，按minus/plus central FD；先float64零点±eps·v，cast float32一次，NumPy float32 tanh一次，不球投影、不重中心化。所有探针在原perpacket `.75*sqrt(5)` 球内部，raw为原[-1,1]。identity192 Q/goal/原normalizer/动作尺度不变。

每次独立从原完整3states/2actions开始，packet1沿packet0新latent与action history，不能用保存prefix替代新forward，也无prefix detach/recompile。复用封存`NativeCuda.rollout(...,'whole_replay')`；不调用旧FD的`collect/material/verify/run`，旧固定任务不可进入此新runner。

每root：nominal1 + 3方向×3eps×2sign=19 H2=38predict/actionencoder。两root共38H2=76predict+76actionencoder；18central comparisons。0新AD/J/VJP/SVD/solve/selection/encoder/projector/physics。原GN156predict含16ADforward、8J/1536数学VJP另列；原D2P132predict/22J/4224数学VJP及先前图像采集/编码成本不抹除，也不乘成新FD成本。prepare/verify仅hash/保存数组检查及NumPy Jv，无新ridge solve。

## numerical与报告

固定同原CUDA模型及tensor SHA，SDP MATH、matmul TF32=false、cuDNN TF32=false；禁下载、CPU fallback、训练、新AD API及encoder/projector。每root fresh nominal必须与保存AD terminal和两context SHA逐字一致，否则先保存该次raw结果再停止，不自动retry，不把不一致当作成功。完整nonlinear sides都持久化，保存FD、saved Jv、J乘实际float32 secant direction、requested/effective误差、midpoint drift、相邻eps差；零分母为null，不拟合floor。导数科学结论`not_assigned`，threshold=null；负结果照存，不据结果调eps。数值差异不是自动科学准入/拒绝门。

attempt先写，返回并CUDA sync后才completed；全局predispatch每module76上限，禁失败重试。gzip journal每事件Z_SYNC_FLUSH+fsync；独占NEW路径，all-output cap2MiB（含partial）+200MiB floor，留128KiB终结空间。源、模型终结分别try/catch，保留首错与secondary；I/O彻底失败不保证final report可写，但不覆盖旧证据。只有38完整rows、76/76实际完成、全部source重验与model unchanged才能status=cuda_fd_completed；该状态不是FD精度pass或P1准入。

## MAIN命令（本agent不prepare、不CUDA）

cwd=`/home/gbk/code/LeWorldModel/le-wm-main`。

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q -p no:cacheprovider tests/test_p1_constructive_gn_fd_cuda_20261027_v1.py
sha256sum experiments/spectral_reachability/p1_constructive_gn_fd_cuda_20261027_v1.py tests/test_p1_constructive_gn_fd_cuda_20261027_v1.py innerdocs/p1_constructive_gn_fd_cuda_20261027_v1.md
.venv/bin/python -m experiments.spectral_reachability.p1_constructive_gn_fd_cuda_20261027_v1 prepare --output results/p1_constructive_gn_fd_cuda_20261027_v1_prepare01.json
.venv/bin/python -m experiments.spectral_reachability.p1_constructive_gn_fd_cuda_20261027_v1 verify --prepared results/p1_constructive_gn_fd_cuda_20261027_v1_prepare01.json --prepare-sha PREPARE_SHA
.venv/bin/python -m experiments.spectral_reachability.p1_constructive_gn_fd_cuda_20261027_v1 run --prepared results/p1_constructive_gn_fd_cuda_20261027_v1_prepare01.json --prepare-sha PREPARE_SHA --output results/p1_constructive_gn_fd_cuda_20261027_v1_cuda_main01
```

`PREPARE_SHA`取prepare真实返回，不能沿用旧值；已存在路径不可重用。mock仅验证工程/算术（RAM recurrence、wrong-J负例、精确anchor、失败非退款、startup终结、cap/floor）；实际saved RAM collect仅验证2points来源接线，不是模型forward验证。真实76predict和导数数值待MAIN执行。无独立witness/阈值/科学准入新增。
