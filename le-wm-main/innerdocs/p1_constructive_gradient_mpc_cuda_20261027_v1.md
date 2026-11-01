# Frozen gradient MPC baseline（MAIN-only CUDA）

本模块是完整 H2 open-loop differentiable MPC **求解器**，不是反馈 MPC 控制器。仅使用已批准两份 blind latent task exports，与既有 sampling/GN 相同原模型、独立 goal、raw normalizer、identity192 meanQ 和两 packet logit 球。无 witness/sourcepack/certificate/物理 outcome 输入，不训练参数/readout/Q，不运行 CPU 神经。实现者只运行纯 NumPy/RAM tests；实际 CUDA/physics 仍由 MAIN 独占。

## 预声明算法

固定单个初始化 `x0=zeros(2,10)`、32次更新、步长 `eta=1.0`；不按已开 outcome 或旧 planner 分数选参数。不随机重启、不自适应步长、不早停、无线搜索，不按是否改善跳过任何计算。

```text
evaluate x0 using complete H2 native chronology
for t = 0,...,31:
    fresh differentiable H2 forward at EXACT evaluated raw=tanh(float32(xt))
    L = mean192((terminal-goal)^2)
    g_raw = scalar autograd.grad(L, raw10x2), one reverse sweep
    gt = reshape(g_raw * (1-raw^2), 2x10)
    x(t+1) = project_each_packet_ball(xt - 1.0*gt)
    evaluate x(t+1) using complete H2 (even if worse or identical)
select stable minimum(cost,iteration) over all33 evaluated iterates
fresh nominal / whole / segmented fullH2 replay of selected iterate
```

两包球半径沿用 `.75*sqrt(5)`，投影沿用冻结 `projected_logits` 的 float32 向内舍入。不加 ridge/action penalty，不归一化梯度、不乘192。meanQ loss 的真实标量反向通过两包 latent **及 action history**；不能 detach 第一包或重编码。native loss/梯度为 float32，保存模型选择代价用原 float64 meanQ；保存两种 loss，不要求其浮点值完全相等。

raw 叶子在已经 NumPy float32 tanh 的**精确前向动作点**求梯度，然后只乘一次解析 logit tanh chain；与原 GN raw→logit 接口一致，不是用 torch tanh 悄悄换前向动作。`NativeGradient` 新实现标量反向，不调用旧 `NativeCuda.jacobian`、GN optimize 或 sampler。

## 计算账本

每根：初值 fullH2 2 predict；32次更新各新 AD-forward2 + 新 proposal wholeH2 2；最后三路径6，总136predict+136actionencoder，**其中64已是AD-forward，不额外加64**。另32次 scalar backward（`autograd.grad` API），32次投影 proposal、1次末尾模型 selection。两根272/272 +64 backward +64 proposal +2selection；0新192行 Jacobian、SVD/solve、图像 encoder/projector、physics/training。

`backward` meter与 `AD_API.grad` 是同一批32次反向的两类 receipt，不能相加成64，也不能把一次标量反向当192个 reverse directions。每个 entry 前 nonrefundable attempt，CUDA synchronize 成功后 completed；意外同步/模型/读写错误保留首错和 partial，不重试、不释放物理 commands。额外编码/采集是历史独立成本，不等于总 pipeline 免费。**不宣称与 GN/D2P/sampling forward 匹配，更不宣称全资源公平**。

## 来源、durability、消费出口

输入固定：`task_00.json.gz` SHA `714f471761278fa654542d4bb0878ce18e3c304df6158005af46d2129533265a`；`task_01.json.gz` SHA `70865d56f9b66f0072b2ad7de01cc09d9a9275047925e188104a13d5182ce1f2`，均位于 `results/p1_constructive_blind_encode_cuda_20261027_v1_cuda_main01/`。继承原 approved blind/task/normalizer/with_target=False/model identity 检查，只读 exports、本地权重配置与源码闭包，不调用 encoding producer、不读原 blind NPZ。

完整 source seal 包含本源/tests/doc、冻结 native/wholeH2 replay/interface 与原模型配置/权重；prepare及收尾均复验。冻结模型参数/ buffers 的运行前后 tensor hash 必须一致；模型未加载时 unchanged=None。数值合同固定 SDP MATH、两个 TF32 flags=false。新 image encoder/projector入口禁止；ADGate只允许 metered scalar backward 里的32次 grad，禁止 fullJac/higher-order/tensor.backward 等其他入口；不会改旧模块全局参数。

每个 native packet（三状态/两动作完整内容）、AD-forward 两包、原始 raw gradient20、logit gradient20、tanh chain、全部33 iterate/logits/commands/cost/终端、所有 attempt/return 都 durable gzip Z_SYNC_FLUSH+fsync。新 forward 在 anchor 验证与 backward 前即时保存，失败不能丢失前向证据。选优后独立三路径真实重算，不复制 endpoint。

预算/存储继承8MiB整包 cap，200MiB reserve，256KiB final reserve，prepare<256KiB。mock完整两根stream/results/plans/report只在内存压缩并读回，实际容量由 MAIN receipt 决定；不删梯度或失败数据凑预算。

`consume --report-sha` 是本 producer 的 saved 物理释放 gate，不生成独立审计器：绑定完整外部report/prepare/source/artifact hash、精确两根 own arm，重建固定更新/模型选优、逐包 history内容/布局、实际梯度 forward、所有计数与三重放，输出带 source_slot/arm/blind_sha256/commands/selection_sha256/whole_run_report_sha256 的独立 ownplans。只能接新物理执行，不复用旧 endpoint。

## MAIN 命令

cwd `/home/gbk/code/LeWorldModel/le-wm-main`，实现方不抢 prepare：

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m pytest -q -p no:cacheprovider tests/test_p1_constructive_gradient_mpc_cuda_20261027_v1.py
.venv/bin/python -B -m experiments.spectral_reachability.p1_constructive_gradient_mpc_cuda_20261027_v1 prepare --output results/p1_constructive_gradient_mpc_cuda_20261027_v1_prepare01.json
.venv/bin/python -B -m experiments.spectral_reachability.p1_constructive_gradient_mpc_cuda_20261027_v1 verify --prepared results/p1_constructive_gradient_mpc_cuda_20261027_v1_prepare01.json --prepare-sha <MAIN_PREPARE_SHA>
.venv/bin/python -B -m experiments.spectral_reachability.p1_constructive_gradient_mpc_cuda_20261027_v1 run --prepared results/p1_constructive_gradient_mpc_cuda_20261027_v1_prepare01.json --prepare-sha <MAIN_PREPARE_SHA> --output results/p1_constructive_gradient_mpc_cuda_20261027_v1_cuda_main01
.venv/bin/python -B -m experiments.spectral_reachability.p1_constructive_gradient_mpc_cuda_20261027_v1 consume --output results/p1_constructive_gradient_mpc_cuda_20261027_v1_cuda_main01 --report-sha <MAIN_ACTUAL_REPORT_SHA>
```

Mock只验证工程与纯数组递推，不能充当真实 backward、CUDA同步或科学收益证据。实际运行亦仅两已知calibration任务，不新增独立根/物理成功/P1准入。
