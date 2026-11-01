# P3 固定 H3 / 144+3 调度校准 v2

本版为可执行 scheduler + CUDA runner + 非神经测试 + prepare，非纯文档。
只增加新文件，旧 sealed 文件不改；本轮仅 prepare，不加载模型、不启动
GPU、physics、神经 CPU、下载或训练。主线程审查后才可显式选择 CUDA 入口。

## 已冻结的最小对照

每个既有 calibration root 两臂，uniform / dynamic；H=3 tokens=15 raw steps。
同一初始 context、goal0、Q、完整16项 raw bank、checkpoint、cached-latent
open-loop feedback 合同。raw bank 沿用旧 runtime 的精确定义：每条编译receipt的
`program_live_commands[0]` prefix command 重复五步；第二条 compiler suffix
不能混入。原数组、参数、命令不裁剪、不 padding、不拟合。

H1完整评估16项。随后扩展 K2 个H1父节点，每父节点全16项；再从H2选择K3
个父节点，每父节点全16项。父节点按 frozen goal-projected model cost 升序、
路径字典序破同分。只能在完整 H3 候选中选 winner，禁止 shallow incumbent。

uniform 固定 K2=K3=4。dynamic 固定为：

`K2 = 2 + floor(4*chi + 0.5); K3 = 8 - K2`

不读旧 P3/physical outcomes，不根据四根结果改阈值或公式。两臂各自重新计算
同一个H1信号；uniform计算但忽略。两臂H1信号不一致则配对 unmatched，
不选一个信号替代另一个。规则可能输出4/4，与uniform相同时如实报告。

## χ 的确切目标几何来源

`x0 = compiled.input_context.emb[0][-1]`；
`g0 = compiled.goal_vectors[0]`；
`Q = compiled.scoring.metric.projection`，本批192维、由既有有限goal目录定义。
H1的16个 `Y[i]` 必须是本次同root完整rawbank预测的真实latent终点，不能读取
编译receipt中旧 `predicted_terminal_vector` 或旧Jacobian充当新信号。

`r=(g0-x0) Q^T`，`D[i]=(Y[i]-x0) Q^T`。
对 D 做一次 `svd(full_matrices=False)`，相对 cutoff 固定1e-6。
`chi=||r-Proj_rowspan(D)(r)||² / ||r||²`，只为数值舍入clip到[0,1]。
这是原 runtime 的 uncentered finite-response residual fraction：包含自主漂移；
不是只看奇异值、不是action Jacobian、不是已识别可控性，也不是物理任务几何。
报告保存 r、D、奇异值、rank；goal/Q来源和完整H1 contexts均由protocol/journal绑定。

prepare验证goal/Q/context维数、有限性、Q对称/幂等（绝对容差1e-8）、非零残差，
但不构造新H1预测或声称χ已可用。运行时H1缺项、失败或响应rank=0则不可调度，
uniform也不fallback；保留失败层、全部尝试，不从成功子集计算χ。

## 精确预算与成本坐标

正常完成每root-arm：`16+16*K2+16*K3=144` planning samples，
另从初始context整条15步raw路径重放3个predict samples做verification。
一个PredictionBudget贯穿两phase，cap147；planning显式止于144。
CUDA batch=1，无跨arm共享预测缓存；每个已尝试失败predict都保留计费。
普通edge失败仍保留该层已预定的全部尝试，但禁止下一层或verification。

四根×两臂：1176 predict samples、1176 action encoder samples、3528 processed
context tokens（T=3）、8次非神经SVD。三者独立坐标，不相加为等价FLOPs。
本次cached-latent流程视觉encoder应为0，误调用由hook记录并拒绝完整性通过。
runtime backward/VJP为frozen no-grad代码策略的0，不冒称已安装autograd计数器；
dense Jacobian记unknown，旧上游编译成本不偷换为0。加载、编译、source hashing
不在每arm向量内，不声明total-pipeline成本相等或walltime优势。

## 运行方式

cwd `/home/gbk/code/LeWorldModel/le-wm-main`。

只读源校验 + 新exclusive prepare（本轮执行的分支）：

```bash
.venv/bin/python -m experiments.spectral_reachability.p3_fixed_horizon_budget_schedule_cuda_20261027_v2 --prepare --output /home/gbk/code/LeWorldModel/le-wm-main/results/p3_fixed_horizon_budget_schedule_20261027_v2_prepared
```

prepare导入链无torch/simulator，读取已绑定编译输入与原协议源清单，不加载权重。
生成 protocol.json（含输入packets及hash）、source_seal、prepare.json。
外部checkpoint/config仅绑定上游hash，本轮不重哈希大权重；CUDA启动前必须重验。
source seal同时归档当前源码及tests/doc，执行前后检查源/归档/args和输入字节。
协议固定四根、顺序、交替arm顺序、formula、geometry、cost scope；不可原地重封存。

主线程review后才能执行以下显式分支，`REVIEWED_PREPARE_SHA256`必须取交付的
prepare.json SHA256并独立核对；此文不代表GPU授权：

```bash
.venv/bin/python -m experiments.spectral_reachability.p3_fixed_horizon_budget_schedule_cuda_20261027_v2 --execute-cuda --prepared /home/gbk/code/LeWorldModel/le-wm-main/results/p3_fixed_horizon_budget_schedule_20261027_v2_prepared --prepare-sha256 REVIEWED_PREPARE_SHA256 --output /home/gbk/code/LeWorldModel/le-wm-main/results/p3_fixed_horizon_budget_schedule_20261027_v2_cuda
```

仅CUDA分支延迟导入torch、验证完整checkpoint字节、加载frozen eval模型并调用现有
CudaReplay，绝无CPU神经fallback。每条edge的attempt/result journal及时flush/fsync，
每arm报告保存predict phase及encoder counters、H1 signal、allocation和H3验证。
全局失败也补齐八个预定arm槽位；中断的未知工作明确unknown，不能捏造0计费。

非神经回归：

```bash
.venv/bin/python -m pytest -q tests/test_p3_fixed_horizon_budget_schedule_20261027_v2.py tests/test_p3_fixed_horizon_budget_schedule_cuda_20261027_v2.py
```

## 科学边界与分母

此前frontier审计为4个既有root：按root 1 full-path / 1 guard-partial / 2 unmatched；
按两次依赖replay记录为2 / 2 / 4。不是8个独立根，新增独立根始终0。
本版预定4 roots×2 arms，没有物理replay，不移用任何旧物理终态到新路径。
本轮prepare的真实新predict=0、physical steps=0；非神经fixtures只证实现合同。

未来CUDA完整性通过也只提供固定H3/同预算的模型内调度校准。模型cost定义为
`mean(((x_H3-g0)Q.T)^2)`；P1尚未通过，不能把它叫已验证物理quality。
只一个预算点，不是budget sweep/frontier曲线；无独立held-out根、P3不准入。
