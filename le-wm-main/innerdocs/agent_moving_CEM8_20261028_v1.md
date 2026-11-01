# 固定八个 teacher_full anchors 的 CEM64 后缀实现

有界实现交付，不是新根确认、完整闭环 CEM-MPC 或官方论文复现。P1/P3=false。
MAIN 执行 CUDA/物理；worker 仅 RAM/mock 与 saved metadata 检查，无神经调用。
使用 experiment-code 技能；按用户范围不训练、不画实验结果、不以零成功改参。
旧四 observed-prefix 的 next 模块与所有旧产物不变，本模块不把它们混入八项分母。

## 输入与身份

固定父 prepare：`results/capability_suffix_factorial_cuda_MAIN_20261027_v2_prepare_MAIN01.json`，
SHA `330412c675844d7edff23ade45012cf4ef4920f64bd47a31a0c49e6872f673f5`。
父真实 CUDA report SHA `fc620c9fdcbb5046c1d076c79cd0703da999275ec921de799f3747c0204d69ab`；
MAIN strict receipt `results/consume_suffix_factorial_MAIN_20261027_v2_MAIN01.json`，
SHA `3b17d6b6a7323545d0fb81034dac5e0c1d479b5a76de307bb2ca1a1ae1d8ad58`。
build 调用父 verify 与 strict saved consumer，重新验证完整模块/raw/normalization/绑定后，
要求返回值与真实 receipt 完全相等。不导入/执行父 CUDA run。

固定次序 `slot(0,1) × condition(off,on) × prefix_arm(line_search_GN,full10_conditional_D2P)`。
每个 ordinal 取父 case 的 teacher_full，保留原 task_id、selection_digest、blind_sha、
原 prefix5、同 condition goal192、teacher latent3 与 action_history 两包。
纯 float32 从 warmup 最后5步及该 prefix5 重算两包 token，必须逐位一致；
latent 前两帧必须等于原 task 的 history[1:]。不走 zero-only interface。
目标 identity192 Q 固定，不拟合。所有8项都进入分母，无按评分/成功筛选。

用户提供的 factorial 物理 report SHA
`cc92465141b400659e024e052b05683153b4dc6c013b81bdf1e3e29355fd2663`
仅为后续比较来源说明；本 planner **不读其标签、成功与评分**，不将其作为选根/选动作输入。
父6162条 bindings 由父 verify 传递验证，自己 prepare 不拷贝上游报告/数组/绑定大表；
直接 seal 父 prepare、report、strict receipt、原 task 和自己执行 closure。

## 事前算法与预算

沿用冻结 next 源 SHA `bc8e588bc11a4ab50572ace8a99ef397f05f0ee8b6edfab5ae5a42f022528e10`
的 CEM64 数学，不读其本次 CUDA 结果。不调参：4轮×16候选，top4，初始 mean0/std1，
float64 PCG64 Gaussian，逐10维向量投影到 `.75*sqrt(5)` logit ball。
对投影后的 elite 求 mean、总体 std(ddof0)，clip std 到 [.05,1]；无 smoothing、incumbent、padding。
投影改变 proposal，未作 exact density correction，不冒称原始无约束 CEM。

新 seed 固定 `20261028 + 800000 + 104729*ordinal`，ordinal=0..7，
独立于旧4-prefix规则，不含 outcome/cost。保存全部请求/投影 bank、64候选完整 context/cost、elite更新。
以全部64成本稳定 argmin（相同成本取最早 index），封存选择后仅独立 replay 一次，
MATH 下完整 context/cost 必须精确相等；失败不改选、不重试。

先在 NumPy 对 projected float32 logits **只做一次 tanh**，产出 literal float32 raw5。
JSON float 精确承载这些 float32 数，TorchBackend.step 仅 tensor(raw) 后进入原 make_transition，
不再 tanh、不 /100、不 clip。模型与 model_plan/physical dispatch 共用这些数。
saved consumer 可重新计算 proposal 以校验，但不重算 CUDA raw 作为替代执行权威。
每次 forward 保留 action_encoder/predict/pred_proj 全部原精度输入输出、attempt/return，
校验 normalized raw→AE→predictor→projector→实际返回 context；不只检查 terminal 收据。

| 新调用轴 | 每 anchor | 全8项 |
|---|---:|---:|
| predict API / predictor samples |65|520|
| action_encoder / pred_proj，各自 |65|520|
| Jacobian / VJP / SVD / solve |0|0|
| encoder / projector / physics |0|0|

上表为新规划执行轴。每次 build/verify 的父 strict consumer 会额外进行48次
**saved bank 纯数组 SVD复算**，单独记为 saved_validation_per_build，不冒充零算术开销；
prepare、pre/final verify、consume 的重建次数会重复付费，但不增加模型/Jacobian调用。

65=64候选+1独立winner replay，都是 H1 后缀，不冒充65完整H2候选。
父 teacher_full 每 anchor 是13predict+1J+192方向+2SVD；同 context/Q/action domain **不等总算力/FLOPs**。
共享历史 fixed-direction 40encoder/40projector/80predict；factorial 全24为312predict/24J/4608方向/48SVD，
teacher8子集104predict/8J/1536方向/16SVD已包含在全24，不可相加重复计费。
更早 source、overlay RGB、原 planner 与真实 physics 另行付费，未伪造完整历史总额为零。

模型全部 frozen eval/CUDA-only、torch.no_grad、SDP MATH、TF32=false；无 CPU neural fallback。
原权重 before/after tensor hash 与父 model hash 必须一致。
工程输出 cap32MiB（含12MiB final reserve），disk floor200MiB；prepare cap2MiB。
八项全模块数组与候选/最终report较旧四项翻倍，因此明确新工程上限，不省略数组来压cap。
实际文件大小需由 MAIN build/run核实；不启动模型测尺寸。
全分母 rows 在执行前建立；首错和 partial arrays/attempt 保留，未执行项不替补。

worker 已实际运行 saved-only `build()`，完整 parent strict consumer 通过，
重建全部8项，canonical prepare 约 **1.384MiB**（未写 prepare）；小于2MiB。
RAM 全8分支 consume 回归保留全部 module/candidate/journal，并核验合成总尺寸加
真实 AE embedding shape 差额仍小于32MiB，最终 report 小于12MiB。
这只是合成工程尺寸检查，不是实际 CUDA 输出大小保证；生产 cap/floor 会继续强制执行。

## 可消费接口与命令

own consume status=`teacher8_CEM_commands_verified_not_admitted`, errors=[]，plans固定8项。
每 plan 包含 root_slot/source_slot、condition、prefix_arm、anchor、task_id、blind_sha256、
原 selection digest、teacher context SHA、literal prefix_commands/suffix_commands/commands10、
自身 selection_sha256 与 whole_run_report_sha256。
`arm=teacher_full_CEM64`；下游必须按(root_slot,condition,prefix_arm) join，不能借旧endpoint或旧consumer白名单。
真实物理/RGB评价尚未执行，actual native success/latentQ 均未评。

以下从 `/home/gbk/code/LeWorldModel/le-wm-main` 由 MAIN 执行，路径须全新。
worker 不执行 prepare/run/物理。

```bash
OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m pytest -q -p no:cacheprovider tests/test_agent_moving_CEM8_20261028_v1.py
.venv/bin/python -B -m experiments.spectral_reachability.agent_moving_CEM8_20261028_v1 prepare --output results/agent_moving_CEM8_20261028_v1_prepare_MAIN01.json
.venv/bin/python -B -m experiments.spectral_reachability.agent_moving_CEM8_20261028_v1 verify --prepared results/agent_moving_CEM8_20261028_v1_prepare_MAIN01.json --prepare-sha ACTUAL_PREPARE_SHA
.venv/bin/python -B -m experiments.spectral_reachability.agent_moving_CEM8_20261028_v1 run --prepared results/agent_moving_CEM8_20261028_v1_prepare_MAIN01.json --prepare-sha ACTUAL_PREPARE_SHA --output results/agent_moving_CEM8_20261028_v1_cuda_MAIN01 --authorization MAIN_FIXED8_TEACHER_CEM64_CUDA_ONLY
.venv/bin/python -B -m experiments.spectral_reachability.agent_moving_CEM8_20261028_v1 consume --output results/agent_moving_CEM8_20261028_v1_cuda_MAIN01 --report-sha ACTUAL_NEW_REPORT_SHA
```

ACTUAL参数必须使用 MAIN 新执行返回的实际 SHA，不能拿父 SHA 替换。
RAM tests 使用纯 NumPy/device-labelled接口 stand-in，不实例化 torch 模型或模拟器。
交付 RAM 测试 **52 passed**；包含全8分支 saved consumer 与 CLI prepare/verify 内存回归，
另实际运行 CLI --help。未创建真实 prepare/result，未运行 CUDA/physics/下载。
即便真实运行全部通过，仍为2 known roots 上8 anchor 校准，freshness/P1/P3未获准入。
