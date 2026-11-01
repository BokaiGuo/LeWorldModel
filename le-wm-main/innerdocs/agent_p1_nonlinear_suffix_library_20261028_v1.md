# 固定42 prefix库：非线性 suffix 动作梯度对照

独占 source：`experiments/spectral_reachability/agent_p1_nonlinear_suffix_library_20261028_v1.py`；tests：`tests/test_agent_p1_nonlinear_suffix_library_20261028_v1.py`。不修改任何旧模型、实验源码或结果。仅 MAIN 审读后主机 CUDA 执行，本 worker 不执行神经/物理/网络。

## 输入 authority 与新干预

固定 `results/agent_allmode_prefix_20261028_v2_cuda_MAIN01/report.json`，SHA `c3e38899a12cd163d75918da79ef029adb14b2479f74f15bdfac0b6ffc57242d`。prepare 真实调用原 `source.consume/verify`，并重新验证四槽全部42候选的 bank、ridge、原动作/上下文与原目标成本。保存原 prepare/bindings、所有原数组与事件 artifact SHA，以及新三件套和复用模块 SHA。每次 verify 完整重建闭包；成本不冒充轻量 receipt reuse。此工程开销计入 run 的1800s whole-stage wall（边界 watchdog，不是进程抢占器）。

known2 roots × off/on 共4槽，42 prefix/槽，总168条；不是168独立样本。全库包括零/重复/null方向，绝不删候选或按效果筛根。每条原 literal rawfloat32 prefix5 逐位不变，在原 root 上重新预测 prefix 得自己的 imagined mid；不使用 teacher/真实反馈中间状态。原 ridge suffix 为 iterate0。只优化 suffix，固定8次 scalar梯度、eta1、原 packet logit cap，保留 incumbent 与全部8 iterate，稳定按 `(nonlinear terminal meanQ, iterate index)` 选择，再按 `(selected terminal meanQ,prefix index)` 全42选择。每条独立 wholeH2 replay，最终赢家另做 whole/segmented 双重 replay。

与原 algorithm 的变化明确分栏：原 winner 使用 stage+ridge reach、beta1；新算法只使用 terminal nonlinear Q，全库重选。原 generation winner 身份保留，新策略不能冒称旧策略原本采用 terminal selector。此对照同时改变 suffix优化及最终评分选择，不能仅凭新旧最终winner差异把收益唯一归因于ridge；每prefix全iterate及成本可支持分别描述优化效应与选择效应。不调 beta、不训练、不拟合Q、不读物理outcomes。

## cap 与舍入域（不修补历史合法动作）

原 solver 在 float64 ball 中求解，部署时 `np.tanh(np.asarray(params,np.float32))`。MAIN 实测168个 incumbent 中81个 postcast float64 norm 略超cap（最大约5.4457e-8），这是原合法solver-cast见证，不能新加 postcast cap gate 拒绝。prepare 对 incumbent 验证原 `bounded_ridge` 完整算术重构和 float64 `norm<=cap`、原 solver norm/params/radius，再核原 cast/chart rawbits；不做 inverse-tanh、不投影iterate0、不加epsilon。保存 solver norm 与 postcast norm，后者仅诊断。只读真实168回归要求81个边界见证仍通过。

新 iterate 使用已审 `gd.screen.projected_logits`：float64更新/球投影、cast float32、必要时既定 inward rounding；严格核新 iterate float32 norm<=cap。这与历史 incumbent 的合法域认证方式不同，明确记录，不偷换原动作。模型调用消费 literal float32 raw，禁止执行处二次tanh。梯度针对 fresh suffix raw[5,2]、原native scalar meanQ，随后解析 `(1-raw²)` 链一次；不是把NumPy float32 chart当逐位可微程序。scalar codec会将0维提升为blob[1]，运行处仍强制loss.ndim==0，saved严格[1]，raw/gradient必须[5,2]。

## 新增固定费用及双账

每条 prefix重放1 + 9 suffix评分 + 8梯度前向 + 选中fullH2独立重放2 =20 predict；每槽最终 whole/segmented再4。总新增3376 predict、1344 scalar reverse。每predict记录 action_encoder、predict API、predictor、pred_proj原生输入输出与资源 reserve/entry/return；每reverse记录scalar grad/VJP/raw callback，三者是同一次reverse的不同计量单位，不能相加。新J/SVD/视觉encoder/projector/physics为0；pred_proj不是视觉projector，正常逐forward计费。原allmode1032predict/172J/33024VJP/176成功helper SVD历史费用另列。FLOPs/backward engine calls未知为null，不称等算力或训练。

run 固定CUDA、冻结model参数与全部buffers、TF32/autocast off、MATH SDP；实例级native module观察器，不patch globals。source/weights/model类闭包先后绑定；run前后重hash所有bindings。资源ledger保留逐实际attempt/return，失败不可退款；168条行及已用账保留，不重试补候选、不把partial变完整。输出512MiB（64MiB final reserve），磁盘floor200MiB，CUDA allocated3GiB/reserved4GiB，whole-stage1800s。

## 严格消费

验全artifact census、原literal输入、全9/8递推及稳定选择、normalizer、完整native三行预测与last-successor滚动、每一步AE→predictor→pred_proj张量、反向输入/梯度/raw callback、完整资源账及journal hashchain。完整事件生命周期约束：每forward start→packet/resource/tensors→end；该phase的reverse只能在自身end后、下一forward start前；selection只能在全部iterate返回后且独立replay前。未知事件/字段拒绝，不filter。事件移位与数据篡改均有RAM回归。receipt有完整4plans仅表示工程模型计划验证；不授权physics，P1/freshness/science始终false。

RAM fixtures为NumPy线性接口替身，不执行torch或环境。测试包括全168实际dispatcher3376/1344计数、单候选完整native strict、真实文件/chain partial consume、失败保分母、真实saved81边界、源码/绑定漂移、event跨phase splice。RAM结果不是模型或物理科学证据；生产prepare/CUDA/strict由MAIN执行。

## MAIN 执行入口

在 `/home/gbk/code/LeWorldModel/le-wm-main`。冻结三件套之后才prepare，之后不原地改。两个SHA占位只用实际输出替换，不预造。

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m pytest -q tests/test_agent_p1_nonlinear_suffix_library_20261028_v1.py
.venv/bin/python -m experiments.spectral_reachability.agent_p1_nonlinear_suffix_library_20261028_v1 prepare --output results/agent_p1_nonlinear_suffix_library_20261028_v1_prepare_MAIN01.json
.venv/bin/python -m experiments.spectral_reachability.agent_p1_nonlinear_suffix_library_20261028_v1 verify --prepared results/agent_p1_nonlinear_suffix_library_20261028_v1_prepare_MAIN01.json --prepare-sha ACTUAL_PREPARE_SHA
.venv/bin/python -m experiments.spectral_reachability.agent_p1_nonlinear_suffix_library_20261028_v1 run --prepared results/agent_p1_nonlinear_suffix_library_20261028_v1_prepare_MAIN01.json --prepare-sha ACTUAL_PREPARE_SHA --output results/agent_p1_nonlinear_suffix_library_20261028_v1_cuda_MAIN01 --authorization MAIN_FIXED42_NONLINEAR_SUFFIX_HOST_CUDA_ONLY
.venv/bin/python -m experiments.spectral_reachability.agent_p1_nonlinear_suffix_library_20261028_v1 consume --output results/agent_p1_nonlinear_suffix_library_20261028_v1_cuda_MAIN01 --report-sha ACTUAL_REPORT_SHA --receipt results/agent_p1_nonlinear_suffix_library_20261028_v1_consume_MAIN01.json
```

未运行：生产prepare、CUDA、physics、新策略任务收益。后续必须与原相同prefix/goal/raw actions合法配对评估，不能从预测Q改善声称物理改善、prefix库充分或P1通过。
