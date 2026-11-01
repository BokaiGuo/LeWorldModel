# Constructive blind forward-count matched sampling v1

独占新增本 stem 源码/tests/doc；旧 sampler 与所有 sealed 输入不改。MAIN 独占实际 prepare/run/consume/CUDA。作者只执行纯 NumPy/RAM mocks，不加载模型、训练或运行物理。

## 固定协议

沿用 `p1_constructive_sampling_cuda_20261027_v1` 的两个已知 calibration blind task（source_slot 0/1）、原 frozen checkpoint、native projected CLS192、identity192 Q、goal、完整三状态/两动作 history、normalizer 和 cap=.75sqrt5 的逐 packet logit 球。仅从固定两份 blind export 读取任务；不读取 witness/sourcepack/native outcome，不重新选根。

固定 source_slot 升序，每根 `random_shooting` 然后 `CEM`。种子仍为 `20261027 + source_slot*101 + arm_index`，random=0/CEM=1，NumPy default_rng/PCG64。新 sampler 不增加独立样本，和旧采样共享随机流是预声明配对，不是新独立确认。

- random：32 个 iid standard Gaussian `[2,10]` logits，使用 sealed screen.projected_logits 的 packet 球投影/float32 inward rounding；不注入零候选、incumbent 或旧 winner。
- CEM：2轮×16，初始 mean0/std1。每轮全部16个模型 fullH2 cost 返回后稳定 top4，使用投影后 logits 的 mean/std(ddof0)，std clip[.05,1]，不平滑、不注入。第2次 update 仍保存但不再产生候选。
- 每候选执行完整 H2=两包×5 raw actions；raw 是 float32 NumPy tanh(logits).reshape(10,2)，不额外 /100 或 clip。模型输入原 normalizer 与完整 rolling history。
- 在全部32槽按 `(cost,index)` 稳定全局 argmin。先 durable 封存 winner，再从原 task root 独立重新执行一次 selected fullH2，严格校验 commands、terminal、cost 与搜索一致。不是 nominal/whole/segmented 三重 replay；final_prediction 不带伪造的 whole_replay/segmented_replay 字段。

每 arm/root：32×2+1×2=**66 predict 和66 action_encoder**。固定四臂总计各264；候选128个、selected replay4个、总132个 H2 evaluation。新 AD/VJP/J/SVD/solve/encoder/projector/physics 均0，历史 blind 编码单独计账。仅 forward 调用次数与 D2P 原66预算匹配；D2P 的 AD/J/SVD、执行时间/内存不等价，**不是全算力匹配**。

## 封存、资源和失败

独立 NAME/schema/prepare/report，复用封存 base Store、NativeCuda guard 和 GN single fullH2 primitive；不修改旧模块全局量。build 绑定当前新源码/tests/doc、既有源闭包、两份 export 与模型 SHA；verify 重建整份 prepare。无 CPU neural fallback，MATH SDP，TF32关闭，frozen CUDA model 前后 tensor hash一致才释放。

predispatch 66 cap，逐实际模块 attempt/return/synchronize 计账；每包完整 context durable，候选 bank 在执行前封存，CEM update 在整轮之后，winner 在全部32 cost之后且重新评估之前。consume 重建随机流、每个候选 raw/ownQ、全H2 history、CEM update、选择顺序、真实33次H2与132个模块返回、完整产物 SHA/gzip footer。plan schema 与原 sampler 同类键，但使用本 NAME；consume 返回 `{schema,rows,P1_admission:false,new_physics:0}`，rows 有 source_slot/arm/commands/task_export/task_sha256/blind_sha256 等，供新实验显式接线，不冒充旧 seal。

输出硬 cap8MiB（含 partial），200MiB floor，256KiB终态预留；启动需要至少208MiB。prepare仅小 hashmanifest（<256KiB），不复制大模型或报告。异常保留首错/secondary、未开始全分母和已完成 candidate，不替补、不重试、不依据 outcome 调整。完整 mock 的体积不保证实模压缩率；实际超限必须失败，不能放宽 gate。

## MAIN 命令

工作目录 `/home/gbk/code/LeWorldModel/le-wm-main`，以下输出必须不存在；作者不执行实际 CUDA。

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m experiments.spectral_reachability.p1_constructive_matched_sampling_cuda_20261027_v1 prepare --output /home/gbk/code/LeWorldModel/le-wm-main/results/p1_constructive_matched_sampling_cuda_20261027_v1_prepare_main01.json
```

MAIN 自行取得 prepare 外部 SHA，设为 `$MAIN_PREPARE_SHA256`：

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m experiments.spectral_reachability.p1_constructive_matched_sampling_cuda_20261027_v1 verify --prepared /home/gbk/code/LeWorldModel/le-wm-main/results/p1_constructive_matched_sampling_cuda_20261027_v1_prepare_main01.json --prepare-sha "$MAIN_PREPARE_SHA256"
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m experiments.spectral_reachability.p1_constructive_matched_sampling_cuda_20261027_v1 run --prepared /home/gbk/code/LeWorldModel/le-wm-main/results/p1_constructive_matched_sampling_cuda_20261027_v1_prepare_main01.json --prepare-sha "$MAIN_PREPARE_SHA256" --output /home/gbk/code/LeWorldModel/le-wm-main/results/p1_constructive_matched_sampling_cuda_20261027_v1_cuda_main01
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m experiments.spectral_reachability.p1_constructive_matched_sampling_cuda_20261027_v1 consume --output /home/gbk/code/LeWorldModel/le-wm-main/results/p1_constructive_matched_sampling_cuda_20261027_v1_cuda_main01 --report-sha "$MAIN_REPORT_SHA256"
```

## 非神经验证与结论边界

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m pytest -q -p no:cacheprovider tests/test_p1_constructive_matched_sampling_cuda_20261027_v1.py
```

测试用192维数组替身、RAM产物及既有 export 只读校验；无大 tmp。覆盖四分支66预算、候选/全H2时序、CEM因果/seed/投影、selected replay、内存端到端 consume、篡改/异常/超预算拒绝。mock不是实际 CUDA 效果，也不是任务/机制准入；P1_admission 与 new_independent_confirmation 恒false。

experiment-code skill 用于预声明实验、固定种子、原始日志及失败测试。用户禁止本轮模型/物理实验，因此不执行 skill 通用训练/画图/准确率模板，不以负结果改方法。skill 所列 ~/.claude 辅助 references 本机不存在，采用已完整读取的主 SKILL 与现有封存接口。

交付验证：新测试52 passed；旧 sampler 回归40 passed（此前联合89 passed，随后仅增加3项新journal故障测试，再跑新52 passed）。全部数组/RAM，无模型/物理。只读 build 对真实固定两份 export 通过，110个 bindings，prepare预计21,561 bytes；本轮未创建prepare或实际结果。旧源码/测试/doc SHA前后不变。真实CUDA预算与效果必须由MAIN实际运行另行验证；本交付不声称已取得实际收益或P1准入。

goal状态：get_goal 返回不兼容的旧offline-fidelity active goal。依用户条件未创建新goal，也未将已停止的旧审查虚报complete；当前交付不是旧goal完成证明。
