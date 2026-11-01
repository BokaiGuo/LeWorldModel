# P4 known-root 双臂 live CUDA driver v1

独立新实现；旧 core/runtime/response/sourcepack/v1/v2 历史均不修改。
本文件与源码及 tests 一起封存。prepare/verify 仅保存数据；run 仅 MAIN 专用进程执行。
无独立 P1 admission，known root0 的双臂是同一批，不是两个独立批次；即使真实执行成功也不声明 P4 通过。

## 来源与因果顺序

固定 root `9173218f71898716eac8fe0f3946b8e2cf20527ac71d84555c20c4e1b7fbf333`。
读取已封存 sourcepack `p4_feedback_root_sourcepack_20261027_v1_root0_prepare01`；
对实际 response `p4_feedback_root_cuda_response_20261027_v1_cuda_main01` 重新执行独立 saved-array audit。
固定原 goal/Q、model/config、normalizer（仅显式 mean/std 传入）、原 root latent history 与原 prefix commands。
response 的 J、winner 和后继 context **绝不作为本次执行输入**；不 admit 原 runtime。
prepare 核验 P3 原 physical prepare/report、四根 reference、collection reset receipt、实际 source bytes 与 model bytes。
源缺失或 SHA 漂移立即 fail closed，不写占位 capability。精简协议，不复制旧大 protocol。

实际 run 创建一个环境；按原 seed/oversample 收集四根并验 fingerprint，选择 root0。
每臂 reset 到原 state/goal，各执行三个 zero 五步 packet（15步 warmup），每一个 primitive 均用原 RoyalBlue RGB guard；
verify 原 whole-stack RGB SHA、goal RGB SHA 与 float32 action history。
从本臂原 prefix 执行五步；固定动作与初始 context 先 fsync 保存，重新 native predict 一次。

每臂 step5/10/15：编码自身三张最近边界 RGB，保存 native float32 3x192 array、像素 SHA 与 action history，验证连续两张重叠 latent。
actual 使用此次观察 latent 三帧；imagined 保留自己连续预测的 context，实际 RGB 不进入其候选排序，但仍用于 guard、观察记录和终评。
每轮新 nominal、新192x10 CUDA AD、新十候选 nonlinear original-Q score、新独立 winner replay。
复用 frozen `spectral_bank` 数学：zero/ridge、两半径×两模式×两符号，固定 damping=.002 与总 logit 半径。
候选与 request/J/goal/Q/完整 context/winner/replay 封存、fsync、readback、写 durable journal 后，才准许一次五步 dispatch。
step20 仅 RGB1 encoder 与 terminal latent/Q-prediction/goal、official/pose 评价，不再编译或反馈。
没有读取旧 step10/15/20 PNG 作为本次帧，没有仅替换 latent 后复用初始 J 的捷径。

## 明确工程预算（不是已经测得的 live 通过结果）

| 单位 | 完整执行 cap/期望 |
|---|---:|
| collector reset / branch reset | 32 / 2，共34 |
| warmup public step / selected public step | 30 / 40，共70 |
| predict / action encoder | 各80（六轮各13 + 两次prefix） |
| processed predictor context tokens | 240 |
| Jacobian API | 6 |
| grad API / engine Python entry | 各1152（每轮192，分开记账） |
| 数学 scalar VJP 方向 | 1152，不是1152张量Jacobian |
| visual encoder / existing projector | 各8：六个RGB3 + 两个RGB1，共20帧 |
| public backward / FD / training / retries | 0 |
| output cap / prepare cap / free floor | 16MiB / 8MiB / 200MiB |

wall/allocated/reserved 必须由 CLI 显式传入，建议沿用 MAIN 已给的600s、3221225472 bytes、4294967296 bytes 工程 cap。
这是新 live prospective cap，不继承旧 response 的资源通过结论。
CUDA-only，冻结参数与 buffers、before/after tensor hash（必须匹配已完成 response 的 model hash）、禁 TF32、math SDPA、offline loader。
所有 hooked module/API 按 arm/boundary/phase 非退款计 attempted/returned，禁止不在预算 slot 的神经调用。
collection 不准 env.step；reset/warmup/selected 各自有限额。每个步骤新 render，guard 无减弱，终止/部分/不可见均停止，无 padding、无自动重选。
每步 PNG 与 physics receipt、所有 intent、selection、module/API/physics entry、错误和失败终态均小文件持久保存。
POSIX alarm +同步显存/时限检查；Python 无法抢占卡死 GPU kernel，MAIN 仍须外部 watchdog。
fsync/磁盘硬故障无法保证最终 receipt 写出，但已成功 fsync 的 journal/partial 文件保留；绝不把缺失回执当完成。

## 命令（从 le-wm-main 执行）

```bash
.venv/bin/python -B -m pytest -q -p no:cacheprovider tests/test_p4_feedback_live_cuda_20261027_v1.py
.venv/bin/python -B -m experiments.spectral_reachability.p4_feedback_live_cuda_20261027_v1 prepare --wall-seconds 600 --allocated-bytes 3221225472 --reserved-bytes 4294967296 --output results/p4_feedback_live_cuda_20261027_v1_prepare02
```

prepare 输出外部 SHA。将其明确填入以下两条命令；verify 是保存数据操作，run 只能 MAIN 审核后在 HOST CUDA 上启动：

```bash
.venv/bin/python -B -m experiments.spectral_reachability.p4_feedback_live_cuda_20261027_v1 verify --prepared results/p4_feedback_live_cuda_20261027_v1_prepare02 --prepare-sha PREPARE_SHA
.venv/bin/python -B -m experiments.spectral_reachability.p4_feedback_live_cuda_20261027_v1 run --prepared results/p4_feedback_live_cuda_20261027_v1_prepare02 --prepare-sha PREPARE_SHA --output results/p4_feedback_live_cuda_20261027_v1_cuda_main01
```

run 输出必须是 NEW，不覆盖任何历史目录；部分失败保留 `receipt.json`、`events.jsonl` 与已封存 partial artifacts。
mock 只验证控制/预算/错误契约；saved prepare 只证明来源可重建；真实 CUDA/physics 仍须 MAIN 执行。
科学准入仍需独立 P1 与独立来源及后续归因审计，不能把这次已知根描述成独立确认。

试准备 prepare01 仅作为开发历史保留（SHA `221473beb0e7674e6641b54dcb214270e0838e143a2db6d66cf16c6668ab3e2b`），
随后静态核对发现 `_frames_tensor` 的 transform/device 为 keyword-only，已在新 driver 中修正并补直接 mock。
prepare01 未运行、不得启动，其旧 source binding 不再匹配工作文件；不修改或覆盖该目录。仅最终 prepare02 可用于本交付。
