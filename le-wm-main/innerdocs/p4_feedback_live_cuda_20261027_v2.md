# P4 live v2：启动前五项补强

v1 源码/tests/doc、prepare01/02 与所有旧 core/response/sourcepack 不修改。
v2 独立 run/finalization 与 prepare/verify；不修改 v1 的 module globals。
保留 v1 sealed workflow、Native、Ledger、逐步 RGB guard 和 frozen compiler math，只有终评 adapter 与错误收尾扩展。
本轮只做 mock、saved prepare/verify；不执行神经、CUDA、physics。MAIN 审阅后独占真实运行。

## 五项精确定义

1. `counts.predict` 计 **model.predict API entry**，不是 `model.predictor.forward` entry、FLOPs 或底层 kernel 次数。
   hook 目标与 frozen Native 调用图一致；不是通用 direct-predictor bypass 检测器。
   独立 mock 让一次 model.predict 内部两次 predictor.forward，确认只记一次 API。
   action_encoder / encoder / projector 分别记各模块 forward entry；grad API 与 engine Python entry 分开计账。
2. v1 的未捕获 journal.close 故障确会阻止 receipt 写出，不修改已封存历史。
   v2 对 final_journal、journal_close、alarm、hooks/env cleanup 追加错误，保留执行 first_error 顺序。
   任一 close 故障都不得完成；`receipt.first_error == errors[0]`，已有 fsync journal/partial 仍保留。
   journal 最后只写 `run_finalizing.pre_close_complete`（关闭前候选状态），最终完成以关闭之后写出的 receipt 为准；另记录 `journal_close_failed`。
   测试强制运行在 torch/gym import 前失败，再让实际 Journal 子类 close 抛错，验证 partial receipt 仍写出且原错误未覆盖。
   若磁盘/文件系统本身不可写，则仍无法保证最终 receipt，绝不把缺失当完成。
3. 不改变 installed native evaluator，也不借其多圈角差 bug 报成功。
   终评 native 标签重命名为 `physical.official_native`（原值原样记录），不再提供含混的 `physical.official`。
   新 `physical.wrapped_diagnostic` 独立使用 abs(atan2(sin(delta),cos(delta)))，范围[0,pi]；
   原 position<20、angle<pi/9 条件分别报告 wrapped official/object-only success。
   同时记录 raw delta、native angle、native-domain-valid 与两种 success 是否一致。
   例如 delta=4pi+0.5 时 native angle 为负且可误报 true，wrapped=0.5 明确 false；不修写 native、不伪成功。
   此诊断只终评，不更改 goal/Q、候选排序或动作，不产生 P1/P4 admission。
4. v2 prepare 先检查 **200MiB+8MiB prepare+16MiB run** 的总 headroom。
   不再复制 v1 的约4.9MiB full source closure：固定绑定原 prepare02 SHA，并在 prepare、verify 和 run 前后完整调用 v1.verify 核验父 seal、来源/model/arrays。
   v2 只新增三个显式源码/tests/doc 的小型 revision seal（源码总和上限256KiB），协议含父 protocol SHA 与 v2 文件 SHA。
   父 prepare/source 缺失或漂移即拒绝，不能仅凭声称继承 source closure 启动。
5. overlap `np.allclose(rtol=0,atol=2e-5)` 仅是 **batch encoder 数值一致性检查**，允许 roundoff，不是 bitwise 或观测 latent 精确同值声明。
   像素身份用 live RGB uint8 SHA，root 用 whole-stack/goal SHA；保存的 native array 另有文件 SHA。
   实测 mock 在容差内故意不同仍通过且差异原样保存；超过容差拒绝，不把近似写成精确观察。

## cap 不变

600s / allocated3221225472 bytes / reserved4294967296 bytes，必须与已封存父协议完全一致。
reset34（collector32+branch2）；public env.step70（warmup30+selected40）；双臂各20step、三轮 step5/10/15。
predict API80、action_encoder80、Jacobian API6、grad API1152、engine entry1152（每轮192、每臂576）。
visual encoder8+existing projector8，RGB3六次+RGB1两次共20帧；predict processed context tokens240。
public backward/FD/training/retry 全0；output16MiB、prepare8MiB、disk floor200MiB。
不降预算；Python alarm 不能抢占挂起的 GPU kernel，MAIN 仍须外部 watchdog。
两臂来自一个 known root，并非两批独立来源；没有独立 P1 admission，不宣称 P4 通过。

## 命令

在 `/home/gbk/code/LeWorldModel/le-wm-main`：

```bash
.venv/bin/python -B -m pytest -q -p no:cacheprovider tests/test_p4_feedback_live_cuda_20261027_v2.py tests/test_p4_feedback_live_cuda_20261027_v1.py
.venv/bin/python -B -m experiments.spectral_reachability.p4_feedback_live_cuda_20261027_v2 prepare --wall-seconds 600 --allocated-bytes 3221225472 --reserved-bytes 4294967296 --output results/p4_feedback_live_cuda_20261027_v2_prepare01
.venv/bin/python -B -m experiments.spectral_reachability.p4_feedback_live_cuda_20261027_v2 verify --prepared results/p4_feedback_live_cuda_20261027_v2_prepare01 --prepare-sha PREPARE_SHA
```

仅 MAIN 审核后执行，输出目录必须 NEW：

```bash
.venv/bin/python -B -m experiments.spectral_reachability.p4_feedback_live_cuda_20261027_v2 run --prepared results/p4_feedback_live_cuda_20261027_v2_prepare01 --prepare-sha PREPARE_SHA --output results/p4_feedback_live_cuda_20261027_v2_cuda_main01
```

v1 prepare02 保持可验证的父证据，但本交付应启动 v2，避免 v1 journal-close 缺口。
