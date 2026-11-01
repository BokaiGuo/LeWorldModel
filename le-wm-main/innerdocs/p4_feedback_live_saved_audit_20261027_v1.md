# P4 LIVE v1 实际 saved-data 独立审计

输入仅 MAIN 已完成的 `p4_feedback_live_cuda_20261027_v1_cuda_main01`，receipt 外部 SHA
`e3c91195540b7465e2fcbf8431d7f182c06bfdff9da6ab160489ae6fe23bbb47`。
不改 live v1/v2、旧 core、任何 sealed 源码/结果或共享总文档；不运行 CUDA、神经、AD 或 physics。
独立 saved audit 只使用 NumPy/Pillow、小型 seal verifier、原独立 response saved auditor 的算术/来源工具。
不 import live producer；sourcepack.verify 仅作保存来源核验，不 dispatch producer/runtime。

## 门与边界

* receipt→实际 execution binding→prepare02 SHA→protocol→所有 source seals、model/config 与原 sourcepack/response/physical report 核验。
* 独立生成198个预期文件名，逐文件 SHA，拒绝未知/缺失文件、路径替换、非 native float32 array；结束再核对防漂移。
* 逐臂三个 warmup packet 15步、原 prefix 5步，再三轮 selected 5步：共34 reset、70 public step。
* 两臂均独立从自身 warmup PNG 验 whole-stack/goal SHA，prefix 新 PNG 与原真实 source pixel SHA 对齐。
* 每轮 observed.npy 对齐本臂最近三张边界 PNG 的 pixel SHA 和原 executed action history；actual 必须把整组三帧观察值精确送入 request，imagined 必须保留自身前次 winner predicted context。
* 重叠 latent 的 atol2e-5 仅是 batch encoder 数值一致性检查，不宣称 bitwise 或观测精确同值。
* 每轮独立核验 J192x10 array binding、nominal/J-forward 一致、SVD eigensystem、canonical signs、bounded ridge、十候选 bank/float32 tanh、全部 causal history、nonlinear original-Q 分数、stable winner 与 replay equality。
* 不用 producer Ledger：独立重建5291条事件及2486个 API entries（含嵌套192次 grad/engine），验证六次 selection seal 在 dispatch 前、每次 dispatch 后恰好五个 physical attempt/return。
* encoder/projector 各8批共20帧，predict/action_encoder 各80、J6、grad/engine 各1152；predict 单位是 model.predict API，不是内部 predictor 或 FLOPs。
* 原 RGB guard 不变，70张步骤 PNG 独立计 RoyalBlue 像素；84条 guard observation 中82条可由已存 PNG 独立重计，另两条初始 warmup render 只有 source-bound recorded count，没有独立 PNG。
* step20 只终评。独立复算15→20 latent MSE、Q prediction MSE、Q goal MSE；也附 step5/10/15 预测与新 observation 的描述性误差。
* 从保存七维 state/goal 独立复算 installed-native official 全字段、abs(atan2(sin,cos)) wrapped angle、object pose 与 wrapped success。native 多圈角差 bug 保留原记录，wrapped 另列，不伪成功。

saved PNG/array/context 连接、原 source 与日志通过，不等于重新执行 encoder/J 或硬件 attestation；恶意伪造执行不在本保存审计证明范围。
仅一个 known-root pair，不是两批；没有独立 P1 admission，报告始终 P1/P4 false。
模型 Q 数值与 object task 指标分开报告，不以低 Q-goal MSE 宣称物理收益。

## 小空间命令

在 `/home/gbk/code/LeWorldModel/le-wm-main`：

```bash
.venv/bin/python -B -m unittest discover -s tests -p 'test_p4_feedback_live_saved_audit_20261027_v1.py' -v
.venv/bin/python -B -m experiments.spectral_reachability.p4_feedback_live_saved_audit_20261027_v1 --output results/p4_feedback_live_saved_audit_20261027_v1_main01
```

unittest 仅内存算术/mock，无 pytest tmp/cache，无大 source copy；源码/tests/doc+实际 report 总和严格<=300KiB。
报告 NEW 独立目录、独占创建；不覆盖输入。先 audit 成功再写小 report；写入前检查200MiB reserve。
