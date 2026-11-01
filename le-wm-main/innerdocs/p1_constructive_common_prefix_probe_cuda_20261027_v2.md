# Common-prefix CUDA v2：分离旧task认证与正式MATH数值协议

2026-10-27。MAIN授权的新v2，旧v1 source/tests/doc/prepare/92enc失败产物全部保持冻结。仅新增本stem源码/tests/doc。不设专用审计agent；MAIN review/prepare/HOST CUDA。实现者仅RAM/mock/saved读，不运行神经/物理。使用experiment-code技能的先固定协议、实际输入接线、逐调用故障日志；不做训练/按结果调参/绘图。

## 已知数值协议失败与修复边界

MAIN提供crossover实际证据：4/4默认SDP编码与旧blindtask逐bit相同，4/4 MATH编码与v1失败run逐bit相同，权重未变。这里据授权修复数值协议接线，不声称新的独立crossover审计或模型/理论负结果。v1将跨backend逐bit相等当作gate，导致92enc后0predict；v2不覆盖该失败、不设容差、不删除来源认证，而以额外4次**原数值协议**编码执行原task exact认证。额外成本完整计入，不称仍为92enc。

## 固定合同

输入仍显式pin physical report SHA `450b182e4ed4b02959f74593099cf85c728619bfc5ddc3678ea1940c846e1210`，physical prepare `70616f76f08f560301cd0b5445cad8fcb41a9e53415bcac7a9d0efbf34cf560e`，冻结v1源码 `a29fe7a195de3c9b4ee0b1b6f45a67dad0af5c6e6595337291fb0ac2f00e9713`。复用其全部saved PNG/file/pixel/state/journal/fullbody/prefix/role/action来源检查、84行固定分母和算术，不调用其失败execute/run，不改producer。

prepare额外沿approvedtask.task_contract.prepare_sha回读旧blind encoder prepare，核source_slot/blind SHA，以及原pixel_sha256[1]/[2]与当前actual warm5/warm10 PNG pixel SHA逐项一致。只读prepare/task和hash，不解析blindNPZ/sourcepack/certificate/witness。默认认证与正式warm使用**完全相同的4 PNG路径和file/pixel SHA**，不得换图、重采样、pixel转换或后验择帧。

顺序与scope固定：

1. `anchor_default`：root0 warm5/warm10、root1 warm5/warm10，共4encoder+4projector；所有SDP backend enabled（PyTorch默认dispatch，非强制某单一kernel）、cudnn TF32=True、matmul TF32=False；float32/batch1/native projectedCLS192。每张输出先durable保存192float32及bits，再与批准旧task对应latent_history[1]/[2] **float32字节逐bit**认证。含signed-zero差异也拒绝，不允许allclose/阈值/自动重试。失败保留输出及认证false，在正式MATH/预测前终止。
2. `image_math`：原92槽顺序与图片不变，MATH-only SDP/cudnn TF32=False/matmul TF32=False，92encoder+92projector。4正式warm不再执行跨backend相等gate；保存MATH−default完整delta、exact_bits、maxabs和MSE，acceptance_tolerance=null。此差值只作描述，不构成新质量阈值。
3. `predict_math`：全部84个固定原H1 probe，MATH/cudnnFalse/matmulFalse，共84actionencoder+84predict。输入只取正式MATH `[warm5,warm10,prefix5]`与原raw`[zero5,prefix5]`经原float32 normalizer；默认认证latent不替换任何输入，terminal仅labels。没有跨backend拼接context、没有额外nominal/replay/选择。

总计**96encoder+96projector+84predict+84actionencoder**，新AD/J/VJP/SVD/solve/训练/physics均0。每模块entry有phase/scope/index、attempt/completion，返回并sync成功才完成，失败不退款。phase caps4/92/84与全局caps96/84同时检查；完整终结要求各phase实际账精确相符。每次数值phase记录实际SDP enabled flags/cudnn/matmul，退出恢复进入前TF32/SDP设置；不是靠报告填写计数。全模型hash前后相同，eval/frozen，offline/noCPUfallback。

## 输出、partial和科学边界

报告独立`anchors[4]`（输出、bits、exact_original_task、原PNG provenance）、`images[92]`、`rows[84]`、`warm_backend_differences[4]`、总/phase资源账；lossless gzip事件逐事件Z_SYNC_FLUSH+fsync，输出在后续gate前保存。原8MiB whole-output cap、256KiB final reserve、200MiB floor，启动208MiB；prepare≤2MiB。源/model finalization分开捕获，I/O完全失败仅承诺已有字节，不保证finalreport落盘。额外4次不是padding：认证旧task身份有独立作用，不能退款。

继承全部21×4固定分母/partial guards。默认4锚任何缺失/不exact明确拒绝，不用新图补值；正式terminal缺则保留missing/null与实际调用账，不能拿早停帧补终态；完整scope为已批准84/84actual，partial仅mock路径验证。native192共同方向仍是批准旧blindtask的goal−original_root，两根各共享，不按新MATH数值更换goal/root/Q；这两个固定参考来自默认backend，backend数值差保留说明，不宣称表示完全跨backend等同。物理u属另一空间，不混量纲。

原pred-vs-real MSE、各position相对自身zero的Δz、192×10中心secant及signed投影全部不变。不以修复假说解释控制效果，不从outcome改probe/方向/阈值。两knownroots、G=D、有限干预；P1/P2=false。

## MAIN执行命令

cwd `/home/gbk/code/LeWorldModel/le-wm-main`，实现者未创建prepare/results或运行CUDA：

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m pytest -q -p no:cacheprovider tests/test_p1_constructive_common_prefix_probe_cuda_20261027_v2.py
.venv/bin/python -B -m experiments.spectral_reachability.p1_constructive_common_prefix_probe_cuda_20261027_v2 prepare --physical-report results/p1_constructive_common_prefix_probe_physical_20261027_v1_main01/report.json --physical-sha 450b182e4ed4b02959f74593099cf85c728619bfc5ddc3678ea1940c846e1210 --output results/p1_constructive_common_prefix_probe_cuda_20261027_v2_prepare01.json
.venv/bin/python -B -m experiments.spectral_reachability.p1_constructive_common_prefix_probe_cuda_20261027_v2 verify --prepared results/p1_constructive_common_prefix_probe_cuda_20261027_v2_prepare01.json --prepare-sha MAIN_PREPARE_SHA
.venv/bin/python -B -m experiments.spectral_reachability.p1_constructive_common_prefix_probe_cuda_20261027_v2 run --prepared results/p1_constructive_common_prefix_probe_cuda_20261027_v2_prepare01.json --prepare-sha MAIN_PREPARE_SHA --output results/p1_constructive_common_prefix_probe_cuda_20261027_v2_cuda_main01
```

测试只RAM/mock与已批准saved inputs，不导入torch/不创建nativephysics实例/不写tmp。mock通过只证明工程接线，不替代MAIN实际96enc/84predict。
