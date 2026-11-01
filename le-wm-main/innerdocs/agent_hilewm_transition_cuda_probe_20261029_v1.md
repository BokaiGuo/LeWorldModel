# Hi normalized transition：固定 HOST CUDA 方程与动作导数 probe

独占source/tests/本文。旧normalized_transition和全部Hi loader/OwnedLow/结果冻结不改。worker无生产prepare/full strict、torch/神经CPU/CUDA/physics/权重加载/网络；只有RAM callbacks与临时saved fixtures。MAIN审读后运行。不是三臂实验，不是原foreign类parity、P2配对、raw inverse或科学放行。

## 固定合同与独立reference

仅既有**root0/off**已验证Hi `z_init`，single-z、显式zero incoming grouped history；B1/S1/H3，完整三个normalized width10 future tokens，无选择/补步/物理动作。父low报告实际SHA `68d23f06b88114d3f3a3df227cefe7ca247cb399ba452defbf5456dbda9f32d0`；持久完整strict SHA `96be2b84aadabc4ca23a8e84f2f54f889b86367d41d4d344e0d7d56b46e3b43e`。prepare从真实report取其实际manifest SHA，重建whole low.consume并与receipt完整相等，再verify父manifest；不接受仅status。输入原NPY、root/source/RGB身份、loader/export/IR/425原始tensor值和原receipt全部绑定。已付编码/high/low历史不变成新免费开销。

reference是本文件 `reference` 对旧OwnedLow和官方 `rollout_low/predict_low` 的独立字面转写：append full a→kernel1 Conv/embed→ctx=min(3,…)→low_predictor→flatten low_pred_proj→取末latent追加。**不调用seam**。其底层仍是同一可信owned节点，验证的是source assembly/新seam，不独立验证整个神经模块方程。whole/segmented才调用已冻结seam，分段(1,1,1)保留完整flat z/a；各自独立图、独立action叶。

固定数值由protocol源码预声明：actions=`((arange30%11)-5)/32`，方向=`((arange30%7)-3)/16`，terminal weights=`((arange192%13)-6)/128`，全部float32、无随机种子/结果选择。诊断标量为float64 terminal×固定weights求和，不是拟合Q/任务成功分数。

三个grad-enabled H3：reference/whole/segmented。逐模块所有原输入/输出、逐步完整z/a、完整预测与标量和30维action梯度保存NPY。全forward值按shape/dtype/bytes**逐位相等**；实际shape/stride/storage_offset/contiguous/device/dtype/requires_grad也跨三route精确相等。三份标量VJP（autograd.grad）完整gradient要求逐位相同，不放宽ULP。不截断hidden、不借seam作为reference。参数eval/frozen且全部buffers同CUDA，但actions叶允许requires_grad；无inference_mode，MATH-only、deterministic、TF32/autocast关闭，逐call保存实际precision。

中心差分只用reference额外两次H3，epsilon固定2^-8，只有一条固定方向，无多epsilon/配置重试。AD_dot由保存的30维梯度和方向用float64独立算术复算，FD由两个真实scalar复算；事前判据：`abs(AD_dot-FD) <= 2e-4 + 2e-2*max(abs(AD_dot),abs(FD))`。此判据只针对float32 forward中心差分近似，不是forward/gradient parity容差；它不保证每坐标导数正确或全Jacobian通过。标量独立保存、以192项float64舍入界复核。若失败留报告，不能按结果改判据或重试挑配置。

## 资源、保存和失败

新增每种AE/low_predictor/low_pred_proj **15 calls**；predictor sample rows15、context tokens30；AE完整history tokens45；projection flattened rows30。三次明确scalar VJP；AD primal为前3route共9 low forwards的inclusive子集，不再次相加。2个中心差分forward rollouts已计入15，encoder/projector/high/physics/predict API均0（未调用root.predict API，直接low节点）；内部backward engine调用/FLOPs未知null。API、rows、tokens、AD、capture、墙钟scope分别列，不称同FLOPs。

新输出64MiB、final reserve8MiB、disk floor200MiB、cooperative wall600s（含初始/末尾完整verify和tensor hash，不能抢占kernel）。每call真实attempt/return、CPUcapture wall及callback+CUDA同步wall；这些范围重叠，不相加当总GPU时间。另记录initial verify/construct/model hash before/dispatch/model hash after/final verify、每route wall及实际peak allocated/reserved。无退款/隐式retry。

原生float32/float64数组，不dtype伪装；全文件SHA/bytes、raw SHA、NPY header/长度/finite/EOF和完整文件/引用census。trace保存用detach+CPU copy仅用于取证，kernel返回原tensor继续图。run开始先排他创建本stem目录；verify/loader/forward/AD/阈值任一步失败保留已完成数组、五route分母/not_started和失败report。finally尽力记录model/source after；磁盘/进程强杀仍可能无法finalize，不能声称绝对持久化。成功时全部weights须等于export425值，完整source/export/input/receipt前后不变。

saved consumer重建整个父authority、manifest、全部native时序/array delivery和上下文，独立算术重算scalar/FD和资源，逐位复核三个route的所有模块输入/输出/layout/gradient；不神经重算，不冒独立梯度证据。fullstrict通过仍依赖MAIN真实运行与源码审查，不认证硬件真实性或模型正确性。

## MAIN准确CLI（worker未执行生产命令）

cwd `/home/gbk/code/LeWorldModel/le-wm-main`。每次用不存在的新本stem目录；SHA必须取实际前一命令输出，不能填未来值：

```bash
OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m pytest -q -p no:cacheprovider tests/test_agent_hilewm_transition_cuda_probe_20261029_v1.py
.venv/bin/python -B -m experiments.spectral_reachability.agent_hilewm_transition_cuda_probe_20261029_v1 prepare --output /home/gbk/code/LeWorldModel/le-wm-main/results/agent_hilewm_transition_cuda_probe_20261029_v1_prepare_MAIN01
# 将prepare实际输出SHA/路径分别赋给 HILEWM_PROBE_MH / HILEWM_PROBE_MANIFEST
.venv/bin/python -B -m experiments.spectral_reachability.agent_hilewm_transition_cuda_probe_20261029_v1 verify --manifest "$HILEWM_PROBE_MANIFEST" --manifest-sha256 "$HILEWM_PROBE_MH"
CUBLAS_WORKSPACE_CONFIG=:4096:8 .venv/bin/python -B -m experiments.spectral_reachability.agent_hilewm_transition_cuda_probe_20261029_v1 run --manifest "$HILEWM_PROBE_MANIFEST" --manifest-sha256 "$HILEWM_PROBE_MH" --output /home/gbk/code/LeWorldModel/le-wm-main/results/agent_hilewm_transition_cuda_probe_20261029_v1_cuda_MAIN01 --main-authorized
# 用run实际输出SHA赋给 HILEWM_PROBE_RH
OPENBLAS_NUM_THREADS=1 .venv/bin/python -B -m experiments.spectral_reachability.agent_hilewm_transition_cuda_probe_20261029_v1 consume --output /home/gbk/code/LeWorldModel/le-wm-main/results/agent_hilewm_transition_cuda_probe_20261029_v1_cuda_MAIN01 --report-sha256 "$HILEWM_PROBE_RH" --receipt /home/gbk/code/LeWorldModel/le-wm-main/results/agent_hilewm_transition_cuda_probe_20261029_v1_consume_MAIN01.json
```

运行只支持绑定的 `.venv` torch和真实HOST CUDA，无CPU fallback，无foreign模型执行或pickle。CLI提供prepare/verify/run/consume，不自动下载/安装。RAM fixtures生成的是明确合成的算术trace/gradient身份，仅测试saved协议；mock CUDA标签不构成真实设备证据。真实prepare及CUDA方程/梯度结果尚未取得，不能预写ready或passed。没有缺失loader接缝需要猜补；若父严格链/配置/425tensor或环境不符，prepare/run实际拒绝并需MAIN诊断，新版本修复、不patch冻结依赖。

实际worker验证：新probe加冻结seam联合 **87 passed in 1.19s**（probe48项、seam39项）；CLI `--help` exit0，scoped `git diff --check`通过。已保存低层report/manifest/receipt仅核字段ABI与实际SHA，无full strict调用。首轮35pass/1fail来自RAM夹具NumPy与torch的transpose签名不同，只修夹具，未改真实方程。完成后冻结三件套；所有真实数值/梯度与HOST资源结果仍待MAIN。
