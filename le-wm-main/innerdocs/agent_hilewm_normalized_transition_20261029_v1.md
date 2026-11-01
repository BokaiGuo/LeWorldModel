# Hi normalized differentiable transition：路线A最小接缝

本独占 source/tests/本文；不改旧 OwnedLow/root/model/loader/global，不训练、拟合、mapper、raw inverse或P2坐标强转。worker仅RAM arithmetic callbacks、source SHA和CLI检查，无torch/权重/NN/physics/full strict。experiment-code技能用于固定接口/日志/回归；不要求训练、收益或图表。MAIN审读后才绑定真实CUDA并验证方程及梯度；本实现本身不是数值等价证书。

## 原时序与严格范围

两个pin源：`agent_hilewm_low_cem_20261029_v1.py::OwnedLow.get_cost`，以及 `results/p1_hilewm_text_contract_20261027_v1/texts/package__code__h_le_wm__models__jepa.py.source.txt::rollout_low`（354–441）、`predict_low`（151–156）。`verify_sources()`只读这两份≤256KiB文本并核原SHA，不调用旧consume。

每步：先向完整a_seq追加本步outgoing width10 → kernel1 Conv1d与原embed MLP作用于**整个a_seq** → ctx=min(3,z长度,emb长度) → 原low_predictor末ctx输入 → hidden flatten(-1,192) → low_pred_proj → reshape后只取最后latent → 向完整z_seq追加。无inference/no_grad装饰、detach、CPU搬运、contiguous、in-place、裁剪或重新构造输入张量。冻结模型参数不阻止输入action的autograd。caller自己决定grad mode；在外层no_grad下调用不会凭空产生梯度。

本版显式接受 `[B,S,Lz,192]`、`[B,S,La,10]`、future `[B,S,T,10]`，不模仿官方rank3中history/sample轴歧义，不自动expand/补零。B1..5、S1..64、T1..16、初始Lz/La1..16；single-frame/zero history由caller显式给，full-context可携带，但不是P2配对认证。normalized动作无物理bounds声明，不clamp；每token表示5个二维primitive的grouped模型坐标，**不是raw5**。

## API（源码位于 experiments/spectral_reachability/）

- `Kernels(cat,stack,check,action_encoder,low_predictor,low_pred_proj,kind)`：无模型的callback接口；kind=`ram_fixture`或`host_cuda`，kind标签不认证设备/真实性。callbacks必须纯、不得改输入/root；调用后返回dtype/shape受检。
- `transition(FlatContext(z,a), outgoing, *, kernels, ledger=None)`：flat `[B*S,Lz,192]`、`[B*S,La,10]`和outgoing `[B*S,1,10]`，返回新完整FlatContext和本步 `[B*S,192]`。原tensor graph保留，不以列表/序列化context重建。
- `rollout(z_hist,a_hist,future_actions, *, kernels,ledger=None)`：完整whole，返回 `Rollout(prediction[B,S,T,192],final,contexts)`，每步full context均保留。
- `segmented_rollout(..., segments=(1,1,1),kernels=...,ledger=...)`：独立重放，正整数partition之和必须精确T；完整flat context跨段传递，不重新encode、不bootstrap、不截断、不padding。future始终原tensor切片；本步严格 `future[:,:,k,:].reshape(B*S,1,10)`，不改成action列表stack。没有声称CUDA layout已经数值等价。
- `check_replay(whole,segmented, *, exact)`：caller提供严格shape/dtype/bytes等值函数，验证完整预测、全部逐步z/a和final；不内置容差、auto-detach或模型重算。返回值仍明确CUDA/gradient_verified=false。fixture通过只证明fixture。
- `Ledger`：逐模块before-attempt/returned/error及input/output shape；失败不退款，错误shape回调如已返回仍保returned。API/sample/input rows/tokens有嵌套关系，不相加为总算力；没有擅自将3模块合称model.predict API。整段AE input tokens含历史重新编码；whole与segmented各自付费。它不认证真实硬件，`real_meter_authenticated=false`；J/VJP/backward/ADprimal/FLOPs未由本seam测量，保持null。

## MAIN后续CUDA接线（代码已可绑定，但本交付无生产launcher）

`cuda_kernels(torch,root,main_authorized=True,trace=None)` 接受MAIN已有可信CUDA runtime/root；不import torch、不load权重或foreign类。先核两source pin、所有节点eval、参数冻结且全部参数/buffer同CUDA、真实pos_embedding `[1,3,192]`和kernel1 action Conv配置；拒绝inference_mode/CPU。check拒绝非float32、inference tensor、错device与非有限值。不改root属性或forward；通过现有owned节点显式执行Conv1d→embed、low_predictor和low_pred_proj。

可复用 `agent_hilewm_rgb_high_low_20261028_v1.py::Trace` 的 `call(kind,inputs,kernel)`，每次全部原生输入输出可由MAIN持久化；传入trace不得修改/断开kernel返回张量。Ledger仅shape账，不能代替full数组/stride/device、model-before/after、loader/export/precision/source闭包。代码没有prepare/run/full consume伪入口，不自动调用已封存四槽实验。

MAIN最短验证顺序：已有可信构造与全部上游strict → 固定normalized S/H/context（先原single/zero，再显式multi-history）→ 同输入逐步对照封存OwnedLow方程，保存所有AE/predict/proj、stride与full context → whole/segmented精确比较 → 固定action输入梯度/JVP检查与冻结参数before/after。需要新独占driver和MAIN审计授权后执行，不能以本RAM比较替代；若出现ULP/stride差异保留失败、不放宽容差。旧原公式sum cost不在此模块重定义；没有goal、Q、target选择、评分、赢家或physical输出。

## 现在可执行的命令

cwd `/home/gbk/code/LeWorldModel/le-wm-main`：

```bash
OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m pytest -q -p no:cacheprovider tests/test_agent_hilewm_normalized_transition_20261029_v1.py
.venv/bin/python -B -m experiments.spectral_reachability.agent_hilewm_normalized_transition_20261029_v1 protocol
.venv/bin/python -B -m experiments.spectral_reachability.agent_hilewm_normalized_transition_20261029_v1 verify-sources
```

CLI只stdout JSON，不创建results、不执行任何callback/runtime。RAM测试包括literal独立循环、先append/ctx1→2→3、投影flatten、完整history seam、非连续sample/time布局、全部分段分母、输入/返回shape篡改、失败attempt账、冷导入禁止torch/NumPy/pickle/physics、真实source pin与tamper。测试不是神经CPU实验，不是CUDA方程或梯度证据。

实际最终专属测试：**39 passed in 0.26s**。另 `verify-sources` 实际exit0、两原source SHA匹配；三件套 scoped `git diff --check`通过。符号shape/dependency callback证明本接口没有丢弃输入action依赖标签，不计算真实导数，也不认证PyTorch autograd/神经数值。未运行生产prepare/CUDA/physics/full strict；未新写results。

仍未知：真实CUDA逐层等值、导数正确性、与P2坐标配对、原始scaler/raw单位认证、三臂optimizer/controller预算接线及native可执行性。科学gate保持false。本文与source均不签任何资产/模型/根certificate。
