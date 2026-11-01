# Hi frozen low：显式可微 facade

仅本 stem source/tests/doc；旧 loader/probe/失败结果完全保留。旧 loader `TrustedTensorNode.forward` 的 inference_mode 是原 inference-only 合同，不是旧 CEM 的失效证据。本模块不修改该合同；只对同一受信 data graph 低层方程建立新执行边界，不证明官方 parity、P2 坐标配对或科学准入。

## MAIN 最小接线

```python
from experiments.spectral_reachability import agent_hilewm_differentiable_low_20261029_v1 as diff
# root,t 来自 MAIN 已认证的 frozen loader/CUDA runtime；不要在 inference context 构造/调用。
# 原出口/收据/全部425项权重 before/after SHA 仍由新 driver 完整复核。
facade = diff.wrap(root, t, main_authorized=True)
closure = diff.source_closure(t)  # 加入新 driver manifest，验证执行前后不变
# 将 facade 而不是 root 传给新版本 probe 的独立 reference 及 seam.cuda_kernels。
# seam.cuda_kernels(t, facade, main_authorized=True, trace=trace)
# reference(t, facade, z_hist, a_hist, future, trace)
facade.assert_frozen()
```

不改冻结 probe，也不在本模块启动新 probe。梯度模式保持 caller-owned：普通 enabled grad 才允许；不偷偷开启、不进入 inference/no_grad、不 detach。原图参数不复制、不注册新参数、不训练。`parameters/buffers/modules/named_parameters/named_buffers/state_dict` 委托原图；其他原图 executable 不暴露。caller 可保留外部原 root 做像素编码，但不允许绕回旧低层 forward 代替本 facade。

## 方程与边界

逐式绑定冻结 loader SHA `df313393f1ead36a512dc549f8883a1da85dd3e7ffa057f690d39627a1e7b411` 的低层 forward，以及 `module.py`/已存官方 jepa 文本（源码内固定 SHA）。Linear、LayerNorm、eval BN（原 running stats/momentum/eps）、GELU approximate、SiLU、eval Identity/Dropout、Sequential、MLP、FeedForward、causal Attention、AdaLN ConditionalBlock、Transformer、ARPredictor 均转写。保留 source reshape/transpose/flatten/pos-slice 与运算顺序，不声称相同值必有相同 CUDA kernel。

`action_encoder.patch_embed` 是数据访问视图，直接返回原 weight/bias 和 literal stride/padding/dilation/groups；reference/seam 的 functional conv 不变。`embed/low_predictor/low_pred_proj` 均新 facade。可选直接调用 action_encoder 时，采用原 Embedder float→permute→functional conv1d→permute→embed；严格只准 `[10,10,1]`、stride1/pad0/dilation1/groups1。ModuleList/Softmax/Conv1d 仅供受支持方程访问，不独立调用。无 pixel/high/macro/raw inverse 扩展。

wrap 先核验源闭包，再检查整个原图所有 node eval、所有参数/buffer同 CUDA/冻结/非 inference tensor。每次调用前后检查 tensor identity/version/shape/stride/offset/dtype/device、node identity/children/literal attributes。它不是内容 SHA 证书：`.data` 等绕 version 的外部变更必须由 driver 的原始425 tensor字节 hash before/after捕获。driver还需绑定原 export/loader/运行时/kernel closure；本模块不接受调用者自称“认证”作为配对证明。kernel闭包包含 torch functional、grad_mode、module、tensor源文件，并返回 source SHA；不读取 tensor/NPY/权重。

## 本地可执行检查

仓库根 `/home/gbk/code/LeWorldModel/le-wm-main`：

```bash
OPENBLAS_NUM_THREADS=1 .venv/bin/python -m pytest -q tests/test_agent_hilewm_differentiable_low_20261029_v1.py
.venv/bin/python -m experiments.spectral_reachability.agent_hilewm_differentiable_low_20261029_v1 verify-sources
.venv/bin/python -m experiments.spectral_reachability.agent_hilewm_differentiable_low_20261029_v1 protocol
```

测试只用 symbolic shape/表达式/调用顺序 witness 和 fake CUDA标签，没有 torch import、CPU神经算术或模型权重。涵盖全部方程分派、因果SDPA/布局、参数原引用、旧 forward 毒化拒调、训练/梯度/版本/属性/device/参数替换变异。RAM通过不等于真实自动微分通过。MAIN需新 driver 进行 source-matched reference/whole/segmented/固定VJP与原预注册FD检查；不放宽判据、不自动换配置。本交付不含生产run命令，不运行NN/physics/fullstrict，不开放P1。
