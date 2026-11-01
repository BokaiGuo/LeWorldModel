# Hi differentiable low v2：仅 Conv1d literal 结构 ABI 修复

旧 v1 facade、loader、probe、seam 与失败结果全部保留。本版完整继承 v1 方程/API/冻结守卫及 caller-owned grad；唯一执行语义差异是三个 Conv1d 结构属性的 list→tuple 解释。未运行模型/physics/fullstrict，不声称 CUDA/梯度已通过。

## 真实来源而非构造器推断

已读 `results/agent_hilewm_inert_pickle_ir_20261028_v1_MAIN01.json`（SHA `709b31e28bee8039d577668cbd9613347be1d070a7619e3db3ca61e3674cabbe`），`nn.modules` 中 path=`['action_encoder','patch_embed']`、node=18441、声明 `torch.nn.modules.conv.Conv1d`。其 BUILD state：

|属性|IR node|IR kind|literal 解码|v2 facade|
|---|---:|---|---|---|
|stride|18522|tuple|[1]|(1,)|
|padding|18524|tuple|[0]|(0,)|
|dilation|18526|tuple|[1]|(1,)|

冻结 loader `LiteralIR.literal` 将 tuple/list 等容器统一解码为 list；`graph_spec` 将解码结果存入 `literal_attributes`，构造节点保存为 `literal_source_attributes`。没有证据表明旧构造器最后 setattr 为 tuple，本文不作该声称。v2 是**新 facade 的显式结构解释**，不是复原旧 node 上不存在的动态属性。

仅 kind=Conv1d 且属性属于上述三项时，要求原 literal 类型严格为 list 或 tuple、长度1、元素 `type is int` 且值精确等于表内固定值，再返回 tuple。拒绝 bool、float、scalar、空/多项、其他整数和 tagged dict；不猜默认值。不改原 literal，不改 kernel_size/groups 等其他字段，不改其他 node 类型。零数值变化，无新 tensor 或参数复制。

## MAIN 接口

```python
from experiments.spectral_reachability import agent_hilewm_differentiable_low_20261029_v2 as diff
facade = diff.wrap(graph, torch, main_authorized=True)
sources = diff.source_closure(torch)
# 新 probe v3 将 facade 交给原 reference 和原 seam；冻结 v1/v2 probe 不改。
# diag.old.weights(facade) 使用委托原 graph 的 state_dict，仍核验全425项。
facade.assert_frozen()
```

`wrap(graph, torch, *, main_authorized=False)` 保持不变；原 parameters/modules/buffers/state_dict 等委托不变；无旧 forward、inference/no_grad/detach、monkeypatch 或旧 _modules 修改。disabled grad 仍拒绝：MAIN 的 FD 原graph推理参考由新 driver 独立控制，不改变本 facade 合同。全部权重内容 before/after SHA 及 authority认证仍由 MAIN driver 承担；版本/identity 守卫不冒充内容认证。

## 实际可运行的非神经核验

在 `/home/gbk/code/LeWorldModel/le-wm-main`：

```bash
OPENBLAS_NUM_THREADS=1 .venv/bin/python -m pytest -q tests/test_agent_hilewm_differentiable_low_20261029_v2.py
.venv/bin/python -m experiments.spectral_reachability.agent_hilewm_differentiable_low_20261029_v2 verify-sources
```

继承全部 symbolic RAM 方程/毒化旧forward/冻结mutation测试，新增三个list→tuple、非法结构与真实saved IR回归；后者只读惯性IR JSON并调用纯元数据 LiteralIR，不加载NPY/权重/pickle或执行神经。测试直接使用冻结seam的 `n.stride==(1,)` 等 guard 字面验证ABI，不用 tuple(n.stride) 掩盖不匹配。MAIN 后续主机CUDA数值判据保持原预注册值；旧实际forward数组逐位比对可作为推理forward parity，但同方程不冒充独立全网络验证。P1/P2配对与科学准入均未开放。
