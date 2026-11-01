# PushT full-packet chart：独立非神经坐标代数

仅新增纯 NumPy 模块和测试，不导入 torch、模型、checkpoint、环境或数据。不修改已有 sealed 文件、中央进展或 stage 门；不提供根因、效能或准入结论。

## 统一坐标与物理尺度

一个 packet 是五条按时间排序的二维相对动作，flatten 顺序为 `[x0,y0,x1,y1,...,x4,y4]`。旧 constant 参数 `a∈R²` 对应五次重复的 `tanh(a)`；不能错误地 repeat-interleave 各坐标。

新增等距 constant 基 `C = tile(I₂,(5,1))/√5 ∈ R^(10×2)`，full 基 `F=I₁₀`。二者列正交归一。系数 `u` 映射为 logits `z=B u`，再映射 raw packet `tanh(z).reshape(5,2)`；因此 `||z||₂=||u||₂`，即相同的十维 atanh-logit L2 预算。约束/正则化在 tanh **之前**；不是 raw action L2、逐坐标 clipping，也不是归一化后模型输入的 norm。

constant 嵌入 full 后动作相同，预算相同。直接沿用旧二维数值 `a` 作为新系数会缩小物理尺度；要恢复旧 constant raw packet，应取 `u=√5 a`。这明确改变了新 constant 的坐标尺度，不能称其参数化与旧版本原样相同。

按主线程最新规范（模块本身不硬编码这些超参数）：候选 norm cap=`√5×.75`，top2 谱模式的正负半径为 `√5×(.25,.75)`，damping=`.002=.01/5`。于是 constant 坐标的 ` .002 ||u||² = .01 ||a||²`，恢复对应旧根候选的物理尺度与正则尺度；谱向量符号或重根方向仍由编译器处理，不在此断言候选身份逐字一致。每臂各10候选：zero、DLS、top2模式×两半径×正负。候选生成不属于本模块。

suffix 两臂使用相同新 norm cap=`√5×.75`，suffix 使用各自同一 chart。这是新共享限制，不等价于历史 unbounded suffix。输入若有多个 horizon，本模块 budget 是**整个传入 suffix 的一个总 L2 球**，不是给每个 packet 分配同样半径；单 packet 调用自然只约束该 packet。

## 稳定公开接口

模块路径：`experiments.spectral_reachability.pusht_full_packet_chart_20261027`。

```python
basis(kind)                                      # (10,d)，kind='constant'/'full'
raw_packet(coefficients, kind)                   # (d,) -> (5,2)
chart_matrix(chart, *, horizon=1)                # (10H,dH)，kron(I_H,B)
params_to_full(params, chart)                    # (...,d) -> (...,10) logits
constant_to_full(params)                         # 等距嵌入 (...,2) -> (...,10)
full_to_params(logits, chart, *, atol=1e-12)      # 严格子空间逆，不静默投影
params_to_raw_packets(params, chart, *, radius=None)  # (...,d) -> (...,5,2)
raw_packets_to_params(raw_packets, chart, *, atol=1e-12)
atanh_l2_norm(params, chart)                      # 所有 packet 总 norm，float
project_params(params, chart, *, radius)         # 总 L2 球径向投影，返回新数组
raw_mapping_jacobian(params, chart)              # (...,10,d)
pullback_jacobian(j_full, chart, *, horizon=1)    # (...,10H) -> (...,dH)
pullback_gradient(g_full, chart, *, horizon=1)    # 同上，行形式梯度
ridge_step(jacobian, residual, *, damping, current=None)
chart_ridge_step(j_full, residual, chart, *, damping, horizon=1, current=None)
```

`d=2` 或10。所有数值运算返回 float64，拒绝 bool、complex、object、非有限值和错误 shape；不修改输入。`raw_packet` 只接收一个系数向量；多 packet 使用 `params_to_raw_packets`。后者提供 `radius` 时超预算直接拒绝，不替调用方偷偷投影。显式使用 `project_params` 后可传相同 radius 再验预算。

逆映射拒绝 `abs(raw)>=1`，不 clip 后伪造 atanh；constant 逆映射拒绝非 constant packet。`atol` 是绝对子空间残差容差，默认 `1e-12`，`rtol=0`。极大系数引起 float64 tanh 数值饱和至 ±1 时，正向 raw 映射也拒绝；逆映射不能恢复已经丢失的位。

## Jacobian、gradient 与 ridge 的准确含义

设 `J_full=d output/dz` 是关于十维 **pre-tanh logits** 的导数。必须在对应点 `z=C u` 求导，才有 `J_constant = J_full C`。H个 packet 使用 `C_H=kron(I_H,C)`，flatten 为 packet-major。梯度列形式为 `g_constant=C_H.T g_full`；本模块接受最后一维为坐标的行形式，因此实际运算为 `g_full @ C_H`。

若已有的是关于 raw command 的 Jacobian `J_raw`，应先乘 `diag(1-tanh(z)²)`，不能直接拿 `J_raw @ C` 当成 logits 导数。`raw_mapping_jacobian` 返回 `diag(1-tanh(z)²) B`；若继续对归一化模型输入求导，还需使用冻结的 mean/std 变换。本模块不引入新归一化统计。

`ridge_step` 解以下问题，`residual` 的符号固定为 **target − current output**：

```text
min_delta ||J delta - residual||² + lambda ||current + delta||²
(J.T J + lambda I) delta = J.T residual - lambda current
```

`current=None` 表示零向量；lambda 是 `damping` 本身，不是平方后的 damping。非零 current 必须用当前 chart 的扁平系数。函数以增广最小二乘求解，不平方条件数；不做 norm cap，调用方应显式径向投影。若约束的是更新后的参数，应投影 `current+delta`，不是仅投影 delta。

`chart_ridge_step` 先做 Jacobian pullback 再解 ridge。constant 解等价于 **限制 full 参数在 range(C)** 的问题，通常不等于“先解 unconstrained full ridge，然后用 C.T 投影”。两 chart 最优解本来可以不同，测试不把这种差异误判为代数错误。球面投影后的方向也不宣称是 trust-region 约束问题的精确最优解。

## 非神经验证

在 `/home/gbk/code/LeWorldModel/le-wm-main` 执行：

```bash
.venv/bin/python -B -m pytest -q -p no:cacheprovider tests/test_pusht_full_packet_chart_20261027.py
```

实际运行结果：**64 passed**。测试覆盖单/多 horizon 正交归一、等距 norm、time-major layout、正逆映射、不修改输入、共享球预算、精确预算边界、sqrt5 与 damping 的旧 constant 尺度换算、非零点 tanh/Jacobian 链式规则、坐标 gradient 有限差分、ridge 正规方程和受限 full 目标梯度，以及非法输入拒绝。随机代数矩阵仅为确定性单元测试，不是神经或物理实验结果。

仍需主线程使用冻结 Q/roots 的 fullpacket compiler 和主机 CUDA 验证真实模型 `Jfull @ C` 对直接 constant J；本模块的数学有限差分不替代该验证。相同候选数不自动构成相同总计算预算，不修改 P1 门，也不声称找到了 P1 根因。
