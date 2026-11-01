# PushT one-shot suffix 公式与坐标审计

2026-10-26。仅只读检查源码及一次 NumPy 纯代数核验；无 GPU、模型加载、物理执行、源文件修改或准入变更。

## 结论

在**同一固定上下文、同一 goal、同一投影算子、同一零 nominal、同一 horizon、同一参数坐标和 λ**下，`pusht_d2p_metric_program_compiler` → `pusht_d2p_nuisance_program_scoring` 的 one-shot 与 `preflight_pusht_d2p_p1_reconfiguration._reachability` 给出同一个严格凸 ridge 问题的唯一解。没有发现这两条路径因 mean/SSE 或 λ 定义而产生数学不一致。

`consistent_metric_suffix.solve` 并非 one-shot：它取 `_reachability` 的初解，再调用四轮 nonlinear GN 和零动作 incumbent 比较。`full_metric_suffix.refine` 也不是保持原 one-shot 的接口，甚至 `iterations=0` 仍会先在初解与零参数中选择。父线程的 noGN、独立 augmented-lstsq 干预不能用这些返回动作充当同解断言。

将预测前缀上下文替换为真实观察后的上下文，是预期的干预变量；即使公式完全相同，J、nominal terminal、残差与解也可能不同。这不是公式错误证据，更不提供 P1 效能或理论结论。

## 1. 精确公共公式

设固定 RGB/action 上下文为 c，latent 维度 D；suffix horizon H，参数维度 m=2H。

- p∈R^m：unconstrained atanh 参数，不是物理命令或模型 normalized token。
- live command u(p)=tanh(p)。每 token 将一对命令按时间重复五次。
- 设 T 将 2H 个命令坐标按 token 展成 10H 坐标，S=diag(public std 的对应 packet 展开)，μ 为同顺序 mean：模型输入 a(p)=S⁻¹(T tanh(p)−μ)。当前路径不额外除 100。
- f_c(p)：经该 packet、action encoder、冻结 predictor 得到 suffix terminal latent；p₀=0，z₀=f_c(0)，J=∂f_c/∂p|₀，r=g−z₀。
- L：源码名 `projection`，是乘在残差/Jacobian 左侧的算子；A=LJ，b=Lr，λ=0.01。

唯一 one-shot 解：

```text
δ* = argminδ ||b − Aδ||² + λ||δ||²
   = (AᵀA + λI)⁻¹ Aᵀb
   = V diag(σ/(σ²+λ)) Uᵀb,  A=U diag(σ)Vᵀ
p* = p₀ + δ*;    executed live command = tanh(p*)
```

这里 λ 就是 ridge 系数，不是它的平方；增广矩阵底块必须为 sqrt(λ)I=0.1I，而非 0.01I。

若 Q 表示通常意义的二次型权重，则 Q=LᵀL，上式为 `(r−Jδ)ᵀQ(r−Jδ)+λ||δ||²`。不能未经区分把一般 Q 当 L 左乘，否则实际权重变成 QᵀQ。当前 finite-goal nuisance 算子 P=I−NNᵀ 数学上对称幂等，因此 PᵀP=P；对当前 P，两种写法在精确算术下吻合，但有限精度下应直接复用同一保存的 L，不重新投影/重建。random sketch 的 L 是矩形、各行单位化而非正交，不能把 LᵀL 误认为正交投影，也不能与 identity/nuisance P 混用。

## 2. 各源码到底计算什么

| 路径 | 实际优化/数值 | 是否与公共 one-shot 相同 |
|---|---|---|
| `pusht_d2p_metric_program_compiler.py`，约 77–119 行 | 每 prefix 先 `_advance`，detach predicted context；suffix nominal=0；对 unprojected terminal 按 p 求 J；float32 模型数组转 float64 后交 scorer | 是，限定同一预测上下文；该 J 的 “raw” 指未投影 latent，不指 raw-action 导数 |
| `pusht_d2p_nuisance_program_scoring.py`，约 84–104 行 | A=PJ，b=P(g−z₀)，SVD ridge；返回 `nominal+δ`；误差、ridge 都除 D | 是；当前 caller nominal 为零；候选选择另含 stage+βV，不改变单个 suffix 解 |
| `preflight_pusht_d2p_p1_reconfiguration.py:144` `_reachability` | 直接对 Lf(p) 求 Jacobian，同一 L 作用残差；torch SVD 同一 λ；返回 p=δ | 是，精确算术；torch float32 与 NumPy float64 不保证逐位一致 |
| `pusht_d2p_consistent_metric_suffix.py:18` `solve` | H=1、λ=.01 的 `_reachability` 初解；之后以 Lf/Lg 调 `refine(...iterations=4)` | 仅初解相同；最终动作不应要求与 one-shot 相同 |
| `pusht_d2p_full_metric_suffix.py:7` `refine` | nonlinear SSE+λ||p||²，先比较 zero 与 initial，再 GN/backtracking | 不同算法；在 affine f、同 L、p₀=0 的特例，one-shot 已是全局最优 |

### 归一化和值

scorer 保存的 `own_reach_value=(||b−Aδ*||²+λ||δ*||²)/D`。共同除 D 不改变最优解。`_reachability.value` 不除 D，它用下面的相同最小值闭式：

```text
c = Uᵀb
V = ||b−Uc||² + Σ_i [λ/(σ_i²+λ)] c_i²
```

所以在相同输入下，scorer.value = reachability.value / D，而不是两个不同解。薄 SVD 包含零奇异值也不破坏此恒等式。对所有同维候选统一乘常数不改变单独 V 排名；若与未同步缩放的 stage 相加，β 的含义会改变，不能跨路径直接混加。当前 scorer 自己的 stage 与 V 都除 D。

若 parent 使用 `mean residual² + .01||p||²`，就不再相同：乘回 D 后实际 ridge 为 D*.01。正确选择是 SSE+.01||p||²，或 `(SSE+.01||p||²)/D`，或增广时同时将 A、b、sqrt(λ)I 全部除 sqrt(D)。不能只把 A/b 除 sqrt(D)。也不应除以保留 rank 或 action 维度后保持 ridge 不变。

### nonlinear GN 的正则中心

`full_metric_suffix.py:41–42` 的步长是：

```text
Δ = (JpᵀJp + λI)⁻¹ [Jpᵀ(goal−prediction) − λp]
```

若传入 projected terminal，则 Jp 为 L∂f/∂p，goal 为 Lg。`−λp` 正确对应绝对零中心的 ||p||²，不是仅惩罚步长 Δ。array scorer 通用接口在非零 nominal 时惩罚 ||δ||²，因此与绝对零 ridge 通常不同；当前 compiler nominal=0，不构成当前 bug。refine 的返回 mean terminal cost 不含 ridge，其接受判据含 ridge；不能拿该诊断字段冒充优化 objective。

## 3. 已确认的差异与非 bug 边界

1. 原 preflight `compile_reconfiguration` 使用 random sketch reachability V，但 immediate stage 用 raw latent MSE；`_reachability` 的 `predicted_goal_cost`、`bounded_suffix_goal_cost` 也都是 raw latent MSE。这些字段与 projected V 不是同一度量。consistent wrapper 已明确标记 legacy raw 字段不用于选择，单独重算 projected cost；不能从字段不一致推导其 one-shot 初解错误。
2. `audit_pusht_d2p_full_metric_bank.py:35–46` 的 legacy random8 模式使用 sketch one-shot，但其报告的 `predicted_goal_cost` 是 raw latent MSE；该模式不进入 GN。只有第三种模式进入 full-latent nonlinear refine，且它使用 identity 初解，因此不能声称此 caller 实际混用了 sketch 初解与 full-metric GN。random8、full192 identity 与 consistent wrapper 要分别识别。
3. scorer 的一次线性解不因后续 nonlinear forecast 而重排；consistent/full refine 则显式评估 nonlinear objective、保留 zero、GN line search，动作不同合理。
4. J 已包含 tanh、五次 packet 重复、std normalization、action encoder 和 predictor 的链式导数。对保存的参数 J 再除 std、乘五或除100会重复缩放。若从 normalized-token 导数重新构造，才需 `Jp=Ja S⁻¹ T diag(sech²(p₀))`。p₀=0 时 tanh 导数为1，不等于可以删除 S⁻¹T。
5. 换坐标必须同步正则。若局部变量 δx=Cδp，目标中的 ridge 应变为 λ δxᵀC⁻ᵀC⁻¹δx；对所有坐标一律 λI 通常不保解。用十维 packet 独立控制代替两维共享重复参数也改变可行子空间。以 tanh(p*) 执行不是截断线性解；改成 clip(p*) 或直接执行 normalized token 都不等价。
6. public mean/std 的训练溯源是否精确，不由公式审计认证。各路径调用相同转换只能证明内部坐标一致，不能证明归一化常数是 checkpoint 真值或物理任务适用。

## 4. parent 如何保证“same solve”

parent 的 observed-prefix、H=1、sameQ λ=.01、noGN 干预建议固定以下可核验合同，不复用 refine：

```python
# z0, J: SAME observed-prefix context, same terminal function at p0=[0,0].
# J differentiates raw latent output with respect to atanh suffix params.
# L: EXACT saved left operator, not a newly fitted metric.
A = np.asarray(L, dtype=np.float64) @ np.asarray(J, dtype=np.float64)
b = np.asarray(L, dtype=np.float64) @ (
    np.asarray(goal, dtype=np.float64) - np.asarray(z0, dtype=np.float64))
m = J.shape[1]  # 2 for horizon=1
p, *_ = np.linalg.lstsq(
    np.vstack([A, np.sqrt(0.01)*np.eye(m)]),
    np.concatenate([b, np.zeros(m)]), rcond=None)
live = np.tanh(p.reshape(1, 2))
```

保存并绑定 root/candidate、observed context RGB/actions、goal、L、z₀、J、p₀、H、λ、参数排列与 dtype；校验 J shape=[D,2]、所有数组有限、L 与目标合同不变。不要重新从 observed 后果拟合 L，不调整 λ 或按物理效果重选候选。相同公式不要求 observed 与 imagined J 数值相同。

独立检查 SVD 解、增广 lstsq 解及正规方程残差 `||(AᵀA+λI)p−Aᵀb||`；比较 linear error、ridge 和 V 的一致性，明确输出 SSE 与 mean 的单位。float64 两种独立解应在按尺度/条件数声明的容差内一致，不预设全部 float32 neural 解逐位相同。记录 float64 解、实际 cast 为模型/执行参数的值与最后 live command；compiler 现有调用会先把 float64 p cast float32 再 tanh，parent 若用 NumPy float64 tanh 后再 cast，可能产生末位差异，需声明实际顺序才能要求执行命令严格匹配。

不做 GN、不做 zero-incumbent nonlinear 择优、不做 clipping/学习/额外约束，不以真实后缀结果更改解。该干预是观察上下文替换后的同一优化规则，不是当前 sealed imagined 解的复现，更不自动开放 P1。

## 5. 本轮实际纯代数核验

使用 `le-wm-main/.venv/bin/python`、NumPy，seed=20261026，D=12、m=2、rank-3 nuisance span，λ=.01；不读取模型或实验大数组。结果：

```text
SVD vs augmented-lstsq max abs       3.3306690738754696e-16
common /D normalization max abs      3.3306690738754696e-16
reach closed-form vs primal abs     4.440892098500626e-16
mean-only residual scaling mismatch 0.021394361245817395
```

最后一项是故意构造的错误归一化对照，不是现有源码观测到的 bug。该核验仅支持线性代数推导，不验证模型 Jacobian、任务度量、物理效力或 P1。
