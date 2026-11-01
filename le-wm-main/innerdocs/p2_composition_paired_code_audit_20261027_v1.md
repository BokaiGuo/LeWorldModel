# Paired v2 负结果：代码根因与实现合同审计 v1

独占只读审计，不是一般结果 consumer，不改任何 paired v2 planner、physical、
测试、协议、封存或结果。输入限定为已经执行的四个 calibration roots：
results/p2_composition_paired_20261027_v2_prepared01 与
results/p2_composition_paired_physical_20261027_v2_main01。
没有新 neural forward、AD、有限差分、编码、physics、训练或方法选择。
experiment-code 技能仅用于可复算代码、记录与测试；不采用其训练/绘图流程，
因为用户明确禁止新实验并要求合同审计。

## 审计问题与代码链

- planner.suffix_solution：残差 Q@(goal0 - nominal)，其中 nominal 是该 prefix
  完整预测上下文下执行 zero suffix 后的终点，不是 prefix 本身的 latent。
  192x10 Jacobian 对 suffix 的 pre-tanh logits 在零点求导；tanh'(0)=1。
  prefix 固定，没有错误地把它当作 suffix 自由变量。
- 全20 logits 的半径 sqrt(10)*.75，suffix 半径为剩余欧氏球半径。
  不是各 packet 独立球，也不是 raw box 的均匀采样。先 logit 求解，
  后 float32 tanh，时序排列 [2,5,2]；没有额外 clipping。
- compiler.bounded_ridge 的目标是
  (||QJ*x - Q(goal-nominal)||² + .002||x||²)/192。
  linear cost、ridge penalty 同除192；这不是把 damping 偷乘192，
  也不是对 latent 再除192。独立 KKT 检查不调用原 suffix_solution：
  Kᵀ(Kx-r)+(.002+lambda)x=0，加上半径/互补条件。
  zero cap 单独检查 singleton，不要求不存在的 multiplier。
- compiler.make_transition 与 RootAnchoredCudaReplay/CudaReplay：
  3 latent boundaries 对2个 between-state normalized action tokens。
  新5x2 raw packet 按 time-major flatten 后使用 public mean/std 逐坐标归一化；
  追加成3个 action tokens 输入模型，然后 latent/action histories 均 shift。
  本审计逐160个 composition 候选的两条 edge 验证 retained histories；
  joint 的 whole-H2 见证检查 retained root boundary 和最终两个 action tokens。
- 原 Q 筛选是 mean((Q@(terminal-goal0))²)，不使用 prefix score、
  ridge penalty 或线性预测替代原 nonlinear terminal score。
  全40候选 stable first argmin，见证与所选10个 raw commands、
  物理 trace、RGB journal、真实 terminal encoding 按 root SHA + arm 绑定。
  独立 full/segmented winner replay 的已有严格相等标记由原 exporter 复核，
  本审计不把 saved-context lookup 冒称新神经 replay。
- RgbObservabilityGuard.dispatch 返回 unchanged command.copy()。
  被物理协议封存的 stable_worldmodel/envs/pusht/env.py 默认 relative=True，
  action_scale=100；step 设置 target=agent.position + raw*100。
  window clip 行被注释，没有执行窗口裁剪。PD dynamics/碰撞不是“动作篡改”。
  本审计不创建 env，不声称额外执行证明。public action normalization
  的归一化一致性不等价于独立证明 checkpoint 训练分布正确。

原 source/current/archive/args、protocol bindings、shared bank 与所有 local solve
由已有 sealed physical.load_prepared -> inputs -> planner.selected_actions 只读验证；
另用独立 NumPy KKT、Q score、rolling history 检查以及 durable branch artifact
核对。所有输入 SHA 在运行结束再次确认。额外 exporter SVD/重求解与本审计
数组运算不属于原 CUDA ledger，不伪装成零成本；新 neural/physics 为零。
源源码封存保证所审阅实现不漂移，不独立证明 saved Jacobian/nominal 真值。

## 独立复算及负结果解释

每个改善均定义为 joint minus composition，正数为 composition 更好。
下表只用于区分根因层级，不是重复全指标结果总表；Q 与 pose 的数值
不可跨单位比较，只比较排序符号。

| root index | 模型 Q 改善 | 真实终点编码 Q 改善 | 真实 pose 改善 |
| --- | ---: | ---: | ---: |
| 0 | +0.0686526744 | -0.0632224211 | +70.7171296703 |
| 1 | +0.3417204420 | +0.3526933327 | -36.5143640828 |
| 2 | +0.1326648822 | -0.0194398917 | -15.7839015776 |
| 3 | +0.1325948820 | +0.2671151445 | -11.0200935146 |

root0/2：预测 Q 与真实 encoded Q 的 selected-arm 排序反转，
是已观察到的 model-to-real objective transfer gap。root0 的 pose 仍改善，
说明 encoded-Q 排序反转也不等价于 pose 一定变差。
root1/3：真实 encoded Q 已改善，pose 却恶化，因此不能把这两根全部归因
于终点编码预测错误；原 latent-Q 与实际 pose 目标失配在这些配对上可见。
这些是所选见证的可观察排序差异，不是某个内部机制的因果证明。

逐候选保存 nominal cost、线性 suffix cost、非线性 suffix cost、
Q 投影后的线性近似余项、tanh 与 logit 偏离、KKT residual、
相同 prefix 的 joint 模型对照。相对 zero suffix 的 nonlinear 模型 cost
恶化候选数分别为0/10/13/4（每根40）。这是有限步局部线性模型没有
非线性下降保证的具体证据，不是选错 winner；原 runner 已重新计算所有
nonlinear costs 后选最小值。不能因打开这些结果就改半径、damping 或加
事后 fallback，尤其不能追认旧 sealed report。

conditional suffix 在预测的 prefix context 而非真实中间观测上编译；
整条物理路径 open-loop 执行10步，没有反馈中途重新编码。这是已声明实现，
不是 rolling-context bug。未执行的候选没有实际终点编码，因此不能声称
本审计找到了物理最优候选、candidate-wide 真实排名或 re-selection 收益。
无新 AD/FD，无法独立证明 derivative 真实值或定位误差发生在哪次预测。

## 结论与版本边界

本轮没有发现可复现的 residual/坐标/动作裁剪/history/Q归一化/
所选见证实现错误。拒绝把“表现差”直接升级成可修 code bug；
不提出新 planner 版本，不改目标或阈值，不利用已打开物理结果调方法。
这里“未发现”不是形式化正确性证明：独立 derivative、nominal 神经真值、
未选候选真实结果均不在本轮证据内。

paired unique 成本每根252 predict +252 action encoder；shared4，
joint exclusive84，composition exclusive164；独立使用归因88/168不可相加。
每根41 Jacobian API calls、7872 scalar-output VJP directions，不冒称
测量的 backward engine API 次数；具体 SVD/solve/resource receipts 原样保留。
不等总预算，不证明 P1/P2，不开启科学准入。

## 非神经复现命令

在 /home/gbk/code/LeWorldModel/le-wm-main：

~~~bash
OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m pytest -q -p no:cacheprovider tests/test_p2_composition_paired_code_audit_20261027_v1.py
OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m experiments.spectral_reachability.p2_composition_paired_code_audit_20261027_v1 --output results/p2_composition_paired_code_audit_20261027_v1_audit01
~~~

CLI 拒绝已存在输出；复现必须换全新独占路径，不重跑或覆盖旧结果。
新 output 自封存 module/test/doc 与直接审计依赖，并绑定所有只读输入。

