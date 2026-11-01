# 已保存 teacher fidelity 误差分解 v1

仅新增本 stem 的 source/tests/innerdoc；旧 teacher v1/v2 与共享 progress/dispatch/closure 不修改。
本模块仅 Python 标准库与 NumPy，不 import 原实验执行器，不 load 模型，不执行神经网络、physics、训练、adapter 或 Q 拟合。

## 固定输入与重算范围

输入 `results/p1_moving_live_feedback_teacher_fidelity_cuda_20261027_v2_cuda_main01/report.json`，
固定 SHA256 `7323347a9e74c8f1751550642503426d07ddce55549ddaef4331f3c362567128`。
要求完成的 CUDA teacher、固定 12 episode、两个任务、四层×O/I/R；不选子集、不换根、不做显著性。
独立解码并验证原 prepare envelope/semantic SHA，hash 全部 prepare bindings（含旧源码/模型，仅读 bytes），
核完整 teacher artifact 文件集合与 SHA、prediction seal、事件 hash chain、先 seal 后评分、48 条 module ledger。
逐 episode 核 input/identity、AE→predictor→pred_proj 保存数组桥接、最后 token 与 prediction 逐位一致、
本 episode encode5/encode10 原 NPY receipt 与 label/prefix 逐位一致；独立重算原五项 metrics，必须精确相等。
全部被消费文件记录文件 SHA；所有 teacher NPY 和使用的原 factual NPY 记录 dtype/shape、原始 C-order 数组 SHA。
goal 是 prepare 内 literal float32 数组，单独记录数组 SHA、容器 SHA 与 JSON pointer。
输出前再次核所有消费文件 SHA。缺失、NaN、错形状、错身份、错 SHA、部分完成一律失败，无 0 替代。

## 时间轴与指标

原输入三 token 为 `[source_z5, source_z10, own_actual_prefix5]`，动作对应 warmup 末5、实际 prefix5、实际 suffix5。
原 `jepa.py:predict` 对每 token 做一步 next-embedding 投影。保存的三投影分别比较：

- token0→source_z10：保存的源 warmup anchor，不称本 episode 自身 factual 标签；
- token1→own actual prefix5：上下文内 teacher-forced 一步；
- token2→own actual terminal10：本 episode suffix 的真正 H1，实际 5→10、五个 physics steps。

`prediction_steps` 逐 token 列 latent MSE 和目标/pred 数组 SHA，三 token 不是三步自由 rollout。
物理中间步6/7/8/9没有 latent 标签，不补造逐 physics-step 误差；真正 endpoint 科学比较仅 token2。

原 Q 为 cost：`Q(z)=sum_j(z_j-goal_j)^2/192`，所有差值先转 float64，再运算。
定义 `e=pred-actual, r=actual-goal`，精确恒等式：

`Q(pred)-Q(actual) = [2<r,e> + ||e||²]/192`。

同时输出未归一化 cross/square、各自除192的两项、重构和浮点 residual/tolerance。
`||e||²/192` 即 latent MSE。cross 可负，不截断；无余弦归一化、数据拟合尺度或任意 epsilon 分母。
另输出 absolute/squared Q error；不计算会在 actual_Q=0 时无定义的相对误差。

四层 R-O 按 `(source_slot,prefix_arm,mode)` 显式配对，核 task/selection/prefix/goal；I 仍全量评分。
margin=`Q_R-Q_O`，负数偏好 R。pred/actual 异号才 strict reversal，恰好零为 tie 而非反转。
margin error 进一步分解为 R-O 的 normalized cross 差与 squared-error 差。
不硬编码预期3/4反转，不用报告里的已聚合值代替计算，不以层数冒充独立任务数。

## 运行与测试

cwd `/home/gbk/code/LeWorldModel/le-wm-main`：

```bash
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -B -m pytest -q -p no:cacheprovider tests/test_p1_teacher_fidelity_error_decomposition_20261027_v1.py
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -B -m experiments.spectral_reachability.p1_teacher_fidelity_error_decomposition_20261027_v1 --output results/p1_teacher_fidelity_error_decomposition_20261027_v1_saved_main01.json
```

输出要求新独占路径（`xb`），不覆盖旧文件；任何输入失败均不发布成功报告。
测试为 RAM mock/NumPy 算术，含完整12行、SHA/缺失/污染/部分完成拒绝、负cross、零Q、tie、配对顺序和源码import边界。
测试通过不是科学证据。真实 saved-array 分析也不是新实验或新控制收益；不支持物理改善、泛化、P1/P4 admission。
任何未来真实实验须 MAIN 审核后 HOST CUDA，本分析不启动该流程。

## 本次实际 saved-array 结果

上述独占测试文件实际运行：`29 passed`。测试全为 RAM mock/已保存数组算术，不是科学验证。

已成功生成上述 `..._saved_main01.json`，报告 SHA256
`483c40c33eb856c45b380e7d0892e555784236d8e7584c198b3881098c16a0e7`。
绑定798个输入文件 SHA、132个实际消费 NPY 的数组 SHA，12行和36个带标签作用域的投影 token MSE 全量保留。
endpoint Q 分解最大绝对浮点恒等式 residual 为 `1.5265566588595902e-16`。

| 层 | predicted Q_R-Q_O | actual Q_R-Q_O | strict reversal |
| --- | ---: | ---: | --- |
| 0 | -0.11464941383908783 | 0.08337655337607058 | 是 |
| 1 | -0.13922638198214798 | -0.14321710547049737 | 否 |
| 2 | 0.045246326523013636 | -0.04219944671806286 | 是 |
| 3 | 0.03922147667041345 | -0.06992661996426389 | 是 |

这是独立数组复算出的3/4（层0、2、3），不是测试期望值或程序输入。
该结果仅解释 latent teacher 误差如何改变 latent-Q 排序，不提供物理收益证据。
