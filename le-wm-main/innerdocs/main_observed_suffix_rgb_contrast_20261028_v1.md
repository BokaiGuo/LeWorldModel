# MAIN：同真实中点、同时间的新旧动作三层分解

生产源 `experiments/spectral_reachability/main_observed_suffix_rgb_contrast_20261028_v1.py`。
新terminal RGB全strict重新运行并比较持久receipt的全部字节，所有原模型、
任务和物理来源由完整闭包认证；旧终态192再从原gzip NPY读取并核file/array
SHA、shape/dtype，不把旧预测当真实编码。保留全部固定四条件（仅两个known根）。

对照是同一个 actual cp5 history 下的原suffix预测（compiler iterate0）与
新suffix预测，不拿旧imagined-prefix预测混入此干预。实际新旧终态均为各自
绑定raw10执行的step10图像，原prefix5及cp5 state/snapshot/PNG必须精确相同。
genuine native stop缺新cp10则该槽保留missing，不补图/步、不筛掉该槽。

复用已独立验证的带符号恒等式：令e0=p0−a0、da=a1−a0、de=(p1−p0)−da，
则 ΔQpred−ΔQactual = mean(2 da·e0 +2(a1−g)·de +2e0·de +de²)。
保留四项符号、相消、实际响应误差和严格共同偏移是否成立；不做残差拟合。
同时独立重算完整原native7任务指标，不以latent距离下降冒充任务成功。

六项纯RAM回归通过；另列上游真实CUDA与物理证据。该工具自身无神经/物理，
不改模型、Q、goal、根或成功阈值。科学门与freshness不从工程验收自动放行。

实际完整重建/执行结果 `results/main_observed_suffix_rgb_contrast_20261028_v1_MAIN01.json`，
SHA `55ee1bef2ef77b16fe2153a4d2eecfa494c909da3b82ef2c514bfd56eac24d22`。
root0两个条件预测→真实编码排序反转；root1两个条件编码Q改善但角度损害。
四对均保留相消、无精确共同offset。此为两个known根的诊断，不是fresh确认。
