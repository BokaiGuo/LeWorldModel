# MAIN：真实/滚动历史的成对目标几何独立复算

这是已知输出后的描述性分析，不是新的预注册实验、独立根或P1 gate。只读取已完成8A主机CUDA的report、完整consume及scores，按真实SHA绑定；全8端点及4个同root/time的endpoints−midpoints比较均保留，0新神经/物理调用。不导入实验producer数学或NumPy，以Python math.fsum独立计算192维identity meanQ、MSE及cross term。

恒等式为 `Q(pred,g)-Q(actual,g)=mean(error²)+2mean(error*(actual-g))`。同根同时间比较的margin误差再分解成两项之差。4096*eps64尺度界只用于核对已保存规约值及恒等式，不作排名阈值或科学门；每个margin的符号还必须与原保存值严格同号，零也不删除。真实标签与goal、实际A预测对已认证CUDA输出逐项join；加载/输入因果/原生IO仍依赖绑定的MAIN完整strict。

MAIN实际12项回归通过（0.17秒）；实际MAIN02 analyze与whole verify成功，SHA `1c7bafae7d57bb146f40128f96ded293009644a70a97eb03faa03b3615955a62`。MAIN01为增加paired-goal与reduction-sign guard前的历史版本，保留不覆盖；当前复现以MAIN02为准。

|root / endpoint|真实编码Q margin|A真实历史预测margin|R滚动历史预测margin|
|---|---:|---:|---:|
|0 / 15|+0.08797557|+0.05766481|−0.02060607|
|0 / 20|+0.10261299|+0.11182847|−0.01244879|
|1 / 15|+0.02433241|+0.01713730|+0.02084729|
|1 / 20|−0.06466051|−0.01063860|−0.01862482|

A四对均同号，R两对反转；两次反转均在root0。R对应MSE差+0.12660303/+0.14318128，被cross差−0.23518466/−0.25824305覆盖，导致margin错误；A的两项合计−0.03031075/+0.00921548未改变该两对排序。八个端点的误差平方项与cross相消计数R6/A2，但这不是相消越少越好：root1 endpoint20的A margin误差反而大于R。原8A的目标绝对误差也仅5/8改善。因此不能把平均MSE下降或4/4局部排序当作统一偏移修复、普遍可靠性或物理收益。

这组证据支持下一实际闭环使用正确时间的真实观察重新编译，并继续检查模型排序到真实编码、真实编码到任务指标两层。原task warmup末端control0与actual cp5不同，后续必须使用真实cp5的绝对5/10/15完整history，不把旧task换名。所有P1/P3准入保持false。

```bash
OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m pytest -q tests/test_main_p3_history_pair_geometry_20261029_v1.py
OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m experiments.spectral_reachability.main_p3_history_pair_geometry_20261029_v1 verify --output results/main_p3_history_pair_geometry_20261029_v1_MAIN02.json --sha 1c7bafae7d57bb146f40128f96ded293009644a70a97eb03faa03b3615955a62
```
