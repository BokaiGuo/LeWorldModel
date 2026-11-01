# P5 固定80对：native/Q方向与JEPA内部误差

固定输入为 MAIN 已保存并验证的 `agent_p5_fixed164_physical_model_persist_20261029_v1_MAIN01.json`，SHA `25cb6846263027b1ceedd39b9060c3158c6cc6a095d323b012be30728e7511c3`。仅重新计算该文件标量；继承其上游strict及关联证据，不重跑fullstrict、模型或物理。保存本分析自身source/tests/doc SHA；验证重建整个输出字节，不是第二套独立数学实现。

固定所有80对（4组×2个h×10坐标），164枝和4 baseline不改；每对按原正负动作索引算 **plus−minus**。分别计算native距离、object XY误差、wrapped角误差各自的差，与LeWM **actual编码的goal Q** 差的符号相同/相反/双方零/单方零/缺失。这里Q不是模型预测的动作效果，native_distance也不是可直接换算的LeWM距离；只比较方向，不做数值相减，不合并成因果或控制收益。符号用精确0，无自定epsilon；null传播，全部80分母及四组各20保留。

JEPA用同一模型及同时间的共同标签MSE：逐端点记录 H2_20−H1_20、short_20−H1_20、short_20−H2_20 和 short_15−H2_15；并分别记录每个预测的plus−minus MSE。不同历史和启动长度仍有协议差异，不称纯horizon效应；15/20标签batch差不能当时间效应。没有把JEPA与LeWM数值相减、没有根据结果筛选。

known roots=2、fresh=0；分支/成对并非独立根。描述性计数不作显著性、置信区间或P1/P5准入；本口径在结果已存在后提出，不伪称预注册。0新NN/physics。source缺失或原完整join不满足则拒绝，不以幸存子集冒充80；字段null保留缺失。

```bash
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -B -m pytest -q -p no:cacheprovider tests/test_agent_p5_fixed80_error_decomposition_20261029_v1.py
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m experiments.spectral_reachability.agent_p5_fixed80_error_decomposition_20261029_v1 analyze
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m experiments.spectral_reachability.agent_p5_fixed80_error_decomposition_20261029_v1 write --output results/agent_p5_fixed80_error_decomposition_20261029_v1_MAIN01.json
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m experiments.spectral_reachability.agent_p5_fixed80_error_decomposition_20261029_v1 verify --analysis results/agent_p5_fixed80_error_decomposition_20261029_v1_MAIN01.json --sha256 ACTUAL_SHA
```

write仅本stem results新文件、排他创建、文件/目录fsync、8MiB上限与200MiB余量；故障可能留下partial，绝不覆盖。verify要求实际SHA及完整字节重算；本轮不自行持久生产分析，MAIN审读后可执行。
