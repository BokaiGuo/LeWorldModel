# P5 固定164物理—模型保存接线

现有 full causal-roll、H1/H2 共同标签重评分和 164 物理/RGB strict 均已完成；不再重跑它们。新增 saved-only 接线：在固定全部164枝、80个正负对及4个原baseline上，将物理 native 终态、LeWM 实测 goal Q 和 JEPA H1/H2/short 共标签 MSE 并栏。它不是另一项神经或物理实验，也不证明跨模型公平、因果或控制收益。

输入固定四个真实文件 SHA：short comparison `bbafa5ec58dde0f0a12d01aded1f2d49a6a869bfe1c73e0d638eb12a6c57ff24`；full-roll strict `3a20eb780028ddd40a88d11c59b8ada9dcbc0c7147bf1df9aa9acde332aa26fd`；physical strict `8df105747cf6835056d41ca1b1fd80f7f5b4170f7f527ec538213addc4c9d0bb`；LeWM RGB strict `d6b6ed08732d7595e740f98e0dc2450adf439e726aa3078854badb78f7d8b4b2`。所有 source/输入及完整 native IO 验收继承上述封存证据，不重跑 fullstrict。

逐行核 index/root/off-on/task/selection、原 raw10 little-endian float32 bytes SHA、原producer job、model task binding、实际 physical plan10 PNG/pixels/state、native final metrics、完整数据状态。JEPA comparison 对同一 physical strict 的字节绑定及其上游原共同标签/动作来源验证由已有保存分析继承；不是从相同 index 单独推定动作同一。所有80对按4组×两个固定 h×10坐标原顺序核对；baseline [0,41,82,123] 保留，不按成功或误差选数据。当前完整源均全10步；缺帧/early stop 不补，不将不完整来源伪装成此完整 ABI，直接 fail closed。

JEPA 的 MSE 是 256×384 patch 特征的共同 H2 标签误差；LeWM 的 Q 是实际编码192相对 goal192 的距离，两者不同空间，不做差或排名。JEPA 输入还含 agent proprio，LeWM 观察权限不同。输出只拼接 native 原指标与各自指标，不计算统计显著性，不将164枝当独立根。实际已知2根，fresh=0，P1/P5 admission=false；模型推断和物理新增调用均为0。

```bash
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -B -m pytest -q -p no:cacheprovider tests/test_agent_p5_fixed164_physical_model_join_20261029_v1.py
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m experiments.spectral_reachability.agent_p5_fixed164_physical_model_join_20261029_v1 analyze
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m experiments.spectral_reachability.agent_p5_fixed164_physical_model_join_20261029_v1 verify --analysis ACTUAL_SAVED_FILE --sha256 ACTUAL_SHA
```

analyze仅stdout，不自动写文件；verify核文件SHA并重算整份分析。测试使用合成全164字段，不执行模型/环境。受已验收结果的强绑定，本工具不是新的采集输入桥、不是未来新根适配器；任何上游变更须新版本审读。
