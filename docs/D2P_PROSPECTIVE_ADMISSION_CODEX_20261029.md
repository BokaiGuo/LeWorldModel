# D2P 前瞻来源：已知历史排重与保守阻断

本文件与同 stem 脚本、测试、结果独占新增；不改现有协议、producer 或封存资产，不运行 GPU、环境或模型，不按结果筛根。

## 现状与结论

已有固定四槽前瞻协议和 `agent_p4_fresh_source_20261028_v2` producer 实现；不是已经生成的四个物理根。原 fresh_candidate prepare 要求外部历史证书，该 producer 的另一种 calibration_only 模式不构成 freshness，且其交付说明禁止在这四个保留槽上运行 calibration。现有四槽实际 root identity 为 null，故不能用 seed 或新目录证明新根；也不能把此筛查当作执行许可。

本次读取固定预注册、历史排重索引与原 root manifest：四对 seed 在这些输入的已记录 seed、半开保留区间和 A–H 条件闭区间中均无命中。此结论仅关于这三个固定输入的已知范围，不是全历史无碰撞、统计独立或原 reset 尝试闭包。四槽全部 `blocked_no_actual_identity`；历史范围、原尝试完整性、前 witness 真实身份及其来源均未获认证。

## 保守筛查规则

新增脚本将预注册四槽及种子固定，分母始终4；拒绝含 outcome、score 等额外字段、seed 错配、非有限 state、缺字段及不合规 SHA 的输入。可接受外部 SHA 固定的候选身份声明作为筛查数据；该文件不因有 SHA 就证明采集过程、时间顺序、记录真实性或实际来源。不能用该接口创造认证候选。

对已有候选按原规则筛查已知历史和批内 pose（XY 每维 `1e-9`、环绕角 `1e-12`）、历史 snapshot/RGB3、root-goal 声明及 batch snapshot/RGB3；已知 seed 与 reserved 半开区间、条件闭区间均按 root/reset 两个命名空间交叉检查。碰撞保留每槽及原分母，双方均标记，不补根。若未命中，只输出 `no_known_collision_unverified`，不能称通过；所有槽的 `admitted`、source/witness 权限与 freshness 固定为 false。prefix 只覆盖历史 observed pose，不冒称完整 root/snapshot/RGB3 集。此前条件包络仍不认证原执行范围。

真正可授权的 source 必须另取得符合原协议的外部历史范围与 reset 证据、source producer/环境真实闭包、原始持久化 attempt 账本、witness 之前实际 state7/snapshot/RGB3/raw10 及其可信溯源，并由已有历史证书入口和责任方审查；不能将本脚本输出作为其替代。实际采集与其权限不由本脚本签发。

## 执行与验证

工作目录 `le-wm-main`：

```bash
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -B -m experiments.spectral_reachability.codex_d2p_prospective_admission_20261029 --output results/codex_d2p_prospective_admission_20261029_LOCAL02.json
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -B -m pytest -q -p no:cacheprovider tests/test_codex_d2p_prospective_admission_20261029.py tests/test_codex_d2p_historical_exclusion_index_20261029.py
```

已生成 `results/codex_d2p_prospective_admission_20261029_LOCAL01.json`，SHA256 `9d2821c8fcc78974589b802e28970d5d26777ba2eec7e578e0623c625a1fdc4c`；四槽均阻断。联合 11 tests passed，包括真实空候选、合成已知 pose/角度别名、批内碰撞、条件 seed 包络、结果字段/seed/NaN 拒绝，以及有身份但无已知碰撞仍不准入。测试身份只为软件验证，不是实际采集。输出 exclusive create；不会覆盖旧产物。
