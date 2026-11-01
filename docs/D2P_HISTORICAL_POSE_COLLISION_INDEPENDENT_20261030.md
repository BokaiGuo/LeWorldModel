# 历史 prefix pose 独立碰撞复核

本次只读取固定的历史排重索引 `codex_d2p_historical_exclusion_index_20261029_LOCAL01.json`，
校验完整文件 SHA，并独立按 little-endian float64 重新计算 80 条 state 与 pose 身份；
没有打开原结果、候选、保留槽或 outcome，也没有运行环境、模型或 GPU。
该索引是已有来源的投影，不是原采样根的完整 ledger。

以四维坐标逐维 `1e-9`、环绕角 `1e-12` 做全部 3160 次直接比较：
360 对碰撞，8 个连通分量，各 10 条；每个分量都是两两碰撞的 clique。
8 个精确 pose key 与 8 个连通分量在本快照中数量相同，但并不证明是 8 个
独立采样根。一般容差关系不具有传递性，代码单独报告非 clique 分量，
不能把连通性当两两相等。

独占脚本为 `le-wm-main/experiments/spectral_reachability/codex_historical_pose_collision_audit_20261030.py`；
测试为 `le-wm-main/tests/test_codex_historical_pose_collision_audit_20261030.py`。
在 `le-wm-main` 运行：

```bash
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' .venv/bin/python -B -m experiments.spectral_reachability.codex_historical_pose_collision_audit_20261030
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' .venv/bin/python -B -m pytest -q -p no:cacheprovider tests/test_codex_historical_pose_collision_audit_20261030.py
```

这只是固定索引内部的独立一致性和已知碰撞复核，不重新核验原始证据文件、
原加载闭包或历史范围。已知 8 来源的 reset 完整性及历史 scope 仍需原运行证据和
责任方裁定；没有据此认证候选、签发 source 权限或声称 fresh。
