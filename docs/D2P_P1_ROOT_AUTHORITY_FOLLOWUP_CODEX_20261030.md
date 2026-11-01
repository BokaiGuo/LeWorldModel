# P1 根来源认证跟进：未解除来源缺口

本次核查现有 `D2P_P1_SOURCE_AUTHORITY_GAP_20261030.md`、训练 split 审计、
历史 pose 碰撞复核、历史公开来源查询记录，以及 checkpoint README 和当前训练配置。
未打开候选、未开放 holdout 或 outcome；没有采集根、训练、模型推理、GPU 或物理执行。

## 训练来源

checkpoint README 声明数据集 `quentinll/lewm-pusht`，当前训练数据配置命名
`pusht_expert_train.lance`；二者都不能证明冻结 checkpoint 实际训练使用的数据版本或成员。
既有的本地权重 SHA256 为
`48938400ae3464c9680731287f583a9cb516f55a8ec64ea13a91be47fb15b607`。
先前审计记录的模型下载 revision `22b330c28c27ead4bfd1888615af1340e3fe9052`
是模型下载版本，不是训练数据版本。

既有公开来源查询的 report 记录：2026-10-26 指定模型 revision、模型 tree 和数据集
metadata 请求因网络不可达而 unavailable；GitHub 仓库和有限历史 metadata 有成功响应，
但不提供 checkpoint 到训练 run 的绑定。这些网络失败不能解释为资料不存在。
本次未获得新的一手训练 run receipt、不可变训练集 revision、实际 split 成员、
normalizer 拟合值及其 run 绑定。现有本地静态配置也不是原训练已解析配置。
上述权重 SHA 本次引用既有审计，未重新计算。

## 不按 outcome 选择的历史排重

复运行既有 `codex_historical_pose_collision_audit_20261030`，它只读取 SHA256
`ac2544512d033ec14fd07808c3abe2dfd96676bb6b82ec89569f498f58256f98`
的固定历史索引，并重算 state/pose 身份及全部直接碰撞，未按 outcome 选根。
结果：80 条 prefix 记录，3160 对比较，360 对碰撞；8 个连通分量各 10 条，
均为 clique，精确 pose key 数为 8。它们不是 8 个独立采样根的证明；该索引
是已有来源的投影，不是穷尽 reset 的原始账本。

本次复运行该审计及来源缺口检查的测试，共 `19 passed in 0.37s`：

```bash
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' .venv/bin/python -B -m experiments.spectral_reachability.codex_historical_pose_collision_audit_20261030
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' .venv/bin/python -B -m pytest -q -p no:cacheprovider tests/test_codex_historical_pose_collision_audit_20261030.py tests/test_codex_p1_source_authority_gap_20261030.py
```

命令在 `le-wm-main` 中运行；前者仅向 stdout 输出。没有新候选输入，且历史范围与
训练成员 authority 未补齐，因此没有执行候选排除，也没有新增重复的排重实现。

## 仍需的前置

需要训练方提供可独立审验的 checkpoint SHA→实际训练 run、代码和解析配置的链，
训练数据不可变身份、完整 split 与过滤/转换成员、实际 normalizer 及拟合来源；
还需要覆盖十次已知 dispatch 的穷尽 reset 账本或可证明完整的加载闭包、失败和重启、
cutoff 后增量及身份映射，并由原负责人和独立审阅者核查完整性。候选与历史、
训练成员之间还须有可验证身份映射。receipt 或 hash 本身不证明来源真实性与穷尽性。

因此训练外、历史新鲜性及 P1 独立根认证均保持 unknown；本次复核不构成启动许可。
