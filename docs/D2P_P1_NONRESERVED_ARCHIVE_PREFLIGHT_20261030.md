# P1 非保留来源：本地 PushT archive 起点预检

## 结论

本机 `/home/gbk/.stable_worldmodel/datasets/pusht_expert_train.h5` 可只读访问，
大小 46,300,921,856 bytes，包含 18,685 个 episode 和 state7。
但它是训练命名的 archive；本仓库 train/eval 配置均指向相应训练数据，
checkpoint 实际训练血缘和数据 split 等价性尚未认证。**它是可检查的候选材料，
不是已证明训练外、独立或 fresh 的来源**。本地 TwoRoom 属于不同环境，不能直接
承接 PushT 的 state7、动作、渲染与 P1 任务合同。公开检索本次无有效返回，
因此没有认证其他公开来源或公开 holdout；这不是证明不存在。

在读取起点之前，代码固定 episode ordinal 100、101、102、103 四项、不替换。
实际检查 archive schema、相邻 episode 边界、起点 episode/step 编号、有限 state7、
原 float32 state 字节 SHA；逐项比较历史索引 80 条 observed-prefix pose 与批内 pose。
只读取这些指定起点的 state 与对应身份元数据，不读取 action、pixels、goal、奖励、
成功或未来状态，不运行环境或模型，也不访问 P4 保留槽。

结果：100 与其余三项未发现批内 pose 冲突；101、102、103 的 state7 字节完全
相同，三者不能当作三个独立物理根。四项均未命中固定的 80 条历史 prefix pose；
索引不是完整历史，所以这不是全历史排重。原四项分母保留，没有按结果补选。
state7 不包含完整物理求解器隐状态；没有 moving RGB/action history、可复验的
源采集记录或 P1 goal/task 绑定。整个 archive 的字节 SHA 未计算：stat 加起点
slice SHA 不能证明公开版本、采集来源或原 checkpoint 训练排除。

既有实验结果已被研究者观察过，不能把这次事后挑选的区间称为盲态预注册；
本次检查未按模型或物理 outcome 选择 episode。source independence、training
disjoint、global freshness 均为 unknown；P1 admission 和 source run 授权为 false。
要取得独立确认来源，仍需可信的独立采集／持有和训练血缘证明、完整历史范围或
适用的新独立来源合同，以及真实 pre-witness 动态历史、任务与原生 ABI 对齐；
这不是通过多抽 archive episode 就能补足的证据。

## 可复核产物

- 预检代码 `le-wm-main/experiments/spectral_reachability/codex_p1_archive_source_preflight_20261030.py`：
  SHA256 `30c13f3d00f02146a92bbac224cef3fb105dfeb4379c172910275f84b4e34c21`。
- 依赖的 pose 比较代码 `codex_historical_pose_collision_audit_20261030.py`：
  SHA256 `025cff1d52d560276c00ff12b65639ef96f97c682073572932c5d07b335d749a`。
- 独占结果 `le-wm-main/results/codex_p1_archive_source_preflight_20261030_LOCAL01.json`：
  SHA256 `494be8895e26c300c725c3f4144cc204143467d79517b98210a483de085acb06`。
- 合成 fixture 与历史碰撞审计联合测试：18 passed；fixture 不构成科研样本。

从 `le-wm-main` 复现时使用新的输出名，程序以 exclusive-create 拒绝覆盖：

```bash
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 .venv/bin/python -B -m experiments.spectral_reachability.codex_p1_archive_source_preflight_20261030 --output results/codex_p1_archive_source_preflight_20261030_LOCAL02.json
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 .venv/bin/python -B -m pytest -q -p no:cacheprovider tests/test_codex_p1_archive_source_preflight_20261030.py tests/test_codex_historical_pose_collision_audit_20261030.py
```

这次预检仅绑定当时 archive 的 stat 和读取切片；重复运行需重新比较 stat，
即使匹配也不是完整文件内容证明。脚本不能签发来源真实性或执行权限。
