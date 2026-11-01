# 强基线 protocol bridge：可执行交接，而非新增效果表

Jason 独占 source/tests/本文。只保存 metadata、hash 与 manifest；不下载、解 pickle、实例化模型、运行神经/物理或打开保留根。不修改任何既有封存或共享表。使用 experiment-code 的固定输入、日志和可复现入口要求；本任务不采用其泛用训练/挑最好结果/零成功即修改规则。negative/unknown 原样保留。

## 新增的可执行功能

现有 `agent_baseline_launch_readiness_20261028_v1` 已能核三个本地工作流及生成 verify/consume 命令；本实现直接复用它（source SHA `6056d61e1463a13db6a20d28467196558d1ad92de4790c25c2f8f08d979bb042`），不再复制扫描器或刷新总表。

1. `check/prepare` 读取实际 frozen task/prepare/模型与报告绑定，提取真实 RGB3 context、两 incoming action packets、非零 raw 历史、完整 goal 与 identity192 Q 指纹。要求各工作流使用同两 task_id/exportSHA/blindSHA；固定 H2/10raw/off 目标、packet logit cap 与原 normalizer。生成可以外部 SHA 冻结的 execution handoff manifest，含完整原算法合同、各自实际 resource 收据和既有 strict verify/consume argv，而不是纸面清单。
2. `verify` 重建全输入闭包与 manifest 精确比较，文件变更拒绝；不重新规划或编码。未调用 own strict consumer 的状态明确为 false，不把 metadata 通过伪称完整结果通过。
3. `request/intake` 提供四方法缺失资产的固定 role schema；新材料到达后可直接校验完整角色、绝对路径、外部 SHA、非临时常规文件和读取期间不变。模板 null 为真正阻断，不能用 `externally_reviewed_complete` 自我声明代替。

仅交接与 intake，无 subprocess/shell 执行器、模型运行子命令。`main_saved_commands` 是 MAIN 可执行的既存入口；prepare 不授权 run。工作流顺序和作者 cell 事前硬固定，不读取 outcome 来选臂。4方法分母始终保留；Hi-LeWM/PlanWAM unavailable 不删，也不填0。

## 现有可用同协议局部对照

固定原 off moving 两根，task SHA `a42bfbb1ea692782355e2572a5df2cd048b72cf3e6c41fee40eb12503674959b` / `101add21a7a40591167a6ab8e8b22c3c62cb1fd700c741fd36cd2073e2d5df74`，来自 `results/p1_constructive_moving_history_encode_cuda_20261027_v1_cuda_main01/task_00/01.json.gz`。实际前五 raw float32 [.1,0]，后五 [0,.1]；不是 zero padding。冻结模型 weights/config 路径及 SHA 沿 task_contract 和原工作流 bindings；不重拟合 normalizer，其 training provenance unknown 不升级。

可直接消费的工作流与 scope：

|工作流|既存 module|实际资源轴 / 比较边界|
|---|---|---|
|moving_grasp|`p1_moving_grasp_baseline_cuda_20261027_v1`|3604 predict/AE、1800 scalar backward；保留 virtual-state/soft-dynamics/sync，不改称普通 GD；非原官方 DINO-WM 表级复现|
|matched_sampling|`moving_prediction_matched_sampling_cuda_20261027_v1`|CEM 与 local nominal-MPPI 各144 API/samples/root；整次576 predictor/AE/proj；只有这两方法同 forward 轴，非官方全 MPPI|
|moving_gradient|`p1_moving_gradient_baseline_cuda_20261027_v1`|272 predict/AE、64 scalar reverse；open-loop，非闭环 MPC，不冒充 LeWM 原表全部 solver|

同 task identity 不等于相同 search support，不等于 AD/FLOPs/wall 公平。selected action-domain 仍交每方法 ownconsume 原生验证，不用 metadata 反推已执行动作的可行性。各报告完整 contract 与 resources 原样进入 manifest，继承 acquisition/physical/RGB 成本不能当0；没有独立测量则不造总 FLOPs。

native success 包含 agent+object 前四维 XY 和原角度判据；native distance 保留全部7维（含velocity）。RGB192 Q 不是任务成功，object-only 投影也不能替代完整任务判据。桥不产生新成功率或方法排序，不覆盖 MAIN 刚完成的 E1 RGB/consume。

## 官方缺失材料：精确阻断与可用接点

- GRASP：官方 solver 文本 commit 已在原 contract；本地 bounded 变体有真实保存结果。仍需逐表 checkpoint/config、原 data split/normalizer、完整预算与许可映射，才能谈 paper-native。保留 virtual-state/soft dynamics，不把普通 GD 改名补齐。
- Hi-LeWM：现有 `results/p1_hilewm_text_contract_20261027_v1/contract.json` SHA `c25edf542f38faa9cf0150df4822198d28753e2cf77a0d243e990ffabe849ded`，33份真实源码/config 文本支持 `rollout_high -> first predicted subgoal -> low solver`，但不代表权重已到。本地 terminal archive report SHA `19c526da4a8e3da6b340addd23289df2bd3a57e5a18169290f9bd13dd4f0783b` 为 failed_preserved。需要完整可信 `package/checkpoints/pusht/main/pusht_hi_lewm_epoch15_object.ckpt`（122424517 bytes，CRC32 d59f865b，实物SHA unknown），依赖/许可、内嵌 low identity、冻结数据统计和正式 hierarchical protocol。原 `agent_hilewm_asset_verify_20261028_v1` 可对 MAIN 交付完整归档做无 pickle 验收；本桥不重新下载。base variant 不无端要求 -C bank；若以后运行 -C，必须另加经验bank/indices/encoder/data证据，不能把本请求改标签。
- LeWM：本地 checkpoint 已可用于上述同任务变体；原 `eval.py` 会 fit scaler 且有 CPU fallback，不在本轮允许运行。官方 Table10 各 solver 的具体 budget/seed/data/checkpoint 与误差定义仍缺，不能从 CEM300x30 推给 gradient。
- PlanWAM：没有已认证本地官方驾驶 runner。旧 PointMaze 同名实验不算官方资产。最小需要 immutable inference source、checkpoint/config、4camera×4history/ego navtest split、NAVSIM-v2/HUGSIM evaluator、inference protocol 与许可。不能把 PushT raw10/成功率换名对齐 EPDMS。

作者值独立来自 registry SHA `52e687811e6d6f728ab3eda60836ea05eff11f36da2bb4af975d1dd19ef380f1` 的四固定 locator，manifest 保留原 record/source（含 unknown），而本地 `local_result_values=null`，不新造数值、不跨栏算差值或排名。完整238-cell registry 继续是权威，本桥不复制整个表或选最好 cell。

## 材料 intake 的严格权限边界

`request --method ...` 生成 role/path/sha256 模板。MAIN/资产来源方只填写 records，其他 schema/variant/gates 字段不得改；以明确外部 SHA 输入 intake。每 role 一个最终常规文件（可为 source/数据文件集的 immutable manifest），单文件4GiB上限、1MiB块 hash，同 fd 前后和 path stat 不变；无解包、无 pickle、无目录扫描或网络。任一缺/多/重复角色、未知字段、null、相对/父目录/临时路径、SHA mismatch 均拒绝。

即便所有字节通过，只返回 `materials_verified_not_runtime_ready`：**不解读所交证书内容、不声称作者真实性、完整 archive CRC、内嵌模块等价、训练数据正确或可信加载已通过**。例如把 failed archive report 填入 receipt role 只能证明该失败文件字节存在，不能关闭 semantic gates，也不能解锁模型。MAIN 下一步须调用原 archive verifier 和 review 后的专用语义检查；这里不凭文件名/布尔标记升级 authority。材料错误/负结果留在新回执中，不能覆盖旧报告。

## 命令（inner le-wm-main，全部仅 metadata）

```bash
OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m pytest -q -p no:cacheprovider tests/test_agent_baseline_protocol_bridge_20261028_v1.py
.venv/bin/python -B -m experiments.spectral_reachability.agent_baseline_protocol_bridge_20261028_v1 check
.venv/bin/python -B -m experiments.spectral_reachability.agent_baseline_protocol_bridge_20261028_v1 prepare --output results/agent_baseline_protocol_bridge_20261028_v1_prepare_MAIN01.json
.venv/bin/python -B -m experiments.spectral_reachability.agent_baseline_protocol_bridge_20261028_v1 verify --manifest results/agent_baseline_protocol_bridge_20261028_v1_prepare_MAIN01.json --sha256 ACTUAL_MANIFEST_SHA
.venv/bin/python -B -m experiments.spectral_reachability.agent_baseline_protocol_bridge_20261028_v1 request --method Hi-LeWM
.venv/bin/python -B -m experiments.spectral_reachability.agent_baseline_protocol_bridge_20261028_v1 intake --method Hi-LeWM --manifest MAIN_FILLED_REQUEST_ABSOLUTE_PATH --sha256 MAIN_REQUEST_SHA
```

check stdout 不创建结果；prepare/request --output/intake --output 仅排他新文件，<=2MiB+200MiB floor、fsync。intake blocked exit1，schema错误异常非0；成功仅 metadata exit0，不能视作四方法已可运行。所有运行命令生成不执行，MAIN只需从 manifest 的 `main_saved_commands` 取现存 verify/consume argv，已有结果无需重跑。

此有界交付仅含 RAM 测试与真实 saved metadata build/prepare/verify。38 RAM tests passed；实际执行命令与新 manifest SHA 见交付回复。未执行 existing ownconsume、模型、物理、网络、候选根采集。实现冻结后更改必须新版本；作者协议资产与独立根仍未齐，P1保持关闭。
