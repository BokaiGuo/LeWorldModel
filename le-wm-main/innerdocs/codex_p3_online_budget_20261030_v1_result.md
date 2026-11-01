# P3 在线信号总预算探索：实际结果

固定内核合同 `codex_p3_online_budget_20261030_v1.md`；预算点 20、30，顺序 uniform、signal、reverse。在线信号各策略每预算点重新计算，不使用前次保存的权重。新 CPU-only 模拟和篡改测试仅用于接口，不是实验数据。

## CUDA 实际运行与计量

主机 prepare `results/codex_p3_online_budget_host_20261030_v1_PREP_01.json` SHA256 `bfab85393221afe02f8ac804130040c4f493d5d3deffdeb5486e14cc6e5cb305`；report `results/codex_p3_online_budget_host_20261030_v1_RUN_01/report.json` SHA256 `7d2ff7ef0c554077cf0a2461ab733d80f0f14a5658bdf1c554fd58ac6f4dacde`。沙箱第一次在 CUDA 初始化及创建输出目录之前设备查询退出 9，未执行实验；随后主机权限下取得固定 inode 合作锁，实际加载冻结模型并完成两个预算点。锁仅约束合作入口。模型前后 hash 一致；实际 GPU 峰值 allocated 141664768 字节、reserved 144703488 字节；整个 host elapsed 21.642 秒，含初始化和加载，不是每策略或每调用的独立 wall。

独立 saved consumer `results/codex_p3_online_budget_host_20261030_v1_CONSUME_01.json` SHA256 `03a3eaa5f7cce26298ab578440c02dac87b7d984ab0e2cb3eccfdc07c28adc67` 重建两个点、六策略、逐步账项、在线零动作 nominal、Jacobian 行、SVD 信号、配额、每候选动作/路径/cost 和赢家。模型模块数组未独立重新计算；动态源码闭包 unknown。它不认证物理或科学通过。

| 点 | 策略 | 两根候选数 | 每类前向轴 | metric | Jacobian/SVD | VJP/grad API/raw callback |
|---:|---|---|---:|---:|---:|---:|
| 20 | uniform | 12/12 | 48 | 24 | 0/0 | 各 0 |
| 20 | signal | 15/5 | 48 | 20 | 2/2 | 各 384 |
| 20 | reverse | 5/15 | 48 | 20 | 2/2 | 各 384 |
| 30 | uniform | 17/17 | 68 | 34 | 0/0 | 各 0 |
| 30 | signal | 22/8 | 68 | 30 | 2/2 | 各 384 |
| 30 | reverse | 8/22 | 68 | 30 | 2/2 | 各 384 |

每类前向轴为 action_encoder、predict、predictor、pred_proj 分别的 attempt/return；上表均完成。合计每类前向各 348、metric 158、Jacobian 8、SVD 8、VJP/grad API/raw callback 各 1536，均为嵌套或不同语义的资源轴，不可相加成 FLOPs。signal 两根权重约 162.324052/156.286715，四次独立在线信号采集得到相同数值；信号成本已经在各动态策略内实付，uniform 用省下的前向预算多评估 4 个候选，但不伪造反向成本。相同分项上限不等于相同实际总算力。

## 物理绑定与执行

`results/codex_p3_online_budget_physical_bind_20261030_v1_01.json` SHA256 `ada4292c6f897c5713295a9c0cffdb1b121337e05160f347fd40fd12927b62b4` 绑定全部 12 槽到已认证根、任务和原始 float32 动作。物理执行规则在模型结果后固定为两个预算点均完整六槽；有已知相同动作仍重复物理执行，每次算资源，不能算作独立任务或独立样本。每槽 warm10 后原始 winner raw10，原生早停即停止无补步。

| 点 | 根 | uniform 位置/角度 | signal 位置/角度 | reverse 位置/角度 |
|---:|---:|---|---|---|
| 20 | 0 | 71.5224 / 0.49984 | 65.0958 / 0.44062 | 70.7867 / 0.47590 |
| 20 | 1 | 76.4027 / 0.08223 | 76.4027 / 0.08223 | 76.4027 / 0.08223 |
| 30 | 0 | 70.7867 / 0.47590 | 70.7867 / 0.47590 | 70.7867 / 0.47590 |
| 30 | 1 | 76.4027 / 0.08223 | 76.4027 / 0.08223 | 76.4027 / 0.08223 |

位置为 object_position_error，角度为 wrapped_object_angle_error 弧度。12 槽均走满计划10步，native success 均 false。每个预算点实际物理资源均为 construct 6、close 6、reset 12、set_state 24、goal_render 12、goal_bind 6、public_step 120、physics_step 1224、explicit_render 132；各类 attempt 与 return 相等。两个点分别为 `results/codex_p3_online_budget_physical_host_20261030_v1_P20_RUN_01/report.json` 和 `..._P30_RUN_01/report.json`；report SHA256 分别 `458a270f69e4448dbab9c16c5206ea4dc472c6909eba53e5eae03d445dc1b4d9` 和 `f3b5324a6f0985ae5f8b1e7545315237808973544d313243c969fa930e52a5b8`。

独立物理 saved consumers 均核验六槽和 132 帧、来源、动作 ACK 和总账。因原摘要不携带输入身份，两份摘要字节相同；额外保存外部身份绑定封套 `results/codex_p3_online_budget_physical_host_20261030_v1_P20_CONSUME_BOUND_01.json` 和 `..._P30_CONSUME_BOUND_01.json`，分别记录对应 prepare/report SHA。consumer 未独立重跑物理。

## 边界与测试

已知根、前述结果已见、物理合同在此次 CUDA 后固定；不声称预注册、独立泛化、等总预算或 P3 通过。模型读取/加载及 CUDA 上下文的独立成本、逐臂 wall、硬件能耗、FLOPs 和完整动态代码闭包 unknown；整个 host wall 和内存峰值只是有限测量。反向计算不跨策略人为补齐，不把上限相同写成实际成本相同。物理源使用旧已认证 reset/goal/warmup，所有旧产物保留。

`tests/test_codex_p3_online_budget_20261030_v1.py`、`tests/test_codex_p3_online_budget_saved_20261030_v1.py`、`tests/test_codex_p3_online_budget_host_20261030_v1.py`、`tests/test_codex_p3_online_budget_physical_bind_20261030_v1.py`、`tests/test_codex_p3_online_budget_physical_20261030_v1.py` 离线测试共 29 passed。
