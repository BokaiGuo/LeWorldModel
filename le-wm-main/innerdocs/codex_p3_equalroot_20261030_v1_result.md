# P3 每根等调用预算探索结果

执行前固定合同见 `codex_p3_equalroot_20261030_v1.md`，不是外部预注册。prepare SHA256 `9cdfff674690bce149eef8a4d09329f0f666be02e1b9be8a7ede40c8490fd252`。第一次沙箱启动在持锁后、CUDA 初始化和输出目录创建之前，`nvidia-smi` 返回 9；主机权限下使用同一 prepare 和固定 inode 合作锁启动，运行完整，report SHA256 `7c1e389681172f897a28463760fe64cbb670fe5c31e6d74a4ecd20aed02534e6`。未重做已产生神经调用的批次。

独立 CUDA saved consumer 核对 60 候选、480 条分类型神经操作 attempt/return、60 metric、赢家和路径；未独立重建模块数组，动态源码闭包 unknown。新任务来源绑定 `results/codex_p3_equalroot_physical_bind_20261030_v1_01.json` 同时保存每根每策略四类调用 attempt/return，各类均 20，metric 返回各 10。四类调用是嵌套轴，不能加总为独立 FLOPs；历史共享信号的获取资源另列于原 prepare，逐臂分摊、加载、上下文、FLOPs 和逐槽 wall 成本 unknown。

新物理 prepare SHA256 `3773ac746162d0f72b6bf0ab07c9dc9c945ff04208dcdfb09f4d3afa85c6a420`，固定重放每槽完整 raw10，暖启动10；native 终止或截断即停、不补步。`results/codex_p3_equalroot_physical_host_20261030_v1_RUN_01/report.json` SHA256 `1c4d358e2743776302b94e1de47120739f3068ebe647f1a2734a72460bc81075`。六槽均完成 warm10 和 plan10、无早停、native success 均 false。attempt/completed：construct 6/6、close 6/6、reset 12/12、set_state 24/24、goal_render 12/12、goal_bind 6/6、public_step 120/120、physics_step 1224/1224、explicit_render 132/132。

| 根 | 策略 | 赢家 index | 模型 cost | 物体位置误差 | 角度误差 rad |
|---|---|---:|---:|---:|---:|
| 0 | uniform | 0 | 0.684316 | 28.970473 | 0.152789 |
| 0 | signal | 0 | 0.684316 | 28.970473 | 0.152789 |
| 0 | reverse | 13 | 0.912382 | 70.786706 | 0.475903 |
| 1 | uniform | 14 | 0.970916 | 125.942600 | 0.552746 |
| 1 | signal | 14 | 0.970916 | 125.942600 | 0.552746 |
| 1 | reverse | 1 | 1.205713 | 114.830087 | 0.358935 |

独立 saved 物理 consumer `results/codex_p3_equalroot_physical_host_20261030_v1_CONSUME_01.json` 核验全部六槽、原始 ACK、源 root/goal、暖启动、132 帧和资源总量；不独立重跑物理。两根的 uniform/signal 赢家动作分别相同，不能作为独立样本或 signal 机制成功。signal 在本合同中只是人为 rank-to-window 干预，不等价于原跨根资源分配；不得宣称等总预算、泛化、P3 科学通过。

离线测试：`PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -B -m pytest -q -p no:cacheprovider tests/test_codex_p3_equalroot_20261030_v1.py tests/test_codex_p3_equalroot_physical_20261030_v1.py tests/test_codex_p3_six_physical_adapter_20261030_v1.py tests/test_codex_p3_six_physical_contract_20261030_v1.py`，17 passed。合成 fixture 只检验接线和篡改拒绝，不是实际神经或物理证据。
