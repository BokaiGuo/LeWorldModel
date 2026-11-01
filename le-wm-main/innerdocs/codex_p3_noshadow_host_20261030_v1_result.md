# P3 无 shadow 固定队列主机 CUDA 探索

固定 prepare `results/codex_p3_noshadow_host_20261030_v1_PREP_01.json` SHA256 `4c2b834778a8b7e18a77e18992c59daebd032048a8d8fd970ec2006e6a34f78a`，先于本次执行生成。两个根和候选在历史运行中已见，因此只称本次执行前固定，不称预注册、独立或 fresh。每臂仅评估资格内的 20 个 H2 候选，不运行 shadow 候选，不训练、没有物理。

`results/codex_p3_noshadow_host_20261030_v1_RUN_01` 主机 CUDA 成功完成。合作锁非阻塞获取，父进程持锁至 CUDA 子进程退出；锁只约束合作入口。report SHA256 `d0f52c7c88156e894517b75611181572825b745363242d8bba4562688c63ff9c`，journal tail `01461a26af34abf33170f62c36a5cc18416a250f46bace3a6ab870f2d99dda58`；模型前后状态哈希均 `4b12c01f5859f4b46bea478f3a8d5e2d3f516df456a622a3feec31bde7e33546`；设备 UUID `GPU-4e8973ff-0278-53f3-2bae-2615d461e441`。

独立 saved 语义消费者按固定队列核对 60 候选、480 神经轴 attempt/return、60 metric attempt/return、路径、模型成本和赢家。其状态为 `semantic_panel_checked_not_certified`：未独立重建模块数组，不认证物理或完整动态源闭包；该程序消费输出为 stdout。

| 臂 | 候选数 root0/root1 | 每类神经轴调用 root0/root1 | 赢家 index / model-Q root0 | 赢家 index / model-Q root1 |
|---|---|---|---|---|
| uniform | 10/10 | 20/20 | 1 / 0.7874349800 | 1 / 0.6231905744 |
| signal | 15/5 | 30/10 | 12 / 0.7485899228 | 1 / 0.6231905744 |
| reverse | 5/15 | 10/30 | 1 / 0.7874349800 | 1 / 0.6231905744 |

四类分别为 action_encoder、predict、predictor、pred_proj；有嵌套，不可相加冒充总独立 FLOPs。每臂这四轴各 40 次，metric 20 次。统一跨两根总数不代表逐根同算力；历史信号共享及其各臂归因、加载与 CUDA 上下文、FLOPs、每臂 wall 未证明相同。整个 host elapsed 8.55s 含模型加载、来源哈希和面板，不是每臂计时。signal 首根不同、第二根碰撞，与已有完整面板子集一致，这是重复已知数据后的模型描述，不是泛化或物理收益证据。门槛不变。

测试：`tests/test_codex_p3_noshadow_20261030_v1.py` 与既有无 shadow queue/journal/source 测试共 9 passed；RAM 只验证接线及失败篡改，不是神经实验。
