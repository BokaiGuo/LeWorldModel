# E1 saved FIRST / history3 offline bridge

本次只读已保存结果，新增脚本、测试、本文与独占 JSON；不训练、不运行模型或 GPU、不拟合 scaler、不执行物理。既有 RGB 高层 FIRST 到原生低层的完整 saved consumer 已存在；此处只重核其中与 history3 接缝有关的数组子集，不声称重做完整 consumer。

脚本固定 report SHA `083d0d9aa190b561e6ccf865b09d7224b97d404b16ba38c792bd8af386f5622a` 和已有 consumer SHA `40f28414682c9540e9e195b361a36dfc514f816358adbeacb379fedc460e4ef0`。对四个固定 root/overlay 行读取完整 selected rollout、FIRST、低层目标、z_init、单帧 z history 和单步零动作 history；验证文件和数据 SHA、float32、维度、finite、无尾字节，逐位连接 `rollout[:,0,0,:]` → FIRST → 原生低层目标，并验证当前帧。行身份以保存的 root slot 和 overlay 标记，不推定是 E1 的新共同目标或新根。旧 consumer 还检查了本工具未重新计算的其它网络、CEM 和来源链。

`check_history(zpath,zsha,apath,asha,current_sha)` 对调用者提供的 NPY 文件进行离线检查，分别要求 `[1,3,192]`、`[1,2,10]`、float32、finite、准确文件 SHA 和无尾字节，且最后 z 帧的原始字节 SHA 等于所选行的 `current_raw_sha256`。不重采样、pad、填零、归一化或调用 tracker；该检查**不能认证**其他历史帧和动作的来源、时间、归一化、同模型坐标或物理配对。调用者自报 SHA 仅检查文件一致性，不能获得 authority。实际 history3 delivery 未执行。

首次重算到一个不存在的新路径：

```bash
cd /home/gbk/code/LeWorldModel/le-wm-main
OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m experiments.spectral_reachability.agent_e1_first_history_bridge_20261029_v1 --output results/agent_e1_first_history_bridge_20261029_v1_MAIN02.json
OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m pytest -q -p no:cacheprovider tests/test_agent_e1_first_history_bridge_20261029_v1.py
```

本轮 MAIN01 JSON SHA `3e4eb4839c8b2cb9e6b4fb272a59d3a295f1fbc225d0ec6ed401afa40fbf489d`；6 项测试通过，包括真实保存数组与合成历史的错 SHA、错末帧、维度、dtype、NaN、尾字节拒绝。测试不构成真实三臂对照。

FIRST 无已证物理时间；不能把 high 五步重规划、D50 最终目标 offset 或共同 tracker 的目标时间套给它。该资产对应的是已有工程集成，非官方完整 policy parity。原历史 action scaler/process 与 checkpoint/数据配对、history3 真实来源、共同目标/时间与预算和最终 raw action inverse 均未认证，缺失成本保持 null；不能以此报告 learned baseline 实测、因果或泛化结论。新的共同 scaler 需另行明确授权，不是历史 scaler 恢复。
