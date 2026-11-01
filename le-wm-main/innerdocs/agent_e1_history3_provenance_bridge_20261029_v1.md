# E1 已存 history3 的来源与动作接缝

本轮只新增本 stem 的脚本、测试、文档与独占结果；未训练、加载模型、执行 GPU/物理或拟合 scaler。实际固定两根是原 known2/off，不按结果选根。

源导出合同标注 observed steps `[0,5,10]`，对应同一个 blind NPZ 的三帧和十个二维动作；两包动作分别按时间顺序取 `[0:5]`、`[5:10]`，展平为各 10 维。脚本锁定 E1 prepare、Hi prepare、两份 task、两个 blind、既有 FIRST 子集检查与跨计算 parity receipt 的 SHA；独立读取实物并核每张 RGB、动作文件/数据身份、task 对应、P2 已记录 mean/std、两包归一化后与记录 token 的逐字节相等，以及最终 latent 和 Hi FIRST 高层输入的逐字节相等。source 文件 SHA 也锁定；归一化计算按该源的 float32 公式，不声明它是 Hi 原 scaler。

此前独立 saved parity receipt 对每根三帧各记录了 P2 和 Hi 两条已计算编码与原 paid P2 latent 相等（每根 6 项）。本工具检查回执的固定 SHA 与这些条目，**没有重新执行编码或完整消费**。实存导出自身标注 `source_admission=MAIN_manifest_only_not_independently_verified`；帧/动作时序是该冻结来源的声明与文件绑定，不提升为独立物理溯源、freshness 或科学准入。仅同输入、同末帧及该固定样本计算 parity，不能推出所有输入、模型或正常化完全等价。

结果包含两根的三帧 latent、两包 raw、两段 normalized、FIRST 和 RGB 的各自原字节 SHA；不保存复制的训练数据或编造新模型输入。`inherited_compute_cost=null` 是本工具未逐事件核算继承开销，不能当免费；FIRST 物理时间、原 Hi process/scaler 及历史配对仍未知。没有调用 history3 tracker，也没有输出 raw 环境动作。真实三臂对照仍需事前冻结来源、目标和预算及明确批准的共同动作域；若采用 P2 统计，它是新工程协议而非原始 Hi scaler 恢复。

复算须使用不存在的新结果名：

```bash
cd /home/gbk/code/LeWorldModel/le-wm-main
OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m experiments.spectral_reachability.agent_e1_history3_provenance_bridge_20261029_v1 --output results/agent_e1_history3_provenance_bridge_20261029_v1_MAIN02.json
OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m pytest -q -p no:cacheprovider tests/test_agent_e1_history3_provenance_bridge_20261029_v1.py
```

实际 MAIN01 SHA `7f15b119c63b96d3b9cdc63326882e38e7af294b2b67eb1d48f1a23213de6a16`。7 项 CPU 离线测试通过：实存两根、原始时间/坐标/两包错序、shape、NaN、零 std、token 和源 SHA 篡改拒绝；合成错误样例只测试检查逻辑，不构成新实验。
