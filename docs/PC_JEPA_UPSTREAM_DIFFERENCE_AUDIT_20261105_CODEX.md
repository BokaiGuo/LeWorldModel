# JEPA PushT 本地实现与固定提交差异核验

2026-11-05。范围为嵌套仓库工作树相对本地 HEAD
`13cf1d9c7e476f53c17714d2e0f1dc239a883ce0` 的差异，以及一次
单任务动作历史改动隔离。远程配置指向 facebookresearch/jepa-wms；
本次未独立完成远端提交与论文评测版本认证。以下不是官方复现证明。

## 五处差异及状态

| 文件 | 本地差异 | 核验结论 |
|---|---|---|
| `app/plan_common/datasets/pusht_dset.py` | decord 不可用时退回 OpenCV | 本机 venv 可导入 decord；预检通常使用原解码分支，未比较两种解码像素。 |
| `app/plan_common/models/dino.py` | 优先加载本地 DINO 源码 | 本次实际载入 `_DinoV2Shim`／Transformers 实现；未证明与 TorchHub 原实现及权重数值一致。 |
| `app/plan_common/models/prop_embedding.py` | 缺属性时默认关闭 MLP、单 token | 兼容旧对象的行为变化；没有证明当前完整评测中分支是否触发或等价。 |
| `app/plan_common/models/vit.py` | 缺 `use_sdpa` 时默认 False | 兼容行为，未全面认证路径与数值等价。 |
| `app/vjepa_wm/modelcustom/simu_env_planning/vit_enc_preds.py` | 动作特征长度不足时前补零 | 与上游切片语义不同；不是已知真实历史。有限单任务隔离见下。 |

本地 PushT checkpoint SHA256
`9beca3eafe0739c3b3adb5d734fa435ccbda0fea8a65d53d4cccec176aaaa0eb`；
基准配置 SHA256
`7ded0cd2da37208c51b7312259b319038d850b48dad1638484b46bda6aaf7d5c`。
哈希仅标识文件，不证明权威来源或训练集独立性。

## 同条件动作历史隔离预检

使用上述配置的开发性缩减版本：1 任务、CEM 8 样本×2 轮、
2 elite、关闭 decoder 和可选图，目标及执行时钟保持配置值。
两臂使用同一数据、checkpoint、随机种子和本地 DINO 适配器；
第二臂从固定提交导出临时副本，保留另外四处本地修改，
只将 `vit_enc_preds.py` 恢复为固定提交版本。它不是完整原版。

两臂均抽到轨迹 7、偏移 47，参考轨迹成功，规划动作最终均失败，
最终距离均为 126.19967；输出 MP4 的 SHA256 均为
`801ae4d274b7b20cf04dc6cd81057d6d51dcaa9ca81477e9ca1fe073a1732595`。
输出分别在
[本地修改臂](../le-wm-main/results/pc_jepa_runner_smoke_termcolor_20261105_codex/)
和[原 unroll 隔离臂](../le-wm-main/results/pc_jepa_upstream_unroll_isolation_smoke_20261105_codex/)。

此结果只说明该单例保存的视频字节和记录指标一致；没有保存逐次候选、
预测张量和动作，因此不能声称所有内部计算一致。没有证明补零对其他
上下文长度、历史及任务等价，也不能从单例失败推断模型性能。
此配置 `action_encoder_inpred=true`，零补的是归一化动作输入；
其他配置若先编码动作，零特征未必等于编码后的零动作。

首次启动在沙箱网络限制下无法创建本机分布式端口；放行后先后缺少
`skimage` 和 `termcolor`。预检使用仓库既有 `.jepa-physics-site`
和临时目录中的系统 `termcolor.py`，没有安装或修改共享依赖。
运行含 NCCL 销毁警告，未认定其影响结果。

## 尚未完成

需要独立核验权威提交与数据/checkpoint 的版本及来源、固定相同输入
比较 TorchHub 与本地 DINO 编码、覆盖多帧上下文的预测与动作历史、
比较解码路径及其余兼容分支、记录动作与中间张量，并运行完整配置的
多任务多 seed 对照。未完成前不得称“权威实现等价”或“官方成绩复现”。
既有评测和用户工作树未被覆盖；旧 gate 不变。
