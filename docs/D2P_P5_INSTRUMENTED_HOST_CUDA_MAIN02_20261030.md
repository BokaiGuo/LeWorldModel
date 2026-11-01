# P5 固定 RGB 输入：受监督主机 CUDA 执行

2026-10-30，主线程对既定 164 枝运行新 instrumented 路径，不修改 checkpoint 或训练额外组件。主机静态前置 `p5_rgb_host_preflight_20261030_codex_v1_MAIN02.json` SHA256 `140ac82af916d0c431014c8559e96ec9eb1aed69ac678f519e2c0ee56d167980`：静态检查通过，GPU 动态身份／资源和外部监督留待启动检查。启动时 `nvidia-smi` 报告 GPU UUID `GPU-4e8973ff-0278-53f3-2bae-2615d461e441`、总显存 16311 MiB、空闲 15763 MiB；这是瞬时观测，不是资源预留。

作业运行于 host systemd unit `lewm-p5-rgb-20261030-main02.service`，总服务限时 1800 秒，子任务限时 1500 秒，runner 限时 1200 秒；最低总显存 15000 MiB、空闲显存 8192 MiB、磁盘静态余量 2048 MiB。服务正常退出，监督器记录子任务运行 4.576 秒，固定 off/on 各 82/82 枝完成；没有重试。独立接受入口读取外部给定的 supervisor 和 journal SHA 并验证完整事件、前后快照及来源绑定，接受结果状态为 `self_report_consistency_only_not_scientific`。

- [接受记录](../le-wm-main/results/p5_rgb_instrumented_supervision_20261030_codex_v1_accept_MAIN02.json)，SHA256 `a9d8a8aecf9b639d3df2d308c0fc8b59aa06fc0c012833657d96bf4a3df9d23e`。
- [监督日志](../le-wm-main/results/p5_rgb_instrumented_supervision_20261030_codex_v1_supervisor_MAIN02/supervisor.jsonl)，SHA256 `f6d93a73b3cf2eaa8fa351c857e68598951b9954dea0d176919b4ef1e060efef`。
- [预测事件](../le-wm-main/results/p5_rgb_instrumented_supervision_20261030_codex_v1_run_MAIN02/events.jsonl)，SHA256 `bd40eea0c274a09429c95faa79f74c7c8e02e92e47ff48ffe2f38234c452b2a3`。
- before/after 状态快照分别 SHA256 `d99f81981f22e3c9ae3a0ddb02c53b5137be9fbd5d90b0b3dd3ab97f2b890b3e` / `e79570c38a9877766f40ec58d8c28d0f491d32c4e08c2d2e37e33ad9ec62eaf0`；哈希不同包括采样时点和日志锚信息，验收程序对实际模型状态内容执行相等检查。

这些是已知固定输入上的预测及自报一致性证据。没有读取未开放的未来标签，也没有完成与真实物理结果的匹配分析、独立根泛化、训练数据来源认证或跨模型优势检验，不能将其称作 P5 科学通过。设备快照不是显存预留；自报日志与两次状态快照不证明中间任意时刻不存在状态变化。系统总体 `degraded` 因 `todesk.service` 失败；该监督服务自身正常退出。
