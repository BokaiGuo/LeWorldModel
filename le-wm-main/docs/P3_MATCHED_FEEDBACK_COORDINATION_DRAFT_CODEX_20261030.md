# P3 matched-order 真实反馈启动协调草稿（未授权）

状态：**未获启动授权，不可执行**。本文件只是待填写的交接草稿，不是资源预留、审批、来源封存或运行回执。2026-10-30 用户明确告知 GPU 正由主线程占用；物理／仿真环境及显示资源的占用与可用性未知。本次没有初始化 CUDA、构造环境、reset、render 或实际运行。协调各方应在执行前单独确认，不能把本草稿或占位文字当作许可。

## 本次只读复核

- 封存 prepare `results/p3_matched_order_feedback_host_v2_codex_20261029_prepare_MAIN01.json` 的 SHA-256 为 `b6c8dcf650d4e752f6306ed4b6f9434861c224ced51c7c8de25c0e7ca6d323f2`；调用 `host.verify` 完整重建通过，包含 19 个实现文件及其来源闭包。此核对不证明未来不会漂移；运行前还会再次重建。
- 四槽固定为 root0/fixed、root0/signal、root1/fixed、root1/signal，两个已知根；运行包含实际 PushT 仿真环境、RGB render、模型 CUDA 和观测反馈，不是新根或真实机器人执行。源码工厂调用 `gym.make`，使用 `swm/PushT-v1`、`render_mode='rgb_array'`、224 分辨率；这不证明渲染不占显示设备，也不证明现有环境能正常运行。
- 本次 `df -hT .` 显示约 22 GiB 可用，只是瞬时值。Store 创建前要求至少 2 GiB 总上限加 512 MiB floor；运行中还检查 256 MiB 终态保留及分类额度。整作业 wall 上限 7200 秒；CUDA 峰值上限 allocated 3 GiB、reserved 4 GiB，不代表当前资源有余量或独占。
- 本次检查时下列拟用输出和 receipt 均不存在；这不是预留，启动前必须再次检查。不能覆盖已有路径。
- `--authorization` 与 `--gpu-coordination` 为调用者字符串；runner 仅核对固定令牌和协调字符串至少 8 个非空白字符，不验证发起人、资源租约、设备、显示或时段。真实授权必须来自独立可追溯记录。
- `lifecycle.execute` 对单槽普通异常或 `NativeInitialStop` 保留该槽、尝试清理并继续下一固定槽；单槽内原生停止之后不继续派发该槽动作。不能宣称任一槽失败自动停止整个面板；协调方必须明确接受此行为，或先协调并设计新版本，不能为改变行为修改封存 runner。不可根据结果挑槽、补槽或擅自重试。

## 协调记录模板（全部待确认）

请在**独立、可追溯的真实协调记录**中填写并由相应负责人确认；不要修改本草稿冒充批准。建议将记录 URI/工单号与版本或 SHA 一同传入 `--gpu-coordination`，并保留批准的原始记录。该参数只记录引用，不自动核验其真实性。

| 项 | 待记录内容 |
|---|---|
| 实验责任人与授权 | 负责人、批准人、可核查批准链接/标识、批准时间、明确包含四槽 PushT 仿真与 CUDA 的范围 |
| GPU 主线程 | 占用作业结束或明确许可共享的证据；设备身份、可见映射 `cuda:0`、预约起止、负责人及冲突处理 |
| 物理／仿真资源 | 明确本次 PushT 仿真及所需设备/环境资源，资源负责人、并发限制、预约窗口；若存在真实硬件相关资源，分别确认，不得假设空闲 |
| 显示／渲染 | 实际渲染后端、显示或 headless 环境、相关设备的占用与访问授权、负责人、预约窗口；不得根据 `rgb_array` 推断空闲 |
| 时间与存储 | 时区、窗口（覆盖最多 7200 秒及收尾）、运行时新鲜磁盘余量、输出和 receipt 独占路径、操作者 |
| 执行条件 | prepare SHA、本草稿与封存协议链接；确认理解失败后继续后续槽、失败分母、不得补槽或擅自重试；任何条件变化先停止并重新协调 |
| 事后留存 | 命令、实际协调引用、开始结束时刻、退出码、stdout/stderr、report/completion 的实际 SHA、消费回执或缺失原因 |

运行前还须确认来源重建仍通过、项目环境与原生依赖可用、显示与仿真相关资源由负责人明确授权，且 GPU 主线程已解除占用或明确允许此实验。实际 CUDA allocation 探针会初始化 CUDA 并分配张量，原生 render 探针也会使用环境／显示资源；**这些探针也只能在相关资源协调完成后执行**。本草稿不授权任何探针。runner 自身会进行 CUDA allocation，并在每槽对 warm10 图像／状态、根及目标身份执行运行时核验；失败不构成已验证通过。

## 将来运行命令（仅协调完成且授权后）

以下命令含占位符，不得原样执行。实际协调记录必须真实存在、覆盖 GPU 和环境／显示，填入唯一可追溯引用；启动前复核目标不存在、封存 SHA 与依赖、时段及资源。若有人要求在任一槽失败后整作业立即终止，现版本不满足该要求，先停止协调版本变更。

```bash
cd /home/gbk/code/LeWorldModel/le-wm-main
env PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -m experiments.spectral_reachability.p3_matched_order_feedback_host_v2_codex_20261029 run \
  --prepared results/p3_matched_order_feedback_host_v2_codex_20261029_prepare_MAIN01.json \
  --prepare-sha256 b6c8dcf650d4e752f6306ed4b6f9434861c224ced51c7c8de25c0e7ca6d323f2 \
  --output /home/gbk/code/LeWorldModel/le-wm-main/results/p3_matched_order_feedback_host_v2_codex_20261029_RUN_MAIN01 \
  --authorization COORDINATED_MATCHED4_HOST_CUDA_PHYSICS \
  --gpu-coordination '<REAL_TRACEABLE_COORDINATION_REFERENCE>'
```

保全 stdout/stderr、退出状态和打印的两个 SHA；partial 仍须保全，不补造终态。成功或 partial 完成后，以实际 SHA 在另一个进程运行独立 saved consumer；receipt 必须是新路径。若 report 或 completion 缺失，保全现场并调查，不伪造 SHA。

```bash
env PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -m experiments.spectral_reachability.p3_matched_order_feedback_host_v2_codex_20261029 consume \
  --output /home/gbk/code/LeWorldModel/le-wm-main/results/p3_matched_order_feedback_host_v2_codex_20261029_RUN_MAIN01 \
  --report-sha256 '<ACTUAL_REPORT_SHA256>' \
  --completion-sha256 '<ACTUAL_COMPLETION_SHA256>' \
  --receipt /home/gbk/code/LeWorldModel/le-wm-main/results/p3_matched_order_feedback_host_v2_codex_20261029_receipt_MAIN01.json
```

完整 receipt 仅为来源可信条件下的已知两根 saved 工程链核查，不是硬件远程证明、fresh roots、等 FLOPs 或 P1/P3 科学通过。部分 receipt 不可冒充完整结果。原协议与封存资产均未改动。
