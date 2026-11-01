# P5 观测刷新监督与 saved 验收交接

本文件不是生产授权。`PREP01` 固定 SHA `46bfde766c6ce6d99c95cba0c37909c4437dcfbdb9e5e049da02ca0d83403409`，监督源码 SHA `d5ab673a87a15460a3875d80266cad75819d37d092070f9605667ac7e832a289`；使用已开放的 164 枝、80 对，不读取未开放 holdout。离线前置重新构建整个输入合同、校验 checkpoint/源码与基线，不执行模型；运行 child 再次重建并逐步 materialize。

监督器先通过指定物理 GPU UUID 查询 `nvidia-smi`，拒绝不匹配 UUID、MIG、无效计算模式和低于显存阈值的设备；采样不保留显存，不能排除并发抢占。以独立进程组启动单个 child，外部 deadline 后 SIGKILL 进程组；不自动重试或删除已有 partial。child 前/后捕获模块、参数字节哈希、冻结状态、归一化和 precision，并将快照锚定在第一次编码及最后一次预测的 attempt 事件。strict saved review 逐项验证固定 164×5×2 调用的身份、哈希链、float32 输出和快照，accept 再校验指定 supervisor 与 journal SHA。accept 和 producer 属于同一源码文件的独立 saved-review 路径，不是独立代码实现或独立硬件证明。

## 主线程必须核验

- 在实际运行环境中重新审核源码哈希、PREP01、基线和 31 项离线测试；确保仅使用已开放历史结果。
- 在同一 host/systemd 设备可见环境生成既有 host preflight，核对驱动、设备节点、Python、磁盘空间及源；报告状态始终为 blocked，不是授权。刷新监督器**没有自动绑定该 host 报告或自动验证 systemd cgroup**，主线程须自行完成这两项；不得将旧五步 preflight gate 称作已接入本刷新协议。
- 检查 GPU UUID、独占/占用、驱动可用性及 MIG，并由主线程选择显存阈值；没有经测量的显存上界或默认安全阈值。检查磁盘限额、日志可写、未使用的 suffix；supervisor/child 无日志体积硬配额。
- 主线程选择外部 child deadline、调用间 timeout、systemd RuntimeMaxSec 和 stop grace；systemd 限额必须容纳启动和前置、child deadline 及终止/日志收尾。检查服务实际 cgroup、`KillMode=control-group`、`SendSIGKILL=yes` 和生效时间限制。外部 child deadline 不覆盖 GPU 查询和 child 启动前的监督器工作，systemd 是更外层限制。
- CUDA、运行时快照和硬件访问在本次均未实测；若 host 或 CUDA 条件不满足，停止并保留现场，不重试或降低分母。

## 命令模板（仅主线程核验并另行授权后）

以下尖括号是主线程选定值，不应直接原样运行。项目根为 `/home/gbk/code/LeWorldModel/le-wm-main`，服务要求机器级 systemd 权限及 GPU 可见。

```bash
cd /home/gbk/code/LeWorldModel/le-wm-main
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 .venv/bin/python -m experiments.spectral_reachability.p5_rgb_host_preflight_20261030_codex_v1 --min-disk-free-mib <disk-floor-mib> --output-name p5_rgb_host_preflight_20261030_codex_v1_<unique>.json
sha256sum results/p5_rgb_host_preflight_20261030_codex_v1_<unique>.json
```

host preflight 有意退出 2；检查内容和 blockers，而不是将退出码或文件存在当授权。

```bash
systemd-run --unit=<unique-unit> --uid=gbk --working-directory=/home/gbk/code/LeWorldModel/le-wm-main --property=Type=exec --property=RuntimeMaxSec=<service-limit>s --property=TimeoutStopSec=10s --property=KillMode=control-group --property=SendSIGKILL=yes --setenv=PYTHONDONTWRITEBYTECODE=1 --setenv=OPENBLAS_NUM_THREADS=1 /home/gbk/code/LeWorldModel/le-wm-main/.venv/bin/python -m experiments.spectral_reachability.p5_rgb_refresh_supervision_20261030_codex_v1 launch --authorization MAIN_THREAD_APPROVED_SUPERVISED_REFRESH_164_CUDA --gpu-uuid <GPU-uuid> --run-suffix <unique> --min-free-mib <reviewed-free-mib> --min-total-mib <reviewed-total-mib> --hard-seconds <reviewed-child-deadline> --runner-seconds <reviewed-between-call-limit>
```

进程完全结束后，人工记录以下两个实际哈希，再运行 saved-only acceptance：

```bash
sha256sum results/p5_rgb_refresh_supervision_20261030_codex_v1_supervisor_<unique>/supervisor.jsonl
sha256sum results/p5_rgb_refresh_supervision_20261030_codex_v1_run_<unique>/events.jsonl
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 .venv/bin/python -m experiments.spectral_reachability.p5_rgb_refresh_supervision_20261030_codex_v1 accept --run-suffix <unique> --supervisor-sha <supervisor-sha256> --journal-sha <journal-sha256> --output-name p5_rgb_refresh_supervision_20261030_codex_v1_accept_<unique>.json
```

partial、超时、快照缺失、哈希不符或任何拒绝都不能验收。通过表示保存自报告的一致性，不意味着准确性、观察权限等同、fresh root 复现或 P5 科学通过。
