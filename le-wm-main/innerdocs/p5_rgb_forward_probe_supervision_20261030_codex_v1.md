# P5 接口/递归探针监督交接

本监督入口绑定探针源码 `ab5b2b3b9e384110ff9578cd84c51613e5cb63289aa2bbd0fd5b949ca6e99e61` 和固定 PREP02 `f0c3aca32dd7a267c43295e26a455212fe374201a5cdaf4b42ccd24dc48c8c27`；同时固定既有 host gate、硬超时监督源码和间接 checkpoint、baseline 与输入来源。子进程重新核验合同并仅支持 CUDA。探针固定四个已知 baseline 和 272 次显式神经 API 调用，详见原探针合同；有限差分不是理论因果、闭环物理或新鲜确认。

launcher 检验预检 SHA/时效、重查静态 host 条件，并调用当前 UID 的 `systemctl --user show`，固定访问 `/run/user/<uid>/bus`，验证用户管理器、服务名、属性及 `/proc/self/cgroup` 精确匹配；按物理 UUID 查询 GPU 的显存、MIG 和 compute mode，映射 UUID 到 child 的 `CUDA_VISIBLE_DEVICES`。child 在单独进程组运行，deadline 触发 SIGKILL 请求；保留 partial，不自动重试。外层 systemd RuntimeMaxSec 覆盖 supervisor 前置及 child。资源查询是采样，不保留 GPU 内存；主线程应排除并发竞争并选择实际内存余量、磁盘预算、时限。

验收以外部 supervisor/journal/preflight SHA 复核启动记录、进程结束、设备和预检绑定，并调用探针 saved-only review 检验完整 4 枝调用、时间顺序、float32 递归状态与扰动、前后 checkpoint/module/precision 快照，输出完整数值。属于同项目代码的自报告一致性，非独立硬件证明；失败、超时、缺失、超出上限或篡改均拒绝。journal 8 MiB 是读取上限而非写入限额；stdout/stderr 与进程内存无硬配额，需主线程另行设置。

## 主线程生产前置

1. 手动审核源与 PREP02 哈希、离线测试及预固定 4 枝/预算；确保结果与监督目录 suffix 未使用。确认只使用已开放输入。
2. 在同一 GPU 可见 host 与用户环境生成既有 host preflight，检查其实际源、驱动、设备节点、Python 与磁盘。该 preflight 即使通过静态项也固定返回退出码 2，不能视为授权。报告最多允许 300 秒，启动器会重查；注意服务启动及 preflight 开销。
3. 选择并单独授权物理 GPU UUID、总显存和空闲显存阈值、外部 child deadline、调用间 deadline、磁盘及 systemd watchdog 限额；尚无实测本探针显存或时间上界。用户服务需以同一个真实/有效 UID 执行、`/run/user/<uid>/bus` 可用，并有正确设备可见/权限。cgroup 必须为 `/user.slice/user-<uid>.slice/user@<uid>.service/.../<unit>.service` 且与自身 cgroup 精确相等；非标准用户管理器布局会 fail closed。`watchdog` 限制 duration 的 systemctl 显示为单单位形式，复杂形式会 fail closed。
4. 外部服务的 RuntimeMaxSec 应覆盖启动、两次 preflight、GPU 查询、完整 child deadline 及日志/复核的收尾。检查 `KillMode=control-group`、`SendSIGKILL=yes`、`TimeoutStopSec`；GPU 可见性/显存校验不保证内存被预留。已有探针直接运行入口可以绕开本监督，主线程必须调用此入口。

## 命令模板（仅主线程另行授权后）

尖括号为待主线程决定的值，不能原样运行：

```bash
cd /home/gbk/code/LeWorldModel/le-wm-main
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 .venv/bin/python -m experiments.spectral_reachability.p5_rgb_host_preflight_20261030_codex_v1 --min-disk-free-mib <disk-floor-mib> --output-name p5_rgb_host_preflight_20261030_codex_v1_<unique>.json
sha256sum results/p5_rgb_host_preflight_20261030_codex_v1_<unique>.json
```

```bash
systemd-run --user --unit=<unique-unit> --working-directory=/home/gbk/code/LeWorldModel/le-wm-main --property=Type=exec --property=RuntimeMaxSec=<service-limit>s --property=TimeoutStopSec=10s --property=KillMode=control-group --property=SendSIGKILL=yes --setenv=PYTHONDONTWRITEBYTECODE=1 --setenv=OPENBLAS_NUM_THREADS=1 /home/gbk/code/LeWorldModel/le-wm-main/.venv/bin/python -m experiments.spectral_reachability.p5_rgb_forward_probe_supervision_20261030_codex_v1 launch --authorization MAIN_THREAD_APPROVED_SUPERVISED_FORWARD_PROBE_CUDA --gpu-uuid <GPU-uuid> --run-suffix <unique> --preflight-name p5_rgb_host_preflight_20261030_codex_v1_<unique>.json --preflight-sha <preflight-sha256> --min-free-mib <free-mib> --min-total-mib <total-mib> --hard-seconds <child-hard-deadline> --runner-seconds <between-call-deadline> --max-age-seconds <1-to-300> --watchdog-unit <unique-unit>.service --watchdog-max-seconds <service-limit>
```

在服务和 child 均已停止后，独立计算 SHA，再运行：

```bash
sha256sum results/p5_rgb_forward_probe_supervision_20261030_codex_v1_supervisor_<unique>/supervisor.jsonl
sha256sum results/p5_rgb_forward_recursive_probe_20261030_codex_v1_RUN_<unique>/events.jsonl
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 .venv/bin/python -m experiments.spectral_reachability.p5_rgb_forward_probe_supervision_20261030_codex_v1 accept --run-suffix <unique> --preflight-sha <preflight-sha256> --supervisor-sha <supervisor-sha256> --journal-sha <journal-sha256> --output-name p5_rgb_forward_probe_supervision_20261030_codex_v1_accept_<unique>.json
```
