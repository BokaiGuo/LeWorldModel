# P5 固定递归探针：主机执行与封存交接

本文件是独占交接，不是运行授权。2026-11-01 的沙箱只读检查中，既有探针及 PREP02、checkpoint 和历史 RGB 来源的固定哈希通过；设备节点在沙箱不可见，不能推断主机是否可用。未查询或占用 GPU。`results` 当时可用 18059 MiB；10 GiB 地板是本交接的操作预算，不是运行峰值的测量。主机必须重新检查。

固定工作量为四个既有分支 0、41、82、123；4 次 forward、4 次 encode、264 次 predict、52 次 reset。5 步、六维 ±0.001 的固定扰动不允许按结果调整。历史 164 条用于来源检查，不是本次 164 分支执行；已见根不是新根。

## 风险和限制

- 读端拒绝大于 8 MiB 的 events.jsonl；它本身不限制写入。下方服务的 `LimitFSIZE=8388608` 是每个普通文件的内核上限，覆盖子进程继承的 journal、stdout、stderr 和 supervisor 日志，但不是总目录配额，也不覆盖其他进程写入；超限导致失败或不完整，不得裁剪补造。服务标准输出/错误不写入 journald。
- 选择的 1800/2100/2400 秒分别是调用间检查、子进程组硬超时和 systemd 服务外部上限，非实测耗时。进程卡在内核不可中断状态时即使发出 SIGKILL 也可能尚未消失；必须独立检查服务 cgroup 和写入进程，禁止自动重试。
- 固定的文件上限与磁盘地板不是磁盘保留、总配额、GPU 显存峰值或物理 GPU 独占。GPU 查询只检查 UUID、显存、MIG、compute mode，不检查全部进程，也不能防止别人随后启动。运行前由主机管理者保证整块 GPU 独占，并在运行期间维持；不改变 compute mode。
- 尚无经测量的 GPU/主机内存峰值。以下 GPU 显存门槛和 `MemoryMax` 必须由主机管理者根据设备及政策填写，不得把任意数值描述为验证过的需求；限制太低会中断运行。
- 文件 `mkdir`/新建提供的是同名冲突排他，不是全局 GPU 锁或可信时间戳。不要覆盖已有后缀，不删除失败产物；时间、哈希和授权记录另行封存。

## 主机命令

以下仅在明确批准本次 CUDA 运行、确认独占以及确定资源阈值后，由有用户 systemd manager 的主机操作员执行。命令不会自动重试；每次尝试使用全新后缀。先设置经确认的值，下面的 `${...:?}` 会拒绝空值。

```bash
cd /home/gbk/code/LeWorldModel/le-wm-main
export PYTHONPATH="$PWD" PYTHONDONTWRITEBYTECODE=1
SUFFIX=P5HOST20261101A
GPU_UUID='GPU-REPLACE-WITH-EXACT-PHYSICAL-UUID'
MIN_TOTAL_MIB=REPLACE
MIN_FREE_MIB=REPLACE
MEMORY_MAX=REPLACE
UNIT="p5-forward-${SUFFIX}.service"
PREFLIGHT="p5_rgb_host_preflight_20261030_codex_v1_${SUFFIX}.json"
```

用主机 GPU 管理工具核实 UUID、空闲显存、MIG 和其他进程；由操作员建立持续独占。运行以下只读检查；它设计为始终返回 2，且 `blocked_not_authorization` 是预期状态，不等于自动放行。除了 GPU 与外部授权两项必然未验证的 blocker，任何其他错误均停止。

```bash
.venv/bin/python -m experiments.spectral_reachability.p5_forward_launch_readiness_20261101 \
  --suffix "$SUFFIX" --disk-floor 10240 --runner-seconds 1800 \
  --hard-seconds 2100 --service-seconds 2400 --overhead-seconds 300
```

接着仅在主机准备启动时创建新的静态前置记录。命令按设计返回 2；必须审读记录，其 blockers 必须恰好是 `gpu_identity_memory_mig_and_runtime_unverified` 和 `external_watchdog_and_run_authorization_unverified`。每次均重新生成新后缀，不得重用或改写旧记录；记录有效期最多 300 秒。

```bash
.venv/bin/python -m experiments.spectral_reachability.p5_rgb_host_preflight_20261030_codex_v1 \
  --min-disk-free-mib 10240 --output-name "$PREFLIGHT"
sha256sum "results/$PREFLIGHT"
```

将实际输出的 64 位摘要填入 `PREFLIGHT_SHA`。检查 unit 尚不存在；一旦启动，后缀及所有产物封存，不重复运行该后缀。下面的授权字符串仅是程序接口值，不代替人的实际批准。

```bash
PREFLIGHT_SHA=REPLACE_WITH_OBSERVED_SHA256
systemctl --user show "$UNIT" --property=LoadState
systemd-run --user --unit="${UNIT%.service}" --wait \
  --working-directory="$PWD" --setenv="PYTHONPATH=$PWD" --setenv=PYTHONDONTWRITEBYTECODE=1 \
  --property=RuntimeMaxSec=2400 --property=TimeoutStopSec=30 \
  --property=KillMode=control-group --property=SendSIGKILL=yes \
  --property=LimitFSIZE=8388608 --property="MemoryMax=${MEMORY_MAX:?}" \
  --property=TasksMax=128 --property=StandardOutput=null --property=StandardError=null \
  "$PWD/.venv/bin/python" -m experiments.spectral_reachability.p5_rgb_forward_probe_supervision_20261030_codex_v1 launch \
  --authorization MAIN_THREAD_APPROVED_SUPERVISED_FORWARD_PROBE_CUDA \
  --gpu-uuid "${GPU_UUID:?}" --run-suffix "$SUFFIX" \
  --preflight-name "$PREFLIGHT" --preflight-sha "${PREFLIGHT_SHA:?}" \
  --watchdog-unit "$UNIT" --watchdog-max-seconds 2400 \
  --min-free-mib "${MIN_FREE_MIB:?}" --min-total-mib "${MIN_TOTAL_MIB:?}" \
  --runner-seconds 1800 --hard-seconds 2100 --max-age-seconds 300
```

`systemd-run --wait` 的返回码不是验收。若服务没有结束、unit/cgroup 仍有进程、状态为 partial、日志缺失/超限或哈希不一致，停止并保存现场，不运行 accept，不把残留标为完成。`systemctl --user show "$UNIT" --property=ActiveState --property=SubState --property=ControlGroup --property=Result` 和主机 cgroup/进程检查须独立确认；不要仅根据 supervisor 自报判断写者退出。

## 结果验证与显式排他封存

独立确认所有写者退出后，用原始文件实测哈希，逐个记录到外部只追加登记处；不要从程序输出猜测或手填哈希。以下路径只读：

```bash
SUP="results/p5_rgb_forward_probe_supervision_20261030_codex_v1_supervisor_${SUFFIX}"
RUN="results/p5_rgb_forward_recursive_probe_20261030_codex_v1_RUN_${SUFFIX}"
sha256sum "results/$PREFLIGHT" "$SUP/supervisor.jsonl" "$RUN/events.jsonl" "$SUP/stdout.log" "$SUP/stderr.log"
```

按观测值填入以下变量，检查前置摘要与最初记录一致，再执行新文件创建式 accept；既有文件会拒绝覆盖。它严格检查固定计划、链、向量哈希和递归输入，但不是独立硬件证明。

```bash
SUPERVISOR_SHA=REPLACE_WITH_OBSERVED_SHA256
JOURNAL_SHA=REPLACE_WITH_OBSERVED_SHA256
ACCEPT="p5_rgb_forward_probe_supervision_20261030_codex_v1_accept_${SUFFIX}.json"
.venv/bin/python -m experiments.spectral_reachability.p5_rgb_forward_probe_supervision_20261030_codex_v1 accept \
  --run-suffix "$SUFFIX" --preflight-sha "$PREFLIGHT_SHA" \
  --supervisor-sha "$SUPERVISOR_SHA" --journal-sha "$JOURNAL_SHA" --output-name "$ACCEPT"
sha256sum "results/$ACCEPT"
```

填入实测 acceptance 摘要，用 CPU 独立入口重读复核：

```bash
ACCEPTANCE_SHA=REPLACE_WITH_OBSERVED_SHA256
.venv/bin/python -m experiments.spectral_reachability.p5_forward_receipt_check_20261101 \
  --suffix "$SUFFIX" --preflight-sha "$PREFLIGHT_SHA" --supervisor-sha "$SUPERVISOR_SHA" \
  --journal-sha "$JOURNAL_SHA" --acceptance-sha "$ACCEPTANCE_SHA"
```

封存完整 preflight、原始日志、acceptance、源码及环境身份、unit 退出与进程检查、真实批准和独占记录；用外部受控存储的不可覆盖对象及校验摘要建立排他副本。若该存储或其不可变策略不可用，则标记“尚未完成外部封存”，不要伪造。校验成功仅是已保存的自报告一致性；不证明新根、科学因果、P5 admission 或独立服务终止。

## CPU 测试

2026-11-01 在项目虚拟环境、禁用字节码和 pytest 缓存下，readiness、supervision、recursive probe 三组共 57 项通过；这些测试不代表真实 CUDA 完整运行。
