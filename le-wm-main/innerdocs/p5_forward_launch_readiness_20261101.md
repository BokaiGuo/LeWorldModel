# P5 递归探针：运行前置和主机交接

新增只读检查 `p5_forward_launch_readiness_20261101`，重用原 PREP02、监督器和 host preflight；不修改固定源码和数据。检查来源与合同、输出名称冲突（包括悬空符号链接）、results 目录访问、用户总线、静态 host 条件以及 runner < child hard deadline、service >= child deadline + 显式预留。它不创建文件、不查询 GPU，永远返回 2；预留只是操作者选择，不是运行时间保证。检查到启动之间可能发生竞争；原 supervisor 的独占 mkdir 和原 runner 的独占目录才是实际防覆盖门，检查不是锁。

2026-11-01 当前沙箱只读检查：合同与来源通过、驱动三个版本一致、可用空间约 18130 MiB、用户总线 socket 可见；三个 `/dev/nvidia*` 节点不可见。不能据此推断主机节点状态。GPU UUID、显存、占用、MIG、实际 user service cgroup、磁盘配额和限时均未验证。原 journal 8 MiB 是读取上限，不是写入限额；日志也无硬限额。以下数值仅作检查示例，不是实测预算。

## 主机运行前

在目标 host 用户环境执行，不访问 GPU：

```bash
cd /home/gbk/code/LeWorldModel/le-wm-main
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 .venv/bin/python -m experiments.spectral_reachability.p5_forward_launch_readiness_20261101 --suffix P5_HOST_01 --disk-floor 1024 --runner-seconds 900 --hard-seconds 1200 --service-seconds 1800 --overhead-seconds 600
```

检查 errors 中除两项明确未验证的 GPU/授权阻断外无其他阻断；根据实际运行策略重新选择数值和全新 suffix。人工核验 GPU 独占/占用、UUID、MIG、容量及用户 systemd 可用性；没有已测得的显存阈值，不能安全地替操作者指定。仅在另行授权且主机检查满足后，按既有 [监督交接](p5_rgb_forward_probe_supervision_20261030_codex_v1.md) 中命令生成新鲜 host preflight，检查内容并记录实际 SHA，在 300 秒内用同一用户的 `systemd-run --user` 启动该监督器。预检返回 2 是预期，不表示运行授权。原监督命令、所有必要参数和 saved acceptance 命令见交接文档；不要调用可绕过监督的 probe run。

检查 service/child 全部停止并留存实际 supervisor 和 journal SHA 后再执行 saved acceptance。失败和 partial 原样保留，不自动重试；新尝试用新 suffix。已知四枝、272 次显式神经 API 调用及 52 次 reset 的结果仅是同输入自报告的一致性和有限差分，不是独立泛化或 P5 科学通过。

## 参数和接收检查补充

完整的 preflight、`systemd-run` 和 saved acceptance 命令见上述监督交接。命令中的 `<unique>`、unit、物理 GPU UUID、`min-free-mib`、`min-total-mib`、磁盘下限、三个 deadline 和预检 SHA 均须由主机操作者填写；`runner < hard`，且 service 至少覆盖 hard 加上明确的启动和收尾预留。`max-age-seconds` 为 1–300，实际预检到启动不得超时。文首 900/1200/1800 秒和 1024 MiB 仅为离线示例，不是安全或资源预算。必须确认实际 GPU 空闲、无竞争、MIG 和设备权限，并另行制定日志及磁盘限制；现有 8 MiB journal 限制只是读取限制。生产运行需要另行明确授权。

验收前在目标用户环境检查 `systemctl --user show <unique-unit>.service --property=ActiveState --property=SubState --property=ControlGroup --property=Result`，并独立检查该 cgroup 中所有子进程已退出；不能仅凭一次 supervisor 记录推断无残留进程。未能确认停止时不要计算最终 SHA 或验收。留存 host preflight、supervisor、journal 的 SHA，按交接命令执行 acceptance，且为下面的接收复核将 acceptance 命名为 `p5_rgb_forward_probe_supervision_20261030_codex_v1_accept_<unique>.json`。

验收文件生成后，独立计算其 SHA 并在同一来源环境执行只读接收复核（无 GPU 查询；所有 SHA 必须是实际留存值，而非示例）：

```bash
sha256sum results/p5_rgb_forward_probe_supervision_20261030_codex_v1_accept_<unique>.json
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 .venv/bin/python -m experiments.spectral_reachability.p5_forward_receipt_check_20261101 --suffix <unique> --preflight-sha <preflight-sha256> --supervisor-sha <supervisor-sha256> --journal-sha <journal-sha256> --acceptance-sha <acceptance-sha256>
```

该复核重新执行固定来源、监督事件和完整 journal review，并逐字节匹配规范化 acceptance；不会创建文件。SHA 应通过可信的独立渠道留存，否则只能发现相对给定 SHA 的不一致。此工具不能证明服务、子进程已经停止，也不能证明实际硬件执行；中途失败、partial、来源变化或任何 SHA 不匹配均不得作为成功接收。
