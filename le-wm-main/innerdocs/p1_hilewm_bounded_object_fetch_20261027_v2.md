# Hi-LeWM bounded object fetch v2（独立冻结协议）

本次仅实现代码和离线 mock；未进行真实网络、模型加载、pickle 反序列化、CUDA 或物理。v1/main01 不修改、不续传、不删除。256KiB 是否能避免真实链路截断尚待 MAIN 探测，不能从 mock 推出网络可用。

## 固定对象和来源

仅访问 `https://zenodo.org/api/records/21353240/files/package.zip/content`，禁止 redirect、非 206、非精确 Content-Range/Content-Length 或压缩 HTTP body；不读取被拒绝响应的 body。

成员为 `package/checkpoints/pusht/main/pusht_hi_lewm_epoch15_object.ckpt`：archive 2299863924 bytes，local header offset 634194328，payload offset 634194454，compressed 113104429，expanded 122424517，CRC32 `d59f865b`，raw DEFLATE。复用既有本地中央目录/元数据绑定，首尾远端 tail SHA 绑定。缺少 ETag 不声称有服务端 immutable revision；最终 CRC/大小和实测 SHA 不替代作者配对证明。

v1 纯 helper 源码在 import 前 SHA 固定为 `961c23a8ab0af587a8169ce0163005cc36eb88ad15493fbcd0b72b9b1614cb29`。仅调用其绑定、空间、解压完整性、哈希和文件持久化辅助函数；不调用静态 pickle 扫描，不执行对象。receipt 明示 `unpickled=false/static_pickle_scanned=false`。

v1 首 4MiB payload 返回正确 206/长度头，但实收 4185588、缺 8716 bytes：证明响应提前 EOF，不证明权重损坏或具体代理故障。其 30s socket timeout 是空闲超时。v2 新增整体墙钟和小块完整接收后提交；没有暗中修复或继续旧 partial。

## 两阶段和上限

| 项 | probe | fetch |
|---|---:|---:|
| payload | 开头固定两个 256KiB | 从头完整 432 段，每段最多 256KiB |
| 控制读取 | tail65557 + header30 + name/extra96 | 同前，结束再读 tail65557 |
| 重试 | 0 | 单 payload 最多 2 attempts，全局最多 16 extra |
| reserved body cap | 589971 | 117429973 |
| phase wall | 150s | 7200s |

一次 probe + 一次 full 合计 reserved cap **118019944 bytes**，小于 120MiB；加已失败 main01 实收 4251271 仍为 122271215，小于 125829120。此历史合计不是允许反复新建任务的无限额度；任何另一次执行须重新授权。

每次请求前按完整请求长度保守扣除 body 预算，失败同样扣；另计实际 received。socket idle 10s，单 attempt wall 45s，Linux 主线程 SIGALRM 覆盖 DNS/open/header/body；phase timer 同时覆盖解压、哈希、发布。终态诊断写入在 phase timer 外，避免失败证据被超时再次打断，不承诺故障磁盘写入的硬墙钟。拒绝覆盖已有外部 timer。

probe 成功仅为 `probe_completed_not_object_verified`，保存两段原压缩字节，不解压、不发布权重。full 要求外部成功 probe report SHA，并校验 report/source/provenance、5/5 attempts、字节预算、journal SHA/hashchain 和两段文件/range/SHA。MAIN 的两次批准独立，CLI acknowledge 不是沙箱/主机授权替代。

full 使用全新 DEFLATE 流，从头请求，不复用 probe/旧 partial。每段完整校验并缓存在内存（最多256KiB）后才喂 inflater，失败段不污染流；仅 transport/truncation/deadline 可有限重试，协议错误不可重试。结束校 CRC/packed/expanded/EOF/实测 SHA，再从磁盘独立复核 SHA；以 exclusive hardlink 发布，不覆盖。`download.part` 留存为同 inode 别名，不是第二份权重。

## 失败与磁盘

独占 `results` 直接子目录，存在即拒绝。每次 attempt/headers/terminal 与每次 commit 进入 fsync hashchain journal。失败返回字节保存独立 `failed_range_NNNN.part`，已提交展开前缀保存 `download.part`；原始首错与 close/journal/source-finalization 次错分列。最终写 report；若磁盘彻底不可写，不保证 report 可落盘，但不以写入次错覆盖执行首错。无自动清理/续跑。

始终保留 200MiB floor；full 预检要求 expanded122424517 + 8MiB控制余量 + floor，probe 要求 512KiB + 8MiB + floor。完整对象约116.75MiB，有限失败段最多约4.25MiB（16次可恢复失败加最后一次失败），journal/report 按小控制产物预留。每次实际落盘再次检查 floor；不复制2.3GB archive。

## MAIN 执行命令（本 agent 未执行网络）

工作目录 `/home/gbk/code/LeWorldModel/le-wm-main`。

只读协议查看：

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m experiments.spectral_reachability.p1_hilewm_bounded_object_fetch_20261027_v2 plan
```

MAIN 先通过主机审批/`require_escalated` 批准且只运行 probe：

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m experiments.spectral_reachability.p1_hilewm_bounded_object_fetch_20261027_v2 probe \
 --output /home/gbk/code/LeWorldModel/le-wm-main/results/p1_hilewm_bounded_object_fetch_20261027_v2_probe_main01 \
 --acknowledge-probe-network
```

仅成功 terminal report 经 MAIN 独立 SHA 审核后，另行批准 full（以下变量须填实际值，不提供虚构 SHA）：

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m experiments.spectral_reachability.p1_hilewm_bounded_object_fetch_20261027_v2 fetch \
 --probe-report /home/gbk/code/LeWorldModel/le-wm-main/results/p1_hilewm_bounded_object_fetch_20261027_v2_probe_main01/report.json \
 --probe-report-sha256 "$MAIN_VERIFIED_PROBE_SHA256" \
 --output /home/gbk/code/LeWorldModel/le-wm-main/results/p1_hilewm_bounded_object_fetch_20261027_v2_full_main01 \
 --acknowledge-full-download
```

## 离线验证与冻结

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m pytest -q -p no:cacheprovider \
 tests/test_p1_hilewm_bounded_object_fetch_20261027_v1.py \
 tests/test_p1_hilewm_bounded_object_fetch_20261027_v2.py
```

实际离线结果：57 passed（v1 30 + v2 27）。覆盖 fixed ranges/caps、200/redirect/wrong headers 拒绝、截断保留/有限重试、外部 probe SHA/文件篡改、deadline/timer恢复、CRC/展开大小/发布与失败首错保留；mock 不执行网络或 pickle。plan 已本地只读执行。未创建真实 v2 probe/full 输出。

source SHA256：`60eed26bc48efec6d8e7d46bc8efa52b0192a8ddb901bd2cca66431fa75eaa92`。

tests SHA256：`d220f7b153ad3e1b283e457438d92c8e93c5cf0fe2345ff9dad2af7edf957fad`。

交付后冻结；变更须另建版本。真实探测、完整获取和静态资产检查仍未完成；下载成功亦不表示坐标/低层配对通过、learned prior 已接入或科学准入。
