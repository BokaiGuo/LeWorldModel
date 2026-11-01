# P1 checkpoint 公开 metadata 有界刷新器

## 本轮交付状态：仅实现，未联网

- 源：`experiments/spectral_reachability/p1_checkpoint_public_metadata_refresh_20261027_v1.py`
- 测试：`tests/test_p1_checkpoint_public_metadata_refresh_20261027_v1.py`
- 本文：`innerdocs/p1_checkpoint_public_metadata_refresh_20261027_v1.md`

独占新增以上三文件；保留全部旧版，不改共享 docs/progress/dispatch/closure。
已阅读 `innerdocs/p1_checkpoint_historical_contract_review_20261027_v1.md`：旧 HF
不可达不等于无元数据；当前代码、公开 config 或本地缓存时间都不能认证 checkpoint
训练 run、依赖、dataset、normalizer、camera/codec。本工具不重复 GitHub 历史查询，
只补充后续经 MAIN 审核批准后可执行的 HF 公开字节身份取证路径。

本轮仅 RAM mock tests 和 `--help`；没有实际公共请求、实际 weights hash 读取、
刷新结果目录或新公开证据。不要把本工具完成解释为 HF 已恢复或 normalizer 已认证。

## 固定范围与匿名网络边界

唯一仓库 `quentinll/lewm-pusht`，固定两个 API URL：

- official：`https://huggingface.co/api/models/quentinll/lewm-pusht?blobs=true`
- mirror：`https://hf-mirror.com/api/models/quentinll/lewm-pusht?blobs=true`

没有自定义 URL、repo、branch 或 token 参数；不读 token 环境变量、不用 HF SDK、
netrc、cookie jar、代理环境变量或认证客户端。请求仅用标准库 HTTPS GET，
显式 User-Agent/Accept/Accept-Encoding: identity；不发 Authorization 或 Cookie。
不跟随任何重定向（包括跨域 CDN / LFS），不扫描链接、不联系作者、不下载 weights/datasets。
不会读取 README 中的 URL，也不执行下载内容。

每个 metadata 响应最多 **2 MiB**，每个小文件最多 **256 KiB**。
timeout 必须为有限正数且 **≤15 秒**，包括父进程的整次请求截止检查；
网络在独立 spawn 子进程中执行，父进程到期终止它，避免 DNS/TLS/慢速读取绕过期限。
进程回收另有最多两次 0.1 秒 join，不是额外网络时间配额；操作系统调度并非硬实时保证。
无重试；最多 `sources × (1 + 选中小文件数)` 次请求。
两来源全部七种小文件均选择时，上限 16 次请求、7.5 MiB 原始 body；
默认只取两个 API，不自动下载小文件。

## 返回 commit 后的小文件读取

先验证 API id 为固定仓库、sha 为完整 40 位小写 hex commit、siblings 有界且无重复路径；
不接受 main、任意 URL、其他仓库或 private=true 响应。JSON 重复键拒绝。
只有 MAIN 在 CLI 明确选择且 API siblings 确实列出的以下文件，才可继续请求：

`README.md`、`config.json`、`stats.json`、`action_stats.json`、`normalizer.json`、
`normalizer_stats.json`、`dataset_stats.json`。

精确构造 `https://该来源域名/quentinll/lewm-pusht/raw/返回commit/固定文件名`，
不采用响应提供的下载 URL。已声明 size >256 KiB 时直接跳过；size 未声明时仍以流式上限
约束。raw 路径不会取 LFS 权重；若 stats 本身为 LFS pointer，最多保存 pointer 原文，
不转到 LFS 取实体，不声称得到 fitted stats。新出现但不在 allowlist 的文件不会自动下载。

## 留痕与失败语义

`--out-dir` 必须是全新目录，父目录须存在；mkdir(exist_ok=False) 排他占用，已有文件、
目录、悬空 symlink 都拒绝，且拒绝发生在本地文件读取和网络前。
所有内部输出使用 xb 新建，不覆盖、不删除，不自动清理失败目录。

每次请求先写 `<source>/<label>.body` 原始 bytes，再写对应 receipt JSON，记录固定 URL、
HTTP status（未知为 null）、body SHA256、rawbody_path、字节数、complete、error 和时间。
成功、HTTP403/404/429、重定向、无响应、超时、截断或超限都保存 raw；
完全未取得 body 时保存空文件，其 SHA 是空 bytes 的真实 SHA，不是缺失填 0。
超时/超限时保存已收到的有界前缀，complete=false，摘要只代表该前缀，不能冒充完整远端文件。
已知 Content-Length 恰等于上限可判完整；没有长度且恰触及上限则保守判截断，不多读一字节。
异常仅记录固定错误代码，不序列化异常文本、响应 headers、认证值或环境值。
stdout 仅输出最终 report 路径、SHA、请求数和 unknown 状态，不打印 body 或 token。
磁盘故障可能阻止留痕落盘；工具不能保证在磁盘不可写时仍保存日志，且不会删掉部分结果。

目录内容包括 protocol.json、每来源独立 raw/receipt、最终 report.json。
已持久化 raw/receipt 不依赖最终 report 是否成功写出。

## 只读 hash 对照及不能推出的结论

默认本地文件是 `/home/gbk/.stable_worldmodel/checkpoints/pusht/lewm/weights.pt` 与
`config.json`，可由 CLI 指定其他明确现有路径。仅读取字节，不反序列化权重、不加载模型。
weights 上限 8 GiB、config 上限 256 KiB；拒绝 symlink/非普通文件，检查读取前后
inode/size/mtime/ctime，变化即 unknown，不继续拿它认证匹配。
缺文件/读取失败也保留公开请求日志，本地 hash 为 null，不填 0。

每个来源分别记录：

- API weights.pt 的公开 LFS SHA256 与本地 weights SHA256 的相等/不等/unknown。
- API config.json 的 blobId 与本地 config 的 Git blob SHA1 的相等/不等/unknown。
- 若选择且取回该 commit 的 config，公开 body SHA256 与本地 SHA256 的对照，
  以及 body Git blob 与 API blobId 的独立一致性检查；不合并或掩盖矛盾。

official 标记 official_domain；mirror 标记 mirror_domain_not_official_attestation。
即使两者 commit/hash 完全一样，cross_origin_independence 仍为 unknown，
镜像不升级为官方背书；commit 不同时分别保存，不拼接两个版本。

**匹配只证明相应来源声明/返回的字节身份，不证明训练统计来源。**
即使公开目录包含 stats 文件，也没有自动的 weights→training-run→fitted-normalizer
认证链。training_stats_provenance 始终 unknown，training_normalizer_certified、
P1_admission、official_export_manufactured 始终 false。
仍需明确绑定 weights digest 的训练 run/source/dependency/dataset/normalizer export；
历史 camera/codec 同样 unknown。工具不生成“官方 export”，不修改部署 stats、camera 或模型。

## 验证与 CLI（网络示例未执行）

工作目录：`/home/gbk/code/LeWorldModel/le-wm-main`。
先 RED 后实现，**50 passed in 0.14s**。全部网络、目录、本地文件 hash 测试使用 RAM mock；
没有真实 socket、子进程网络或真实结果目录。覆盖固定 URL、token 不发送、重定向不跟随、
worker/父进程超时、失败 raw 前缀、大小预算、API commit/重复路径拒绝、NEW 排他、
CLI 参数、本地只读 hash、两个来源分离、hash 矛盾以及 unknown 不升级等。
测试不是科学证据，也不是实际 HF 可达性证据。

本轮已执行的无网络命令：

```bash
.venv/bin/python -m pytest -q -p no:cacheprovider tests/test_p1_checkpoint_public_metadata_refresh_20261027_v1.py
.venv/bin/python -m experiments.spectral_reachability.p1_checkpoint_public_metadata_refresh_20261027_v1 --help
```

以下 **仅供 MAIN 全文审核且明确批准网络之后** 使用；本轮没有执行：

```bash
.venv/bin/python -m experiments.spectral_reachability.p1_checkpoint_public_metadata_refresh_20261027_v1 --main-approved-network --out-dir results/p1_checkpoint_public_metadata_refresh_20261027_v1_MAIN01 --sources official mirror --timeout 15 --small-files README.md config.json stats.json action_stats.json normalizer.json normalizer_stats.json dataset_stats.json
```

不带 --main-approved-network 即被拒绝；该参数是操作者对已有 MAIN 批准的明确声明，
不是工具自行获得批准的机制。网络/沙箱权限仍须正常授权，不能借此参数绕过。
本轮无训练、adapters/Q 拟合、CPU 神经实验、model/physics、删除或发布。
