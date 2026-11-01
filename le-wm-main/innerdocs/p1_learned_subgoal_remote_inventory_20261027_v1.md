# Learned-subgoal 远端 inventory 与 ZIP 中央目录实查

2026-10-27；本轮网络调用均使用 require_escalated。仅新增独占目录
`results/p1_learned_subgoal_remote_inventory_20261027_v1/`；未改原 metadata、接入文档、封存源码或总账。
按 citation-check 区分官方 registry 声明、HTTP 可达性、目录证据和真正权重验证。

## 1. 实际结论

**Hi-LeWM 的真实归档文件名/大小阻断已部分解除**：官方 Zenodo 206 Range 成功，完整中央目录有414项。
未下载整包、未读取任何 entry 内容、未解压、未反序列化 pickle、未执行模型或物理。
两个独立尾部请求共 **131,114 bytes**；连同官方记录 JSON，共 **136,155 bytes** body；远低于2MiB。

**SAGE 的服务器库存阻断未解除**：HF官方 API 与3个目标 HEAD 均 `URLError: Network is unreachable`；
hf-mirror.com API 与3个目标 HEAD 均 HTTP403。不能将403解释为404，不能从网络失败推断文件不存在。
两端实际 inventory、immutable model revision、模型 license 仍 unknown。未使用额外代理/镜像/域名绕过。
`pusht_visual_generator.pt`、`pusht_lewm_object.ckpt` 的存在性均**尚未验证**；原 registry 的 local_only 也不构成服务器不存在证明。

## 2. 请求、范围与版本绑定

官方记录 `https://zenodo.org/api/records/21353240` HTTP200，5041 bytes，ETag `"3"`；
body SHA256 `2fc221128c12b609b86dfb7810eda46f2399bc796348f36c5d7e016ab25dd773`。
record license 声明 `cc-by-4.0`，不外推为未读的包内全部模型/代码许可证。
唯一 `package.zip` 2,299,863,924 bytes，记录 MD5 `aaf78b8b4fd6f5d1f5e720927951152f`；未读取整包，故**未验证整包 MD5**。

归档 URL：`https://zenodo.org/api/records/21353240/files/package.zip/content`。
两次请求均显式 `Range: bytes=2299798367-2299863923`，`Accept-Encoding: identity`；
HTTP206、Content-Length65557、Content-Range `bytes 2299798367-2299863923/2299863924`、Accept-Ranges bytes。
两次尾部 SHA256 完全相同：`43a0c87d4f1b652caccbc26ac5e1b1ed73c0c6cf97c8e00683ecc1bdd8c16aef`。
服务器未提供该响应 ETag/Last-Modified；明确不声称具备服务器强 validator。
中央目录 offset **2,299,812,772**，长度 **51,130**，完整落在尾部内；EOCD414项，无需再读取目录范围。
中央目录 SHA256 `75e0036bcb4014b30c297583fb1d96983cdcc3dbfa09a77fcf7f058e73eda67b`。

原始尾部保存 `archive_tail.bin`；自写struct解析与 stdlib `zipfile.infolist()` 独立核对414项及checkpoint大小/CRC。
只调用 infolist，未调用 read/open/extract entry。若 Range 返回200，代码在 body read 前关闭；mock测试验证这一点。
headers、URL、range、status、body字节数/hash及错误均在 report/corroboration 中；HEAD不读body，未跟随redirect。

## 3. 实际 checkpoint 文件清单

以下路径统一相对 `package/checkpoints/`，压缩/展开大小单位均 bytes。
均为中央目录声明，不是已下载权重的实测 SHA。CRC32 是ZIP校验值，不是密码学配对证明。
AppleDouble `__MACOSX/.../._*` 与目录项不列为模型。

| 实际路径 | 压缩 | 展开 |
|---|---:|---:|
| cube/main/cube_hi_lewm_epoch15_object.ckpt | 113206737 | 122437603 |
| cube/main/cube_hi_lewm_epoch15_weights.ckpt | 204242779 | 222367003 |
| pusht/main/pusht_hi_lewm_epoch15_object.ckpt | 113104429 | 122424517 |
| pusht/main/pusht_hi_lewm_epoch15_weights.ckpt | 203833486 | 222331803 |
| pusht/fixed_stride_dim8/pusht_hi_lewm_fixed_stride_dim8_epoch15_object.ckpt | 113067762 | 122425355 |
| pusht/fixed_stride_dim8/pusht_hi_lewm_fixed_stride_dim8_epoch15_weights.ckpt | 203608302 | 222218651 |
| pusht/fixed_stride_dim32/pusht_hi_lewm_fixed_stride_dim32_epoch15_object.ckpt | 113104264 | 122462778 |
| pusht/fixed_stride_dim32/pusht_hi_lewm_fixed_stride_dim32_epoch15_weights.ckpt | 203749911 | 222329627 |
| pusht/probes/phase_b/pusht_decoder_probe_phase_b.pt | 16907743 | 18294079 |
| pusht/probes/phase_b/pusht_decoder_probe_phase_b_last.ckpt | 163466630 | 177126187 |
| pusht/probes/phase_b/pusht_decoder_probe_phase_b_epoch20.pt | 16907980 | 18294981 |
| pusht/probes/phase_a/pusht_decoder_probe_phase_a_last.ckpt | 163407384 | 177126123 |
| pusht/probes/phase_a/pusht_decoder_probe_phase_a_epoch10.pt | 16894283 | 18293909 |
| pusht/probes/phase_a/pusht_decoder_probe_phase_a.pt | 16894020 | 18293071 |
| pusht/vq/vq128/pusht_hi_lewm_vq128_epoch50_weights.ckpt | 204978106 | 223828979 |
| pusht/vq/vq128/pusht_hi_lewm_vq128_epoch50_object.ckpt | 113633778 | 122926367 |
| pusht/vq/vq16/pusht_hi_lewm_vq16_epoch50_weights.ckpt | 204926690 | 223807475 |
| pusht/vq/vq16/pusht_hi_lewm_vq16_epoch50_object.ckpt | 113627121 | 122916955 |

完整414项含config/code路径保存在 corroboration.json；report.json含相应local header offsets、压缩方式、CRC32。
PushT main object CRC32 `d59f865b`；main weights `7e80276b`。二者是两种序列化资产，**不能仅根据同目录将其称高层/低层配套模型**，也不先断定必须两者都下。

## 4. 最小可定位组件与下一次授权边界

不应为了节省下载量改选18MB probe，它不是已证可替代正式 learned-subgoal generator 的资产。
标准 PushT 首选候选路径为：

```text
https://zenodo.org/api/records/21353240/files/package.zip/content
  ::package/checkpoints/pusht/main/pusht_hi_lewm_epoch15_object.ckpt
  ::package/checkpoints/pusht/main/config.yaml
```

其中 `::` 表示ZIP成员定位，**不是服务器独立可GET URL**。
object 展开122,424,517 bytes；同目录 config.yaml 压缩916、展开2198 bytes；
两者候选压缩payload合计113,105,345 bytes（不含local-header开销），展开122,426,715 bytes。
这是后续选择性取件的体积估计，不是本次已经批准或已完成的权重下载。
仍需首先读取并审查小型控制资料：

| 成员路径 | 压缩 | 展开 |
|---|---:|---:|
| package/checkpoints/README.md | 696 | 1738 |
| package/code/h_le_wm/checkpoint_registry.yaml | 353 | 1450 |
| package/checkpoints/pusht/main/config.yaml | 916 | 2198 |

**本轮没有读取/解压这些成员**，因为授权明确限定中央目录。目录没有成员内容SHA256：
generator真实接口/正式horizon、需要object还是weights、低层配套来源/URL/hash、内含tensor坐标与本地LeWM兼容性仍待这些源码/配置确认。
因此当前可以给出精确资产定位与体积，不能声称已取得可直接比较的完整组件闭包。
下一次最小申请应是这些小型成员的有界Range+内存解压审查（另行授权），而非整包或神经运行；审查后才决定是否选择性获取约113MB object。
不凭 `.ckpt` 后缀执行pickle，不凭paper超参数推断实际bundle接口，不将Hi-LeWM资产冒称SAGE visual generator。

SAGE若取得可用官方库存/作者小型配对manifest，先前已固定源码声明的最小visual组件仍是
`pusht_visual_generator.pt` + `pusht_lewm_object.ckpt`，声明总301,927,570 bytes；但本轮未验证这两个HTTP资源可获取。
不能给出带假immutable revision的下载命令，不能把现有本地weights.pt配成已认证generator backbone。

## 5. 文件与离线检查

独占结果目录下：probe.py（有界读取器）、report.json、zenodo_api.json、confirm_directory.py、
archive_tail.bin、corroboration.json、verify_saved.py。没有HF成功响应文件，失败收据已保留。

```bash
cd /home/gbk/code/LeWorldModel/le-wm-main
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B results/p1_learned_subgoal_remote_inventory_20261027_v1/verify_saved.py
```

实际5tests通过：200不读body、错误Content-Range不读body、目录截断/数目错拒绝、
saved尾部/hash/独立解析核对、总输出≤2MiB及非神经声明。测试无网络/tmp/模型/物理。
仅证明元数据读取与目录证据；不是模型兼容、learned-prior有效性或Experiment1科学准入。

report.json SHA256 `7feaf480934462619149c72e293de7df09cf55835f1001a8f7d88f9194b26c66`；
corroboration.json SHA256 `26ff285fd9bc6206f91a855e2de02d73597d918ce683a8785cc8756081635a48`。
