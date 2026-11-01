# Hi-LeWM main object：有界选择性取件器（MAIN review交付）

2026-10-27。只新增源码/测试/本说明。**本轮没有实际网络、checkpoint取件、pickle执行、神经或物理**。
应用api-and-interface-design的边界验证与分离状态原则：对象字节验证、静态容器扫描、模型可信加载/配对是三种不同状态。
作者没有配对SHA不再作为下载阻断；本次取件将产生实测对象SHA，随后MAIN另行认证内嵌低层。

## 固定输入，不允许换权重/镜像/URL

- 唯一官方URL：`https://zenodo.org/api/records/21353240/files/package.zip/content`。
- 唯一成员：`package/checkpoints/pusht/main/pusht_hi_lewm_epoch15_object.ckpt`。
- ZIP总长2,299,863,924；local-header offset634,194,328；method8，实际flags8（使用data descriptor）。
- 压缩113,104,429；展开122,424,517 bytes；CRC32 `d59f865b`。
- 本地完整中央目录与原report/原tail的硬编码SHA共同验证；命令行不接受其它成员、offset或大小。
- source SHA：`961c23a8ab0af587a8169ce0163005cc36eb88ad15493fbcd0b72b9b1614cb29`。
- tests SHA：`d8f222e77daee4f2889c5b6a58bf73a595767f84cee53be235e046c1f6c23bb8`。

## 实际运行合同

下载前检查 `free >= 340,528,325 bytes`：122,424,517对象 +8MiB控制产物余量 +200MiB磁盘floor。
逐次写expanded文件与journal前再检查200MiB下限；共享盘被外部进程并发填满时无法提供原子空间预留，检查失败即停，不删资产。
正常最终对象占约116.75MiB，控制产物很小；不保留113MB压缩副本、不复制整73MB既有报告。

总请求body保守预留cap **120MiB=125,829,120 bytes**，包括archive localheader/name/extra及两次tail；
不是仅对权重payload计cap。HTTP传输协议本身的headers/TLS不是该应用层body计数。
压缩payload最多4MiB/range，response流每次≤64KiB，解压每次≤1MiB且总expanded严格限122,424,517。
不自动重试、不resume，不为“补齐预算”读取无关数据。

执行顺序：

1. 绑定已保存414项完整中央目录，拒绝目标重复、symlink或实际元数据变化。
2. 再读官方65,557-byte尾部并核对冻结SHA。所有请求只用同一固定HTTPS URL，禁止redirect；要求206、精确Content-Range/Content-Length、identity编码。
   返回200/3xx/403/错误Range时正文不读。ETag/Last-Modified若服务器提供则跨请求保持一致；未提供不虚构强validator。
3. 精确取30-byte localheader、name/extra，验证签名/flags/method/name；非descriptor模式也必须核对header CRC/size。
   此目标flags8意味着CRC/size以中央目录为准，流结束重新核对；不需要读后续data descriptor或整包。
4. 连续精确Range只取该成员压缩payload，raw DEFLATE流式写`download.part`。维护compressed SHA、object SHA、CRC与两种size。
   接收块先计应用层received，再交解压器；异常留journal与已有partial，绝不替换其它checkpoint。
5. 全量CRC/size/DEFLATE EOF通过、再次远端tail SHA通过、持久化文件独立重读SHA相等后，fsync并用`os.link`独占发布正式`.ckpt`。
   **保留`download.part`作为同inode别名**，没有删除操作，通常不另占122MB。勿按两个pathname机械相加估计真实磁盘块。
6. 自动做静态容器inventory和pickle opcode扫描；写独占`static_inventory.json`与`report.json`。

`events.jsonl`每条带sequence/previous/hash，flush+fsync；有range attempted/headers/completed/partial及每payload range持久化checkpoint。
失败不发布未通过字节校验的正式对象；如果静态扫描或后续收据失败，已经字节验证的对象可能存在，但整体report明确failed。
first_error保留执行首错，final-source-hash/journal/close异常独立列secondary_errors，不覆盖首错。
如果磁盘/设备已无法写诊断，journal/partial仍尽力保留，不能承诺在I/O彻底失败时生成完整report；原异常不会被report写错覆盖。

## 下载后静态检查不等于加载

不导入torch，不调用pickle.load/loads，不import GLOBAL引用，不执行REDUCE/BUILD/构造函数。
`zipfile.infolist()`列内部名字/大小/CRC；禁止内部traversal/重复路径/symlink/加密。
只读取累计≤8MiB的`.pkl`供`pickletools.genops`分析，**不读取tensor storage**。
记录GLOBAL/INST及能静态确定的STACK_GLOBAL；不确定项计数，所有class_list_complete保持false。
ZIP成员CRC由zipfile读取检查；内层pickle解析错误在扫描结果保留，不能称兼容或安全。
legacy/non-ZIP仅扫描前8MiB内最多8段protocol pickle前缀，不触及反序列化，标complete_inventory=false；可能无法完整识别全部类。
报告是类/模块**引用**清单，不是对象重建、完整依赖闭包或任意pickle可安全加载的证明。
静态扫描阶段异常不追认配对；字节SHA/CRC收据与加载认证独立。

## MAIN审查与执行命令

仓库根是 `/home/gbk/code/LeWorldModel/le-wm-main`。只读plan已实际执行，验证固定中央目录，不创建输出/不联网：

```bash
cd /home/gbk/code/LeWorldModel/le-wm-main
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m experiments.spectral_reachability.p1_hilewm_bounded_object_fetch_20261027_v1 plan
```

离线测试已实际 **30 passed**；小型mock ZIP/DEFLATE fixtures，无模型/模拟，默认pytest临时目录仅用于小fixture：

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m pytest -q -p no:cacheprovider tests/test_p1_hilewm_bounded_object_fetch_20261027_v1.py
```

以下是 **MAIN review后、另经host网络approval才可执行** 的唯一对象取件命令。本agent没有运行它，也没有创建这个目标目录。
如该目录已存在，停止，使用新独占后缀；不删除、不覆盖、不恢复旧partial。

```bash
cd /home/gbk/code/LeWorldModel/le-wm-main
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m experiments.spectral_reachability.p1_hilewm_bounded_object_fetch_20261027_v1 fetch \
  --project /home/gbk/code/LeWorldModel/le-wm-main \
  --output /home/gbk/code/LeWorldModel/le-wm-main/results/p1_hilewm_bounded_object_fetch_20261027_v1_main01 \
  --acknowledge-network-download
```

CLI确认flag是防误操作标记，**不替代工具require_escalated/host approval**，也不伪装可在沙箱绕过网络。
fetch成功后静态扫描已自动完成，无需重跑；可人工阅读report.json、static_inventory.json及events.jsonl。
仅需独立重新静态打印时（输出stdout，不另取网络）可用`static --object <已核SHA的object路径>`；该子命令本身不认证输入来源。

30 tests覆盖实际固定目录、路径拒绝、200/302/403/错range/长度/编码/host均不读body、下载cap、truncated计账、
journal secondary不覆盖consumer首错、validator变化、流式CRC/SHA/膨胀/尾随拒绝、floor、zip/legacy纯opcode、
真实localheader flags8 fixture、全链mock成功/失败/不覆盖、source消失保留首错、preflight在network前拒绝。
mock成功不是实际download成功；当前尚无此对象实测SHA、类清单或tensor配对结论。

成功取件后只解除资产可用与字节收据门；MAIN仍需可信加载审批、类依赖固定、内嵌低层逐tensor认证，才可CUDA接口测试。
本交付不要求作者先提供配对SHA，不训练，不另下flat/weights，不触发Experiment1/P1科学准入。
