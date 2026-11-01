# Hi-LeWM整包离线验证工具 v1

本模块只验收MAIN已关闭下载并交付的**最终文件**，或只读已有终态JSON诊断失败。不读取、stat、轮询或续传 `results/hilewm_archive_MAIN_20261028_v1/archive.part`，不发网络请求、不调用pickle/torch、不自动extract、不执行包内代码。仅新增同stem source/tests/本文；旧证据及模型不修改。

## MAIN整包下载已终态失败（更新，不再等待/重试）

只读`results/hilewm_archive_MAIN_20261028_v1/report.json`，实际SHA `19c526da4a8e3da6b340addd23289df2bd3a57e5a18169290f9bd13dd4f0783b`：schema `MAIN.hilewm_whole_archive_result.v1`，status `failed_preserved`，HTTP200、curl exit18，实收16,634,096 bytes，缺**2,283,229,828 bytes**。curl transfer time276.626614s（外层wall276.659385s）。这是正确区分的**HTTP200 early EOF截断，非403**；无整包MD5/实测SHA验收，不可解包。旧part保留，不自动重试或换传输。report的`actual_saved_bytes`是MAIN收据值，本agent没有重新读取part验证。

直接执行以下只读诊断，**预期exit1**表示拒绝截断资产，不是诊断器故障；exit2为收据SHA/schema等不符合。不会打开receipt中`filename_effective`指向的part：

```bash
cd /home/gbk/code/LeWorldModel/le-wm-main
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m experiments.spectral_reachability.agent_hilewm_asset_verify_20261028_v1 receipt \
  --download-report /home/gbk/code/LeWorldModel/le-wm-main/results/hilewm_archive_MAIN_20261028_v1/report.json \
  --download-report-sha256 19c526da4a8e3da6b340addd23289df2bd3a57e5a18169290f9bd13dd4f0783b
```

receipt mode只接受指定schema的terminal失败、固定URL、actual_saved_bytes/curl size_download/exitcode一致；200+18还需errormsg缺失字节精确一致。输出完整性false、ZIP读取不准入、baseline unknown；不把这16MB当任何有效checkpoint或macro bank。

## 已有材料中的候选来源（不是下载队列）

|候选/入口|已存出处与实际范围|交给MAIN的选择/缺口|
|---|---|---|
|`https://zenodo.org/api/records/21353240/files/package.zip/content`|冻结zenodo_api.json列出的唯一整包对象；旧Range两次截断，MAIN整包又200/18截断|同一资产已失败，**不安排重复请求**。如MAIN明确另授权不同transport/供件主机策略，仍须同size/MD5/tail验收；本文无可用性承诺|
|`https://zenodo.org/records/21353240` / `https://doi.org/10.5281/zenodo.21353240`|API已有self_html/doi；论文缓存HTML也指此DOI|官方元数据/作者联系入口，不是独立镜像，不能把两个URL当不同资产源。可请求作者离线供件或官方可验证镜像及digest；不自动跟latest|
|`https://huggingface.co/quentinll/lewm-pusht`|已存`texts/package__code__h_le_wm__checkpoints.py.source.txt:18`明确是flat低层config+weights来源|**不是Hi-LeWM高层整包/对象镜像**。已有本地候选，无需再下；不能解除high/macro bank缺口|
|`https://github.com/lucas-maes/le-wm.git`|已存`THIRD_PARTY_LEWM.md.source.txt:19`和code README|上游LeWM源码依赖，不是高层checkpoint镜像，不clone latest补权重|
|作者另发高层对象/完整包镜像|已读33文本、API及论文缓存未认证具体独立mirror URL/revision/digest|**unknown**；最小外部输入是作者/MAIN提供实际来源、许可、不可变版本、整包MD5或与官方成员绑定的SHA。不虚构镜像链接|

历史`report.hf.mirror`的403针对未认证HF generator/flat目标，并非此次Zenodo HTTP200/18，也非Hi-LeWM高层镜像存在证明。hf-mirror不进入新请求清单。来源清单纯离线整理，未打开任何URL或复探失败。

## 三层状态

`report.layers`各字段为true/false/null（unknown），不得合并成“baseline复现成功”。

1. `download_complete=true`：全包字节数2,299,863,924和官方MD5 `aaf78b8b4fd6f5d1f5e720927951152f`通过，报告另存**实测**SHA256；官方API没有给成员SHA。大小/MD5失败前该字段null。失败详情含expected/actual/missing字节，未完成下载不当成0分。
2. `zip_safely_readable=true`：冻结tail、414项完整目录元数据、所有路径/类型、全部成员实际展开大小/CRC、33份已存文本的SHA全部通过。含所有checkpoint在内，外层ZIP逐成员流式读到EOF，总展开2,500,619,520 bytes，最大单项223,828,979 bytes；不会打开checkpoint内部容器或反序列化它。这里“safe”仅指有界ZIP字节读取/路径检查，**不是pickle安全证明或恶意文件沙箱**。
3. `baseline_runnable=null`永远保持unknown：仍缺可信类依赖、embedded low逐tensor/config配对、原数据/normalizer身份、具体protocol及-C bank绑定、第三方许可/环境revision审核、MAIN CUDA接口验证。`baseline_missing_evidence`逐项列出；对象/config同包同源并不证明内部参数/配置相容。P1/freshness不开放。

完整成功状态为`archive_verified_baseline_unestablished`，科学准入false；任何读取/校验失败为`rejected`，错误code/detail保留，CLI exit1。reference/CLI/输出错误exit2。不得只检查exit0就宣布可运行baseline。

## 冻结来源与配对范围

`load_binding(project)`只读现有小参考文件：

- `results/p1_learned_subgoal_remote_inventory_20261027_v1/zenodo_api.json`，SHA `2fc221128c12b609b86dfb7810eda46f2399bc796348f36c5d7e016ab25dd773`，固定record21353240、URL、size、MD5、record级CC-BY-4.0。
- 同目录`archive_tail.bin`，65,557 bytes，SHA `43a0c87d4f1b652caccbc26ac5e1b1ed73c0c6cf97c8e00683ecc1bdd8c16aef`；完整414项中央目录的冻结来源。尾部ZIP视图offset加2,299,798,367还原真实整包offset。
- `results/p1_hilewm_text_contract_20261027_v1/contract.json`，SHA `c25edf542f38faa9cf0150df4822198d28753e2cf77a0d243e990ffabe849ded`；其sources下33份文本均校本地SHA、目录大小，整包对应成员逐一校同SHA。包括PushT main config、registry、models/jepa、latent_action、eval、policies、正式matrix、checkpoint README及第三方说明。

主对象固定为`package/checkpoints/pusht/main/pusht_hi_lewm_epoch15_object.ckpt`，header offset634,194,328，packed113,104,429，expanded122,424,517，CRC`d59f865b`，method8/flags8。全目录每项比较name/size/packed/CRC/method/flags/header offset/external_attr/create_system；不允许用其它variant、training weights替代。关键对象与33份文本都是required；缺成员在`missing_members`和inventory mismatch错误中列全。源文本不匹配为`source_config_mismatch`；不是宽松YAML字段猜测。

zip路径拒绝绝对/父目录/空或点段/反斜杠/冒号/NUL/控制字符；拒绝重复、symlink/设备/FIFO/socket、加密和非stored/deflate算法。全部成员按冻结大小流式读取，zipfile还检查local filename、overlap和CRC；无路径写入，因此不自动安装包。

## MAIN准确调用

只看协议/小参考来源（本agent可执行，不触碰候选整包）：

```bash
cd /home/gbk/code/LeWorldModel/le-wm-main
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m experiments.spectral_reachability.agent_hilewm_asset_verify_20261028_v1 plan
```

**等MAIN最终文件且下载进程已关闭后**，MAIN填真实最终绝对路径，不得仍为`.part/.partial/.tmp/.download`（任意suffix包含即拒绝，未open候选）。`--acknowledge-final-file`是MAIN交付声明，不自动推断下载完成，也不是主机授权替代。不存在的最终路径输出FileNotFoundError，不轮询。

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m experiments.spectral_reachability.agent_hilewm_asset_verify_20261028_v1 verify \
  --archive "$MAIN_FINAL_HILEWM_ARCHIVE_ABSOLUTE_PATH" \
  --acknowledge-final-file \
  --output /home/gbk/code/LeWorldModel/le-wm-main/results/agent_hilewm_asset_verify_20261028_v1_MAIN01.json
```

输出仅exclusive `x`新文件，父目录须存在，已有文件拒绝覆盖；不指定output则stdout。不要把输出指向MAIN下载目录或旧sealed证据。应用层报告cap2MiB；读块1MiB；默认墙钟900s（CLI可缩短不可加长），在每块/成员边界协作检查，不保证故障磁盘阻塞的硬中断。输入通过O_NOFOLLOW/O_NONBLOCK只读打开、要求regular file，用同一fd贯穿hash与ZIP读，结束比对fd dev/inode/size/mtime/ctime及path身份；并发变化拒绝并撤销download_complete。仍要求MAIN保管不可变输入，不是对恶意并发管理员的安全边界。

RAM helper `inspect_stream(stream,binding,report,tick)`允许tiny fixture注入以覆盖破损流；CLI不接受自定义MD5/size/成员清单/跳过CRC，生产binding必须从上述固定SHA加载。RAM计数/fixture通过不是实物验收。完整归档校验至少顺读2.3GB并解压外层约2.5GB，这是离线压缩/哈希算术，不是CPU神经推理。

## 测试与未完成

```bash
OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m pytest -q -p no:cacheprovider tests/test_agent_hilewm_asset_verify_20261028_v1.py
```

实际**61 passed（0.15s）**。测试使用小型RAM ZIP与已存小参考文件；不打开MAIN的part、不读真实checkpoint、不联网。覆盖三层分离、缺成员/配置篡改、MD5/size/tail、危险路径/特殊类型/加密/算法、元数据变异、实际body CRC损坏、local header、墙钟、part预拒绝、fd/path变动撤销准入、关闭资源及report独占拒绝覆盖。另实际运行plan成功；上述真实terminal JSON的receipt诊断返回预期exit1，精确标注HTTP200/18截断和2,283,229,828-byte缺口。未生成实物ZIP验收结果。冻结SHA在交付回复中给出。

未完成：MAIN当前整包下载已终态失败，无完整最终文件供正向验收，整包实测SHA未知；可信反序列化、依赖闭包、许可细分、low配对、数据统计/-C bank、CUDA/physics及科学准入均未完成。本agent不补下载、不训练、不自行运行后续步骤。
