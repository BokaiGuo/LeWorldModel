# Hi-LeWM storage → NPY：data-only v1

独占source/tests/innerdoc，使用experiment-code的固定输入、最小实现与回归要求；不训练、不运行neural/physics、不执行pickle/foreign classes。worker仅RAM合成ZIP与已封存JSON读取，不生产导出。MAIN审后运行。

## 固定来源和输出含义

唯一IR：`results/agent_hilewm_inert_pickle_ir_20261028_v1_MAIN01.json`，6,103,904 bytes，SHA `709b31e28bee8039d577668cbd9613347be1d070a7619e3db3ca61e3674cabbe`。绑定IR source SHA `6752c9eb0b4351e780a62198ab15948b4cb2bd181ebfd8325d07db253c8aec10`、三文件闭包、原static report与decode/source/license链；从已保存符号IR重新遍历nn元数据并逐字段相等，不读取/执行pickle。

唯一ZIP仍为`results/agent_hilewm_bounded_deflate_decode_20261028_v1_MAIN01/decoded.unverified.ckpt`，122,424,517 bytes，SHA `b87805747d40037841877ce7b99b7dda3ebe7a52202c0ba46bf0006ab5d6f008`。生产export/verify前后全文件SHA与fd/path身份检查，复用冻结ZIP bounded preflight；只打开明确byteorder与425个data成员，不打开data.pkl，不解包foreign代码、不加载模型。

实际saved IR：491项含66显式None，425 storage，声明storage字节122,154,832。保存全部根/路径，不按参数值或outcome选择。所有storage成员必须正好对应定义，不漏掉ZIP data成员；每entry复核dtype/storage identity、非负shape/stride/offset与范围，含空/scalar/noncontiguous/broadcast/alias。每个storage与byteorder验证原ZIP目录size/CRC/flags/compression/header offset，读取完毕验证实际CRC/长度。byteorder必须实际为`little`或`big`；无默认host字节序，无默默转换。

输出每storage一个一维plain numeric NPY（FloatStorage→f4，LongStorage→i8，显式端序），每非None entry一个原shape、C-order NPY。值按原stride/offset从storage逐位复制，不经Python浮点运算，不改变NaN payload/负零/精度。保留全部原source_entry、storage/key、shape/stride/offset/location、requires-grad与hooks节点ID；保留node/storage/module alias表。张量NPY本身是contiguous值快照，不假装保留view共享内存；可信自建加载方可由storage NPY和原view描述重建alias，不可仅靠逐key load丢失alias后声称等价。storage与entry分别记录NPY完整file SHA与C-index、source-endian的payload value SHA；CRC正确不代表作者认证。

`state_dict_paths`用精确module component/name产生候选key，拒绝含点歧义/重复key；从已保存BUILD state字面量读取`_non_persistent_buffers_set`（含惰性builtin.set语法），缺失/未知拒绝而非默认空。None与nonpersistent buffers不列入literal state_dict候选，仍保留原条目及其数据。该包为MAIN可信自建模型提供逐key值数据和完整storage/view映射，不调用nn.state_dict、不执行任何hooks，不推断自定义extra_state、class version或自定义state_dict行为。**不认证runtime state_dict/model equivalence、low/high pairing、可信pickle加载、官方权重等价或E1/P1准入。** 这些字段永久false/unknown。

## 限额、失败与saved verify

唯一新输出目录`results/agent_hilewm_storage_npy_export_20261028_v1_<suffix>`，绝对路径、直接results子目录、O_EXCL新建、所有文件排他/NOFOLLOW/fsync，不覆盖旧文件；无删除/清理/自动续导出。单storage/view<=128MiB、<=64维、JSON<=32MiB；流式64KiB读写，非连续view O(rank)索引内存，不将range变成巨大列表。单storage暂驻RAM，逐storage释放；不生成第二份checkpoint/原始storage副本。预计新磁盘上界为每storage/view字节+各512字节header+32MiB报告+1MiB journal reserve。预算完整计入原asset_resume树、全部decoder目录、scanner/IR报告、本stem全部目录（含失败/重复输出），总量<=3GiB，写入后free>=10GiB；不删旧chunk/失败。MAIN需避免同scope并发写入，容量检查不是跨进程原子quota。

异常保留所有已写NPY及partial文件，尽力排他保存failure.json、attempt/return、异常，不续写为成功；成功report.json只在全部完成后落盘。异常前尚未创建输出的preflight拒绝不产生报告；突然SIGKILL/ENOSPC可能只留下partial文件，不能认成功。每storage journal记录CRC/arrays operation attempt-return，未return不补账。合作wall(0,600]默认300s，不是阻塞I/O硬中断。

verify需MAIN提供成功report SHA，重新绑定IR/ZIP，逐storage重新CRC、逐array按原view重算并与saved NPY全字节相等（包括header、payload、无trailing），比较完整关键manifest字段/分母/alias。verify无写入，不提供从NPY执行/反序列化的入口。值hash不能跨不同端序直接比较；不将数值近似容差用作相等。

## MAIN命令

```bash
cd /home/gbk/code/LeWorldModel/le-wm-main
.venv/bin/python -B -m pytest -q -p no:cacheprovider tests/test_agent_hilewm_storage_npy_export_20261028_v1.py
# metadata-only，无checkpoint/storage payload读取
.venv/bin/python -B -m experiments.spectral_reachability.agent_hilewm_storage_npy_export_20261028_v1 plan
# 以下仅MAIN审后运行，worker未执行
.venv/bin/python -B -m experiments.spectral_reachability.agent_hilewm_storage_npy_export_20261028_v1 export --output /home/gbk/code/LeWorldModel/le-wm-main/results/agent_hilewm_storage_npy_export_20261028_v1_MAIN01 --wall-seconds 300
.venv/bin/python -B -m experiments.spectral_reachability.agent_hilewm_storage_npy_export_20261028_v1 verify --output /home/gbk/code/LeWorldModel/le-wm-main/results/agent_hilewm_storage_npy_export_20261028_v1_MAIN01 --report-sha256 <MAIN实际report_SHA> --wall-seconds 300
```

测试仅工程证据；生产byteorder/CRC/值hash/导出成功与模型接入均unknown，MAIN实际执行后再记录。后续需要MAIN审核可信自建架构、严格key/shape/dtype/persistence/version、原encoder/normalizer/动作ABI、storage alias以及low/high来源配对；不通过pickle解封这些前置。冻结后修订必须新版本。

worker实际metadata-only plan：425 storage / 491 entries / 425 literal state_dict候选，new-byte上界279,347,872；当时scope used361,807,943 bytes、free59,253,039,104 bytes，满足两项空间限制。此为时点容量检查，MAIN export前重新检查。actual byteorder仍null；未生产导出/未读取storage。新测试56项RAM/metadata通过（含大小端完整synthetic export+verify、CRC破坏、alias遗漏、NPY/header/payload/manifest/ledger/额外文件变异、persistence未知、磁盘/时间cap、排他路径/失败保留与冷禁止加载）。
