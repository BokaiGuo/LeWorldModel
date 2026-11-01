# Hi-LeWM ZIP/pickle静态清单 v1（不是trusted loader）

独占source/tests/本文；MAIN审计后实际scan。worker只测试RAM合成ZIP/pickle opcode bytes及已封存JSON/text闭包，不扫描真实checkpoint，不反序列化，不导入h_le_wm/torch，不执行pickle reducer、外部代码或网络，不运行神经/物理。使用experiment-code技能的固定输入/记录/复现要求，不使用训练或挑结果规则。

## 唯一实物绑定与入口

唯一输入 `/home/gbk/code/LeWorldModel/le-wm-main/results/agent_hilewm_bounded_deflate_decode_20261028_v1_MAIN01/decoded.unverified.ckpt`，122424517 bytes，SHA `b87805747d40037841877ce7b99b7dda3ebe7a52202c0ba46bf0006ab5d6f008`。固定decode报告同目录report.json，SHA `f399b23e07f37960127433d10e14c8c7b7eae6c75ff329accb75a52f701edcc9`；其status/CRC d59f865b/EOF/长度/数据非加载状态与固定源码、license/config文本闭包匹配才允许进入静态扫描。复用冻结decoder标准库reader/hash/path工具，source SHA `81aa18081378ee073e267702742566b49744365d321409dff0dccfbe6d45d30c`；不调用其decode/run。MAIN consume尚未给固定SHA，不能捏造；此扫描独立绑定MAIN实际decode报告和完整实物hash，不将其称为已完成decode strict消费证明。

CLI无input/expectedSHA/caps覆写入口。`plan`只读JSON/文本/自身源码，不打开ckpt/deflate；`scan`固定同一fd全量size/SHA prepass后seek，读容器元数据与唯一data.pkl，前后核fd/path identity、metadata及source/tests/doc闭包。没有任何extract/load功能；不是“weights_only”加载器或许可/可信反序列化的替代品。

## 有界范围与保真字段

- 先只读最多65557-byte尾部EOCD，再打开ZipFile：最多8192成员、中央目录4MiB；多磁盘拒绝。ZIP64 locator/EOCD先核长度、磁盘、目录大小与普通EOCD一致性，避免ZipFile依据恶意ZIP64覆盖值分配巨量目录。超cap明确incomplete，绝不截短伪称完整。原始pickle/自解压前缀不fallback；ZIP尾随字节拒绝。
- 完整列出每个中央目录项name、file/compressed大小、声明CRC、compression、flags、external_attr、local_header_offset/create_system及issues，不按结果过滤。路径绝对/drive/backslash/NUL/control/空/点/父段、重复名、symlink/特殊类型、加密、不支持方法、非法offset等保留issues并拒读data.pkl。total declared<=512MiB、单成员<=256MiB属于工程cap，不是科学门槛。
- 仅一个`data.pkl`（根名或精确`/data.pkl`尾部）可读；多个/缺失拒绝，不猜目录名。file/compressed各<=16MiB，64KiB增量读取，长度与ZipFile CRC严格检查。**storage payload不读取、不解压、不抽样**，各CRC仅声明值，除data.pkl外`CRC_verified=false`；绝不声称完整tensor/storage字节语义验收。全文件SHA读过storage压缩/原始字节，但不等于逐成员CRC或tensor验证。
- `pickletools.genops`只解析静态opcode和其数据参数，最多200000 opcode、stack和memo各200000、符号最长4096。保存全部opcode position/name及完整计数，不打印普通值/score/大型bytes参数。malformed/opcap/STOP后尾随/未知memo/栈不闭合等保留已解析前缀与errors，`opcode_inventory_complete=false`。FRAME范围有界，但静态列表不证明所有pickle运行时类型、FRAME语义或可执行性有效。
- `GLOBAL/INST`直接按原两行解析module/name，不按空格误拆；`STACK_GLOBAL`只解析UNICODE字面量、memo/DUP复制的明确字符串，保留string origins。REDUCE/NEWOBJ/BUILD/动态构造结果一律opaque；无法证明的STACK_GLOBAL、EXT码明确unresolved/null，不查询copyreg，不import名字所指module，不调用persistent_load，不打开out-of-band buffer。
- globals只是文件中的声明符号（可能函数而非class），不是已验证class manifest，也不保证可达/全部动态引用已枚举。REDUCE/BUILD/INST/OBJ/NEWOBJ/PERSID等敏感opcode位置单列，永远不执行；没有依赖白名单、自动allowlist或“安全即可load”的捷径。

即使完整记录，也只返回`static_inventory_recorded_not_trusted_loadable`，始终`trusted_loading_allowed=false/static_scan_proves_safety=false/baseline_runnable=null/scientific_admission=false`。动态STACK_GLOBAL/EXT未解时status为incomplete，CLI exit1，但仍保存完整已取得inventory及未知。缺来源、文件hash或源闭包等前置直接非0，不写假清单。扫描通过仍不能证明安全；尤其REDUCE可以调用文件声明的任意函数，静态工具不执行它且不能靠“未见危险字符串”认证。

墙钟默认120s，参数仅(0,600]，hash/元数据/每opcode合作检查，不保证阻塞磁盘/解析单个有界参数的硬实时；报告cap32MiB，超限拒绝输出，不静默删成员/ops。只有新排他 `results/agent_hilewm_pickle_static_inventory_20261028_v1_MAIN01.json`等路径，O_EXCL/O_NOFOLLOW、fsync；不会写出data.pkl或模型副本。落盘前合计既存asset_resume、全部decoder输出、本stem已有报告和计划报告<=3GiB，另保留>=10GiB空闲。并发外部写入不属于全盘原子预算；MAIN应串行保管输入/输出。失败/incomplete报告也保留，不删除旧partial或chunk。

## MAIN准确命令（cwd inner le-wm-main）

```bash
.venv/bin/python -B -m pytest -q -p no:cacheprovider tests/test_agent_hilewm_pickle_static_inventory_20261028_v1.py
.venv/bin/python -B -m experiments.spectral_reachability.agent_hilewm_pickle_static_inventory_20261028_v1 plan
# 仅MAIN审计后执行，worker未运行真实scan
.venv/bin/python -B -m experiments.spectral_reachability.agent_hilewm_pickle_static_inventory_20261028_v1 scan --output /home/gbk/code/LeWorldModel/le-wm-main/results/agent_hilewm_pickle_static_inventory_20261028_v1_MAIN01.json --wall-seconds 120
```

stdout为实际新报告path/bytes/SHA及status，报告中给全member/opcodes/symbols/未知/errors和三文件SHA。MAIN再据actual证据审可信loader依赖/不可变revision、许可、embedded low、高层结构及冻结normalizer/data/protocol；这里不新增loader、不签approval、不运行E1。RAM回归不是checkpoint实物扫描或科学证据。冻结三件套后修改必须新版本。
