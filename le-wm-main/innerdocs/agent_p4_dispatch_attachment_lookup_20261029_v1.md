# P4：固定十次dispatch的有界附件定位结案

2026-10-29；只新增本文。未改冻结工具/结果/gate，未生成null草案、wrapper、response或批准；未执行神经、物理、网络或日志中的命令。采用documentation-and-adrs技能记录查找边界，不新增科学前置。

## 1. 实际范围与方法

唯一入口 `results/agent_freshroots_resume_20261027_coverage.json`，实际SHA `00a3fffeaaa8943e851d129b941a7f07feaa1db387117beb98110e4ec95f5eeb`。只取其 `/recovery/dispatches/0`…`/9`、`/log_bindings/0`…`/1`及 `/sources/0`…`/7`的原report目录。八目录只列直接子项，唯一子目录E/postrun_implementation内再列一次；未递归其他results。对原日志已经明确列出的两个后续seal，只核其manifest和列明的14份snapshot，不追随源码import、原live路径或其他邻居文件。

两日志只读原catalog封存前缀，逐字节SHA匹配，**未读追加尾部**。原目录名作为固定字节锚，定位显式路径及`.npy/.npz/.safetensors`、load/loaded manifest、reset/attempt journal/ledger、prerun/postrun关键词；只解码日志type/name/call_id等metadata，字符串中仅抽路径，不解码outcome、数组值或报告全文。命中关键词不等于存在附件。

首次只读正则定位耗时异常，已终止本worker自己的检查进程，未写结果；不计为完成覆盖。改为固定字节锚和有界路径匹配后完整通过，两前缀定位耗时1.92s。此工程检查不是科学实验。

|日志|实际绝对路径|已核前缀bytes / JSONL行数|前缀SHA256|
|---|---|---|---|
|L0|`/home/gbk/.codex/sessions/2026/10/23/rollout-2026-10-23T21-46-56-01a167e2-eca3-7f63-bb9b-93030b5d366c_01a16903-c194-71a3-972b-7a268203b373.jsonl`|508228945 / 58752|0a2c097548ef4bd9818affcf82ecb53102ddfa965498b966c13aa7067efdf70c|
|L1|`/home/gbk/.codex/sessions/2026/10/26/rollout-2026-10-26T12-10-35-01a167e2-eca3-7f63-bb9b-93030b5d366c_01a17667-2cec-7780-b530-fb3aa5acdd59.jsonl`|227482902 / 31891|cad9cf6253c2b466fcc48bddda725368426e796f8337326733de175db8b5ee22|

两前缀分别有165/56行含原目录名。显式原目录路径只有目录本身、report.json以及E/postrun_implementation；未定位到这些原目录内的ledger、state NPY/NPZ或loaded manifest路径。该查找穷尽的是**所列字节范围内的显式路径/固定关键词线索**，不是对被跳过科学子树、无文件名inline记录或未引用资产的全语义穷尽。

## 2. 十次原dispatch逐项定位

全部位于L0，call_id与原catalog指定行逐条相符。SHA是原JSONL整行字节（含末尾换行），不是源码SHA。B三次共用当前B报告目录，不把最终报告追认为前两次失败时已产生。

|ID|call_id|L0行|实际行SHA256|本次附件结论|
|---|---|---:|---|---|
|B1|call_a41c0e8d73624b16959e29b9f05fd4ed|22791|13f6855e076b47608631f13d5b59cc8bbda176766d1b6560e9f3fa89e4b61f25|原目录仅report；无原attempt/state/load附件|
|B2|call_3d96838206a84aba9b97fe0ce82bf8e4|22814|ea6d26b3d5023d00c9e7a71b8c87d0cedd3615b339c5369476d0b701906edc9b|同B目录；不合并为B3的执行历史|
|B3|call_4b779dfb4ad045b696db7409d861c35f|22850|9a45ce29e980668cb6e8e76075cb0a38f1683d1571a497c7e7fb5316ad567842|同B目录；无新增附件|
|C|call_b0f211dffc4e451fab05b0827fdfa415|23599|11a9be87ae7e2329f757c154f890d6e155775055f57899da486f7d67669896cd|原目录仅report|
|F|call_e0f5ab66af7848a89e59e21a975c4165|23658|86e6c6a2ec8b41824b7dff84b6d04f910cf9bd495f1d71bfd7edd16dc4d203c3|原目录仅report|
|G|call_a0928b48326a4f43b88a25a223834196|23730|9a179f3229e604f0372a4fb4df008669324faa1d1530bcc1c9616dcd39eb02bd|原目录仅report|
|D|call_871afe66f51744c3a8964fe49fab6b08|23773|f310459779529c9ad63da4ed1e001aa84599599d68495a6b7a4fab164285cdf0|原目录仅report|
|A|call_981f76dba2b241d79588468b36e979d7|58279|1c6b55788112adf7cda9adeb79b321d862ddea836e7a5d319cb2661c9004657b|原目录仅report；E保存的同hash源码不补原加载证明|
|E|call_7c4638e67c234a65ad8b02be380375cb|58307|f575714e375a11117fab383916e4717daabb482ef7ae411f02d289018ad12080|找到三份postrun源码，非原attempt/state/load附件|
|H|call_947bf4c5b1ce4004a5ab5c4e2ef3d0e5|58363|f12a7a5955f31220482ac2f6aa3ddb18abcd139d4822ff434fee4597b2eb34f2|原目录仅report；后续physical seal不等H原加载manifest|

## 3. 已核原目录文件及SHA

本节路径均相对 `/home/gbk/code/LeWorldModel/le-wm-main`。八report只核文件字节SHA，不解码科学结果；其原身份pointer与缺字段分类继续引用冻结recoverability文档§3–4，不重复投影outcome。

|ID|原report路径|实际SHA256|
|---|---|---|
|A|results/pusht_d2p_p0_packet_contract_v2_20261026_cuda/report.json|2f29a195f291d040a666cc3db6673f9be12c315c96162b126ddaef9bcdb5e135|
|B|results/pusht_d2p_p0_preflight_20261024_cuda/report.json|b1b0fbc29ceb050939ae1403c7644d0ca704b133e47a2411c079dbd626ee0eaf|
|C|results/pusht_d2p_p0_preflight_packet_layout_repair_20261024_cuda/report.json|c499ececab67135a2f4d4d698359e6f62b557be66f9132480ef95b14a0dce080|
|D|results/pusht_d2p_p1_outcome_admission_20261024_cuda/report.json|a9266fb3cd3e68ee197ec187e8b7916cf3ca8c11966778df3d4c92083b7d6b54|
|E|results/pusht_d2p_p1_packet_contract_v4_calibration_20261026_cuda/report.json|a0c9c53bd0161946ec4466878de49795abdf6db102dbb97b174035ed4028a006|
|F|results/pusht_d2p_p1_reconfiguration_replay_bound_20261024_cuda/report.json|4911ae7e945abfb042b332f71d7cda0163e14c9727ee8131ff7881bc67257a14|
|G|results/pusht_d2p_p1_rollout_alignment_repair_20261024_cuda/report.json|1c5fdd414a5104b6487c2c6e5cb38d05943f902a9d7a695b891ae73202ef7abb|
|H|results/pusht_d2p_p1_root_anchored_v5_calibration_20261026_cuda/report.json|99cbb6b6dc4653eb629427353f6f6d05022c219333f3f7362cf4ca43a3076bc2|

E唯一子目录 `results/pusht_d2p_p1_packet_contract_v4_calibration_20261026_cuda/postrun_implementation/` 恰有以下三regular files，全部实测hash：

|文件|SHA256|结论|
|---|---|---|
|preflight_pusht_d2p_p0.py|5215f0b49b772d8def14997996774bfd49db614b52b9f615c164271591f7b83b|候选源码字节；非原现场ledger|
|preflight_pusht_d2p_p1_reconfiguration.py|80b9674063360a2f51d2e146c830c29a5399bb53b9086b177f3b3f2dc3b0d5e4|同上|
|run_pusht_d2p_p1_physical_calibration.py|7aa9f156484ec3ab483ce3f8308466f60b09a512232f21cd2bcd02f25944f114|同上|

原日志L0:58354明确指向postrun目录，整行SHA `821e36ee32679127e52a75fb0e9d105481ab6d860e608a40b06cb8bb8dc5f7ce`。L0:58383为消息引用（行SHA `69bc5b0023e5c5df2004b169e163a6d36624ab1398685176c0c16ad56162894e`），不把消息叙述当原加载receipt。原八目录加E子目录共11文件，未发现其他文件/子目录、state数组或journal。

## 4. 定向核实关键词命中，未外扩搜索

L0的`.npz`命中行：23000、24170、29047、29142、33151、34957、35366、44556、58425。多为compacted/汇总文本；可抽取的数组路径例为reacher readout、cube model_inputs及 `results/pusht/lewm/latents.npz`。它们不是八原目录内附件，未取得十次原dispatch→这些数组的原state/attempt绑定。本轮**未打开**这些数组，不能断言其内容、原state不存在或可用于补证。

L1的`attempt_journal`命中18713/18953，`attempt_ledger`命中29566/29569/29682；路径提取没有获得与原十次调用绑定的ledger文件地址。这些关键词本身不算真实ledger，亦未据此读取输出中的数组或outcome值。其他NPY命中可定位到后续P4 terminal20或GN RGB编码路径，未将其作为原根数组，不打开它们。没有沿compacted文本中的外部附件目录或其他session继续扫描。

L1:29793/29795明确引用两个已有后续seal，行SHA分别为 `531e8c30d3d9d49c6c156178b06c31b2ad402e7e5047a92283d6edd5ad56cc54`、`7dced9b97bbf29ee1101e4299021b87af1ea5956693284375324a0edd20bf83e`。只核以下manifest和各自明确列出的七附件：

- H后续physical：`results/pusht_d2p_p1_root_anchored_v3_physical_calibration_20261026_cuda/prerun_seal.json`，SHA `4e024c0ded185e0bb58c6ceaeb33a0ab1d784165a962995c28f6216d439ef81e`，pointer `/inputs/0`…`/6`；schema为`pusht_physical_prerun_seal.v1`。
- 后续program archive：`results/pusht_d2p_sealed_program_execution4_v1_20261026_cuda/prerun_source_seal/manifest.json`，SHA `0f0f5b8b8a525990dcffd2393ada1a49270dd50a3d4865800343cfd9b8a39aef`，pointer `/files/0`…`/6`；schema为`d2p_prerun_source_seal.v1`。

下表路径相对各manifest所在目录；14附件实际SHA全部匹配manifest，没有执行源码或解码snapshot中的report结果：

|seal|附件路径|SHA256|
|---|---|---|
|H|prerun_inputs/00_run_pusht_d2p_p1_physical_calibration.py|c6bc52e9f138c1123b316b99a18246e50d1a4aaeba9656192ec4f0a8bbb7084f|
|H|prerun_inputs/01_preflight_pusht_d2p_p0.py|745a13d47934f1862a2b43a62b20bc07fc7308075b7b6fea26c3236d9ffa4c98|
|H|prerun_inputs/02_preflight_pusht_d2p_p1_reconfiguration.py|6dc263e6f69840479daedd92a064bead928b1b1b80e1628f46e4c44964525806|
|H|prerun_inputs/03_pusht_d2p_packet_contract.py|eda631348e929b45591807dac0adcbb72d10d84b7d48a7a558a492eab9c04aef|
|H|prerun_inputs/04_checkpoint_compat.py|50b01e20864dc929943df7cb3566c271a771e69c06ea0fdbc9b598c59e6e9dff|
|H|prerun_inputs/05_gpu_preflight.py|98447092df2bb9968f3ab7844e0dabffa933f8bfd3d6e60fba62abd1629e6442|
|H|prerun_inputs/06_report.json|99cbb6b6dc4653eb629427353f6f6d05022c219333f3f7362cf4ca43a3076bc2|
|archive|sources/000.source|3ae4c0593b738c46dfdd918c9017b04d8af8da1a1eefb42401f48bef7660a360|
|archive|sources/001.source|d0b66b26a5c539cd1055192ab179939aa7ac809bf78b1cbf58250aca05bd3a0b|
|archive|sources/002.source|3a4ec300ad7a178640a98accdee551102e4b29ee9f50725a49f915708a5fbbf42|
|archive|sources/003.source|745a13d47934f1862a2b43a62b20bc07fc7308075b7b6fea26c3236d9ffa4c98|
|archive|sources/004.source|eda631348e929b45591807dac0adcbb72d10d84b7d48a7a558a492eab9c04aef|
|archive|sources/005.source|50b01e20864dc929943df7cb3566c271a771e69c06ea0fdbc9b598c59e6e9dff|
|archive|sources/006.source|73871055f10b09aea15f03acf87b04e79e30d546668d44bb8160c7be3c439a67|

结论：两seal是真实存在的后续source/input快照，不是A–H原十次运行的loaded manifest；其列出的附件也不是原attempt ledger或原state数组。manifest所列附件核完即止，未展开后续物理report或遍历其父目录其他结果。

## 5. 停止点与仍缺什么

本次已核完：两原日志指定前缀的显式路径/关键词线索、十条原dispatch行、八个原目录及E唯一子目录、日志明确引用的两份后续seal及14列明附件。各项没有取得可用的**原十次运行**attempt ledger、state原数组或完整loaded manifest。原报告/代码候选/后续seal不提升为原现场身份认证。

因此本有界定位结束，不再扫其他机器、会话、目录、checkpoint或未绑定数组。未知仍是：是否有本范围外的原资产、无文件名inline身份材料、原env/wrappers加载依据及相关未保存尝试。不能把“指定引用核完”写成“历史从未记录任何附件”，也不把无限外域记录加成新科学必要条件。

要推进原fresh合同，只能由原来源方交付明确的真实原附件/原绑定依据，或确认具体call/字段未记录或遗失；无新输入时不再生产同义wrapper。本文不解除16引用阻塞、不改变freshness门槛、不自签MAIN批准，不授权source采集或科学P1/P4。

本轮实际校验是上述原SHA、十行call_id、prefix长度/hash、11目录文件及14seal附件的只读断言，均通过；未新增源码/RAM测试实现，未用测试充当fresh证据。原冻结三件套和既有结果保持不动。
