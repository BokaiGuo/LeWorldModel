# 未来 holdout：原始训练 split／manifest 有界核查

日期：2026-10-30。结论：**本轮核查范围内，没有取得能直接证明当前 P1 checkpoint
训练排除的不可变成员 split／manifest，也没有建立 checkpoint 到该清单的可核验连接。**
训练排除状态保持 unknown；不将现有或未来根标为训练外。

## 逐项证据

|检查对象|本轮实测／原始声明|可用范围与缺口|
|---|---|---|
|本地 P1 权重与模型配置|`/home/gbk/.stable_worldmodel/checkpoints/pusht/lewm/weights.pt`，72,290,721 字节，实算 SHA256 `48938400ae3464c9680731287f583a9cb516f55a8ec64ea13a91be47fb15b607`；`config.json` SHA256 `2564086e961e7b5c7c04dffc451091115b389a590645ff19653c64fd0bc16e09`|字节身份可核验。指定 checkpoint 顶层只有 `.gitattributes`、`README.md`、`config.json`、`weights.pt`；下载缓存只有四个对象的 metadata，无独立训练成员清单。未检查权重内部或其他所有目录。|
|checkpoint README 和下载 metadata|README SHA256 `660de4e79b109ea18602618d90c72c44630957978ebd8474910713d896340292`，声明数据集名 `quentinll/lewm-pusht`；weights metadata SHA256 `d43af1fea219fa74d8e2db75c0b836901f919af3c7cbaf20a40c6614e50bdb26`，记录下载 revision `22b330c28c27ead4bfd1888615af1340e3fe9052` 与上述 weights SHA|这两项分别是数据集名称声明和模型下载对象元数据，不是 checkpoint → 训练 run → 数据 revision → episode 成员的证明。|
|当前训练代码及配置|`train.py` SHA256 `c5b4ced2295113186d2834671020f92104b56793839c7fce490ba0050bef90d4` 的 `run` 中用 `cfg.seed` 对 dataset 做 `random_split`；`config/train/lewm.yaml` SHA256 `fef2e609e3e1be367400e69e0929229ee0b6bd95cc4d4681c075ec9ab7d4cf89` 声明 `train_split: 0.9`、`seed: 3072`；`config/train/data/pusht.yaml` SHA256 `dcdd082ebdc628cdc8a88f15cb647fce6eec87343f9fe190271946ff5d52f757` 声明 `pusht_expert_train.lance`|当前源码／静态配置不是已解析的原训练配置或成员清单；未知该 checkpoint 实际使用的代码、dataset revision、样本映射、变换与抽样状态。不能凭种子重建或认证其训练排除。|
|本地 PushT HDF5|`/home/gbk/.stable_worldmodel/datasets/pusht_expert_train.h5`：46,300,921,856 字节，inode 19417810，mtime 2026-10-14 01:56:11.576170341 +0800。仅打开 HDF5 容器元数据：根属性为空；根级 `action, ep_len, ep_offset, episode_idx, pixels, proprio, state, step_idx` 均为 dataset 且属性为空|根级元数据没有 split 成员清单；只列成员及 shape，不读取数组。未完整 hash 46 GB 文件，stat 不是不可变身份；也未证明它与训练时的 Lance 文件或 checkpoint 一致。|
|此前保存的数据镜像 tree|`le-wm-main/results/agent_hilewm_normalizer_asset_route_20261029_v2_mirror01/response_01.bin` SHA256 `a937385559f06c58259847eca48456eb5c36445c67dc365aee66352dd341221f`；所保存 candidate revision `655cd446b9929369d7d406001da85c15d1457850` 的 tree 只列 `.gitattributes`、`README.md`、`pusht_expert_train.h5.zst`；压缩数据的 LFS SHA256 为镜像声明 `7cfbd6d90fa2f27876379a5ff169715a36ed82edbda64f9e5b5bfa34d212f318`|只复核已有镜像响应的字节内容，不代表本轮访问官方站、核验大数据字节或覆盖其他 revision／私有附件；该 tree 本身没有独立成员 split 清单，也未绑定原 checkpoint 训练 run。|
|SAGE 缓存的统计文本|`le-wm-main/results/p2_learned_subgoal_asset_metadata_20261027_v1/sources/data__stats__pusht_train_seed42.json.source.txt` SHA256 `c5b8e64d124ca09481854140820684fd90e573ee0f3aa0484ba5a327fc70e42d`，其中 `split` 字段引用 `data/splits/pusht_episode_split_seed42.json`|这是 SAGE 缓存统计文本中的路径声明，不是成员文件或当前 P1 checkpoint 训练清单。本轮检查 `le-wm-main/data/splits`、该来源缓存下对应 split 文件及已列定数据目录，未找到该文件；不声明全机不存在。HDF5 与 SAGE Lance 的 episode/frame 身份等价亦未认证。|

## 可复核的缺口

要在既定 checkpoint 上声称某个未来数据对象训练外，至少需要可信来源提供并绑定：

1. 上述权重 SHA 与实际训练 run、训练代码及解析后配置的来源记录；
2. 实际完整训练数据的不可变版本／内容身份，以及覆盖全部训练来源的说明；
3. 原始 train/validation/test 成员表、过滤／重复／转换规则和数据身份算法；
4. 候选来源对象到该成员表身份空间的可核验映射，以及独立来源真实性和完整性审读。

前述材料若不可获得，应记录具体不可恢复项；新目录、名称、seed 或未来产生的 UUID
都不替代训练成员证据。另一个可能方案是由独立保管者证明新采集发生于冻结模型之后
且未复制旧数据；那需要独立采集和训练血缘证据，本轮没有提供或认证。

## 操作范围及限制

本轮没有运行模型推理、训练、GPU、物理或 reset；没有读取 HDF5 数组，也没有
生成或挑选根、访问 P4 保留槽。**操作偏差：**核查 SAGE 统计文本来源时，打开的一份
已有资产说明包含公开论文的结果表，超出用户禁止读取任何 outcome 的范围；
表中数值没有用于选择或判断，本报告不复述它们。此偏差不改变结论的证据边界。
仅做上述指定目录及文件名搜索，找不到不等于全局不存在；本次没有请求或声称
发布方补证，亦未建立假证据连接。
