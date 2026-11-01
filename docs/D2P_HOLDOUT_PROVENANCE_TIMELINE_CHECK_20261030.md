# Future holdout 时序与来源链检查器

状态：仅元数据声明检查，始终 blocked。适用对象是未来采集方案，不认证现有根、
训练排除或物理状态互斥，也不授予采集、推理或 P1 权限。

工具 `le-wm-main/experiments/spectral_reachability/codex_holdout_provenance_timeline_20261030.py`
只从标准输入接收一个 UTF-8 JSON 文档，不遍历或打开数据文件、路径、P4 或结果。
调用方必须提供标准输入原始字节的 `--input-sha256`；这只防止意外输入改变，
不是可信来源证明。最大输入 256 KiB，拒绝重复键、非有限数字、多余和缺失字段。

顶层精确字段：`schema`（`codex.holdout.provenance_timeline.v2`）、
`checkpoint_sha256`、`protocol_sha256`、`source_sha256`、
`dataset_manifest_sha256`、`events`。四个身份均为不同的小写 SHA256；checkpoint
必须等于已核查的 P1 权重身份
`48938400ae3464c9680731287f583a9cb516f55a8ec64ea13a91be47fb15b607`。
此常量只绑定历史核查的字节身份；本工具不读取当前权重重算，也不证明冻结时间。

`events` 必须恰为以下顺序，每项精确字段为 `role`、`subject_sha256`、
`evidence_sha256`、`parent_evidence_sha256`、`not_before_utc`、
`not_after_utc`：

| role | subject 的声明绑定 |
|---|---|
| checkpoint_freeze | checkpoint_sha256 |
| protocol_commitment | protocol_sha256 |
| beacon_publication | protocol_sha256 |
| acquisition | source_sha256 |
| dataset_seal | dataset_manifest_sha256 |

首项 parent 必须为 null，之后必须等于紧邻前项的 evidence SHA；证据 SHA 必须两两
不同。事件时间区间采用秒精度 UTC，每项起点不晚于终点，前一项终点严格早于后一项
起点，最后每项终点均不晚于运行机器的 UTC 时钟。时间区间只是调用方声明，时钟
本身不可信；相邻端点相等也拒绝，避免声称可确定的严格先后。

链中的 SHA 和 parent 仅是声明，不表示工具实际读取证据、验证其签名或证明证据
内容含有所声明的连接。尤其 beacon 项绑定协议摘要不证明信标轮次、值、签名或者
承诺中确实预先固定轮次；dataset seal 不证明 manifest 成员及采集过程。RFC 3161
上界不能单独作为采集发生后的下界。调用方不得提供含 outcome 的输入。

一致时返回 `status=blocked`、`structurally_consistent_unverified` 和具体缺口；
任何错误退出码为 2，认证和授权始终为 false。要推进真实认证，仍需外部可信的
checkpoint 冻结记录、协议的事前承诺、指定信标签名及可信时序、独立采集及 no-copy
保管证据、manifest 内容与来源绑定，以及针对训练排除和历史状态的独立审查。
这里未取得这些材料，也未建立真实的来源链。

可用标准输入传递用户已经准备好的、严格限于上述字段的元数据；本工具不会创建输入
或产生候选。验证只使用合成声明：

```sh
cd /home/gbk/code/LeWorldModel/le-wm-main
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' .venv/bin/python -B -m pytest -q -p no:cacheprovider tests/test_codex_holdout_provenance_timeline_20261030.py
```
