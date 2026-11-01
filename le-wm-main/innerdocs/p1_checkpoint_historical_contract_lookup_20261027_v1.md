# P1 checkpoint 历史 normalizer / camera 合同：有界只读查询

固定 weights SHA256=48938400ae3464c9680731287f583a9cb516f55a8ec64ea13a91be47fb15b607、HF revision=22b330c28c27ead4bfd1888615af1340e3fe9052。仅检查本地 digest 与公开官方小型元数据/文本，不执行任何训练、神经、物理或 transform fixture；P1=false，不更改旧封存。

使用 experiment-code 的先协议、记录、冻结要求，但不进行性能实验或绘图。网络发现已定位作者官方代码仓库 lucas-maes/le-wm（其主页与论文网站一致）；本轮候选源仅该仓库、两项 galilai-group 官方依赖以及 quentinll/lewm-pusht HF 模型/数据元数据。不是向作者发送消息。

GET only，无 credentials，不自动跟随 redirect。每响应最多512KiB、总正文最多8MiB、最多60请求、12秒timeout、4 workers；所有原始正文与 URL、headers、HTTP状态、时间、SHA256 落盘。禁止权重/数据正文、release assets、git clone及大包。网络错误保留为 unavailable，不解释成文件不存在。源协议先 seal；结束检查本地 checkpoint 与本模块/tests/doc 未变。

预先固定查询：HF pinned revision metadata/tree/card/config 与 model commits、dataset metadata；GitHub repo/head/releases/tags，以及 train/utils/data-config 的最多30条历史（保留 Link）；抓取固定 head 与所返回最早 train.py revision 的有限文本。它们是 source-history 对照，不是 outcome 或模型性能选择。若历史分页/树截断，不能说全历史查完。查询两依赖 head/tags 及有限 source 仅为候选合同，不能声称它们用于固定 checkpoint。

最关键的证据层级：远端 pinned LFS SHA256 匹配仅能建立对象身份；还必须有 checkpoint→训练 commit/run、dataset revision、拟合 normalizer、camera/codec 版本的额外 witness，才允许历史绑定。公开训练代码、README数据链接、架构 config、当前依赖 main、上传时间先后均不单独满足此条件。缺失保持 unknown，不能把算法定义当作精确训练统计量。

```bash
.venv/bin/python -m pytest -q tests/test_p1_checkpoint_historical_contract_lookup_20261027_v1.py
.venv/bin/python -m experiments.spectral_reachability.p1_checkpoint_historical_contract_lookup_20261027_v1 --out-dir results/p1_checkpoint_historical_contract_lookup_20261027_v1_public_v1
```

程序输出 report.json（保守 unknown 与源码摘录）、protocol.json、source_seal、原始 responses 和逐请求 journal。执行后另加独立 review 文档，精确指出远端实际找到的合同以及无法建立的链条；不修改已冻结 collector 或 report。当前变换已审计，不重复执行。
