# SAGE 一手来源身份核验：本轮仍未解除

日期：2026-10-27。独占新增本文及同 stem `.sources.json`，不改旧来源账、封存、
代码、runner或其他agent文件。按 `literature-search` 技能进行互补查询并核查一手
材料；技能示例脚本路径不适用本机，使用实际可用的 arXiv 工具，无额外脚本/网络替代。

## 判定

**`unresolved_access_blocked`，不是认证成功，也不是查明论文不存在。**
两次 arXiv 搜索都返回限流，未收到论文元数据；因此连“外部搜索零命中”都不能写。
本地有范围检索未找到可认证的SAGE原文。当前未取得多个一手候选，不能声称已经
鉴别出某两篇同名论文；真正未解决的是提案简称与唯一论文身份之间的映射。

| 要认证的字段 | 本轮结论 |
| --- | --- |
| 真正完整题名、作者 | unknown；不能将下述候选标题升级为正式题名 |
| arXiv/DOI、版本、日期、会议或preprint状态 | unknown |
| 官方项目、代码URL、revision、许可 | unknown；不是已证实“无代码” |
| 官方已发布权重、配置、normalizer、许可 | unknown；不是已证实“无权重” |
| 已核验SAGE reported数值 | 无新增记录；不是性能为0 |
| 本地同协议配对结果 | 无新增记录；本轮未运行模型 |

候选字符串 **SAGE: Subgoal-Conditioned Action Generation for Latent World Model Planning**
直接来自 `docs/D2P_REPORTED_BASELINES_PRISM_IMWM_20261026.md:15` 的内部附件转录。
该文自己明确标为未独立认证，不能通过在新文档再次引用就变成一手确认。
“冻结LeWM、多时域subgoal/条件动作proposal”同样仍是提案的识别描述，不是本轮
确认的论文方法或训练合同。没有猜作者、ID、题名变体、代码域名或权重地址。

## 实际检索轨迹

本轮只有 arXiv search/read 系列文献工具可用；没有可调用的通用web、Semantic
Scholar或OpenAlex检索工具。没有改用shell联网、代理或镜像绕过限流。

| 实际动作 | 返回及解释 |
| --- | --- |
| `search_papers`，`ti:"Subgoal-Conditioned Action Generation" OR (ti:SAGE AND (abs:"world model" OR abs:subgoal))`，最多15项，relevance | `isError=true`；`Error: arXiv is rate limiting this IP. Please wait 60 seconds before retrying.`；不是返回15篇或0篇 |
| 限流期间核对repo材料、调用 `list_papers({})` | 工具缓存153个paper IDs，**没有题名/作者**；不能据此称已检索全部缓存。未逐篇读取153篇，也未用可能调用神经embedding的semantic search |
| 完成本地复核后重试 `search_papers`，`ti:SAGE AND (all:"world model" OR all:LeWM OR all:subgoal)`，最多20项，relevance | 相同限流错误；未取得任何可供`get_abstract/read_paper`定位的SAGE ID，因此不编造ID继续取正文 |

按技能的互补检索设计，候选全题名与方法描述两条实际请求均已记录，但受服务阻断
而没有成功覆盖。没有做S2/OpenAlex交叉认证或全网穷举，不能写成多数据库查无此文。

本地检查：repo docs/innerdocs及非结果源码中的SAGE引用、现有四论文原文缓存、
Overleaf `references.bib`。对
`le-wm-main/results/p2_reported_baseline_table_20261027_v3/sources/` 的txt/html运行：

```sh
rg -n -i '\bsage\b|subgoal.conditioned' le-wm-main/results/p2_reported_baseline_table_20261027_v3/sources -g '*.txt' -g '*.html' --stats
```

实测 **12个文件、2,719,390 bytes、0匹配行**；该目录涵盖GRASP、Hi-LeWM、LeWM、
PlanWAM的既存一手缓存。指定BibTeX同样无匹配。这不穷举本机匿名PDF、压缩包或
全部153篇工具缓存。旧文档的本地资产清单仅作线索，不冒充本轮重新认证权重缓存。

## 已排除的一种不当替代：方法相似不等于身份相同

本地一手缓存可确认另一个工作：
[Mind the Gap: Promises and Pitfalls of Hierarchical Planning in LeWorldModel](https://arxiv.org/abs/2607.12547)，
方法名 **Hi-LeWM**，缓存PDF正文标 `2607.12547v2`。
摘要明确描述冻结低层LeWM、高层latent-subgoal规划，因而与提案轮廓部分重合。
但它不是候选SAGE题名，已查正文/HTML没有SAGE字符串，没有一手证据说二者为别名。
**不能把Hi-LeWM的作者、ID、代码、权重或结果填入SAGE字段。** 本轮不将该文数字
再次抄入SAGE结果表，也不宣称已从语义相似性判定提案写错了名字。

## main可使用的精确阻断与纳入条件

1. **身份仍是首要阻断。** 需要可读原论文，或作者关联的稳定论文URL/唯一ID，能
   同时确认题名、作者、版本及上述方法轮廓。只有另一份无出处的比较附件不解除。
2. **官方资产必须单独认证。** 论文/作者页面指向代码，再核revision、许可、推理
   接口；权重需包含实际subgoal/proposal模块、配置/normalizer及许可。仅有LeWM骨干
   或同名GitHub仓库不够；只有训练脚本不代表已有可用checkpoint。
3. **训练边界不变。** 如果原文确认额外学习模块，应披露其已有训练成本；没有发布
   可用模块而必须新训练时，需要用户另行授权，不能静默补训练或直接删掉强比较。
4. **reported只按原协议保存。** 认证后才逐行提取环境/版本、模型/checkpoint、
   观察权限、goal offset、内部rollout horizon、真实执行长度、反馈、搜索/编码/梯度
   预算、训练成本、统计单位及原表页。缺字段保留unknown，不算跨论文提升率。
   公开论文结果始终与本地配对结果分开；不能进入当前root的paired排序。

旧附件的Table1/state-assisted、Table18/visual-only、H为goal offset、manifest SD与
pooled query统计的差别，仍只是**待原文核实的约束线索**，本轮未把任何一个细节或
其中数值写成论文事实。机器JSON的 `reported_results=[]` 与 `local_paired_results=[]`
表示无认证记录，不是成绩零或缺失训练预算等于零。

## 小型来源JSON与验证

`docs/p2_sage_primary_source_resolution_20261027_v1.sources.json` 保留两次真实工具
参数/错误文本、缓存清单限制、本地SHA、全部unknown身份字段、排除替代项及解除条件。
`identity_gap_resolved=false`，不得把交付文档的存在计为baseline身份准入成功。

只读/检索与两文件写入之外无副作用：无下载大资产、无模型/物理运行、无训练，
不触碰P3/GPU。最终结果是有实录的**身份检索受阻**，不是任务目标已经实现。
