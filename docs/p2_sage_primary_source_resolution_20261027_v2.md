# SAGE 来源解析 v2：论文身份已确认，发布资产仍需绑定

2026-10-27。仅新增本文和同stem source JSON；v1原样保留。本轮按 literature-search
技能补做其他provider与一手摘要核验。普通sandbox三provider均DNS失败后，通过工具
`require_escalated`审批执行只读请求；后续官方metadata请求亦申请审批。每响应最多
读取65,536 bytes、不跟随redirect、不传token/cookies。无PDF、论文正文或权重下载。

## 真实身份与证据链

**用户提案中的候选题名确实对应一篇可认证论文，不再是只有内部转录的名称。**

- 题名：**SAGE: Subgoal-Conditioned Action Generation for Latent World Model Planning**。
- [arXiv:2607.17973](https://arxiv.org/abs/2607.17973)，当前 **v2**；v1提交
  2026-07-20 14:10:56 UTC，v2提交2026-09-28 04:41:40 UTC。会议接收状态未认证。
- 官方摘要页citation metadata作者：**Letian Cheng、Qi Zhang、Yisen Wang**。
- DOI：`10.48550/arXiv.2607.17973`；OpenAlex：`W7169840123`。

先由OpenAlex的 `SAGE LeWorldModel` 检索成功发现准确题名和ID，再读arXiv官方摘要页，
题名/作者/ID及方法描述一致。摘要说明指定duration的中间latent subgoal引导动作候选，
多duration priors结合局部/长程规划，由冻结world model评估和refine；LIBERO段明确
比较vanilla LeWM。因此与提案的“冻结LeWM多时域subgoal/proposal”身份轮廓相符。
不是将Hi-LeWM改名，也不是凭简称关联任意同名SAGE。

一手摘要原句：

> At each planning stage, a goal-conditioned generator predicts the next intermediate latent subgoal for a specified duration, which is then used to condition the generation of candidate action sequences.

> Then the frozen world model evaluates these proposals against the same subgoal and guides their refinement before execution.

## 官方实现声明、权重发布入口与尚未认证部分

GitHub官方索引按确切arXiv ID检索，返回两个仓库：`PKU-ML/SAGE`与一个daily
aggregator。前者README明确写 **Official implementation**，链接确切论文与Models。

| 资产 | 可定位来源与当前边界 |
| --- | --- |
| 实现仓库 | [PKU-ML/SAGE](https://github.com/PKU-ML/SAGE)；核验的是仓库官方声明与准确论文关联，所查arXiv摘要页没有反向项目链接，不额外声称完成作者账号归属认证 |
| 固定源码版本 | `43c0a655bd2bbd33d8047c7588360e193671e9f0`；随后按该commit重读README，blob仍为`62cd308ee7b8f42c63ca4074b520bce15de9d0da` |
| 源码许可 | GitHub metadata标MIT；本轮未独立审阅LICENSE全文，不能外推到模型/数据许可 |
| README发布的模型入口 | [CLTRAY/SAGE](https://huggingface.co/CLTRAY/SAGE)；HF metadata请求SSL握手超时。模型文件、配置、normalizer、不可变revision、SHA与许可仍unknown；不是“无权重”，也不是已下载可运行 |
| README发布的数据入口 | [CLTRAY/SAGE-data](https://huggingface.co/datasets/CLTRAY/SAGE-data)；未访问或下载数据 |
| 真实可复用入口 | 固定commit的 `scripts/reproduce_main.sh`，blob `a53731c4e4167d5bea1c15c8ee1439c4c83c2947`；先audit_release，再download_checkpoints，再eval_pusht/eval_cube，最后汇总。它会下载与运行环境，**本轮未执行** |

**作者元数据差异必须保留：** README BibTeX为Cheng、Zhang、**Qixun Wang**、Yisen
Wang四人，arXiv官方citation metadata为上述三人。引用论文时优先保存官方版本作者表，
另记仓库差异；不自行补人、删人或推断差异原因。此差异不抹掉准确题名/ID的确认。

README明确额外训练subgoal generator和action priors，world model冻结不等于整系统
training-free。已有训练发布资产可以作为外部参照披露成本；新训练仍需另行授权。

## 原协议能确认到什么程度

固定commit的 `REPRODUCING.md`（blob `d7f0c55b54860502ac26d75bfaa59f3de23a31df`）
记录以下**仓库协议**，不冒称已逐cell对应论文表格：

| suite | 查询 | 候选/CEM | 执行budget |
| --- | --- | --- | --- |
| LeWM PushT | 每horizon，3 seeds×50 | 300/30 | 2H |
| LeWM Cube | 每horizon，3 seeds×50 | 300/30 | H |
| Visual-only PushT/Cube | 相同固定manifests | 300/30 | 对应上两项 |
| DINO-WM PushT/Cube | 相同固定manifests | 128/6 | 对应上两项 |

文档另说明：DINO Cube的generator/prior使用完整28维observation，不是visual-only；
不能混不同encoder checkpoint；DINO Cube指定OSMesa，EGL跑通不能认证相同renderer。
这些候选×迭代不等于全pipeline算力，亦未绑定到本地P1/P2根。

## 数值只保留摘要原claim，不能当配对结果

以下确实出现在官方 **v2摘要**，不是附件数字；没有读取正文表页，所以不标成已完成
table/page验证。原环境、指标、offset及缺失预算字段逐项保留。

| 原环境/指标 | offset | 摘要baseline→SAGE（%） | 尚缺的可比性字段 |
| --- | --- | --- | --- |
| PushT success | 150 | 4.7→64.7 | 该句未显式命名baseline；具体checkpoint、cell预算、统计单位和表页unknown |
| OGBench Cube success | 150 | 20.7→67.3 | 同上；不能把repo通用预算追认成该cell合同 |
| LIBERO Scene2 full-episode success | unknown | vanilla LeWM 0→48.7 | 原episode分母、seed/误差定义、预算/权重unknown |
| LIBERO Caddy full-episode success | unknown | vanilla LeWM 0→58.0 | 同上 |

统一 `context_only_not_local_paired`；不计算跨论文排名或本地提升率。旧附件的Table1/
Table18、state-assisted/visual-only、SD或pooled-query定义仍未由原表核实，不能随身份
解析一并升级。摘要数值也不纳入下面四论文的“已定位原表值”归档。

## 请求实录与交付状态

新source JSON记录17次HTTP尝试：3次普通sandbox DNS失败；审批后的14次中，11次
HTTP200、2次429、1次HF SSL timeout。成功provider包括Crossref、OpenAlex、arXiv
摘要页和GitHub；Crossref前5个模糊候选未命中，不代表不存在。OpenAlex返回11项的
第一页5项即含目标；没有声称穷举全索引。最大实际响应43,341 bytes，全部低于64KiB。
S2和arXiv export的429各自保留；v1旧工具错误与本地检索实录保持不变。

**已完成：论文身份解析、准确实现仓库/版本与发布入口定位。未完成：权重可用性/许可/
hash、作者表差异消解、原表统计协议、实际本地复现或科学准入。** main下一步应先拿
不可变checkpoint清单并匹配各encoder/组件，再决定原协议reported-only或另封本地对照；
不能直接运行会下载的reproduce脚本。本轮不再继续外网取资产，无CUDA/物理/训练。
