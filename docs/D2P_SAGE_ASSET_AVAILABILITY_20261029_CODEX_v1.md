# SAGE：公开资产可用性与受控下载交接

核查日期：2026-10-29，Asia/Shanghai。本轮只读查询和检查缓存；未安装、下载权重、克隆仓库、加载 checkpoint、运行 GPU 或向已有封存目录写入。本文为独立新文件，补充[原文和复现前置](D2P_SAGE_REPRODUCTION_PREREQUISITES_20261029_CODEX_v1.md)。**目前不能确认作者权重可下载、模型许可或完整复现。**

## 来源与实际可用性

| 来源 | 本轮实际结果 | 身份和许可边界 |
|---|---|---|
| [PKU-ML/SAGE](https://github.com/PKU-ML/SAGE) | GitHub API 成功返回固定 commit `43c0a655bd2bbd33d8047c7588360e193671e9f0`，tree `da65a63b807d24cc8d59a224f35798e63acc40af`；commit 签名状态 unsigned | 仓库 README 自称官方并链接论文，历史缓存 [LICENSE](../le-wm-main/results/p2_learned_subgoal_asset_metadata_20261027_v1/sources/LICENSE.source.txt) 为 MIT 软件许可；不自动适用权重或数据。确认提交可读不代表已下载完整仓库，也不认证 commit 签名 |
| [CLTRAY/SAGE](https://huggingface.co/CLTRAY/SAGE) | 官方 HF 模型 API 本轮 20 秒超时 | 作者仓库声明的发布入口；实际模型清单、不可变 revision、可下载性、模型卡和权重许可仍 unknown。超时不是不存在或不公开的证据 |
| [hf-mirror.com](https://hf-mirror.com/CLTRAY/SAGE) | 模型 API 与 README 精确请求均 HTTP 403 | 第三方镜像服务，不是原作者资产来源或许可来源；本轮没有确认镜像副本、版本或 hash，不提供未经核实的镜像下载 URL |
| [ModelScope](https://www.modelscope.cn/) | 精确候选 `CLTRAY/SAGE` 模型 API HTTP 404 | 仅证明该精确路径请求返回 404；不代表平台全局无其他同名或镜像，也不能证明作者背书 |

普通 sandbox 的 GitHub、HF 和 hf-mirror DNS 均失败；随后分别以审批后的只读 HTTP 请求核查。没有传凭证、调用下载端点或执行远端代码。上述状态是本次观测，不保证之后不变。

## 本地可见资产与声明值

`/home/gbk/.cache/huggingface/hub` 当前没有 `models--CLTRAY--SAGE` 目录；按文件名在已有结果目录和 PushT checkpoint 目录检索未发现以下 SAGE 组件。检索是限定路径，不声称全盘不存在。现有 [LeWM weights](/home/gbk/.stable_worldmodel/checkpoints/pusht/lewm/weights.pt) 与 config 存在；它们不是已认证的 SAGE 训练配套 backbone。

固定源码 [checkpoint registry](../le-wm-main/results/p2_learned_subgoal_asset_metadata_20261027_v1/sources/configs__checkpoints.json.source.txt) 声明如下信息：**这是源码中的预期值，不是本轮对远端文件或本地权重的验证。**

| 文件 | 声明字节数 | 声明 SHA256 |
|---|---:|---|
| `pusht_generator.pt` | 233878994 | `0b3647a3a41435969d750ec58176ef5f92a419c4eacae2b5cda74b35e63f90da` |
| `pusht_action_prior.pt` | 163404922 | `03ecab8f9d757eb8b3fc15e93481e830325428c24ede813c18f8692cc9b4bd80` |
| `pusht_far_action_prior.pt` | 163404602 | `60ed6831750e478b22c259b69e671236b41164f90284cb0422fe56d99e8c1425` |

前两项是主 SAGE PushT 组件；far prior 用于额外对照，不是主 action prior。另在 [assets registry](../le-wm-main/results/p2_learned_subgoal_asset_metadata_20261027_v1/sources/configs__assets.json.source.txt) 中 `pusht_lewm_object.ckpt`、`pusht_visual_generator.pt`、`pusht_visual_prior.pt` 都标为 `local_only`：记录中的 repo 字段不能推断这些资产已经公开；应向作者确认发布状态、取得途径与许可。主 state-assisted 和 visual-only 权重不可互换。

## 可复现下载命令：待前置条件满足后执行

以下命令**未执行**。需先核对远端文件清单、权重许可、权威固定 revision，并由操作者确认网络、磁盘、目标目录和下载范围；不指向已有封存目录。当前文件系统约 30 GiB 可用（本轮 `df`，会变化）。`mktemp` 创建独占临时目录；复制或归档前再检查可用空间和目的地，不能覆盖已有成果。

### 1. 固定源码版本，独占目录

```bash
SAGE_STAGE_DIR="$(mktemp -d /tmp/sage-assets-XXXXXXXX)"
printf '独占目录: %s\n' "$SAGE_STAGE_DIR"
git clone --no-checkout https://github.com/PKU-ML/SAGE.git "$SAGE_STAGE_DIR/code"
git -C "$SAGE_STAGE_DIR/code" checkout --detach 43c0a655bd2bbd33d8047c7588360e193671e9f0
git -C "$SAGE_STAGE_DIR/code" rev-parse HEAD
git -C "$SAGE_STAGE_DIR/code" status --short
```

确认 HEAD 正是给定 commit，且本地工作树干净；这只取得源码，不安装、不运行 `reproduce_main.sh`。`git clone` 获取的完整仓库体积本轮未知，下载前需确认空间与许可。临时目录可能被系统清理，需持久保存时由操作者选择新的、非封存路径。

### 2. 作者权重：先确认版本，后逐文件获取

当前未取得作者模型 immutable revision 和模型许可，因此不能给出已验证的固定版本直链。下列模板中 `AUTHOR_HF_REVISION` 必须先替换为作者模型 API 返回并核实的**完整不可变 commit**；必须先检查该 revision 的真实文件清单及许可，不得使用 `main` 冒称可复现。只选两项；无安装动作。

```bash
AUTHOR_HF_REVISION='REPLACE_WITH_VERIFIED_IMMUTABLE_COMMIT'
curl -fL --retry 3 --output "$SAGE_STAGE_DIR/pusht_generator.pt" \
  "https://huggingface.co/CLTRAY/SAGE/resolve/$AUTHOR_HF_REVISION/pusht_generator.pt"
curl -fL --retry 3 --output "$SAGE_STAGE_DIR/pusht_action_prior.pt" \
  "https://huggingface.co/CLTRAY/SAGE/resolve/$AUTHOR_HF_REVISION/pusht_action_prior.pt"
sha256sum "$SAGE_STAGE_DIR/pusht_generator.pt" "$SAGE_STAGE_DIR/pusht_action_prior.pt"
wc -c "$SAGE_STAGE_DIR/pusht_generator.pt" "$SAGE_STAGE_DIR/pusht_action_prior.pt"
```

逐项与上表和取得时可信来源的 SHA/大小比对；任何不符、404、访问控制、partial 或许可不明均停止，不加载或改写 registry。此模板假定与上一段同一个 shell，`SAGE_STAGE_DIR` 已指向新建独占目录；运行前检查变量非空且目录为本人创建。若原作者文件未公开，不应用其他来源的同名权重填补。

### 3. 镜像只作为候选传输来源

当前两个候选未证实有对应文件，**不提供可直接执行的镜像权重命令**。如以后找到镜像，先保存镜像平台、仓库所有者、不可变 revision、模型卡与权利说明；逐文件对照作者固定 registry 的完整 SHA256 和字节数，明确“镜像传输、作者声明 hash 一致”，不能因此推断镜像由作者运营或具有授权。若作者权利不明确、hash 不一致或来源不能确认，停止使用并联系作者。

## 完整复现仍缺什么

除可核验的 generator/prior，仍需与其训练配对的 backbone、该 backbone 的可信参数身份、数据和 episode split、三个 held-out manifests、action 与 lowdim stats、许可和兼容运行环境。PushT 原文主表允许 11 维 state/proprio；visual-only 是不同协议。即使上述权重的 byte hash 日后匹配，也只完成资产门，不证明接口兼容、配套模型相同、数值复现或 D2P 同协议优势。
