# Hi-LeWM 原始 action normalizer：官方资产获取路线 v1

日期：2026-10-29（Asia/Shanghai）。仅新增本文；不修改冻结 source/results，不拟合统计、不借用其它 checkpoint 的 stats、不运行模型/物理、不反序列化或执行包内脚本。目标是定位作者已经保存的原始 process，而非重新估算一个 scaler。本文不是资产已取得/已认证、完整官方 baseline 或物理放行证明。

## 结论与最小下一动作

本轮新增实质定位：官方 archive 内 `setup_paper_datasets.sh` → `setup_datasets.sh` 明确将 PushT 路由到 **dataset 仓库** `quentinll/lewm-pusht`，与同名 **model 仓库** 是两个不同 namespace。脚本发布的是数据获取路线，不是 frozen normalizer 下载器；不能运行脚本后把现场 fit 结果称为作者原始资产。

经正规提权的只读网络探测：Zenodo 官方 versions API 返回200、total=1，当前公开版本仍为21353240，唯一顶层文件仍是原 `package.zip`。未发现新发布的独立 frozen-process 文件入口；该观察不声称遍历全部第三方或私人资产。HF dataset/model 两个 API 本轮均 `Network is unreachable`，**线上 siblings、revision、是否另有 process sidecar 保持 unknown**，不能据网络失败断言仓库不存在该文件。

最小下一动作：在可达官方 HF 的已授权网络环境，仅查询下述两个官方 metadata API，固定 returned commit 后检查**实际列出的** process/normalizer/statistics sidecar。若没有作者原始 frozen 文件，转向官方 Zenodo record 的作者联系入口请求原始 process 数值导出与 provenance；不要退回下载完整训练数据重新 fit。没有候选资产 URL/size/hash，故本轮不新增下载器、不下载数据/权重、不做第二份整包传输。

## 1. 已确认的官方来源链（不是推测 URL）

工作区基准目录 `/home/gbk/code/LeWorldModel/le-wm-main`，下文仓库内路径均相对此目录。

| 官方原文 | 精确位置 | 本轮确认事实 |
|---|---|---|
| `results/p1_hilewm_text_contract_20261027_v1/texts/package__README.md.source.txt` | 19–34，尤其28 | 复现入口 `source scripts/setup_paper_datasets.sh --home "$STABLEWM_HOME"` |
| archive `package/code/scripts/setup_paper_datasets.sh` | 33–44 | wrapper 固定 pusht,cube；44显式 source 同目录 `setup_datasets.sh` |
| archive `package/code/scripts/setup_datasets.sh` | 111–123 | 官方 collection `https://huggingface.co/collections/quentinll/lewm`；119将 `pusht` 解析为 `quentinll/lewm-pusht` |
| 同脚本 | 160–194 | 请求 `https://huggingface.co/api/datasets/{repo}`；仅选择后缀 .h5/.hdf5/.zst/.tar/.tar.gz/.tgz |
| 同脚本 | 197–214 | 实际 payload URL 模板 `https://huggingface.co/datasets/${repo}/resolve/main/${relpath}?download=true`；relpath 来自实际 siblings，不可猜文件名 |
| `.../texts/package__code__config__eval__hi_pusht.yaml.source.txt` | 14–19、93–99 | cached action/proprio/state，原 eval.dataset_name=`pusht_expert_train` |
| `.../texts/package__code__h_le_wm__eval__hierarchical.py.source.txt` | 309–331、505–509 | 从该 dataset 全列排除含 NaN 行后现场 `StandardScaler().fit`，再传 process；这条代码路径没有读取 frozen process |
| `.../texts/package__code__h_le_wm__planning__policies.py.source.txt` | 890–895 | 从 action buffer 取出并 reshape 为原 env action shape，然后 `process['action'].inverse_transform` |
| `.../texts/package__checkpoints__pusht__main__config.yaml.source.txt` | 59–66、104–118 | embedded low 来源及训练数据名/frameskip；配置中数据名不等于固定数据 revision 或已保存 scaler 数值 |

不能照原 setup 脚本直接操作当前工作区：其208使用浮动 `main`，203–205仅按文件存在就跳过；221含 zstd `--rm -f`，245删除压缩源，269/276/283直接 tar extract。本任务只把它作为 URL/provenance 原文，**没有 source/bash 执行它，也不推荐以此脚本取得证据**。

同名 model 地址 `https://huggingface.co/quentinll/lewm-pusht` 来自已冻结 `.../texts/package__code__h_le_wm__checkpoints.py.source.txt:18`，可作为 embedded low 发布方补充线索；但 model 仓库的一组常数不能自动认证为 Hi archive 本次 eval 的 process。新 installed `stable_worldmodel.data.normalization` 自有 scaler 实现也不等于原 eval 的 sklearn StandardScaler。两者都不得替代本次缺失原资产。

## 2. 本轮恢复的两份官方脚本文本（只读、内存、有界）

使用已封存 `results/p1_learned_subgoal_remote_inventory_20261027_v1/archive_tail.bin` 的414项 ZIP central directory，按精确 member 名定位；随后只读 `results/hilewm_archive_MAIN_20261028_v1/archive.part` 的已存在字节。核 local header signature/method/name、packed/unpacked 长度、bounded raw-DEFLATE EOF/no trailing、中央目录 CRC；未保存解包文件，更未执行脚本。

| Member | local-header offset | packed / decoded bytes | CRC32 | 本轮 decoded SHA256 |
|---|---:|---:|---|---|
| `package/code/scripts/setup_paper_datasets.sh` | 59090 | 534 /1115 | `49340427` | `b41f243e7daaf60fad8f912d2047ac09d456f1d4b16578dbc757cfe8ebafaf65` |
| `package/code/scripts/setup_datasets.sh` | 53769 | 2764 /8106 | `1274852a` | `792fd71e97ff766f50af5ba74d2cd108e85d37192c97a25ee26c98062ec7d1df` |

这些是本地文本字节身份与已引用目录的 CRC 对应，**不是作者提供的 member SHA 签名、完整 ZIP MD5 验收或 normalizer 认证**。旧 partial 没有续传、修改或删除。

相关封存字节 SHA256（本轮实读）：

```text
zenodo_api.json             2fc221128c12b609b86dfb7810eda46f2399bc796348f36c5d7e016ab25dd773
archive_tail.bin            43a0c87d4f1b652caccbc26ac5e1b1ed73c0c6cf97c8e00683ecc1bdd8c16aef
archive.part                1307db333b49c5607f9541945eb128a3bf426d68602eec8d847790ddbcddf555
package__README.md          ae9262b5b5c6d39f2d4762002cf79d568c74b5ffdd81c1208a6627d79551af17
eval__hierarchical.py       0a2c2b2688b73a388ff6011c4b32081704a7641b31efa1645a47b2f1d34eab3c
config__eval__hi_pusht.yaml  35b02a4f6e22dd9fecf034d664e799b2c5c85f157750ba53e0d69ba0be1091cc
planning__policies.py       d6e7d476b45e47c5015494775b4d96727b8ebb8c801f719a100ba32878f288d1
pusht__main__config.yaml    dff6ac0dc9e7c4cfb7d76b268ceb568f6fc0916eaa4a725e6e1a5b53587e9075
```

前两者位于 `results/p1_learned_subgoal_remote_inventory_20261027_v1/`，partial 位于 `results/hilewm_archive_MAIN_20261028_v1/`；其余 source 文本均在前述 `results/p1_hilewm_text_contract_20261027_v1/texts/`，完整 basename 见上表。

## 3. 实际网络观察与尚未知

均通过工具 `require_escalated` 的只读 metadata GET；无 credential、无绕权限、无镜像替换、无模型/data payload 下载。HF 单请求读取上限2MiB、socket timeout25s、redirect 禁止；Zenodo 上限1MiB、socket timeout20s、redirect 禁止。无自动重试；失败保留。

| UTC 起始 | URL | 实际结果 |
|---|---|---|
| 2026-10-28T16:35:01.150557+00:00 | `https://huggingface.co/api/datasets/quentinll/lewm-pusht?blobs=true` | URLError `[Errno 101] Network is unreachable`；无可用 HTTP 状态/body/commit |
| 2026-10-28T16:35:26.201319+00:00 | `https://huggingface.co/api/models/quentinll/lewm-pusht?blobs=true` | 同上；不是403，不证明文件不存在 |
| 2026-10-28T16:36:46.221434+00:00 | `https://zenodo.org/api/records/21353240/versions` | 200，5176 body bytes，SHA256 `42a8c0bce637ba1859f2f29bf8d29a18ee19b4ff20a7cee86880d6d2b1515089`；hits.total=1，id21353240 |

Zenodo 返回唯一 file `package.zip`：size2299863924，publisher `md5:aaf78b8b4fd6f5d1f5e720927951152f`，URL `https://zenodo.org/api/records/21353240/files/package.zip/content`，与封存 API 同对象；没有为此重新下载2.3GB包。record DOI `10.5281/zenodo.21353240`，concept/parent `21353239`。这是本次可见 versions 响应的观察，不是将 latest 永久视为不变。

网络 JSON/错误作为本轮工具 stdout 收据记录，本文保留上述观测与 raw response SHA；**没有新增独立 raw-response 文件，不能将本文当作可重验原响应或作者认证**。如 MAIN 后续需要可复核 metadata 闭包，应在新独占目录重新授权取得并保存原 body/headers，不覆盖本轮或旧收据。

## 4. 安全获取路线与精确停点

### 路线A：官方 HF 实际 sidecar（可达后优先，未取得）

1. 只读查询上述 dataset/model API；保存完整 HTTP status/headers/body、UTC、URL、原 body SHA和 returned commit SHA；保留失败且不凭 body 长度推断存在。检查 JSON duplicate keys、非有限数和合理大小。
2. 仅沿实际 siblings/tree 返回的文件名读取 metadata。若 model/dataset 文件清单没有可验证 frozen process，停止该分支，不猜 `normalizer.pkl`/`stats.json` URL，不把常用同名路径当作已存在资产。
3. 有候选时把 `resolve/main` 改成**实际 API 返回的固定 revision**，固定 file size 与 LFS sha256/官方 digest（若有）。HEAD/Range 只用于确认传输；size/ETag/HTTP200 不是 process 内容语义或与 checkpoint 配对证明。当前 commit、filename、size、digest 全部 unknown，本文不填未来值。
4. 小 JSON/NPY 数值 sidecar 可在授权的新独占目录按长度/内容 SHA、有界/排他/无覆盖方式下载；若只提供 pickle/joblib/PT，先仅保存字节并按已有 static/inert 规则检查，**不 load、不动态 import 类**。不直接解压 archive 到工作区。
5. 若官方仅提供训练 HDF5/archive，它只能认证数据来源，**不满足“取得原 frozen scaler”**。本任务禁止重新 fit，故不下载多GB训练数据来掩盖缺口。数据 identity 也不由名称 `pusht_expert_train` 充分确定。

### 路线B：请求作者公开原始 process（当前最直接的缺件请求，未发送）

官方入口 `https://zenodo.org/records/21353240` / `https://doi.org/10.5281/zenodo.21353240`。缓存与本轮 metadata 的 creators 包含 `Castelli, Niccolò`，ORCID `0009-0006-1295-7178`，以及 Massafra/Punzo/Lo Sardo/Pantelidis。仅将 record/ORCID 作为作者身份线索，未臆造邮箱、未发送消息、未借作者同意自签认证；论文显示名与 metadata 拼写差异不自行消解。

建议由 MAIN 联系作者的最小请求（以下是请求内容，不是已存在文件 schema）：

> 请提供 paper-facing `pusht_hi_lewm_epoch15_object.ckpt` 所用 `eval.build_process_map` 的**当时已保存原始** `process['action']`（若没有保存请明确说明），优先公开为 data-only 原精度 NPY/JSON，附 dtype/shape、原字节 SHA、mean_/scale_/var_、with_mean/with_std、n_features_in_/n_samples_seen_、sklearn/NumPy版本、XY与raw单位。请绑定该 checkpoint 实际字节 hash、数据仓库 revision及文件 hash/列与原数据筛选流程，说明是否来自完整 `pusht_expert_train` 去NaN行，而非训练split或重新拟合；如已有原 frozen export/运行 archive，请提供官方固定版本 URL/文件清单/size/checksum及其对应运行。此请求不要求重新训练或重新 fit，也不将新算统计冒称历史原资产。

官方源码确实现场 fit，因此“作者当时没有保存独立 scaler”是可能的，不能预设一定存在。若作者确认未保存，当前授权内只能记录这个事实与 provenance，**不能在本文或旧 low-CEM 中擅自解除 raw inverse 的 hardfail**；是否允许另一种复现协议须 MAIN 另行决定。

### 下载前共同约束（本轮未执行 payload 下载）

先固定真实候选 URL、版本、文件大小、必要磁盘预算和 checksum，再申请网络/磁盘授权；新排他文件/目录，不覆写/混拼旧 partial。保持 HTTPS 与重定向主机审查；遇403或权限失败不绕过。Range 必须206且 Content-Range/总长/实际body长度吻合，若200则不能当 range拼接；整文件检查最终实际长度与 SHA，有作者 digest才检查作者digest，否则只称 transport complete。可恢复句柄不因 poll 超时重启。压缩件先有界元数据检查再另授权解码，不执行外来代码。没有已授权下载预算，本文不提供会误触发完整数据下载的 `curl -L` 命令。

## 5. 交付检查与状态边界

实际执行：两份 setup 原文只读恢复各 exit0，精确长度/CRC/EOF/无trailing/SHA检查通过；官方源文本及引用行号检查；两项授权 HF metadata 请求失败、一项授权 Zenodo versions GET成功。没有神经测试、没有 scaler fit、没有物理、没有新数据/权重传输；无需新增运行代码或重复已有 missing-normalizer 审计。旧 low-CEM report/strict 和所有冻结三件套均保持不变。

仍缺：可读取的官方 HF 最新固定 revision/实际 siblings（本轮网络不可达），以及与该 Hi checkpoint/eval 明确绑定的**原始已保存** process 数值/flags/provenance 或作者对其未保存的确认。当前 `raw_inverse_authority=unknown`、physical release 仍拒绝；本路线不代表已取得/认证 normalizer。
