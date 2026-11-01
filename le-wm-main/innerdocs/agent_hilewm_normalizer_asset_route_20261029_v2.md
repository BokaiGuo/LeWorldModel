# Hi-LeWM normalizer 资产路线 v2：镜像实际 metadata/tree

日期2026-10-29，工作目录 `/home/gbk/code/LeWorldModel/le-wm-main`。本轮只新增同名前缀 source/tests/本文和两个排他 results 目录；v1 route、全部 Hi loader/CEM、模型和旧结果不改。无训练、stats fit、模型/物理运行或大 archive/weights 下载。

## 新结果，而非重复缺件审计

经 `require_escalated` 正规批准，**hf-mirror.com 这次可达**。实际4次只读 API GET均200，随后沿实际 tree 列出的固定 candidate revision 读取3份小型 README/config，均200且精确长度/候选 git-blob identity 匹配；总响应body4682bytes，其中API3055bytes、三个小文件1627bytes。没有重启低层 CEM、没有取13.1GB训练包或72.3MB权重。

本次镜像快照没有列出独立 `dataset_info`、`process`、`scaler` 或 normalizer sidecar；实际 config 也没有 action scaler 的 mean_/scale_、flags 或数据 provenance。**这是这两个镜像快照的可见范围，不是官方历史/其他revision/文件内部/私有附件的穷尽否定。** `raw_inverse_authority=null`，physical release 仍拒绝。

镜像 candidate commit、`X-Repo-Commit`、git blob SHA1和 LFS SHA256都是该 provider 的声明/内部一致性线索。**没有据此认证官方对应revision、历史process或与 Hi checkpoint 的配对**，也没有开启 inverse。v1官方HF网络不可达的记录仍保留，不将镜像成功倒填为官方请求成功。

## 实际可见仓库和文件

| Namespace | 镜像 candidate revision | 实际 tree 全部 file 路径/声明长度 |
|---|---|---|
| dataset `quentinll/lewm-pusht` | `655cd446b9929369d7d406001da85c15d1457850` | `.gitattributes`2504；`README.md`136；`pusht_expert_train.h5.zst`13136247974 |
| model `quentinll/lewm-pusht` | `22b330c28c27ead4bfd1888615af1340e3fe9052` | `.gitattributes`1519；`README.md`178；`config.json`1313；`weights.pt`72290721 |

两份 API cardData 仅给license，model还声明datasets=`quentinll/lewm-pusht`；没有内嵌 dataset_info/process 数值。API siblings 与对应固定revision tree 在本轮快照中一致；分页 Link 未给下一页，工具记 `tree_complete=true`，同时保留 `exhaustive=false`：这不是官方或全历史资产穷尽证明。

大文件只记录声明，未读取内容：dataset LFS candidate SHA256 `7cfbd6d90fa2f27876379a5ff169715a36ed82edbda64f9e5b5bfa34d212f318`；model LFS candidate SHA256 `48938400ae3464c9680731287f583a9cb516f55a8ec64ea13a91be47fb15b607`。不宣称其中是否有原process，不打开/解压/加载它们，也不下载数据以重算统计。

## 保存的实物与来源

所有路径相对此工作目录；均为新排他目录，不修改旧partial。

1. `results/agent_hilewm_normalizer_asset_route_20261029_v2_mirror01/report.json`
   - SHA256 `f8eeae02c04ef66449243f687c199799c117fb438da9922451bdd1853d6d2256`，实际4.3408s，status=`metadata_recorded_not_scaler_certified`。
   - 同目录 `response_00.bin`917bytes：`https://hf-mirror.com/api/datasets/quentinll/lewm-pusht?blobs=true`。
   - `response_01.bin`511bytes：`https://hf-mirror.com/api/datasets/quentinll/lewm-pusht/tree/655cd446b9929369d7d406001da85c15d1457850?recursive=true&expand=false&limit=1000`。
   - `response_02.bin`1039bytes：`https://hf-mirror.com/api/models/quentinll/lewm-pusht?blobs=true`。
   - `response_03.bin`588bytes：`https://hf-mirror.com/api/models/quentinll/lewm-pusht/tree/22b330c28c27ead4bfd1888615af1340e3fe9052?recursive=true&expand=false&limit=1000`。
   - 每次UTC、HTTP状态、实际body长度/SHA、选定非敏感headers、候选revision/tree/原始cardData与错误均记录；raw bytes已保存，非仅stdout摘要。report绑定本轮实际执行的工具source SHA及冻结v1文档SHA。
2. `results/agent_hilewm_normalizer_asset_route_20261029_v2_small_metadata01/report.json`
   - SHA256 `1de9c82ac42613de2973d0d5cdb4d83b34239612dfab6b92aba85f889f423034`，实际3.2859s；绑定上项report完整SHA，保留每个实际URL/headers/字节收据。
   - `metadata_00.bin`136bytes：dataset `README.md`，SHA256 `ebf5f6c780c07bd365ff8b947acb33e11e06e0ded471de73893d0ec545ed7ccd`。
   - `metadata_01.bin`178bytes：model `README.md`，SHA256 `660de4e79b109ea18602618d90c72c44630957978ebd8474910713d896340292`。
   - `metadata_02.bin`1313bytes：model `config.json`，SHA256 `2564086e961e7b5c7c04dffc451091115b389a590645ff19653c64fd0bc16e09`。
   - 使用实际candidate revision的 `/raw/{revision}/{实际文件名}`，禁止redirect，每项上限4KiB。完整字节长度与tree声明相等；`sha1(b'blob '+decimal_length+b'\0'+body)`等于同镜像tree oid。该检查仅为git blob字节一致，不构成官方签名或原scaler认证。

两份 README 只给license/LeWorldModel论文链接；model README额外列dataset。config顶层仅 `_target_`, encoder, predictor, action_encoder, projector, pred_proj，后两项 `norm_fn=BatchNorm1d` 是神经模块构造配置，**不是action inverse的StandardScaler**。没有因它出现“norm”就采用它作统计参数。没有导入或执行任何 `_target_` 字符串。

## 工具边界、运行命令及测试

`experiments/spectral_reachability/agent_hilewm_normalizer_asset_route_20261029_v2.py` 是stdlib短工具，不安装全局命令、不改credentials/环境、不执行foreign code。API host仅官方HF或hf-mirror；仅本repo metadata与实际40hex revision tree，API工具本身禁止resolve/raw/payload。必须显式 `--network-authorized`，但该flag**不代替工具外的网络审批，更不签任何资产authority**。每响应上限1MiB，最多6请求，总输出cap16MiB、floor200MiB、整轮150s且socket20s、不重试、不跟redirect；请求失败/403与已收原字节保留，分支不伪造revision/size0，轮末写report。这里只进行GET，无写远端API；分页必须同host/同固定revision并防cycle/重复file。

collector exit0仅表示至少一个tree保存完成；exit2表示metadata blocked但失败报告已保存。二者都不表示取得scaler。后续可在独占新目录、再次获得网络批准后运行（不得覆盖本轮目录）：

```bash
.venv/bin/python -B -m experiments.spectral_reachability.agent_hilewm_normalizer_asset_route_20261029_v2 --host hf-mirror.com --output /home/gbk/code/LeWorldModel/le-wm-main/results/agent_hilewm_normalizer_asset_route_20261029_v2_mirror02 --network-authorized
```

本轮三个small metadata通过单独有界stdlib命令读取，不在collector自动下载范围内；只选择前一实际tree的两个README及config，按精确revision/size/gitblob检查，原命令在chat工具收据可见，没有新建泛化下载器。API工具不能用任意URL绕到raw或大文件。没有为尚不存在的process编造文件路径。

实际离线RAM回归：

```bash
.venv/bin/python -B -m pytest -q -p no:cacheprovider tests/test_agent_hilewm_normalizer_asset_route_20261029_v2.py
```

22passed in0.15s：metadata-only路由/HTTPS/固定revision、重复key/NaN/cap、恶意member/path、pagination换host/revision、size缺失不填0、403真实保全、fake candidate revision不能开authority、排他输出、全JSON往返。fixture不运行模型/网络，不构成官方来源证据。本轮还实际复核全部7响应SHA/长度、三个small gitblob、tree/siblings一致和v1SHA未变；既有low-CEM不重跑。

## 下一最小动作与停点

在这个已实际读取的镜像snapshot中，优先项 dataset_info/process sidecar 并未列出，已取的小config无原scaler；不重复下载大训练包，不从其它项目拷贝mean/scale，不fit、不打开weights寻求反序列化捷径。要继续开inverse，仍需作者公开/提供与Hi checkpoint和当时eval/data provenance绑定的**原始已保存**process数值及flags/原精度/字节hash，或官方可认证同对象archive里明确的实际成员。可沿v1已确认的Zenodo record/作者入口发送缺件请求（本轮未发消息）；若作者未保存原process，明确记录事实，不能把另一份新算统计改名成旧资产。

尚未知：镜像candidate revision与官方同revision的独立认证；是否有不在本快照中的原frozen process；原process的实际数值/原始dtype/flags/训练数据版本与checkpoint配对。所有这些unknown保持，不以镜像200、tree_complete或自称Official的README开门。
