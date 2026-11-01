# P2 stageB 真实历史 triplet inventory

独占新stem source/tests/本innerdoc；旧lineage/suffix及MAIN封存均不修改。使用experiment-code的闭包/回归部分，不运行神经、physics、网络或子代，不接触reserved/fresh路径，不训练、不调Q/normalizer/动作/门槛。测试不是科学证据。

## 当前资产与明确停点

原全4 known根8臂，每项均有真实P2 planning-root PNG（warmup15/相对0）、actualstep5 PNG、journal中的真实7维状态及step5命令。通过原 physical report/prepare/模型/源码闭包认证；step5像素必须等于已paid actual-z5的输入像素身份。

额外定位到P4 live run `results/p4_feedback_live_cuda_20261027_v1_cuda_main01/receipt.json`，SHA `e3c91195540b7465e2fcbf8431d7f182c06bfdff9da6ab160489ae6fe23bbb47`。此run为原root0的imagined/actual两个分支，各自保存warmup三个packet的每步PNG及命令，最终receipt列文件SHA。根/history匹配不是靠文件名：

1. P4 terminal成功、source_verified、错误为空、模型/prepare闭包、真实warmup计量；
2. prepare reference与原P2 root SHA/history整组SHA/goal像素SHA完全相等；
3. 两臂各15个实际zero command记录，逐步PNG文件和像素SHA，三个packet确实complete5；
4. 三个distinct observation的warmup5/10/15像素stack SHA精确等于原history stack；goal像素精确一致；
5. warmup15像素与原P2真实root PNG一致，step5复用本臂旧物理记录。

文件名仅定位由已封存schema产生的地址，不能授权身份。每项保留两个等价history来源，不按outcome挑来源。可相同像素，但必须是不同真实observation记录和真实文件引用；绝不复制帧或用root重标warmup10。P4没有保存warmup10 state7，输出null；其RGB时间来源由实际命令/边界和stack认证，不能冒称有该时刻原生状态快照。

当前可认证 **root0两臂完整triplet**；其余root1/2/3的root15和step5有，warmup10精确观测闭包未建立。完整8项分母保留，6项missing为明确字段，非补零、非跳过。

因此不交部分根stageB CUDA runner，也不重复stageA。全8齐备前`stageB_execution_ready=false`。此任务走用户授权的“缺资产先给计量重建差项”分支，不伪造可执行adapter或科学报告。

## 发现层与证据层

CLI inventory额外用`rg --no-ignore`扫描results内JSON/JSONL的原4根root/history精确字符串，单文件64MiB以内。排除路径名含fresh/reserved；不跟随symlink，候选不按文件名认领。只读匹配文件并记SHA；对可能含PNG/NPZ/frame引用的元数据给出schema候选列表。未知schema仍discovery-only，不变成真实观测证明，也不据其未知就宣称磁盘不存在。

它不是全球磁盘穷举：无身份sidecar的孤立PNG、无root/history身份的NPY/NPZ、超cap/其他格式不能证明可执行时间身份，不自动接受。身份索引候选若需新schema适配应新版本处理，不能无声默认。inventory错误保留且不放行。

本版本自己的输出路径不纳入发现输入，防止analyze输出污染后续inventory；verify核已保存输入集合并从已认证P2/P4资产精确重建rows/history/reconstruction合同，不重扫动态目录。verify是saved inventory snapshot完整性核验，不是新的神经/科学execution authority。

worker实际全量saved discovery读取2886份身份匹配元数据，其中1511份含图像/数组引用候选。保留1项旧`p3_horizon_residual_physical_20261027_v4_main01.stdout.json`的Extra data解析错误，未改该文件或跳过错误声称成功。状态为`blocked_missing_or_unclassified_history`。未分类schema可能含其他资产，需新身份适配后才能认领，不能将本版本缺项断言升级为物理上不存在。

## 所缺的最小计量重建

只重建缺历史的root1/2/3，一根一次，身份通过后可供两个原臂共享warmup。按原完整fixed4 collector/seed/count4/oversample4生成，不挑根、不打开reserved4fresh；已有运行实账collector为32 resets。

计划公共API轴：constructor尝试1；原collector resets32；缺根reset3；真实zero warmup steps45（3根×15）；selected prefix/suffix steps0。返回数尚未运行为null，不能写成功计数。原helper显式render至少15次（每根3个边界前后pure render＋current render）；实际reset内部render/space.step/setter等必须另挂attempt/return，不把该lower bound当总render。factory返回后才可安装的hook不得声称constructor前覆盖。

需实际保存每根warmup5/10/15三张边界PNG、原goal PNG、逐step raw command和7维state/snapshot，并严格核原history整组SHA、goal像素SHA、root state7身份。任何不一致、native stop、工程失败都保留完整8分母，不重试换动作、不padding、不静默rerender。

重建本身：0encoder/projector/predict/AD。尚无MAIN授权或已审计重建adapter；wall/output/floor必须在新adapter准备前封存，本版本不invent cap，不声称已可运行physics。disk和实际wall未知null。

全8闭包后stageB才实施原缓存teacher full3 + 真实triplet的public-rollout vs manual：每项每route各1predict，共16predict、16AE、16predictor、16pred_proj；0encoder/AD/physics，不重编码、原goal/Q/动作/precision保持。public `T=H=3`使n_steps=0但官方loop n_steps+1，所以只有1个suffix packet，不能多拼future token。fullnative buffer、失败分母与计量由后续独占runner实现。原历史逐帧编码证明问题与本cached API比较分开，不因raw RGB找齐就声称表示完全正确。

## MAIN 命令

在 `/home/gbk/code/LeWorldModel/le-wm-main`：

```bash
OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m experiments.spectral_reachability.agent_p2_midpoint_triplet_inventory_20261028_v1 inventory --output results/agent_p2_midpoint_triplet_inventory_20261028_v1_MAIN01.json
```

将实际返回SHA放入`P2_TRIPLET_INVENTORY_SHA`，只读核验：

```bash
OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m experiments.spectral_reachability.agent_p2_midpoint_triplet_inventory_20261028_v1 verify --inventory results/agent_p2_midpoint_triplet_inventory_20261028_v1_MAIN01.json --sha "$P2_TRIPLET_INVENTORY_SHA"
OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m pytest -q -p no:cacheprovider tests/test_agent_p2_midpoint_triplet_inventory_20261028_v1.py tests/test_agent_p2_midpoint_history_lineage_20261028_v1.py
```

worker实际运行仅saved/RAM，最终测试终态及三SHA见冻结回复。无模型/physics结果，无P1/P2/freshness/科学准入。交付完成是inventory/缺项/测试/文档，不是完整stageB或研究完成；冻结后三文件不再追加，后续新版本。
