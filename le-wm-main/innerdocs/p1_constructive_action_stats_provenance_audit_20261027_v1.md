# Constructive action stats：独立只读来源审计

日期：2026-10-27。仅新增本文，不改sealed源码、task、normalizer、结果或共享文档；无网络、下载、神经、
物理、训练、scaler.fit或新planner统计。没有加载／反序列化任何权重、pickle、states.pth或rel_actions.pth。
本轮读取本地源码、小型配置／cache metadata，以及H5的action shape/dtype/attrs；**未重新扫描action值**。
下面全action数值由MAIN提供，本文只对这些既有统计做标量算术，不冒称本轮独立重算了全库均值。

## 结论

1. **数值确实不同**：本地全H5的std约比frozen public大3.25%／3.27%。不是四位小数舍入或N与N−1差异。
2. **找到public常数的本地可核文本出处**：JEPA-WMs的PushTDataset写死相同mean/std，注释仅为
   `precomputed dataset stats`；未给这些数的dataset SHA、revision、split／过滤或统计脚本。
   本地constructive从封存run_pusht_counterfactual AST提取的是这组数值约定，而不是从权重提取training stats。
3. **当前LeWM源码的pipeline不是该JEPA loader**：train.py先对所加载dataset完整列做row-NaN排除及
   mean/std，再train/val split；eval.py对所加载H5完整列做StandardScaler。没有看到从公共常数cache恢复scaler。
4. **单位证据不支持把这3.3%差异解释为遗漏×100／÷100**：JEPA的rel_actions先÷100才标准化；
   当前native relative command已是该相对尺度，H5 action量级O(1)，不是O(100)的像素位移列。
   不能据此说H5和JEPA数据逐动作等价，但也不能再将H5 raw除100补“修复”。
5. **subset／stats版本是尚未闭合的候选解释，不是已证原因**。checkpoint→training run→不可变dataset／
   preprocessing／stats的绑定仍unknown。差异本身既不能证明当前frozen public是checkpoint的错误normalizer，
   也不能证明它正确；不可改旧seal，不可把本H5统计自动替换进planner。

优先项RGB v2独立review已完成，文档为
[p1_constructive_selected_rgb_v2_independent_review_20261027_v1.md](/home/gbk/code/LeWorldModel/le-wm-main/innerdocs/p1_constructive_selected_rgb_v2_independent_review_20261027_v1.md)，
SHA `a04220ddfe16bef88f811c18688bdb283e379429b7a26643bfdca752d93f8391`。
未发现阻断matched/feedback真实RGB编码及actual-Q复算的问题；预测reference未桥接，null不是0误差。
review不代替MAIN后续8+4实际运行的receipt审计。

## 1. MAIN实测与本轮核验边界

路径：`/home/gbk/.stable_worldmodel/datasets/pusht_expert_train.h5`。
本轮metadata核验：46,300,921,856 bytes，action `(2336736,2)`、float32，file attrs={}、action attrs={}。
未读pixels/state、未重hash整个46GB归档；这些metadata本身不提供训练run或数据revision。

| 统计 | MAIN报告的本地全action | frozen public |
| --- | --- | --- |
| mean x/y | −0.00781256438 / 0.00686068723 | −0.0087 / 0.0068 |
| std x/y | 0.208467442845 / 0.206748626374 | 0.2019 / 0.2002 |
| raw min x/y | 约−1.4947 / −1.4655 | 常数未声明min |
| raw max x/y | 约2.04267 / 1.79206 | 常数未声明max |

mean差为`[0.00088743562,0.00006068723]`；std差为`[0.006567442845,0.006548626374]`；
std比值为`[1.03252819636,1.03271042145]`。这只是输入统计差，不是模型效应大小或成功阈值。
四位小数round后H5约为mean `[-.0078,.0069]`、std `[.2085,.2067]`，不能round成public数组。

对于N=2,336,736，样本std与总体std之比`sqrt(N/(N−1))=1.000000213974`，约0.0000214%，
远小于这里的3.3%。MAIN若有NaN／有效行数细节，应继续随原实测receipt保留；本文不推定已过滤多少行。
当前utils.py采用float32 Torch reduction，MAIN的高精度chunked算术并非逐bit相同实现；
本轮未运行Torch reduction，不能声称完全排除所有数值实现差异，但没有证据把大差异归为常规roundoff。

## 2. frozen public常数究竟来自哪里

### 2.1 Constructive的直接文本绑定

[p1_constructive_blind_encode_cuda_20261027_v1.py:106](/home/gbk/code/LeWorldModel/le-wm-main/experiments/spectral_reachability/p1_constructive_blind_encode_cuda_20261027_v1.py:106)
先核对run_pusht_counterfactual.py的SHA，再AST读取PUBLIC_ACTION_MEAN/STD，转float32。
返回对象明确`training_provenance_verified=False`及raw已是relative尺度、no second division。
直接来源文件SHA `0a4eb4592951139e596b35becede17d175df7b48a2fde16679ab0aac7c4dce34`。

[run_pusht_counterfactual.py:34](/home/gbk/code/LeWorldModel/le-wm-main/experiments/run_pusht_counterfactual.py:34)
给出相同常数，但没有不可变训练统计manifest。
其中旧normalize_public_action函数还写raw再÷100，属于旧函数合同，**不能当作constructive实际调用**：
constructive只AST取常量，并通过新packet路径对raw执行`(raw−mean)/std`。
[preflight_pusht_d2p_p0.py:80](/home/gbk/code/LeWorldModel/le-wm-main/experiments/spectral_reachability/preflight_pusht_d2p_p0.py:80)
已明确二次÷100曾是坐标bug，public统计只是约定而非checkpoint认证统计。本审计不改历史函数或旧报告。

### 2.2 JEPA-WMs public loader文本

[external/jepa-wms/app/plan_common/datasets/pusht_dset.py:27](/home/gbk/code/LeWorldModel/le-wm-main/external/jepa-wms/app/plan_common/datasets/pusht_dset.py:27)
注释`precomputed dataset stats`后直接写死同mean/std。其constructor：

- relative=True时读取`rel_actions.pth`，否则`abs_actions.pth`；先float再÷action_scale（默认100）。
- 可按n_rollout与dset_fraction截取episode；normalize_action=True时始终使用上述常数，**不按当前subset重估**。
- 再执行z-score；`get_all_actions`按seq_lengths取有效动作、拼接。未知常数最初计算是否也排除了padding。
- load_pusht_slice_train_val分别读data_path/train、data_path/val，两者用相同常数；n_rollout默认50、
  split_ratio形参存在，不等于这些常数就是在“前50条／80%”计算出来的。

本轮只读文本，没有导入该模块、torch.load或pickle.load。
可确认**相同字面数值与相同尺度约定**，不能据此认证LeWM权重由JEPA loader训练，更不能把本地vendor
副本的注释升级成publisher出具的checkpoint training lineage。

## 3. 当前LeWM train/eval与库normalize pipeline

### 3.1 当前train.py / utils.py

[train.py:58](/home/gbk/code/LeWorldModel/le-wm-main/train.py:58)先`load_dataset`，再对非pixels列调用
[utils.py:28](/home/gbk/code/LeWorldModel/le-wm-main/utils.py:28)：

`dataset.get_col_data → np.array → torch.from_numpy → 排除任一维NaN行 → mean(0)/std(0) → (x−mean)/std`。

std使用Torch默认样本校正；无÷100、clip、百分位归一化、按outcome过滤或bootstrap。
normalizer建立在`random_split`之前；当前`train_split:0.9`不能解释为“统计仅用90%训练split”。
训练clip会重叠／存在frameskip，不意味着统计按这些clip的出现次数加权。

当前YAML为`pusht_expert_train.lance`、frameskip5、action列cache，而本地获测资产是H5。
未找到此处Lance资产与该H5逐列相同的manifest；当前YAML也不是发布checkpoint历史run绑定。

### 3.2 HDF5/Lance get_col_data、frameskip与cache

安装库`data/formats/hdf5.py:140`的get_col_data返回原整列或内存_cache整列；cache由`f[key][:]`建立。
不是episode筛选后的clip列，不计算stats、不自动÷100。`_load_slice`只有非action列按frameskip下采样；
action保留每primitive后由Dataset.__getitem__按packet reshape为num_steps×10。
Lance `get_col_data`同样加载完整列到_cache；该cache是原始数组，不是持久化normalizer版本cache。
因此“frameskip5所以统计std应该缩小”或“packet十维所以要再缩放”不符合当前这两条源码路径。

### 3.3 当前eval与新版库scaler

[eval.py:90](/home/gbk/code/LeWorldModel/le-wm-main/eval.py:90)当前`stats_dataset=dataset`，不是comment里另一路stats集；
对完整非NaN列`sklearn.preprocessing.StandardScaler.fit`。按episode/goal_offset选evaluation起点发生在其后，
故当前num_eval=50／goal-offset合法起点筛选不参与这份scaler统计。
config/eval/pusht.yaml指定pusht_expert_train；没有硬编码public mean/std或stats文件路径。

安装库`data/utils.py:333`另有column_normalizer→ZScoreScaler，其NumPy std默认ddof0；
源码没有把这个helper自动替换进本repo train.py所调用的utils.get_column_normalizer。
两套API共存，不能只看函数名猜本批采用哪套。上述ddof0/1差在该N规模不足以解释3.3%。

## 4. 单位：raw relative、pixel displacement和token不得混用

安装native `envs/pusht/env.py:318`：relative时`target=agent.position+100*action`，后续PD积分；
这个100是动作到目标位移的环境尺度，不是normalizer再除100的理由。

JEPA路径的`rel_actions.pth/100`意图把像素相对位移还原到raw relative command，随后z-score。
LeWM H5 loader直接读action且训练utils不÷100；MAIN报告H5列的std约.2、范围约−1.5到2，
支持这份归档已经是relative量级，而非尚待÷100的像素位移。不能仅凭范围证明每条action的生成器来源，
但对本次3.3%差异没有100倍单位转换的正证据。

H5 expert动作部分超出[-1,1]也不等于像素绝对目标；constructive的tanh/logit球是另一个明确的研究动作支持，
不能把数据更宽的support偷换为可任意放宽本批动作球。支持不完全相同可能影响任务覆盖，非本统计审计已证根因。

仅用给定数值，令z_H=(raw−mean_H)/std_H，则

`z_public = [1.03252819636,1.03271042145] * z_H + [0.004395421595,0.000303133017]`。

raw=0的两种token分别约`[.043090639,−.033966034]`与`[.037476185,−.033183714]`。
这是潜在输入坐标差的描述，不是模型效果测试；不能由此推断Q反转或success0的因果效应。
本轮没有调用任何一种normalizer、没有把H5新数保存成scaler或供planner输入。

## 5. Cache／来源元数据查到什么，没查到什么

限定只读范围为本地dataset cache根、LeWM checkpoint根及其HF metadata、HF cache内stats/normalizer相关文件名，
和上述源码；不是遍历全机后声称不存在副本。

- `/home/gbk/.stable_worldmodel/datasets/`中有pusht_expert_train.h5，未见同名Lance目录。
- dataset `.cache/huggingface/download/`仅见零字节`pusht_expert_train.h5.zst.lock`；没有这份H5的download
  metadata／revision／已保存scaler证书。lock不是下载完成或统计来源证明。
- checkpoint目录有weights.pt、config.json、README.md及HF下载metadata，没有本次范围可识别的独立stats.json、
  normalizer.pkl或training run config。未反序列化权重查看是否有额外buffer，不声称扫描了所有binary内部字段。
- config.json仅模型结构；README声明dataset为quentinll/lewm-pusht，不给dataset revision、split、统计值。
- config和weights的HF metadata首行同为`22b330c28c27ead4bfd1888615af1340e3fe9052`；weights metadata第二行
  `48938400ae3464c9680731287f583a9cb516f55a8ec64ea13a91be47fb15b607`与封存文件SHA声明相同。
  本轮没有重读weights内容；这些是下载cache metadata，不能转授成数据版本／训练run证明。
- 当前load/save_pretrained源码保存state_dict和给定config；未见必需normalizer manifest的恢复流程。
  这解释为何结构可加载仍无法恢复normalizer训练来源，不证明历史publisher一定没有其他记录。

subset、episode版本、collection条件、padding／有效步过滤、不同数据格式转换均可能导致统计差异；
尚无来源证据指出具体哪一种。当前LeWM源码已经排除“训练90% split或eval50起点直接导致当前统计”的简单解释；
JEPA loader虽支持subset，但它不按subset计算常数，故也不能反推public常数来自哪个subset。

## 6. 能判定的等级与缺失来源链

| 问题 | 结论 |
| --- | --- |
| H5全action与public数字是否相同 | 否；基于MAIN完整列统计，本文标量复算差异 |
| 当前constructive是否使用H5新统计 | 否；它绑定AST public常数并明确training_provenance_verified=False |
| 是否发现100倍单位错配解释这3.3% | 未发现；现有loader/env文本与MAIN范围不支持该解释 |
| 是哪个subset或stats版本导致 | unknown，尚无不可变来源证据 |
| checkpoint真实训练normalizer是哪组 | unknown；当前train源码不是历史run证明 |
| 是否应修改sealed planner | 不应；差异本身不足以决定正确替代值或归因 |
| 是否解释feedback排序反转／native失败 | 未测因果，不能宣称 |

要闭合而不事后fit，优先寻找**已经存在的**publisher/run材料：

1. checkpoint不可变revision／tensor身份→真实training run config／代码revision。
2. 对应dataset不可变revision／action field SHA、episode清单、format转换、raw单位与过滤/padding规则。
3. run实际使用的scaler数值、dtype、ddof、计算时机、subset、保存manifest及SHA。
4. 发布eval的process对象究竟是训练scaler恢复还是评估集重算，及与训练统计的偏差来源。

这些缺失不阻止继续只读接口／有限物理诊断，但阻止把public或H5数值称为已认证training stats。
若MAIN未来授权normalizer敏感性实验，只能NEW独立variant，明确固定比较与原raw动作边界，不能覆盖旧seal、
不能依native outcome挑常数，也不能把其结果追认为原批无偏确认。本文没有授权或启动该实验。

## 7. 本轮文本证据SHA

以下均本轮直接只读hash，无网络；路径相对`/home/gbk/code/LeWorldModel/le-wm-main/`：

| 文件 | SHA256 |
| --- | --- |
| utils.py | 048701806dfc045f16d33a85475861a9b4c78b314b7dda84161f5d018f17e9d7 |
| train.py | c5b4ced2295113186d2834671020f92104b56793839c7fce490ba0050bef90d4 |
| eval.py | fee483b98d060ac5b5e202843e2a8d7c01414c6d31629fb94ece5935ae7f5af1 |
| config/train/data/pusht.yaml | dcdd082ebdc628cdc8a88f15cb647fce6eec87343f9fe190271946ff5d52f757 |
| config/train/lewm.yaml | fef2e609e3e1be367400e69e0929229ee0b6bd95cc4d4681c075ec9ab7d4cf89 |
| config/eval/pusht.yaml | 66eca39ebc1034d0cf746fef80e1c6309c07dc1fbd6fa067779ac97c446d2856 |
| external/jepa-wms/app/plan_common/datasets/pusht_dset.py | 30070c5a9f4770b84e3d6b6e5323df01d9c3405042cb988af07fcdcf43b6bf92 |
| .venv/lib/python3.12/site-packages/stable_worldmodel/data/formats/hdf5.py | afe836ec706acc6c55450f3c936a5898b57ed65a999695af25b4b59cd14d10f3 |
| .venv/lib/python3.12/site-packages/stable_worldmodel/data/formats/lance.py | 5643a80df28fc8b658b7323bbfcc7684c4ce4280f1ecd24a160ef823c280d331 |
| .venv/lib/python3.12/site-packages/stable_worldmodel/data/normalization.py | 5de5a217774cf634dee33d8d082b38c1825013e80f3fe2cb1c5faa8cc9958f5e |
| .venv/lib/python3.12/site-packages/stable_worldmodel/data/utils.py | a132734367c219c0fdf59cd9d3631ed83bfe58bc427aa41e85146e26ac79ab4c |

无需pytest或模型运行来认证本文；实核范围是源码／metadata读取和给定数值的算术。
全H5统计的完整row过滤、action field SHA与scan实现应由MAIN原receipt单独保存，不用本文替代。
