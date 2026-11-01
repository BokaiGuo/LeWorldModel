# Dense suffix：独立saved审计（包含真实partial）

本审计不import MAIN producer，不调用其collect/verify/analyze，不构造环境或模型。
使用experiment-code技能组织纯NumPy/mock与saved测试；不训练、推理、运行物理、生成新控制候选。
复用旧独立saved审计中的纯task/pose/Reader工具；不改这些冻结工具或任何旧材料。

## 固定输入与可复验边界

MAIN真实report：`results/p1_dense_suffix_physical_main_20261027_v1_main01/report.json`，
SHA `76fb65359a9ebf9e42a25a89aaab48ef2ae6918640e7b5e18ddcfcda5675d9a6`。
prepare01 SHA `0ad44a85d15a14ede21f1dcc79fdf041f549aa27e5a31e7885c9f1057f5e7e4f`。
生产审计CLI不内置actual路径或SHA：四项都须外部显式提供，防止将pending/new outcome默认为当前证据。
tests对上述已给定actual作只读fixture核验，所有故障篡改仅在内存中；无pytest tmp。
source/model/goal/raw source/parent/witness bindings先hash，算术后再hash。
源码/tests/doc哈希写入新审计报告；输出exclusive、≤128KiB、保留200MiB floor。

## 独立检查

独立生成zero＋(.5,1.5)×8方向的17项bank，方向顺序E/NE/N/NW/W/SW/S/SE，
单位方向在五时刻重复，除sqrt(5)控制整packet范数。float32若超名义半径则按冻结算法乘
`float32(1-4*eps32)`向内舍入；不是事后clip或改scale。
逐字段重建102个plan、原prefix/raw tanh/selection SHA；保持same root/task/Q/目标/动作尺度。
角色从原saved 16个step5 model-Q/64个H2 winner重算，不从物理结果重选。
本dense是在sparse结果之后的新诊断，非独立confirmation；大半径同时改幅度和时序支持，不称纯消融。

逐分支检查所有实际已尝试/返回的命令与原native/object指标。完整行复用独立纯数据fullH2校验；
失败行单独验证有界trace、原动作、native终止、guard first absence与不继续dispatch、partial endpoint算术。
102计划分母不变。只要row与本组zero参考都status completed且full_horizon，才形成fullH2配对响应；
其他endpoint留在partial层。即便返回了10步，末帧guard失败也不能冒充完成。
所有到达step5且留有snapshot的行（含partial）比较state7/RGB/两body动力学强等；
缺snapshot明确列证据缺口，不补造，不认证solver caches。

journal流式验证index/previous/event SHA，白名单事件与逐分支操作语法，所有attempt/return/global
counter、step receipt、reset recipe/placement options、fresh RGB、snapshot、guard、close逐项交叉检查。
102个row的step计数必须与global计数相等。完整run才要求204/1020/1122满额；partial按实际事件复算，
缺durable事件时独立列出exact-accounting缺口而非伪造完整账。真实本次14,056事件完整可读，
204 reset/1010 step/1112 render均attempt=return，102 branch closed；不把未执行10步padding为满额。

每张已保存PNG核文件SHA、uint8 RGB SHA和guard颜色像素计数，初始RGB exact绑定原source。
306是上限，不是本次实际图数：实际302 PNG（两条失败仍留step10，四条缺step10）。
完整artifact集合才要求306PNG+3文件；partial保留可辨识的未引用资产并报告，不删除。
独立复算producer analysis全部102项；新报告保存六组full/partial分母、object位移/角度描述与响应digest。
原native指标与object-only保持分开，没有物理赢家选取或P1准入。

## 实际partial与RGB失败的证据分层

MAIN top status为`partial_or_failed`、top errors=[]；失败仍在各branch的first_error中，不能据top errors=[]说全通过。
完整行96；root0 greedy的probe4/5/6/12/13/14为guard失败，该组11/17完整，其余五组17/17。
全部102行都保留step5 snapshot，所以六个失败行也参加prefix等同性核验。

| root0 greedy probe | 最后返回step | 实际agent x | 失败帧PNG保留 |
| --- | ---: | ---: | --- |
| 4 | 10 | -18.632742 | 是 |
| 5 | 9 | -21.831789 | 否 |
| 6 | 10 | -18.632742 | 是 |
| 12 | 7 | -19.526632 | 否 |
| 13 | 7 | -29.641871 | 否 |
| 14 | 7 | -19.526632 | 否 |

六条末态都可由state证明agent中心越过左画布边界；原native默认circle半径15（scale40×.375）下，
这些末态甚至满足x+15<0。这是源码默认几何的推断，不是重建几何/运行物理所得的新测量。
两张失败step10 PNG可以直接验证RoyalBlue像素为0；其余四帧像素未保存，visual failure mode应写unknown。
state支持边界/out-of-view解释，但不假称已从缺失RGB排除遮挡/颜色改变，也不宣称唯一因果分类。
只记录guard first missing frame和该步后停止；没有放宽guard，不补跑失败分支。

96条完整行提供的object响应反例不能因partial top status被抹去：root0 greedy/two-stage的dense
fullH2配对object位移max约33.6284/31.4723，wrapped角位移max约.1874/.3029；root1 selected仍0。
这些是固定bank内的描述性最大响应，不是选出并部署的控制策略，也不说明原任务成功/全局可控。
稀疏探针零响应不能据此外推为物体不可驱动；反过来dense有响应也不证明目标改善、准入或泛化。

## 运行

```bash
cd /home/gbk/code/LeWorldModel/le-wm-main
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -B -m pytest -q -p no:cacheprovider tests/test_p1_dense_suffix_saved_audit_20261027_v1.py
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -B -m experiments.spectral_reachability.p1_dense_suffix_saved_audit_20261027_v1 --prepared results/p1_dense_suffix_physical_main_20261027_v1_prepare01.json --prepare-sha 0ad44a85d15a14ede21f1dcc79fdf041f549aa27e5a31e7885c9f1057f5e7e4f --report results/p1_dense_suffix_physical_main_20261027_v1_main01/report.json --report-sha 76fb65359a9ebf9e42a25a89aaab48ef2ae6918640e7b5e18ddcfcda5675d9a6 --output results/p1_dense_suffix_saved_audit_20261027_v1_main01.json
```

MAIN复验必须换新output名称，不能覆盖报告。文件级错误/identity或算法合同漂移严格拒绝；
可解释的真实partial则保留全分母并给`...partial_audited_not_admitted`，不拒整个run、不冒充全通过。
