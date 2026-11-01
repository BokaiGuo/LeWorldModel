# Same-prefix有限探针：接触机制与目标度量诊断

## 结论及证据边界

当前最直接的解释不是“40个动作没有执行”，而是：**selected prefixes之后，已测后缀没有重建物体接触；
agent仍对每个probe作出正确坐标/符号的运动响应，object在所有已存public-step样本中保持静止。**
这与prefix将agent带离物体、随后只测raw-zero附近的小幅单时刻单坐标脉冲相一致。
它是这两根、这些prefix和这些有限探针的机制证据，不是完整动作域不可控、谱方法必然失败或根因已唯一识别。
不改旧producer/协议/Q，不执行任何神经或物理，不按结果选新尺度、阈值、根或动作。

实际输入是`results/p1_same_prefix_physical_main_20261027_v1_main01/report.json`，
SHA `d975ef03f1b7fa540b449a403fec2fa83cf5eb9a89d4dcedbaf6555efe32c4fe`。
既有独立audit SHA `dcc18b35b5a4327d264a2c14a3d78edd88b3b021e6356cfa36177c3e92670f66`。
其40 tests、246完整分支、34,196事件及738 PNG校验不替代本次机制分析。
本新脚本绑定该audit及其source hashes，再复核saved tasks/plans/每行轨迹；仅新增描述性算术。

## 1. Object-only与接触几何的实际量化

root0=`02a139…0d06`；root1=`05844c…89a7`。距离/位移使用原生512画布坐标，角度rad。
下表的object最大位移/旋转是suffix 5步全部public样本相对step5的最大值，包含zero参考；
各组endpoint相对自身zero-suffix的最大object响应在本actual中也等于相同数字。
不能拿不同role的zero-suffix终态相互减去后当作单一后缀响应。

| root / role(bank index) | step5 agent–object中心距 | suffix样本中心距范围 | 最大object XY位移 | 最大object角位移 | native正contact receipt的suffix步数/205 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0 zero | 31.451 | 17.714–48.412 | 17.616811 | .078752 | 127 |
| 0 greedy(10) | 153.238 | 145.522–177.363 | 0 exact | 0 exact | 0 |
| 0 two-stage(3) | 86.013 | 75.116–112.741 | 0 exact | 0 exact | 0 |
| 1 zero | 75.429 | 73.760–93.291 | 20.701879 | .141427 | 36 |
| 1 greedy(11) | 194.410 | 180.958–218.224 | 0 exact | 0 exact | 0 |
| 1 two-stage(11) | 194.410 | 180.958–218.224 | 0 exact | 0 exact | 0 |

四个selected-role组共820个suffix public-step receipt的`n_contacts`全部0；其step5及step10物体
线速度/角速度均0。root0的prefix本身并非没有作用：greedy已移动object 2.038465、旋转+.023415；
two-stage已移动7.536122、旋转+.047099；这些发生在step0→5，不能错误归到后缀或抹去。

`env.py:312`每个public step清零contact accumulator，`_handle_collision`累加post-solve contact points；
但`_get_info`的分母是`int(1/dt*control_hz)=1000`，而真实每step substeps为`int(1/(dt*control_hz))=10`。
输出是ceil后的原生receipt，**不是精确接触点数/冲量，也没有agent–object pair标签**；本文仅用“是否正值”。
全0与无已记录碰撞相符，不能从正值宣称一定是物体接触。中心距也不是T形表面gap；
本分析没有构造Pymunk几何、没有声称全substep间连续最短间隙已复算。
原生`space.damping=0`亦是既有环境设定，不是本consumer新增；不能凭变量名把它读成无阻尼，
更不能在本诊断中擅自更改。准确的持续时间/接触动力学归因仍需要另行授权实验。

## 2. 构造/坐标/投影/重中心化逐项排查

封存producer `p1_same_prefix_physical_main_20261027_v1.py:25`定义
zero + 10坐标×2个绝对半径(.25,.5)×±，共41；坐标按5×(x,y) time-major展开。
`:78–82`将float32 logits经NumPy tanh后直接拼接原prefix raw commands，球只作合法性断言，
不调用投影/缩放/clip。cap是`.75*sqrt(5)=1.677050983`，因此两半径只占cap约14.907%/29.814%，
**不是**cap的.25/.5倍，更没有覆盖整球。single-coordinate raw幅度约.244919/.462117。

`env.py:318–327`的动作含义是当前agent位置加`100*raw`成为该public step的PD目标，
不是绝对xy、世界坐标位置或object-relative移动。期望目标偏移约24.492/46.212，
不等于实际位移；PD为kp100/kv20，每次10个dt=.01 substeps。每probe只有一个时刻一个轴非零，
其余4步raw零；raw零是当前位置目标并制动来速，不是reset速度、固定step5全状态或持续向物体推进。
selected suffix期间agent相对step5最大采样位移约24.11–27.69；单坐标probe终态相对自身zero参考
的同轴有符号响应为7.24154–18.83873，全部40/40为正，异轴响应max=0。
所有组在预定probe时刻之前的agent state都与zero参考相同。
因此saved证据不支持x/y颠倒、符号丢失、步序错位、全零dispatch或统一projection把probe压掉。

物理执行使用原raw，不应再套模型action normalizer。模型中的`(raw-mean)/std`仅形成predictor token；
其均值/标准差约[-.0087,.0068]/[.2019,.2002]，source任务绑定保持不变。

**确有覆盖局限，而不是发现了实现偏离：**后缀以raw-zero为logit中心，不以原winner suffix为中心，
也不将动作朝物体重定向。原two-stage Jacobian本来也是在各自预测prefix context的zero suffix重编译；
这次物理测量继承其zero-center解释，但实际prefix状态可能已失去接触。
同prefix exact state7/RGB/两body动力学检查排除了已记录前态差异；并不序列化solver warm-start caches。
稀疏单坐标短脉冲、未测多坐标组合/持续多步/全cap/其他中心，可能使测试无法跨越重新接触所需轨迹。
**尚未实际执行那些反事实，不能断言它们一定接触或改善。**

## 3. Roles重复与有效分母

root0 greedy index10与two-stage index3不同；root1两者都是index11。
root1对应41对commands、完整actual trace和三checkpoint动力学均逐值相同。
故246计划分支必须完整保留，但只有205个不同的(root,10-command序列)、5个不同(root,prefix)组，
不是6个独立prefix，也不是205个独立任务；新独立roots仍是0、原calibration roots是2。
root0两个selected prefixes虽都零后缀object响应，其物体基准位置/角度不同，不能说是同一物理状态。

## 4. Goal-Q包含agent：已知路径与尚未证明的因果归因

原生reset在`env.py:279–282`先`_set_state(goal_state)`再render `_goal`。
render会画agent与object；两根saved goal RGB中RoyalBlue agent像素分别41/42，
初始RGB分别40/47。agent像素centroid(224坐标)从(57.675,68.400)指向goal(38.268,172.829)，
以及从(120.021,142.000)指向goal(189.690,21.381)。因此goal RGB不是object-only目标。
`p1_conditional_direct_screen_20261027_v1.py:346`编码整张goal RGB；MAIN direct v2 `:190`
固定Q=identity192，原native projected CLS，没有agent mask或object-only readout。
identity192只是潜空间各坐标等权，不等于物理state7、逐像素RGB loss或agent/object独立通道。

| root / prefix | saved预测Q5 | 实际agent目标位置误差 step0→5 | 实际object目标位置误差 step0→5 | 实际object角误差 step0→5 |
| --- | ---: | ---: | ---: | ---: |
| 0 zero | 1.366917 | 242.044→242.044 | 9.167→9.167 | 3.017657→3.017657 |
| 0 greedy | 1.230223 | 242.044→197.085 | 9.167→10.974 | 3.017657→2.994242 |
| 0 two-stage | 1.270322 | 242.044→192.949 | 9.167→15.389 | 3.017657→2.970557 |
| 1 zero | 2.141044 | 318.983→318.983 | 13.691→13.691 | 1.782513→1.782513 |
| 1 greedy / two-stage | 1.876719 | 318.983→281.621 | 13.691→13.691 | 1.782513→1.782513 |

Q5是saved model预测，后三列是实际物理指标，不能等号互换。greedy只在原16个model Q5中稳定选最小；
two-stage继承原64个H2 model-Q赢家prefix，没有拿这次物理结果重选。
原native position SSE中agent份额：初态约99.857%/99.816%，selected step5约99.368%–99.764%。
这是**物理位置误差的分解，不是latent Q贡献比例**；native state7距离还混入速度，
例如root0 greedy/two-stage step5 native距离330.789/412.192，可在agent位置变好时反而增加。

已知：目标RGB包含agent、当前Q未显式消除agent、所选动作在真实轨迹中更接近agent目标且丢失后缀接触。
相容假说：优化器可能偏好agent/nuisance变化，未保持object可操纵接触。
未证明：latent Q改善由agent贡献多少、agent是唯一根因、改mask/Q必然改善控制。
没有无agent counterfactual encoding或独立目标消融，不应把相关性升级为因果结论。

## 5. 可执行复验与限制

新增纯saved分析脚本/test同stem；利用code-debugging的坐标/尺度/时序检查来排除实现假说，
不按技能的修复流程修改封存实现。报告只输出描述性数值、重复关系、输入SHA和限制，不生成新控制候选。

```bash
cd /home/gbk/code/LeWorldModel/le-wm-main
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -B -m pytest -q -p no:cacheprovider tests/test_p1_same_prefix_contact_diagnosis_20261027_v1.py
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -B -m experiments.spectral_reachability.p1_same_prefix_contact_diagnosis_20261027_v1 --output results/p1_same_prefix_contact_diagnosis_20261027_v1_main01.json
```

复验用全新output路径；输出≤64KiB、保留200MiB floor；tests无tmp、无神经/物理。
后续若授权覆盖扩展或目标消融，须另立预冻结协议和对照，不能把此saved物理标签送入planner，
也不能据此筛根/调阈值。P1/P2等科学准入均不由本诊断改变；有限零响应不支持全局不可控。
