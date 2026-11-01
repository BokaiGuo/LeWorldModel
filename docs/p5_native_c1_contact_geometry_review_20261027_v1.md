# Native C1 三臂接触几何只读复核

日期：2026-10-27。仅新建本文；不改源码、总账、seal 或实际产物。不运行神经、AD、
环境构造或 physics；仅 stdlib/NumPy、保存 JSON 与多边形算术。使用 data-analysis
的数据完整性、数值公式、逐表及跨表一致性复核；不对单个已知根添加显著性检验。

## 结论与因果边界

三臂 suffix 的物体终态相同并非“整个轨迹从未接触”。共同 prefix 的 control1/2/3
有原生接触回调，物体发生约0.228090 px的原点位移及−0.049953 rad转动；从frame4
起物体原点/角度逐位不变。决策frame10时，agent圆盘到实际T碰撞几何的净间隙
为 **8.450732336 px**，已经脱离接触。

后三臂五个control的接触计数全部为零。保存帧与原生PD式的确定性算术重建一致：
compiled/nominal/factual_reference的50个suffix子步最小净间隙分别为
**7.095282560 / 7.404601575 / 3.152982266 px**，均没有碰到T。
compiled先略接近，随后离开；nominal主要消散既有agent速度；factual虽一度更近，
也没有真正到达接触面。这足以解释本次无物体推动和相同终态，不能推出理论失败。

未发现本次已执行链的控制单位、动作映射或接触过滤协议错误。当前编译器优化的是
混合场景patch目标，不包含“先建立接触”的硬约束；局部模型goal-MSE下降不是接触
或物体进展保证。这是现有合同边界，不在结果后追加新成功门、更改目标/阻尼/动作，
也不以此挑新的有利根。本文没有检验模型JVP对真实几何距离的导数。

## 输入、身份与完整性

固定输入：
`le-wm-main/results/p5_native_c1_executor_20261027_v2/native_main_v1/`，
compiled、nominal、factual_reference三臂各frame00..15，共48帧及45条command。
frame00是post-reset，control01把frame00推进到01；不是15帧漏掉初始状态。

本次核对：

- top report绑定ready manifest的文件SHA；三臂status=completed、close_receipt.status=completed，prefix gates=0..10。
- 三条journal文件SHA、sequence、previous_sha256和每条entry_sha256；每臂16条assets_durable记录。
- 48帧×5文件（RGB/state/proprio/full_public_state/transition）共240个实际文件SHA与journal一致。
- 三臂frame0..10五类文件SHA均与ready manifest的prefix_identity一致。
- 45条实际command逐项等于冻结branch的prefix+suffix；agent_xy_before、100*u、absolute_target和pre_public_sha256逐项核对一致。
- 三臂各保存帧的agent/物体shape均为非sensor、collision_type=0、group=0、categories=mask=4294967295，没有关闭agent—物体碰撞过滤。

这些是保存内容完整性检查，不是本次执行认证或重新封存全部软件依赖。没有读取
其他新根/outcome来选择方法。三臂只算一个固定根，不把身份重放当独立样本。

## 几何定义：实际T的两块多边形，不是中心距离

使用public-state的float64 body.position、angle和shape.get_vertices导出的局部顶点。
agent是radius=15的kinematic Circle、offset=[0,0]；block为dynamic，两个Poly的
radius均为0，局部几何如下（原生add_tee实际固定scale=30）：

- 长柄：x∈[−15,15]，y∈[30,120]。
- 横梁：x∈[−60,60]，y∈[0,30]。

顶点世界变换为`world=local @ R(angle).T + body.position`，与native
`pymunk_to_shapely`所用body.local_to_world语义一致。body.position是局部原点的
世界位置；不能再减一次center_of_gravity=[0,45]，也不能把它当T的几何中心。
frame10世界CoG约[258.623678,127.546239]，与body.position不同。

对每个凸Poly，计算agent中心到各边线段投影的最短距离，并用有向叉积判断中心
是否在内部；中心在内取负号。T的circle间隙取两个Poly的最小有符号距离，再减15。
本次中心均在Poly外；初始负间隙表示圆与边界轻微重叠，不表示中心位于T内。
最近点属于长柄远端，而不是横梁；frame10长柄最近点为
[313.496391,180.828369]，长柄净间隙8.450732336，横梁净间隙82.039167301。
以物体中心或AABB距离替代T边界，会给出错误接触判断。

## 完整15-control轨迹：保存帧间隙与原生接触累计

单位px。表内contacts是public `env.n_contact_points`，并非只有终帧的触点数。
三臂frame0..10逐字节相同。后五帧全部contacts=0。

| frame | compiled净间隙 | nominal净间隙 | factual净间隙 | contacts（三臂） |
| ---: | ---: | ---: | ---: | ---: |
| 0 | −1.636094 | −1.636094 | −1.636094 | 1（reset子步） |
| 1 | −1.002716 | −1.002716 | −1.002716 | 10 |
| 2 | −0.720479 | −0.720479 | −0.720479 | 10 |
| 3 | 1.933605 | 1.933605 | 1.933605 | 4 |
| 4 | 2.678602 | 2.678602 | 2.678602 | 0 |
| 5 | 1.004628 | 1.004628 | 1.004628 | 0 |
| 6 | 3.942737 | 3.942737 | 3.942737 | 0 |
| 7 | 8.542908 | 8.542908 | 8.542908 | 0 |
| 8 | 8.604484 | 8.604484 | 8.604484 | 0 |
| 9 | 10.332099 | 10.332099 | 10.332099 | 0 |
| 10 | 8.450732 | 8.450732 | 8.450732 | 0 |
| 11 | 7.098801 | 7.406878 | 3.611756 | 0 |
| 12 | 7.586716 | 7.413691 | 4.875843 | 0 |
| 13 | 9.034141 | 7.413646 | 5.551360 | 0 |
| 14 | 10.982102 | 7.413647 | 4.062537 | 0 |
| 15 | 13.028521 | 7.413647 | 6.895483 | 0 |

native `_handle_collision`在post_solve中累加contact_point_set.points长度；step开始
把n_contact_points清零，十个Space.step累计，info.n_contacts=ceil(total/10)。
control1/2/3的info.n_contacts因此均为1；4..15均为0。回调匹配collision_type0/0，
也可能包含其他同类型形状，不是专门标记agent—block的接触对日志。不能仅凭非零
计数断言确切碰撞对或力大小；本例初始重叠、物体变化与计数同时出现，支持早期
接触解释。suffix零计数加正几何间隙则不支持“接触了但未推动”的归因。

物体frame0→1/2/3原点位移模长分别0.056354277/0.200051402/0.228089769 px；
转角分别−0.015026336/−0.044495254/−0.049952703 rad。frame3..15物体位置/角度
逐位一致，后续保存的velocity/angular_velocity均为0；不是state.npy float32量化
掩盖了float64 public-state内的细小运动。frame10/15物体原点
[220.81840526099847,103.13801555577723]，angle=−13.563869813600146 rad。
angle未wrap；比较采用原生float64值，不将mod2pi后的float32当原始角度。

## 控制单位与“目标点碰到”不等于实际碰到

当前native合同为relative=True、action_scale=100、sim_hz=100、control_hz=10、
k_p=100、k_v=20。每个control的目标为dispatch前agent位置`p0+100*u`，随后
**固定该目标做10个dt=.01子步**。u不是速度，也不是直接位移命令，不能认为一次
control实际就移动100*u，更不能额外除100。`None`会跳过step的子步，零u则仍做
10个PD/physics子步，二者不相同；本次nominal是五个显式[0,0]。

冻结compiled五条u如下，括号不是新动作提议：

| control | u_x | u_y |
| ---: | ---: | ---: |
| 11 | 0.074194421835 | 0.004488597797 |
| 12 | 0.061738033060 | 0.022857507078 |
| 13 | 0.040149643438 | 0.039690232716 |
| 14 | 0.027666843752 | 0.047993912724 |
| 15 | 0.017617270883 | 0.048549222019 |

compiled control11的100*u=[7.419442184,0.448859780]，固定target为
[314.696322009,203.888166984]；实际该control位移仅
[2.873769638,−0.766877200]。初始agent velocity=[22.719971204,−30.054724329]，
惯性/PD制动影响实际轨迹；不能只用命令方向判断首段实际位移。

factual control11的u=[.05,−.12]，target相对T的圆盘净间隙为−4.319212645 px，
但真实frame11间隙仍3.611755798 px；其目标朝向确实可以指向物体，0.1秒的实际
控制尚未到达。control12就切换新相对target，没有把上一target继续保持到接触。
因此“目标点伸入物体，所以一定发生了接触”是具体解释错误，不是执行器单位错误。

nominal control11虽u=0，实际agent仍位移[.680049917,−.899592372]，之后才趋稳；
它是当前位置PD制动，不是静态freeze。suffix累计agent位移分别：

| arm | Δagent_x | Δagent_y | frame15净间隙 |
| --- | ---: | ---: | ---: |
| compiled | 9.489709341 | 5.226155510 | 13.028520998 |
| nominal | 0.675623011 | −0.893736315 | 7.413646632 |
| factual_reference | 3.379239360 | −0.900456646 | 6.895482553 |

native `_setup`沿用space.damping=0、gravity=0；本次没有覆盖damping。这是冻结
原生动力学的一部分，不是本review提出的调参项。agent是kinematic，由PD显式更新，
block为dynamic且已无速度、无接触；不能期待agent隔空移动使物体自行趋近目标。

## 子步算术交叉核对及其限制

不导入pymunk，不调用Space.step。对每个保存control前的float64 agent p,v及实际
target，仅按原生循环的kinematic更新做确定性代数展开：

```python
target = p_before + 100 * saved_u
for substep in range(10):
    v = v + (100 * (target - p) - 20 * v) * .01
    p = p + v * .01
```

对三臂45个control，重建后的p和v与下一public-state的最大绝对差均为0（本机
float64 NumPy运算）。这是保存轨迹的算术一致性检查，不是新physics实验。suffix
物体几何固定，因此可在50个重建子步上用上述polygon算术计算gap：

| arm | 最小重建子步gap | 出现位置（control,子步1..10） |
| --- | ---: | --- |
| compiled | 7.095282560 | (12,1) |
| nominal | 7.404601575 | (11,9) |
| factual_reference | 3.152982266 | (12,3) |

这些不是新保存的substep观测，不据其恢复未记录的arbiter/private contact cache，
也不证明任意连续时间swept path无碰撞。此处离散step判定与control内累计post_solve
零记录相互支持。不能跨模型把agent可解析运动等同block碰撞动力学已被模型学会。

## coverage为何始终为零

用相同局部两矩形分别变换到实际block pose及官方goal=[256,256,pi/4]，逐对做
凸多边形裁剪交集、shoelace面积求和。两块Poly仅共享边，面积相加不重复。
48个保存帧复算coverage均精确为0；45条transition.info.final_coverage亦均为0。

frame10以后实际T的世界AABB为
[188.274107335,52.730985565]至[329.768539725,180.828368905]；
goal T为[160.540584540,213.573593129]至[298.426406871,351.459415460]。
y方向已有32.745224224 px分离，故交集为零，不是coverage阈值或union计算把小幅
成功抹掉。AABB仅在此作零交集的充分证据，前文接触距离仍使用实际多边形。

## 可复核的最小几何公式

下列核心计算只依赖NumPy；输入必须来自本次固定48帧，不筛帧。对凸多边形外的
agent中心，本例每一帧都满足外部判定：

```python
def rotation(a):
    return np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])

def circle_polygon_gap(p, vertices, radius=15.):
    edge = np.roll(vertices, -1, axis=0) - vertices
    t = np.clip(np.sum((p-vertices)*edge, axis=1) /
                np.sum(edge*edge, axis=1), 0., 1.)
    nearest = vertices + t[:, None]*edge
    d = np.linalg.norm(nearest-p, axis=1).min()
    cross = edge[:, 0]*(p-vertices)[:, 1] - edge[:, 1]*(p-vertices)[:, 0]
    inside = np.all(cross >= -1e-10) or np.all(cross <= 1e-10)
    return (-d if inside else d) - radius

# For every frame: min over its TWO actual block Poly shapes.
world = local_vertices @ rotation(block_angle).T + block_position
```

有限数据的结果都是确定性描述，不报独立样本CI。本次未找到可据此要求修正sealed
执行器的具体协议错误；发现的是当前单次短后缀没有建立真实接触，以及从模型图像
改善跳到物体进展的推论缺口。不建议依据这些outcome修改步长、保持时长、目标、
初始root或选择新的有利样本；任何后续协议须独立版本、事前声明并由main授权。

## 固定证据SHA

路径相对`le-wm-main/`：

- `results/p5_native_c1_executor_20261027_v2/native_main_v1/report.json`：`e2f8245f8f69e41eeb35ec5507d8e744f7f32e1e8f76948756927ddf42c44ae6`。
- `results/p5_native_c1_executor_20261027_v2/manifest_ready_frozen.json`：`1099ee3b001ddb9b88fad4d9c31e1fac39fef07b1120501ccee44a907e6c95b3`。
- `external/jepa-wms/evals/simu_env_planning/envs/pusht_env/pusht_env.py`：`6ba34790b6c2ce017a76322c2f1c205ef97d4d3b08a620f844a57075c608a723`。

240个frame文件SHA另逐个对上三条journal。按相对项目根的路径字典序排列，拼接
`relative_path + ':' + file_sha256`并以换行连接（末尾无换行），其集合digest为
`e18b92e598546b4cd75231912e62f0e2c0501d5075b7dd6f26ca9d7c6113fbb5`。
无新输出报告/新代码，仅本文；未修改任何既有结果、模型或总账，P5/C2不由本review开放。
