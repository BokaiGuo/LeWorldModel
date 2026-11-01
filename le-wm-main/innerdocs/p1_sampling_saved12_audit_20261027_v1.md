# sampling实际12帧：原latent/同Q/物理目标独立保存复算

本交付仅新增同名审计module、tests、本文及小JSON报告，不改任何sealed输入或先前ontology文档。只读JSON/原始latent/PNG和纯NumPy算术；没有模型加载/神经计算、CUDA、physics、下载、prepare或tmp目录。旧ontology文档中“sampling12RGB待CUDA”是当时快照；MAIN现已完成这一槽位，本页记录新增实际证据，不追改旧snapshot。

## 审计对象与复算合同

真实目录 `results/p1_sampling_rgb_cuda_20261027_v1_cuda_main01`（本页inner路径均相对 `/home/gbk/code/LeWorldModel/le-wm-main`）。report SHA `ae0fe77b628c35a129bd6035ef59771876fe4791bd12774f1e9123e30a6ef34b`；其绑定analysis SHA `53fb7110e1a2ad8e2c1db4892dbfe22dcc50fefe5dcb73c15a79ab18d6362e53`、journal SHA `54e7f645502bc0f1962e7dbf109b2e2ed289c07f3fc1036ad73bd1bf8b639be9`。独立重算三个文件SHA后再读数组。小prepare `results/p1_sampling_rgb_cuda_20261027_v1_prepared01.json` SHA `485040cd82418b1f395de9dcc578611c78d152e89f146094c6261afeb966955a`。

冻结producer的纯verify重建source/sampling/physical/PNG/model manifest；不调用其analyze/run。另独立按每帧5条事件核对全部root/arm/step/pixel/frame-order，encoder/projector各1次attempt及whole-encode完成收据，两个durable raw192副本逐数组相同。拒绝少帧、额外调用、跨帧scope、错task、pixel或latent拷贝。完成语义保留原合同：encoder/projector completed同时来自整体encode完成收据，不假装具有任意失败路径下逐module-return的精确计数。

由原task读取Q/goal，二次型独立复算 `e.T@e/192`，不调用producer.model_cost：predicted Q、actual Q及native prediction MSE全部匹配analysis，最大绝对差 `4.440892098500626e-16`。按 `(root_sha256,arm,step)` join分析和物理checkpoint，另解码12张224RGB PNG、核文件及pixel SHA。原task_id/goal_Q_context摘要、原选取prediction和实际physical selection均绑定；没有新Q、重选、按结果换root或把预测冒充actual latent。

完整分母2root×2arm×3checkpoint：encoder12/projector12，0predict/AD/新physics，model前后hash相同。每root两臂initial pixels与latent逐位完全一致，最大差0；重复样本依然各付编码，不能增加独立root数量。对照旧four-arm的Q/goal/model/normalizer/root context逐数组一致；旧24帧对照值明确继承原已封存saved24审计，不谎称本轮重新编码旧24帧。

## 实际结果：模型误差、目标距离、native reward与success分列

|root/arm|predicted Q MSE|actual RGB Q MSE|native state distance|object pose cost|
|---|---:|---:|---:|---:|
|0 CEM|1.1711415692|1.2216672577|248.811838|69.520206|
|0 IID|1.1821036195|1.2290427129|202.466196|69.520206|
|1 CEM|1.9014355004|1.9878753069|380.016748|49.341614|
|1 IID|1.8618007308|1.9401233446|525.087763|49.341614|

同根CEM/IID的predicted/actual Q排序均一致；与native连续reward排序均相反。root0 CEM Q更低但native距离比IID大46.345642；root1 IID Q更低但native距离比CEM大145.071014。不能将每次排序差都叫模型预测错误。

相对原two-stage：root0 CEM/IID真实Q分别改善0.036338096/0.028962640；root1分别恶化0.262109703/0.214357740。这支持两根不同结论，不支持“谱编译普遍优于采样”。forward126匹配不等于AD/SVD/solve、总时间或全部算力匹配。

**native reward含速度，success没有速度条件。** 原installed evaluator：

```text
native_reward = -norm(state7-goal7)
distance² = front4_position_error² + raw_angle_error² + agent_velocity_error²
native_success = (front4_position_error <20) AND (native_angle_term <π/9)
```

前四位置包含agent和object；success不要求速度归零。独立wrapped角仅诊断，不静默覆盖native-stop或改其阈值。四终态native与wrapped success均false；此前40步独立物理审计也未出现负native角或native/wrapped成功分歧。不能把native速度惩罚解释为额外success门。

root0 CEM的速度误差平方36968.233027，IID13640.775608；CEM位置平方反而更小（24928.434028 vs27341.121168）。速度项是这对native距离排序反转的明确分量。root1 IID的速度平方81768.368744 vsCEM3891.781691，但IID的front4位置误差也更大（440.373175 vs374.834218）。所以**并非消除速度项就能普遍修复Q与几何目标的排序**。例如root1 CEM相对原zero，actual Q从2.0772934423降至1.9878753069，front4误差却从319.276476增至374.834218，存在独立于速度项的几何错配。

两根采样终态的object pose均与原zero相同，0/4成功，不凭末态判断整轨迹接触情况。RGB-Q下降既非success certificate，也非真实能力改变证明。旧ontology对单帧RGB未显式携带速度的接口观察仍成立，但不意味着RGB不能判定位置/姿态success，也不把速度缺口称为全部Q失配的唯一原因。

B8 direct已实际完成、GD持续改善且root1模型cost1.716405低于two-stage1.720243，这一结论不受本12RGB补测改变。本报告只用原four-arm的实际物理/24RGB作继承对照；没有B8-GD实际物理端点，禁止拿旧B3-GD端点替代。独立新任务族的direct stall资格、constructive reachability和同prefix物理能力探针仍未建立，P1false；本次不是独立确认、理论负结果或阈值校准。

## 文件、测试、实际输出

新增：

- `experiments/spectral_reachability/p1_sampling_saved12_audit_20261027_v1.py`，SHA `d396e19212afc72153251e8da8d523cc0b573a40a678d521b89718c54aeddd28`。
- `tests/test_p1_sampling_saved12_audit_20261027_v1.py`，SHA `c1d9b074cd751529c434a687cf8cce130600785e8c4a6ee7695f4217f373870f`。
- 本新innerdoc。
- `results/p1_sampling_saved12_audit_20261027_v1_audit01.json`，11,831 bytes，SHA `a0c0b276eb6e25fdaf1f6f1c27829abba7c9e809cd37f2f6af181d9c13f25194`。

16 lightweight tests通过（0.47s），含真实saved数组、scope/count/raw副本篡改、合法非identity投影、身份乱序/重复、速度改变reward不改变success、多圈角native风险保留。非identity mock最初误用了不满足投影合同的2I，严格validator拒绝；修正为合法rank1正交投影，**没有放宽生产Q validator**。mock不作为效能证据；真实审计另已完整运行。

cwd `/home/gbk/code/LeWorldModel/le-wm-main`：

```bash
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -m pytest -q -p no:cacheprovider tests/test_p1_sampling_saved12_audit_20261027_v1.py
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -m experiments.spectral_reachability.p1_sampling_saved12_audit_20261027_v1 --output results/p1_sampling_saved12_audit_20261027_v1_audit02.json
```

第二条复跑必须新filename；首次实际输出为audit01，不覆盖。report≤100,000 bytes并保留200MiB；新source/tests/doc/report合计低于150KB，没有pytest tmp fixture/cache或pyc，未删除任何文件，未触碰共享进度文档。证据边界是保存收据与原始数组的独立算术验证，不是独立模型重编码。
