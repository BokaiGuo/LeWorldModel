# Independent saved RGBv2 auditor

NEW独占source/test/doc，无producer import或核心调用。仅stdlib/NumPy/PIL，禁止神经/物理执行。
输入为外部显式report SHA；只读report/binding/prepared/events、native192 NPY、原PNG、两个批准task exports，以及父physical相应binding/prepare/manifest/hashchain。
不打开模型weights/config、sourcepack/source prepare、certificate或witness；不训练、不重编码、不新rollout、不选优。

## 独立复算与绑定

1. report完整terminal/errors[]/first_error=null，artifact map与实际目录精确census，2MiB含所有artifact；每文件SHA重新验证。
2. binding.json受report artifact SHA保护，prepare文件SHA必须同时等于report.prepare_sha256；inputs/contract精确匹配。
3. 两task外部SHA、gzip唯一完整流、task_id=默认spaces JSON hash、contract semantic SHA，source_slot0/1、blind aliases、native root身份为null、with_target=false。
4. root完整三历史/两normalized token、10raw zero动作、192原生维数、native float32 lossless数值、identity192 Q/no_fit、公共normalizer一致。
   task checkpoint元数据与RGB prepare bindings相等；仅比较metadata SHA，不加载或重新哈希模型权重。
5. 父physical外部SHA→execution binding→prepare→manifest；journal逐index/previous/SHA独立hashchain；operation attempt/completed逐嵌套签名匹配。
   全root×arm分母、完整10steps、literal selected/attempted/returned commands、branch_started/closed、实际计账均核对。
   独立root_state+构造goal float64字节hash与evaluator roots一致；task goalRGB SHA在所有物理returned step持续绑定。
   每图必须来自本branch真实prefix5/terminal10 checkpoint，returned state/snapshot/guard/PNG file/pixel SHA一致，不替代早停终点。
6. RGB固定root-major×arm×[prefix5,terminal10]；不去重，即使像素/latent重复也逐图完整计费。
   每图事件严格为image_attempt→encoder attempt/return→projector attempt/return→image_saved；对应row/provenance/receipt逐字匹配。
7. NPY按file/row/artifact三方SHA绑定；float32[192]、有限、无pickle/尾随bytes、canonical own filename；禁止无声float64转换或串行替代。
8. NPY native32只先无损提升float64，独立计算：

```text
actual_Q             = mean((actual_native192 - task_goal192)**2)
root_Q               = mean((task_root192 - task_goal192)**2)
actual_Q_change      = actual_Q - root_Q
actual_vs_root_mse    = mean((actual_native192 - task_root192)**2)
```

全部与report exact float相等，不用allclose；不以epsilon容忍、无新Q拟合。
predicted_Q/predicted_vs_real_mse/Q_improvement_sign_reversed必须null，prediction_status必须unknown_not_recomputed，provenance.predicted也必须null。
**unknown不是0，不是预测准确，也不能据此说排序反转。**

## 证据边界

model before/after tensor SHA相等仅核对producer已封存声明，不能以离线数组审计重新证明神经执行或model不变。
实际encoder/projector计数是对durable module receipts的重计，不是再次执行模型。完整输出没有RGB journal hashchain；其整文件SHA由外部report/ artifact map绑定，
不伪称producer没有记录的hashchain。父physical journal另有独立重建的hashchain。
父物理这里核绑定/分母/commands/计账/检查点，不代替完整native指标或source科学审计；sourcepack/证书路径声明不读取。
不重审compiler候选数学，也不认证P1/P2/plateau/任务机制/公平预算/科学效能；global freshness仍unknown calibration only，不增加样本数。

## MAIN命令及小型输出

cwd `/home/gbk/code/LeWorldModel/le-wm-main`：

```bash
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -B -m pytest -q -p no:cacheprovider tests/test_p1_constructive_selected_rgb_saved_audit_20261027_v2.py
.venv/bin/python -B -m experiments.spectral_reachability.p1_constructive_selected_rgb_saved_audit_20261027_v2 --report results/p1_constructive_observed_suffix_rgb_20261027_v2_cuda_main01/report.json --report-sha 2cc16ea94dc76640a09ddb6d3d3cf8c70f171d9423b3896702715d48132306b1 --output results/p1_constructive_selected_rgb_saved_audit_20261027_v2_feedback_independent01.json
.venv/bin/python -B -m experiments.spectral_reachability.p1_constructive_selected_rgb_saved_audit_20261027_v2 --report results/p1_constructive_matched_ridge_gn_rgb_20261027_v2_cuda_main01/report.json --report-sha f6dbf29177769e8ec0be9e5ea38b928d201812a5ea44c07fbcedd0ebd3ebabbe --output results/p1_constructive_selected_rgb_saved_audit_20261027_v2_matched_independent01.json
```

默认stdout；output为NEW xb独占、<=100KiB、200MiB floor、fsync文件+目录，禁止写入sealed RGB或父physical输出目录。
上述文件若已存在不可覆盖；MAIN复跑使用新名字。不创建tmp/prepare，不复制输入资产。
认证或语义不一致直接拒绝、CLI非零退出，不将partial包装为complete。

## 本轮actual saved结果

36项saved/RAM测试通过，覆盖两actual、Q/rootQ/MSE/unknown字段篡改、资源/事件/缺行/串臂、NPY dtype/尾随bytes、父physical绑定、task合同/Q/history。
两实际都得到saved_rgb_v2_bindings_metrics_verified/findings=[]。所有指标从原NPY+task数组重算一致，无新神经/物理。

|input|images|encoder/projector|父physical reset/public/physics|native success|
|---|---:|---|---|---:|
|feedback|8|8/8，各attempt=completed|8/80/816|0|
|matched_ridge_GN|4|4/4，各attempt=completed|4/40/408|0|

root_Q：root0=1.453853658668115；root1=1.4617977689897848。

|input/root/arm|actual_Q prefix5→terminal10|terminal actual_vs_root_mse|
|---|---|---:|
|feedback/0/GN|1.0826473758→0.8255423006|0.3829180573|
|feedback/0/D2P|0.9120425903→0.8754358977|0.3473302012|
|feedback/1/GN|1.4123117543→1.5379207428|0.1811039678|
|feedback/1/D2P|0.4896678650→0.4283274221|1.2440116969|
|matched/0/GN|1.3532602864→1.3422899182|0.4651152338|
|matched/1/GN|1.4777243988→1.4875178880|0.0113709148|

该表仅为保存指标，不是跨不等预算方法的效能结论；预测误差/符号反转均保持未知。
