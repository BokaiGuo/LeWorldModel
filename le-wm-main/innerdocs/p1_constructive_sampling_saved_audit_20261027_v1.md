# Constructive sampling：独立saved算术审计 v1

只新增本stem源码/测试/本说明及独占小报告。不导入producer、optimize、consume，不加载torch/模型、sourcepack、witness、blind NPZ、PNG或物理结果。现有独立saved-audit utils仅提供JSON/hash/数组合同、float32 packet球投影、action history与Q成本算术。producer源码仅读字节核SHA，不执行。

## 外部真实证据

- report：`results/p1_constructive_sampling_cuda_20261027_v1_cuda_main01/report.json`
- report SHA：`0c2db96dc506a7a2d71e4dcd43f5990fe0cc8ad0ad27f8483388cecdf47b9698`
- prepare SHA：`4c209a1a2fb1b6c9a9e3067b21b5bc3dd6e3364234cada2278b6071d6668b4ce`
- producer SHA：`5fe2af6256071d4a056a9f40908ee616c284e7b72e61a5c1fc5a79e359b86862`

审计严格核prepare/report、110项source/export/model字节绑定、完整产物清单与每项hash、固定2slot×2arm的taskID/blindSHA join；不以任意数组行位置冒充root身份。数据入口只允许两份固定approved latent export，拒绝跟随其他results/NPZ/certificate绑定。

## 独立复算范围

显式PCG64 + standard_normal后affine变换重建随机流，不调用producer proposals/rng.normal/elite_update。逐候选核 `[2,10]` packet-ball cap .75sqrt5、float32投影与tanh `[10,2]` rawcommands；CEM逐轮16槽cost排序取top4、projected logits population std与[.05,1]约束，重算全部8次更新。

逐event严格语法核bank先冻结、每candidate request→两段实际action_encoder/predict attempt/return→native_packet→evaluation_complete→candidate_saved；重建每段float32 normalized history与latent移位，从保存terminal按原identity192 Q重算cost。未知/缺失/重复/越序event都拒绝，不只统计event数量。

全部64cost完成后全局稳定argmin(cost,index)，winner seal在三重replay之前。独立核nominal/whole/segmented的实际六段context/terminal/cost和保存final_prediction；plan/row/selection SHA/rawcommands交叉绑定。资源逐次index/return复核134各/arm/root，总536predict与536action_encoder；候选128加winner6，不漏记replay。零AD仅为预声明合同和journal无AD事件，不是外部GPU profiler证明。

## 实际结果（模型cost，不是物理效果）

| source_slot | arm | seed | winner index | 原Q模型cost |
|---|---|---:|---:|---:|
| 0 | random_shooting | 20261027 | 2 | 0.35086892138499165 |
| 0 | CEM | 20261028 | 53 | 0.38684771044486177 |
| 1 | random_shooting | 20261128 | 33 | 0.39574330879212155 |
| 1 | CEM | 20261129 | 52 | 0.1308949204487526 |

共256candidate、268完整H2 evaluations、536逐packet transition。保存数组与本预声明协议一致，未发现本范围内RNG/拼接/normalizer/history/候选选择/记账错误。此结论不证明模型forward数值真实或预测正确，不保证CEM强于random，不涉及正在执行的物理结果。无独立确认/P1准入或等D2P总算力声明。

## 命令与产物

工作目录 `/home/gbk/code/LeWorldModel/le-wm-main`：

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m pytest -q -p no:cacheprovider tests/test_p1_constructive_sampling_saved_audit_20261027_v1.py
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m experiments.spectral_reachability.p1_constructive_sampling_saved_audit_20261027_v1
```

不带output仅内存复算/打印。首次生成独占小报告可加：

```bash
 --output /home/gbk/code/LeWorldModel/le-wm-main/results/p1_constructive_sampling_saved_audit_20261027_v1_main01.json
```

报告≤150KiB且200MiB floor；存在即拒绝、不覆盖。测试仅实际saved数组与内存篡改，无tmp fixture、模型/物理/网络；不从mock结果推断真实效能。后续真实物理/端点编码绑定与预测—真实排序属另任务，本审计不读取它们。

实际51 tests通过（完整真实saved审计、四分支核算及内存篡改拒绝）。报告还记录审计器自身和两个独立utils源码SHA，并在复算前后核不变。独占交付报告为 `results/p1_constructive_sampling_saved_audit_20261027_v1_main01.json`；无需把已有CUDA/物理任务重新运行。
