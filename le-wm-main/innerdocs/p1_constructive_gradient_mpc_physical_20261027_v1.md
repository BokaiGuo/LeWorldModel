# gradient MPC → generic physical selection adapter

薄接线实现，不新增物理循环、模型或优化器，不写独立审计器。唯一命令权威是已封存 `p1_constructive_gradient_mpc_cuda_20261027_v1.consume`。实际 report SHA `8eadbc6068ee4949f280892442eda45298934ece87dd7ae38a44b4f93d78d728`，prepare SHA `83371ef03c1b18727d8632b6a69085f04638189d7991ecd7831fa3affb62ca8f`，编译器源码 SHA `2c874aa56cc0b7727dbc20fbcbe60c3c0265bd368a6f5e1d050301f095be5be6`。不读取其模型候选池自行重新选优，不把模型 Q 当物理成功。

固定两批准 opaque taskID/blindSHA/exportSHA/source_slot 对应，仅 `gradient_MPC` 一臂，每根完整10条 float32 raw commands；即使两根动作相同也不去重、不clip/round/pad。generic selection的 JSON Pointer直接指向本 actual report 的 `/rows/0|1/{commands,source_slot,arm,blind_sha256}`；row selection digest在进入 physical 前封存，并且每次prepare/run都重新调用原consume检查完整 ownplans。

## 原 source identity 与执行边界

评价器 sourcepack SHA `9ad3a414fd2148258524d4a7e9d4f070f59532ec2e5943cfa61f9145b3a41bd9`，其 prepare SHA `98ffb806018a270f5ef27ffa76be929393a74355c213bc274bf158c72d80b086`。原 evaluator root SHA固定为 slot0 `f7aec52327fa6a6c8fb444919ba426a467ec2382b57da6d8a4f3701f82c65962`，slot1 `f47e9958781f7eac4c810f9dfe24232b635f41d5b4c56423d8c2623f9fd5a328`；不把这些隐藏物理状态或证书路径传回模型/控制选择。

selection builder只调用原consume和generic的纯 JSON Pointer selection绑定，不调用sourcepack/certificate loader。MAIN运行 adapter prepare 时，原generic `material/load_source`才在 **offline evaluator** 侧读取原已批准观测state/RGB/goal字段，验证原blind/source identity；不读取witness commands，不生成候选。run仍由原generic source重建及execute完成，不能使用旧 physical report/endpoint替代。

两根各10warmup+10own commands：上限40 public steps，其中own20、warmup20，reset4、set_state8、goal_render4、explicit_render44、goal_bind2、physics_step408（这是原generic内部step计数上限，不是40额外步骤）。0predict/AD/image encoder/projector/training。wall600s，output8MiB，200MiB floor。保留原target-free RGB224、原goal attachment、state/RGB/dynamics重建、canvas/visibility/state/render guard、逐步durablePNG/state/journal；guard/terminal提前停都不填满、不复制终点，所有两分支census与失败保留，真实数量以generic report为准。

adapter prepare的新schema为本stem`.prepare`，在原generic prepare基础上添加adapter/已封存compiler closure bindings，**不修改generic源码/verify**。adapter自己的verify重建全部新prepare，run授权后直接调用原generic `execute`，而非调用只接受旧schema的generic CLI verify。generic执行report仍使用原generic `.report` schema，execution_binding指向本adapter prepare；这明确表示复用同一执行器而非新物理算法。

选择manifest与adapter prepare均为独占新文件（各<256KiB）；closure包括adapter源码/tests/doc、封存gradient closure、actual report/prepare、generic及原source closure。所有metadata由MAIN生成；实现者无GPU/physics/prepare/output抢跑。RAM tests包含mapping/篡改拒绝、授权先行、delegate不重算、纯数组generic40step与guardpartial；不构成真实物理或科学准入证据。

## MAIN 命令

cwd `/home/gbk/code/LeWorldModel/le-wm-main`：

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m pytest -q -p no:cacheprovider tests/test_p1_constructive_gradient_mpc_physical_20261027_v1.py
.venv/bin/python -B -m experiments.spectral_reachability.p1_constructive_gradient_mpc_physical_20261027_v1 selection --output results/p1_constructive_gradient_mpc_physical_20261027_v1_selection01.json
.venv/bin/python -B -m experiments.spectral_reachability.p1_constructive_gradient_mpc_physical_20261027_v1 prepare --manifest results/p1_constructive_gradient_mpc_physical_20261027_v1_selection01.json --manifest-sha <SELECTION_SHA> --output results/p1_constructive_gradient_mpc_physical_20261027_v1_prepare01.json
.venv/bin/python -B -m experiments.spectral_reachability.p1_constructive_gradient_mpc_physical_20261027_v1 verify --prepared results/p1_constructive_gradient_mpc_physical_20261027_v1_prepare01.json --prepare-sha <PREPARE_SHA>
.venv/bin/python -B -m experiments.spectral_reachability.p1_constructive_gradient_mpc_physical_20261027_v1 run --prepared results/p1_constructive_gradient_mpc_physical_20261027_v1_prepare01.json --prepare-sha <PREPARE_SHA> --output results/p1_constructive_gradient_mpc_physical_20261027_v1_main01 --authorize-physics
```

没有本任务的CUDA命令。实际物理效果仍未执行，不预写success；只支持两既有calibration来源，P1_admission保持false。
