# 固定分散episode的offline fidelity阶段A v2

本NEW版本独占同名源码、tests与本文，不改v1、已封存结果或共享文档。实现阶段A，不实现paired render B。
本次仅RAM mock测试及本地H5指定slice纯读取；不prepare、不运行CUDA/CPU神经或物理。实际CUDA由MAIN审读后执行。
遵循experiment-code的持久日志、失败保留原则；用户固定协议优先，不训练、不调优、不画结果图、不新增科学阈值。

## 固定样本、身份和解释边界

唯一源为本地`/home/gbk/.stable_worldmodel/datasets/pusht_expert_train.h5`，E须为18685。
固定ordinal `[0,floor(E/3),floor(2E/3),E-1]=[0,6228,12456,18684]`，原relative RGB0/5/10/15/20与action[0:20)。
检查ordinal与episode_idx映射、连续step_idx、前后offset/末episode文件尾、length≥21、dtype/shape/finite。
缺失、短episode、身份错位直接拒绝，不替换、不根据任何outcome筛选，不去重分母。
每帧pixels和PNG SHA、动作array entry SHA、精确global/local slices、metadata、字段名/attrs、文件stat及版本写prepare。
只读选定slice，不扫描46GB全文件，不声称全文件SHA或已认证数据revision。
source/import closure、模型config/weights、public normalizer来源及原设计文档纳入绑定；run前后完整重建核验。

四个episode ordinal不等于四个独立演示。prepare保留六对raw action最大差及相同步RGB列表，仅描述相关性，
不据此换样本或扩大样本数。collector lineage、training membership、checkpoint训练run的stats provenance均unknown。
MAIN的v1 first4相关性发现是保留此限制的理由，不是v2改样依据。原raw专家动作允许超constructive球；
不clip/tanh、不再除100，不能将其说成constructive合法部署计划。

## 两teacher-forced窗口与三条件

| 窗口 | 真实latent history | raw packet starts | 仅评分label |
| --- | --- | --- | --- |
| W10 | z0,z5,z10 | 0,5,10 | z15 |
| W15 | z5,z10,z15 | 5,10,15 | z20 |

每episode先W10 F/Z/M再W15 F/Z/M。F用三包真实action；Z只把outgoing raw五条命令变零，再public normalize；
M仍用F三包并实际调用action_encoder，只clone其输出并令`[:, -1, :] = 0`，前两history embedding逐位不变。
M不是attention mask或合法raw命令。W15不是open-loop H2，绝不使用W10预测替代z15。
public float32 mean[-.0087,.0068]/std[.2019,.2002]保持不变、training_provenance_verified=false，不新fit或择优。
packet为5×2 time-major flatten10；本批仅承诺local loader alignment，不将collector causality unknown写成verified。

预测接口白名单仅slot/episode/window/condition/latent_history/raw_packets/tokens，无target、goal、witness、state。
所有24输出durable保存后才评分层join分离labels。此为代码接口隔离，不是OS进程隔离；z15可合法作为W15过去，
z20不进入任何模型预测输入。编码只接单RGB，同原image_transform和existing projector。
`model.predict`内部实际调用predictor和pred_proj；不能以未投影predictor输出冒充native192。

## 成本、持久化和失败

固定20 encoder +20 existing projector，24 action_encoder +24 model.predict API +24 predictor.forward +24 pred_proj.forward。
predict API与predictor分别记录，不相加为48次预测。每调用3 tokens，各72 latent/action packet positions；pred_proj共72 samples。
0AD/grad/engine/physics/reset/render/optimization，禁止CPU fallback、重试、去重及额外候选。
CUDA batch1 native float32、MATH SDP、TF32关闭、eval/no-grad，模型tensor SHA前后相等。
内部wall600s（含verify/init/final检查），外部建议timeout650s；allocated3GiB=3221225472、reserved4GiB=4294967296。
显存是dispatch前后及load后读取CUDA peak的拒绝cap，不冒称可在kernel分配前硬拦截所有瞬时内存；独立新进程运行。
prepare<256KiB，PNG单张≤1MiB；run总32MiB（含partial/journal/final），其中1MiB为final保留，每写检查200MiB floor。
开始须free≥232MiB；不是仅剩200MiB仍绕开floor启动。

每20真实native192和原RGB、raw20×2 durable；每24请求保存history/raw packets/normalized tokens，
完整prediction3×192与last192、embedding mask前后3×192、file SHA+arrayentry SHA。
module进入与完成都有Z_SYNC_FLUSH+fsync journal，文件新增亦目录fsync。输入先封存再dispatch。
失败保留attempted/completed/未开始分母，不重试；异常后close/final source/model检查错误单列，不能覆盖first_error。
外部kill可能只有可解压前缀journal而无final report，不能据退出/部分NPY宣称完成。
source末尾重验和modelhash是强制；final report失败会非零退出并向stdout提供错误，不伪造成功receipt。

评分所有8窗口EF/EZ/EM、EZ−EF/EM−EF、persistence MSE、真实latent change L2及F与Z/M输出MSE。
固定四episode等权，八window等权；未进行独立样本推断。只有F有真实标签对应轨迹，Z/M不是物理反事实truth。
归档fidelity和动作敏感性不是live泛化、native success、模型规划/能力改变或P1严格通过。
paired render B缺同状态/goal/render来源合同，未实现、不得借本结果改变with_target合同。

## MAIN命令（项目内层根目录）

```bash
cd /home/gbk/code/LeWorldModel/le-wm-main
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m pytest -q -p no:cacheprovider tests/test_p1_constructive_offline_fidelity_cuda_20261027_v2.py
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m experiments.spectral_reachability.p1_constructive_offline_fidelity_cuda_20261027_v2 prepare --output results/p1_constructive_offline_fidelity_cuda_20261027_v2_prepare01.json
```

prepare打印实际SHA，MAIN复核后将下列`PREPARE_SHA`替换为其64位SHA；没有预造实际prepare SHA：

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m experiments.spectral_reachability.p1_constructive_offline_fidelity_cuda_20261027_v2 verify --prepared results/p1_constructive_offline_fidelity_cuda_20261027_v2_prepare01.json --prepare-sha PREPARE_SHA
timeout 650s env PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m experiments.spectral_reachability.p1_constructive_offline_fidelity_cuda_20261027_v2 run --prepared results/p1_constructive_offline_fidelity_cuda_20261027_v2_prepare01.json --prepare-sha PREPARE_SHA --output results/p1_constructive_offline_fidelity_cuda_20261027_v2_cuda_main01
```

prepare/run输出路径必须NEW，不覆盖旧文件；verify纯saved，不导入torch，不实例化模型。
测试是RAM synthetic接口/算术验证，不是模型/物理实验结果。

## 本次交付验证（尚无v2 actual CUDA结果）

- v2独立51 tests通过；与冻结v1联合76 tests通过。全部RAM mocks，无tmp fixture、pytest cache或pyc。
- 原H5实际`material()`纯RAM collect通过，未导入torch、未prepare、未写result。115个source/model绑定，
  当次canonical prepare约34KB（最终版本由MAIN prepare重新给出精确bytes与SHA）。
- 固定索引[0,6228,12456,18684]；长度[109,141,83,113]，offsets[0,781895,1562684,2336623]。
  六对选定同step RGB均无逐像素完全相同；raw20×2最大绝对差依次约
  0.548613/0.504318/1.171454/0.491817/0.927616/1.100412。仅描述，不认证独立来源。
- public training provenance仍false；没有改v1或任何旧结果。P1原严格门槛完全保留。

联合回归命令：

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m pytest -q -p no:cacheprovider tests/test_p1_constructive_offline_fidelity_cuda_20261027_v1.py tests/test_p1_constructive_offline_fidelity_cuda_20261027_v2.py
```

不落盘的实际collect/重复重建命令（非神经，不生成prepare）：

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -c "import sys; from experiments.spectral_reachability import p1_constructive_offline_fidelity_cuda_20261027_v2 as m; a,_=m.material(); b,_=m.material(); m.require(a==b,'repeat saved reconstruction'); print(m.canonical(dict(episode_indices=a['episode_indices'],prepare_bytes=len(m.canonical(a)),bindings=len(a['bindings']),torch_imported='torch' in sys.modules)).decode())"
```
