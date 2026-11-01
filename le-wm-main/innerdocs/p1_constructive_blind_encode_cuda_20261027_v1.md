# Constructive blind native CUDA encoding → GN/D2P tasks

独占新producer，完整只读审查constructive source源码/doc之后实现。部署只接MAIN approved blind
白名单与外部SHA，不读sourcepack、journal、commit、certificate、state或witness，不把其路径放到prepare。
这是代码级读取边界，不是OS sandbox。全量sourcepack独立准入仍由MAIN完成；当前仅收到两份blind访问
授权，不把它自动升级为reachability/完整source admission。没有运行模型/physics，没有actual task_id。

## 当前允许的实际输入（仅这两项）

| source slot | blind NPZ | external SHA |
|---|---|---|
| 0 | results/p1_constructive_reachable_source_20261027_v1_main01/s0_blind_inputs.npz | d12ef7b66b69e53444ae6a4bde51c3830bacd6ad449d79845030ea70e2908da3 |
| 1 | results/p1_constructive_reachable_source_20261027_v1_main01/s1_blind_inputs.npz | 9c0f6451de986bbb48c8251a8cfaab2b3fed60ce15698a9c6454c6206e228957 |

load_blind独立实现严格3字段NPZ whitelist：rgb_history uint8[3,224,224,3]、
raw_action_history float32[10,2]全部zero、goal_rgb uint8[224,224,3]。
allow_pickle=False，压缩文件≤2MiB、精确3个唯一member、每项≤512KiB/总展开<1MiB；
拒绝extra state/witness/action候选/object dtype/错误shape、symlink与非blind文件名。
按外部SHA验证读取前/读得字节；不从同目录发现/列举其他研究输入。

## MAIN whitelist 接线与准入边界

正式prepare需要MAIN独立source审计后传入另一个小JSON及其外部SHA；exact fields如下。
`admission_sha256`是MAIN审计的opaque内容ID，本consumer没有审计文件路径、不打开它：

```json
{
  "schema": "p1_constructive_blind_encode_cuda_20261027_v1.approved_blind_whitelist",
  "approved_by": "MAIN",
  "purpose": "calibration",
  "source_admission": "MAIN_independently_admitted",
  "admission_sha256": "MAIN_SUPPLIES_64_HEX_AFTER_INDEPENDENT_AUDIT",
  "with_target": false,
  "history_steps": [0, 5, 10],
  "raw_action_steps": 10,
  "slots": [
    {"order":0,"source_slot":0,"blind_npz":"ABSOLUTE_S0_BLIND_PATH","blind_sha256":"S0_EXTERNAL_SHA"},
    {"order":1,"source_slot":1,"blind_npz":"ABSOLUTE_S1_BLIND_PATH","blind_sha256":"S1_EXTERNAL_SHA"}
  ]
}
```

示例中的占位值不是合法prepare输入。pending admission、unknown keys、sourcepack/certificate路径字段
都拒绝。build限定这版恰好为上面两份MAIN授权文件；不替换/增加slot，不根据模型或结果筛根。
白名单loader泛化支持1–4slot便于mock，但本版真实build严格固定上述pair。
外部hash与MAIN声明是授权接口，不是签名证书或本consumer独立可达证明；完整失败分母由MAIN保留。
不能声称consumer仅凭blind像素验证了warmup来源、with_target渲染开关或native reachability。

## 固定模型/时间/任务合同

checkpoint `/home/gbk/.stable_worldmodel/checkpoints/pusht/lewm` 的原config/weights SHA固定；
native eval/frozen，model tensor hash必须与既有冻结官方模型一致。原raw mean/std从source-bound
run_pusht_counterfactual.py AST解析，mean[-.0087,.0068]、std[.2019,.2002]按float32。
raw relative动作已等于位移/100，不再除100；10个raw zero归一化后一般不是zero！

每slot严格按observed0/5/10/goal四次单帧model.encode pixels-only调用，每次1encoder+1existing projector。
同像素仍4次独立执行，不padding、不共享goal编码；本实际pair预算8encoder+8projector，
0predict/actionencoder/newAD/physics/training，no CPU neural fallback。with_target=False是新task
合同而非旧with_target渲染桥，goal_RGB来自MAIN授权blind，模型看不到goal pose/witness。

保存每张原始float32 native192 NPY，先fsync/readback/SHA再seal task。H2 task采用已审计native
compiler布局：root_context latent_history[3,192]、action_history[2,10]，后者由10rawzero按5×2
两包time-major归一化，代表0→5和5→10 between-state actions；root是observed10而不是observed0。
goal独立第四次编码。identity192 Q固定、无readout/adapter/训练或outcome调参。

task字段严格满足core.validate_task；task_contract_sha256真实绑定本新prepare/白名单SHA、
with_target=False、source slot/blindSHA、时间合同、原normalizer/model与identityQ。
task_id只在4个新latent确实返回并持久化后通过core.task_id计算，不生成占位ID/不冒用旧根taskID。
目前没有物理native root hash授权，`blind_source_sha256`显式只标文件身份，
`native_root_sha256=null`，不能将NPZ SHA改名物理root_sha256。

## 可实际接编译搜索的接口，不仅是编码

每根`task_00.json.gz`包含task、真实task_id和task_contract。load_task(path,external_file_SHA)
验证后返回：

- gn_task：直接接screen.run_direct或screen.NativeCuda(model,device,gn_task)。
- d2p_packet：只含root_context/goal；接p3_goal_patch_consistent_compiler_20261027_v1.compile_arm
  (packet,q,backend,journal)。该数组入口不要求物理root_sha256。
- q：同task identity192；task_id/blind_source_sha256/with_target单列给新runner做identity检查。

mock测试实际调用sealed GN core完成两次更新，以及sealed full10 conditional compile_arm完成
10prefix/各fresh suffix/全路径replay（66packet calls），不是仅检查shape。
它们是NumPy stand-in，不是神经执行/科学证据。
旧D2P顶层source_inputs/execute_cuda硬绑旧四根；**不能复用旧CLI**或旧selected_actions冒充新源。
MAIN后续仍需新的独占搜索driver：从本export加载接口、同prepare model/normalizer/Q、原sealed compiler
数组入口、独立meter/新预算/新结果seal后才能physical。这里不计假想未来compile为已执行。
conditional编译单臂预算原66predict/actionencoder、11J/12SVD/2112数学VJP；GN预算由未来冻结配置
决定，均另计，不混入本producer的8image预算，不宣称全pipeline等算力。

## Durable / partial / tests

encoder/projector分别attempt/return/error，CUDA batch1 shape/inference检查及4n顺序/cap；
predict/actionencoder/pred_proj与AD入口禁止。projector失败不抹去encoder已返回；sync失败不假称
completed。每root partial图像、首错、后续blocked slot都保留，无重试或替根。restore失败追加不覆盖
首错；model未load时model_unchanged=None。运行后重建全部whitelist/blind/code/model绑定。
NPY/tasks/event/report总cap2MiB、final预留128KiB、200MiB floor，prepare<256KiB且不复制PNG/模型。
fsync磁盘硬故障仍可能无法保证report存在，不准入partial。

unit tests不读privileged actual，只有两项已授权blind及固定model/source字节的saved检查；其他均synthetic
小NPZ、白名单拒错、normalized chronology、真实array级GN/D2P接口、CPU/grad/cap/partial/fsync。
完整sourcepack独立audit报告由MAIN负责，不由这里重复。

```bash
cd /home/gbk/code/LeWorldModel/le-wm-main
PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -B -m pytest -q -p no:cacheprovider tests/test_p1_constructive_blind_encode_cuda_20261027_v1.py
# 等MAIN完成source admission并交付白名单之后；当前不prepare：
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -B -m experiments.spectral_reachability.p1_constructive_blind_encode_cuda_20261027_v1 prepare --approved-whitelist APPROVED_JSON --whitelist-sha EXTERNAL_SHA --output results/p1_constructive_blind_encode_cuda_20261027_v1_prepare01.json
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -B -m experiments.spectral_reachability.p1_constructive_blind_encode_cuda_20261027_v1 verify --prepared results/p1_constructive_blind_encode_cuda_20261027_v1_prepare01.json --prepare-sha PREPARE_SHA
# MAIN独占HOST CUDA：
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -B -m experiments.spectral_reachability.p1_constructive_blind_encode_cuda_20261027_v1 run --prepared results/p1_constructive_blind_encode_cuda_20261027_v1_prepare01.json --prepare-sha PREPARE_SHA --output results/p1_constructive_blind_encode_cuda_20261027_v1_cuda_main01
```
