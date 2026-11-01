# Exp1/P2 六槽同 job normalized target controller

严格实现 MAIN 已批准协议 `agent_p2_six_target_controller_protocol_20261029_v1.md` SHA `306305ea86638d120760f7a497859319e3c716cee2b61ddf2522a46af18b7f1e`。只新增本 stem source/tests/doc；所有旧source、四槽controller、paid replay、旧missing和产物不修改。不训练/fit、不新physics/encoding，不认证Hi scaler，不授予raw inverse/物理部署或科学准入。

## 固定接点与前置

源码 REFS 固定已实际存在的原 delivery SHA `82b36fa3…`、old prepare/report/整份strict receipt，以及 Nash prepare `3aaffe495a3cfddda0d88403a39b1c937121942c64f79be5a574499579384ad0`、report `7431e2b851856fbf5e24d2db3b3762bb6298532be7316e599d6bd32de739c278`、完整receipt `1e47155d480c23a0bda9cae50e633f0fe7ef56c992d3f489a31e255e43286ffc` 与 completion SHA `7d2edf45b88a85c9d8e62ea4de63333e3520c204b354ab2cc0f6e07c29839007`。完整路径/全SHA见 `refs()` 和已批准协议，不使用未来hash。

- `authenticate_arbitrary()` 调冻结 companion.consume(prepared,ph,**folder**,rh,receipt_output)，精确重建整份原receipt且核新写文件SHA相同。每次排他创建 `results/<本stem>_auth_*/receipt.json,proof.json`，永不覆盖/删除失败认证记录。纯数值/保存消费的loss130/analytic_gradient128/update128另列，不当新神经AD或controller成本。
- `build()` 调原delivery.build/old.build与old.analyze_saved重建整份原四槽receipt（补其真实report/manifestSHA后全等），不是仅status/部分localchecks。读取原input NPY dtype/header/EOF/file/data SHA，old4 z/a/target逐byte等值（含signed zero）；原row保持完整嵌入新slots.source。oldfullstrict只在MAIN实际prepare/verify等运行，worker测试不实际调用它。
- arbitrary两槽从receipt.plans0/1的三键payload取得，FIRST/float32 bytes匹配target32SHA、原task/source、z/a原字节。同job slots0..5固定顺序为两根各prior_FIRST/program_first_postcondition/arbitrary；全部new dispatch。只向模型传z `[1,3,192]` /a `[1,2,10]` /target `[1,192]`，没有future witness、macro、旧mean或metadata warmstart。
- `verify()` 返回 `(manifest,new_authentication_ref)`，每次重新完整认证；原manifest只绑定prepare时的认证proof，后续verify proof是旁证，不能用不同path重写原manifest。所有认证目录保留、elapsed/CPU次数披露。runtime和consume各自绑定新proof。额外认证文件不属于CUDA payload cap，但每次受原companion16MiB输出和floor约束；本runner proof为bounded JSON，不读取pickle。

模型接点原样复用 `tracker.parity.rgb.loader.construct_cuda`、`tracker.validate_construction/state_hashes/GPUResourceGuard`、`core.track_cuda` 和 `core.validate_saved`。新六槽外壳独立schema/array filename reader/store/whole consumer，不调用旧四槽run/execute_slots，不猴补旧globals/torch方法。425完整state/source/input/export与constructor ownSHA三件套保持before/after强等。

## 作业与所有27轴

H3/N8/I2/topk2/var1/seed0/segments[1,2]；每槽新solver seed0、完整zero init，同一已构造graph串行六次。不改成batch6，不继承前槽RNG/mean。搜索2轮N8完整H3；每个实际native returned mean另付S1 whole和segmented[1,2]各一次。所有候选/模块IO/三latent槽与完整causal history、CEM callback/native返回/actual mean均逐slot封存，不去重、不复用旧输出。每槽12API/54samples，整批72/324；其中mean双replay共36API/36samples。

`AXES`为本schema显式27轴（不是声称旧链原来已有这个命名标准）：

| 组 | 轴名 | 完整六槽实际计划 |
|---|---|---|
| predict | predict_API / predictor_samples | 72 / 324 |
| AE生命周期 | AE_attempts / AE_dispatched / AE_returned / AE_captured | 各72 |
| AE工作 | AE_rows / AE_tokens | 324 / 1296 |
| predictor生命周期 | predictor_attempts / predictor_dispatched / predictor_returned / predictor_captured | 各72 |
| predictor context | predictor_context_tokens | 972 |
| projection生命周期 | lowproj_attempts / lowproj_dispatched / lowproj_returned / lowproj_captured | 各72 |
| projection工作 | lowproj_rows | 972 |
| 搜索 | candidate_cost_calls / candidate_trajectories | 12 / 96 |
| mean重放 | mean_paid_replays / replay_trajectories | 12 / 12 |
| AD | AD_primal / J / VJP / backward | 各0 |
| 未知算力 | FLOPs | null |

27轴逐槽从实际events/searches/replays/caps重建，不以planned覆盖。额外 `side_axes`：image encoder=0/high=0/physics=0/SVD=0、constructor425 copies；每槽capture列保存array count/payload bytes、capture_sync wall、core wall、host-entry wall、outer slot wall。**all_cpu_transfer_calls=null**：冻结core没有完整记录每一次额外byte/hash/exact传输，不能拿array count伪造总CPU捕获次数。保存capture同步时间是core既有范围而非全部CPU费用；认证elapsed和CPU算数费用另留proof。实际failed attempts不退款；invoke意外抛出时本槽未知轴=null，已知前槽subtotal保留，后槽not_started零；FLOPs始终unknown。partial留report/trace/已写NPY，完整consume拒绝partial，不称72/324成功。

## 边界与落盘顺序

CUDA allocated3GiB/reserved4GiB；原venv、deterministic、TF32off/autocastoff、MATH SDPA、caller grad-enabled但模型及输入均不求导，AD0。禁止CPU neural fallback。

整批1024MiB含128MiB末尾reserve；每root256MiB含32MiB；floor200MiB；原core每槽32MiB payload/core300s、outer slot300s（含前后hash/拷贝/持久化）、总1800s（起于Store、含verify和认证、模型构造、全slot/源hash）。6×300不保证加上前后置仍能完成，先到整批wall即失败保留，不扩wall/补算/换根。

顺序：dispatch_manifest → **全部18输入NPY** → input_seal → constructor/receipt持久化 → slot0 trace/全arrays → … → slot5 trace/全arrays。consumer核全文件引用和journal顺序；trace先于arrays保存以便后续cap/timeout仍保留实际debits。TorchDispatchMode每op前后查GPU/whole cap，SIGALRM固定1800s，合作式guard不声称抢占CUDA kernel。final footer允许越时留失败。

report.whole_wall_seconds_before_report不含report fsync；completion.json绑定report SHA，并以`whole_wall_through_report_fsync_hash`明确记录至report真实fsync/hash的整批1800s门，completion acknowledgement自身写入不在该边界。完整consume同时核report和completion；若report写盘越时，即使六槽算完也不接受完整job。两文件及所有NPY均纳入总byte census。disk full/进程被杀可能无完整receipt，failclosed。

## MAIN精确CLI

仅MAIN审读后执行；worker未运行任何实际prepare/verify/fullstrict/模型或生产。

```bash
cd /home/gbk/code/LeWorldModel/le-wm-main
OPENBLAS_NUM_THREADS=1 .venv/bin/python -B -m pytest -q -p no:cacheprovider tests/test_agent_p2_six_target_controller_cuda_20261029_v1.py
.venv/bin/python -B -m experiments.spectral_reachability.agent_p2_six_target_controller_cuda_20261029_v1 protocol
.venv/bin/python -B -m experiments.spectral_reachability.agent_p2_six_target_controller_cuda_20261029_v1 prepare --output results/agent_p2_six_target_controller_cuda_20261029_v1_prepare_MAIN01.json
# MH用上述实际返回SHA，不是未来占位hash。M/R固定如下；MH/RH由MAIN填实际值。
M=results/agent_p2_six_target_controller_cuda_20261029_v1_prepare_MAIN01.json
R=results/agent_p2_six_target_controller_cuda_20261029_v1_cuda_MAIN01/report.json
.venv/bin/python -B -m experiments.spectral_reachability.agent_p2_six_target_controller_cuda_20261029_v1 verify --manifest "$M" --manifest-sha256 "$MH"
CUBLAS_WORKSPACE_CONFIG=:4096:8 OPENBLAS_NUM_THREADS=1 .venv/bin/python -B -m experiments.spectral_reachability.agent_p2_six_target_controller_cuda_20261029_v1 run --manifest "$M" --manifest-sha256 "$MH" --output results/agent_p2_six_target_controller_cuda_20261029_v1_cuda_MAIN01 --main-authorized
.venv/bin/python -B -m experiments.spectral_reachability.agent_p2_six_target_controller_cuda_20261029_v1 consume --report "$R" --report-sha256 "$RH" --output results/agent_p2_six_target_controller_cuda_20261029_v1_consume_MAIN01.json
```

仅同controller工程接通。prior物理时距unknown保持null，三proposal generation预算不齐，同controller不等公平、不等FLOPs/时间对齐。known2不当独立确认；Hi scaler未知；P1/P2/E1/scientific gate=false、raw_inverse=false。无生产接通/成功结论，实际prepare/HOST CUDA/full saved consume均待MAIN。experiment-code技能只影响固定循环/失败账/回归组织，不采用训练或效果择优建议。

交付 RAM 实测：本文件所列专属测试命令 **76 passed in 12.07s**。包含真实小metadata/原input NPY的ABI核对（不fullstrict）、文件级prepare→verify fixture、完整六槽callback与saved消费fixture、整份receipt非status字段篡改拒绝、inclusive completion/fsync边界、27轴partial与unknown、全部18输入先封存与禁止RAM冒HOST。fixture认证替身明确不等实际上游认证或模型证明。尚未运行实际prepare/verify/生产fullstrict/NN/CUDA/physics；待MAIN审读独占执行，未知仍保留，无新增authority缺失的虚构结论。
