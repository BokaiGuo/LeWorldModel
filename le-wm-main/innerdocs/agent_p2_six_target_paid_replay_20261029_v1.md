# Six-target own-six saved domain / paid deployment-token replay v1

仅实现MAIN批准的模型付费slice。独立domain saved artifact → own prepare → HOST CUDA raw inverse/re-normalize → whole H3/segmented[1,2] → own complete receipt。不是六槽物理driver；无物理许可，无训练/normalizer训练身份/P1/P2准入。旧封存不改，不patch globals、不拼旧4+新2。

## 固定authority与低层复用

三份批准protocol SHA由源码PROTOCOLS逐项绑定：base97d81f…、sources4c114b…、authority1e39e8…（完整值见源码）。实际six prepare `cccfadc20911dc1be008a047c57bca7400272f6602deb435fb0341a0c8bbc8f9`、report `b81f12ee3602835bc14fd2e22d7e3644b8376f7ba8b0b265948bfb18429b6934`、`.receipt` `592062bded7d6eb1716594b21fac4b5ef6255cf93d5285c2361f0ec2686c0103`固定于REFS；不猜flags、不拿status替代完整receipt。

MAIN的domain/verify-domain/prepare/run/consume会调用冻结six.analyze_saved及其纯saved认证，重建全部原receipt（含已持久authentication/completion）并与实际receipt整份强等；不调用six.run/solver，不重新生成arbitrary，不重复神经。认证/完整NPY解析/哈希成本真实计入各调用墙时；worker测试不执行真实upstream/fullstrict。six实际模型425与输入/source闭包及completion inclusive wall一并认证。

公共stats/native Box AST authority、原physical prepare和program receipt作为mapping证据；核原真实warm10 → 一次sub/div → chronological action2，六槽task/blind/source相同。Hi训练scaler仍未知。domain逐槽只取新six真实trace的selected.mean，核native_return.actions/mean字节相等及原NPY/file/data SHA，输入三键逐位核，不按结果取elite或换mean。

低层显式复用旧paid.execute/validate_trace/persist_trace及原rollout/capture算术；嵌套trace保留旧paid.trace schema，协议明示仅为未变的低层算术见证，不是旧四槽report/receipt。外层domain/prepare/report/completion/receipt、six文件reader与分母均自有，不伪造fixed4 schema。

## 两阶段与blocked

`domain`输出独立不可覆盖`.domain` JSON，含完整来源receipt、controller prepare、六mean源/完整坐标/bit/ULP/zero signs、任务warm映射和source闭包。CPU NumPy分步float32 mirror不是CUDA证据。任何超界保留全部六槽及blocked_slot_ids，不clip/tanh/换mean；prepare整批拒绝，不能挑合法子集。domain外文件布局有误同样fail closed。

CUDA各槽从相同z[1,3,192]/a[1,2,10]开始，mean[1,3,10]按时间reshape15×2；独立float32 mul/add/sub/div，禁止FMA和额外/100。raw120bytes对本次CPU参照精确，包括signedzero；重新产生的tokens而非旧mean进入两条完整H3重放。原history2不偷换、不加入future witness。全6新付费，即使前4结果重复也不抵扣。

全部六槽prediction数组durable后才写prediction_seal，再计算并逐槽持久化描述性target float64 fsum-MSE（不是GPU模型额外调用、不是选择；consumer独立重算）。无new encoder/AD/J/VJP/backward/physics/training。36predict/36samples、AE和lowproj各36、AE tokens144、predictor context/lowproj rows108。失败trace/debits先落盘、arrays随后；unexpected未返回账为unknown不补planned，partial不能签完整receipt。runtime发现blocked即停止成功闭合，绝不部署成功子集。

## 工程cap与精确墙时边界

总768MiB含96MiB末尾reserve，每root384MiB，trace payload64MiB，metadata32MiB，floor200MiB，开始需cap+floor；whole1350s、每槽180s含转换/425检查/持久化和本槽score，等待其他槽seal不计本槽但计whole；CUDA allocated3GiB/reserved4GiB。HOST pinned venv、deterministic、TF32/autocast off、MATH SDPA，无CPU neural fallback。

Store start在domain验证前；report记录before-report时间，report真实fsync及SHA完成后取inclusive elapsed写completion。completion绑定report SHA，complete仅当producer完成且inclusive≤1350s；completion自身ack写入不在该墙时门，但在768MiB文件census内。共享源/425 before/after强等。SIGALRM/逐opGPU guard是合作式，不能抢占kernel；失败footer允许越时写partial但完整consumer拒绝。未returned的attempt不退款。磁盘/杀进程/输出reserve耗尽可能没有可完整consume产物，保失败不造receipt，不承诺所有partial完整闭合。

## MAIN CLI（仅审读后运行；worker未生产）

```bash
cd /home/gbk/code/LeWorldModel/le-wm-main
OPENBLAS_NUM_THREADS=1 .venv/bin/python -B -m pytest -q -p no:cacheprovider tests/test_agent_p2_six_target_paid_replay_20261029_v1.py
# 每次SHA均取上一条实际输出；没有未来hash。
MOD=experiments.spectral_reachability.agent_p2_six_target_paid_replay_20261029_v1
D=results/agent_p2_six_target_paid_replay_20261029_v1_domain_MAIN01.json
M=results/agent_p2_six_target_paid_replay_20261029_v1_prepare_MAIN01.json
R=results/agent_p2_six_target_paid_replay_20261029_v1_cuda_MAIN01/report.json
.venv/bin/python -B -m "$MOD" domain --output "$D"
.venv/bin/python -B -m "$MOD" verify-domain --domain "$D" --domain-sha256 "$DH"
# domain如有blocked槽，在此停止；不发起CUDA。
.venv/bin/python -B -m "$MOD" prepare --domain "$D" --domain-sha256 "$DH" --output "$M"
.venv/bin/python -B -m "$MOD" verify --manifest "$M" --manifest-sha256 "$MH"
CUBLAS_WORKSPACE_CONFIG=:4096:8 OPENBLAS_NUM_THREADS=1 .venv/bin/python -B -m "$MOD" run --manifest "$M" --manifest-sha256 "$MH" --output results/agent_p2_six_target_paid_replay_20261029_v1_cuda_MAIN01 --main-authorized
.venv/bin/python -B -m "$MOD" consume --manifest "$M" --manifest-sha256 "$MH" --report "$R" --report-sha256 "$RH" --output results/agent_p2_six_target_paid_replay_20261029_v1_consume_MAIN01.json
```

worker仅RAM算术/Store→真实临时文件→reader/fixture严格消费、篡改、partial与真实saved ABI检查；实际mean NPY仅核dtype/shape/byte身份及task mapping，未作domain或收益分析，不以fixture冒HOST证据。测试记录：新模块53 passed / 1.27s；最终与旧paid replay联合95 passed / 3.43s，均exit0。三件套交付后冻结。真实domain、prepare、CUDA和full consume由MAIN审读后执行；物理adapter/运行不在本次授权。
