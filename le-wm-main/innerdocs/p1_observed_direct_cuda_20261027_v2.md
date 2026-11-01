# Observed direct B4/B8 v2：仅无损持久化层变更

v1源码/tests/doc/prepare保持原封存，不修改、不重seal。新增v2复用v1的真实source审计、
observed RGB编码、normalizer、任务template、NativeCuda及sealed fullH2 direct core。
不读取旧direct/witness收益作选择。调用图/算法/数组/事件不减：同两root、sameQ/native192 goal、
原每packet .75sqrt5 logit球、GN/GD、B1=4/B2=8、zero restart1、step1、damping.002、无额外line search。
新task_contract_sha绑定v2新prepare，因此task_id也随协议hash改变；不是改动科学输入或沿用旧task_id冒充新执行。
来源observed_history是真实采集，可能完全静止，与原padding数值相同不增加独立样本；freshness unknown。
P1准入/eligible仍false，B8不自动建立stall、没有τ/独立witness/物理效用保证。

## 固定资源

每臂70predict+70action_encoder、8完整H2J、1536数学VJP；两根两方法共280/280、32J/6144VJP，
GN合16SVD/16solve，32proposal/32selection；新native encoder8/projector8。新physics0/training0。
source此前真实采集208内部physics等成本照旧继承。AD forward计在70内，不换用初始J、不薄化轨迹。
原run_direct原样调用，全部原事件和direct_finished整份report也照样保存，report另存一份不去重。

共同image计账限制：封存screen.native_image_encoder的attempt来自encoder/projector pre-hook，
但completed仅在model.encode整体返回且receipt写入时一起确认。若encoder已return而后projector/encode/
receipt故障，不能从该helper得出每个module精确return完成数；0 completed可能只是整体receipt缺失，
不能解释为encoder未实际完成。v2原样保留这些事件/计数，不声称general exact encoder-return故障账；
成功路径要求8/8整体receipt，失败路径必须结合partial journal且不准入。本轮未修该共同helper。

## 存储协议预声明

总输出cap24MiB，disk floor200MiB，启动完整预检224MiB，不允许调小预算/增大cap绕gate。
普通写预留512KiB用于final报告及gzip footer。所有写均重新检查共享总quota及200MiB floor。
上游小prepare仅hash manifest，不复制大closure；v1 source/tests/doc与新三文件全部绑定。

image journal与每臂journal均gzip level6。每个原始canonical JSON事件先完整序列化，
compress+Z_SYNC_FLUSH，再文件flush+fsync；结束另写Z_FINISH。没有精度转换、字段过滤、数组删减。
task/report整对象gzip（原canonical JSON完整字节），source_audit/final report仍小JSON。
每臂路径为root_INDEX_METHOD/direct_journal.jsonl.gz、report.json.gz，任务task_INDEX.json.gz；
不是旧.json/.jsonl路径，旧消费者不得直接假装兼容。最终report包含压缩文件SHA及路径。
image事件路径image.events.jsonl.gz。read_gzip严格解码完整对象；recover_events返回完整行与gzip EOF布尔。
崩溃后仅恢复完整JSON行，eof=false必须标partial；CRC/结构损坏抛错。fsync不能克服介质硬故障。
写失败后的压缩器状态不再用来append/footer，避免制造假的complete stream。

## 容量证据边界

此前actual B3四臂journal每轮约0.756–0.766MiB，终态report会在journal和独立文件重复；
未压缩B8按该结构约42.1MiB，仅为估计，不能据此绕v1的256MiB预检。
v2在运行前冻结24MiB cap，以完整四臂B8 dense mock作工程测量，要求输出≤12MiB（至少2倍裕量）。
测试使用192维dense float64 contexts/192x20 dense J，全部6144 row事件、32次完整recompile、
36个trajectory节点及所有native/AD/replay事件。原始每条canonical byte流hash必须等于gzip解压后byte流hash，
report逐对象exact相等；不得仅测零数组或丢弃direct_finished重复报告。
本轮完整mock测量：四臂原始journal+report 44,093,621 bytes，全部压缩产物9,669,636 bytes（约9.22MiB），
预声明cap 25,165,824 bytes，约2.60倍裕量；该数值是mock而非actual run产物大小。
这是保守实规模mock证据，不是对任意有限float64序列压缩比的数学保证，更不是actual CUDA证据。
若该完整mock不满足2x裕量或MAIN审查不能接受该容量依据，不启动；不能事后调cap。
即使预检/测试通过，actual仍可能因并发磁盘占用/不可压缩数据触发硬cap而partial，不能冒充完成。
执行前MAIN应确保至少224MiB加并发写入余量，暂停其他大产物写入，完整预算不变。

## 命令（MAIN独占CUDA）

```bash
OPENBLAS_NUM_THREADS=1 .venv/bin/python -m pytest -q -s tests/test_p1_observed_direct_cuda_20261027_v2.py
.venv/bin/python -m experiments.spectral_reachability.p1_observed_direct_cuda_20261027_v2 prepare --sourcepack results/p1_observed_history_source_20261027_v1_main01 --source-sha 0b89d77494acc6cf03bb0110e87231c1c240a253e844326acc668b7e4f93d546 --source-prepare results/p1_observed_history_source_20261027_v1_prepare01.json --source-prepare-sha ecc5f4a488290b6d0eade4d3ff2034d256729de94c4d3a14b12d6eebad783385 --output results/p1_observed_direct_cuda_20261027_v2_prepare01.json
.venv/bin/python -m experiments.spectral_reachability.p1_observed_direct_cuda_20261027_v2 verify --prepared results/p1_observed_direct_cuda_20261027_v2_prepare01.json --prepare-sha PREPARE_SHA
# MAIN only after full review, test capacity acceptance, and disk preflight:
.venv/bin/python -m experiments.spectral_reachability.p1_observed_direct_cuda_20261027_v2 run --prepared results/p1_observed_direct_cuda_20261027_v2_prepare01.json --prepare-sha PREPARE_SHA --output results/p1_observed_direct_cuda_20261027_v2_cuda_main01
```

实现agent只测试/prepare，不执行模型或物理。独立saved-array audit需识别新gzip合同及新task_contract，
并按稳定root身份核对，不能把mock/文件压缩成功解释为科学增益或P1通过。
