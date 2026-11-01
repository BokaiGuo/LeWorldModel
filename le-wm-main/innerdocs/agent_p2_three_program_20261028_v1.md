# P2 三程序 imagined-context 重编译 / H3 joint：实现交接

仅本stem source/tests/本文新增。旧contextual H2的8计划、mid反馈/guard/2×2及实际负结果不改不复跑。本worker无生产prepare、torch/权重、CUDA、物理、网络或下级代理。MAIN审读后实际执行。experiment-code用于固定协议与严格saved验收；不按结果调参或生成额外图表。

## 固定算法，非旧二边别名

原2known roots，仅off，原root10三latent/两个between-action tokens、原warm raw10、原goal192、identity192 terminal MSE、原checkpoint/normalizer全部沿用。2根×joint/three-program=4计划；每臂42候选，全168分母不剪枝。没有fresh/P1/P2准入。

root在H3 zero处用原冻结`compiler.make_transition`做nominal3+functional Jacobian-forward3（192×30）。原42词生成规则扩展至joint30：zero、原bounded ridge、两radius×原模式0..9×符号−/+；完整30维SVD保留零模态，不按响应删词，不手工造技能。joint独立评完全部42 full15，稳定terminal Q,index择优。

composition对同42个root bank提案固定原首raw5，不把首段代价计入选择。在各自第一包产生的完整imagined context处，以该提案剩余H2为合法incumbent，重新求literal raw192×20 Jacobian。固定方向/alpha生成4提案，与incumbent一起全量非线性terminal评分，stable best5，平局保旧，再独立replay选中H2；取其第一包后的完整cp10 context，再以最后H1为incumbent重求192×10 Jacobian、同固定best5。最后全42条full15 terminal Q stable选；两个winner均完整full15和三次单包独立重放，所有三个中间context必须字节同值。

局部方向沿用既有guarded proposal语义：raw Jacobian乘连续float64 tanh chain，方向朝剩余球半径一半的负梯度点，非负斜率则零方向；alpha=[1,.5,.25,.125]。这是原冻结动力学上的新上下文编译，不是observed teacher15接口或旧suffix重评分；不把量化映射当连续可导函数。每次强制模型Q不劣于合法incumbent，不保证真实收益。

## 明确的新H3域与舍入合同

双方共同joint30 logit L2 cap=.75√15，radii=.25√15及cap；不是旧joint20 cap=.75√10，也不是每包独立球。新root-bank浮点向量在生成时显式向内径向舍入（必要时nextafter），保存原向量与运输向量；这是事前声明的新bank规则，不修补任何旧incumbent。后续incumbent始终保原logits和rawbits；原literal chart为NumPy tanh(float64).astype(float32)，不逆atanh、不clip、不宽epsilon、不事后修复。残余域和整个30维域同时硬验；失败保全分母。

## 预算与durable路径

每根shared6；joint42×3+winner6=132独占、standalone138；组合42×(首包1+H2优化14+H1优化7)+winner6=930独占、standalone936。paired unique1068/root、2136总predict；action_encoder/predictor/pred_proj同数，predict API为inclusive不与模块相加。每根85 Jacobian（root1+42×2），16320数学VJP方向；总170/32640。原root bank含2 SVD/1ridge；新suffix无SVD。内部backward engine调用/FLOPs/历史获取成本null，不称等FLOPs或等算力。继承原四条件16encode为已付共享成本，新视觉/物理0。

MAIN HOST CUDA only，MATH/TF32off/autocastoff，batch1，冻结weights。边界wall1800s（初始verify也计时、final重hash；不能抢占native调用），output256MiB/final reserve16MiB/floor200MiB，allocated4GiB/reserved6GiB硬守卫。无重试、无失败退款，首cause与部分records/Journal保留，后根not_started，不释放部分计划。

每forward intent→完整context/raw→三模块intent/input/return及真实predict API→native output滚动；AD intent→完整原forward→独立NPY持久化→return。gzip journal单LF、逐条index/previous/hash，实际写入flush/fsync后才下一操作。完整module输入/输出float32 bytes、J、全部候选、goal/action/source/每边context都封存。consumer先完整重建原输入authority和loader immutable buffer，再**无模型重放整个planner控制流**，逐个请求/全部数组/score/guard/replay和完整event chronology对账；未知、重排、缺失均拒绝。不把hash日志等同计算认证，AD数值正确性仍需独立FD而非此consumer虚构。

## MAIN CLI（inner repo；冻结后）

```bash
cd /home/gbk/code/LeWorldModel/le-wm-main
OPENBLAS_NUM_THREADS=1 .venv/bin/python -m pytest -q tests/test_agent_p2_three_program_20261028_v1.py
.venv/bin/python -m experiments.spectral_reachability.agent_p2_three_program_20261028_v1 prepare --output results/agent_p2_three_program_20261028_v1_prepare_MAIN01.json
.venv/bin/python -m experiments.spectral_reachability.agent_p2_three_program_20261028_v1 verify --prepared results/agent_p2_three_program_20261028_v1_prepare_MAIN01.json --prepare-sha ACTUAL_PREP_SHA
CUBLAS_WORKSPACE_CONFIG=:4096:8 .venv/bin/python -m experiments.spectral_reachability.agent_p2_three_program_20261028_v1 run_cuda --prepared results/agent_p2_three_program_20261028_v1_prepare_MAIN01.json --prepare-sha ACTUAL_PREP_SHA --output results/agent_p2_three_program_20261028_v1_cuda_MAIN01 --authorization MAIN_THREE_PROGRAM_KNOWN2_HOST_CUDA_ONLY
.venv/bin/python -m experiments.spectral_reachability.agent_p2_three_program_20261028_v1 consume --output results/agent_p2_three_program_20261028_v1_cuda_MAIN01 --report-sha ACTUAL_REPORT_SHA --receipt-output results/agent_p2_three_program_20261028_v1_consume_MAIN01.json
```

不填未来SHA；receipt为本writer独占新文件，不调用旧parent全目录cap writer。stdout仅小receipt；持久receipt含root-major/joint-then-composed全部4完整plans。

## 物理handoff与未完成边界

consume plans含原task_file/SHA、blind绝对NPZ/SHA、warm raw10、root full context、原goal/Q identity、normalizer、模型hash、full15 literal rawbits、3条逐context边与witness SHA、compiler节点[0]/[0,5,10]、selected bank index、完整原terminal预测/Q、prepare/report/selection SHA。joint三条边只是replay边，**program_count=1**；组合program_count=3。物理runner后续必须新full15 native adapter、同原reset/warmup/goal/renderer，核原动作meter与cp5/cp10/cp15及完整真实指标；旧full10或teacher15 outcomes不可复用。当前无物理执行入口，physical_release/P2_mechanism/P1始终false，成功/任务收益未知。

RAM测试使用无模型加法响应/NumPy布局callback与temp saved文件，不是神经CPU推理。完整saved consumer通过只代表工程闭包；主机CUDA、原生物理、RGB预测对照、独立根及真正非贪心机制仍未完成。
