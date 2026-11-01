# Constructive reachable PushT source v1 — MAIN审阅后执行

这是可执行探索source producer，不是P1 admission或独立确认实验。原生success是可达完成标签，
不要求direct困难/plateau/独立tau authority才探索。无训练、模型、encoder、CPU/CUDA neural。
使用experiment-code做持久化/失败测试，experiment-design隔离构造、资格与后续盲测阶段。

## 固定小设计

固定2–4候选，每候选一条确定witness、无重试无补根、无physical-outcome选role。
root/reset seeds由MAIN显式给定并和当前metadata exclusions比对；unknown provenance仍unknown。
root recipe重用已封存sample_state的近接触几何，但本任务不随机抽最终goal：真实witness端点
直接提供完整native state7 goal和真实RGB。新任务`with_target=False`（移除目标叠图，不移除goal
agent），不得与旧render合同混淆。native env源码SHA固定，geometry/control频率不变。

每slot：

1. 构造episode：reset指定seed取原始随机几何参考goal；sample_state一次（proposal/attempt均0）；
   placement reset seed0，按实际placement开始执行10步zero observed warmup，保存0/5/10真实RGB。
2. 从实际warmup末root几何计算radial=(object-agent)/norm，tangent为90°方向；前5步logit=.5radial，
   后5步logit=.25radial+sign*.5tangent，sign按slot奇偶固定。每包logit范数≤.75√5，float32 tanh。
   command/goal生成规则在prepare封存，具体logits/commands在未来动作前另写commit。几何计算是
   **privileged offline dataset construction**，不是待测部署算法，也不是模型baseline。
3. 10步真实端点state7+RGB成为goal。中途guard、native terminal或truncated则该slot失败，不继续。
4. 两次独立verification episodes均从相同reset/sample/placement+真实warmup重建，逐数组比较history/
   state/已记录block dynamics，不恢复saved poststate。然后只调用native `_set_goal_state`并设置`_goal`
   为真实端点RGB；该setter源码仅赋goal，无physics。绑定前后snapshot精确不变，禁止`_set_state`
   设goal后再恢复root。`with_target=False`避免goal叠图及旧goal_pose污染这一路径的观测。
5. 原生goal和success开始生效，root若已success则拒绝平凡任务；逐步重放同一witness并与构造轨迹
   对齐，只执行到原生首次success（可早于10步），honor terminated/truncated；不补剩余未来。
   两次均native成功才标reachable。完整端点goal包括agent、物体、角和测得agent速度，不把速度
   归零、不改native reward/success。物体位移/转动独立描述，不要求阈值、不假称接触重配置。

此goal attachment是新source recipe的一部分，不是旧consumer许可或旧source格式bridge。
未来盲测runtime也必须从该recipe重建，在自己的root绑定同goal；不得admit旧runtime。

## 隐藏数据防火墙

`sN_blind_inputs.npz`严格只有`rgb_history`(3,224,224,3)、`raw_action_history`(10,2 float32)
和`goal_rgb`；`load_blind`白名单+外部SHA拒绝state、witness及额外字段。normalization与模型绑定
由后续独立冻结模型阶段处理，本producer不编码或假造model task_id。
construction_certificate.npz、journal、commit及sourcepack含witness/state，仅给dataset builder/evaluator，
不得把整个results目录当selector输入。root/source哈希用于身份，不能泄漏证书动作/未来帧作为候选答案。
这是工程文件分离，不是操作系统访问控制；后续consumer须只接收白名单payload，不给证书路径。
consumer必须先审完整sourcepack及journal，要求`sourcepack_complete=true`且该slot reachable，
再传blind文件及外部SHA给loader；loader自身不是整包物理证书验证器。失败包文件不可单取冒充准入。

## 固定cap与durable输出

每slot上限3episodes×(10warmup+10future)=60public steps、6reset、63explicit render、6reset内部goal
render、12set_state、612实际space.step、2goal_bind。2slot上限120public/12reset/126explicit render/
12goal render/24set_state/1224physics/4goal_bind。4slot逐项×2。physics hooks在每个reset新space上
安装，含失败操作的部分子步；attempt不退款，completed仅返回计数。重放提前成功可少于cap。
factory一次、close一次；native ctor不推进physics，显式caps指上述受meter控制的资源。

wall由MAIN显式传1..600秒，每operation前检查，无法抢占单个卡住的native调用；MAIN应外加timeout。
output总cap8MiB、final预留128KiB、启动200MiB+8MiB；每次写前200MiB floor。hash-chain journal每个
operation前后fsync；每步freshRGB guard都保存PNG/file+pixelSHA及state/snapshot，再准下一动作。
目标/证书和blind source立即NPZ压缩fsync；写JSON/NPZ/PNG均exclusive。无下载、删除、resume。
候选拒绝保留完整固定分母与已落盘帧；工程异常fail-fast，后续not_started，first error不被close/restore
覆盖。final磁盘写失败不能保证receipt存在，不宣称completed。源码/历史metadata运行前后重hash。
`execution_binding.json`记录外部prepare文件SHA，`prepare_id`是canonical语义ID，二者不混用。

## MAIN命令（本交付不prepare，不启动physics）

cwd `/home/gbk/code/LeWorldModel/le-wm-main`。下面两对seed为固定探索候选，prepare会实际查历史冲突；
不是独立freshness证书。若拒绝须MAIN审批新候选，不暗中挑seed。wall建议300s仅工程限时。

```bash
.venv/bin/python -B -m unittest discover -s tests -p 'test_p1_constructive_reachable_source_20261027_v1.py' -v
.venv/bin/python -B -m experiments.spectral_reachability.p1_constructive_reachable_source_20261027_v1 prepare --root-seeds 202798101 202798102 --reset-seeds 202798201 202798202 --wall-seconds 300 --output results/p1_constructive_reachable_source_20261027_v1_prepare01.json
.venv/bin/python -B -m experiments.spectral_reachability.p1_constructive_reachable_source_20261027_v1 verify --prepared results/p1_constructive_reachable_source_20261027_v1_prepare01.json --prepare-sha PREPARE_FILE_SHA
timeout 350s .venv/bin/python -B -m experiments.spectral_reachability.p1_constructive_reachable_source_20261027_v1 run --prepared results/p1_constructive_reachable_source_20261027_v1_prepare01.json --prepare-sha PREPARE_FILE_SHA --authorize-physics --output results/p1_constructive_reachable_source_20261027_v1_main01
```

run只物理，无GPU需求，不可与MAIN其他physics同时擅自启动。无需P1阈值authority；可0/2reachable。
mock测试不证明真实Pymunk重放或固定witness会操纵T，actual由MAIN决定执行后审计。

## 后续模型盲测与P1分离

source先由真实可达见证建立任务完成前提，再由另一个固定model/transform/normalizer/预算配置
只消费blind payload，从自己的新RGB重编译程序，选择先封存后执行。先报告整个可达探索池上的
模型程序成功/失败，无须筛成direct困难。若之后专门研究P1 kill experiment，再独立校准和冻结
direct困难/非贪心/能力改变门，并在新confirmation根检验；不得用探索结果调门后追认独立检验。
旧两根的262分支0成功与前提未证历史保持不变；本producer不训练、不改旧success、不宣称P1通过。

MAIN另报告dense actual root0 greedy/twostage已有物体非零响应；本交付未独立核验该新结果。
旧稀疏null仅描述当时有限探针，不代表不可控，不作为本固定新source family的选根依据。
历史排除表中的`metadata_authority_content_id`是sealed resolver给出的语义摘要，不是文件路径；
完整exclusions由prepare外部SHA绑定，其余真实metadata路径逐项重hash，不重新收集未来结果。
当前完整历史metadata超过200KB，prepare使用JSON envelope内gzip/base64无损保存，仅元数据，
不复制旧大protocol。envelope包含展开plan语义SHA，解码限2MiB、外层文件限200KB；
CLI verify/run自动解码再逐项核验，不能把外层JSON直接当展开plan。
