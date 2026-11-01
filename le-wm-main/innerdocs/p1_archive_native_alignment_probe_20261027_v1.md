# Archive → native action-time replay feasibility v1

## 固定问题，不是新增 benchmark

检验 factual archive 的 action-time 对齐与 native 重建可行性。不能从 archive 模型预测好、live
失配直接认定代码错/理论错；本实验不调用模型、不修改 action scale、不调 label、不选最佳 lag
追认修复。MAIN 先批准固定协议，再由本实现产生执行资产；本交付只有 source、RAM/mock tests、
本文档，未 prepare、未运行任何 physics/torch。使用 experiment-code 的预声明/持久化/故障测试
模式，不采用其超出本轮授权的训练、图表或正向结果要求。

固定 episode ordinal **[1,6229]**，分别 lag0、lag1，顺序 (1,0),(1,1),(6229,0),(6229,1)。
不同于原 offline [0,6228,12456,18684]，但不声称全局 fresh、训练外、独立 demonstration。
H5 attrs{}，collector 的 render/target/action 配置、block hidden velocities、训练成员关系未知。
全部未知写入 prepare/report；本实验的数值吻合或失配都不是因果证据，亦不自动选择 lag。

## Archive 固定读取及 source/time seals

H5 `/home/gbk/.stable_worldmodel/datasets/pusht_expert_train.h5`，只读打开，禁止网络/下载/大文件复制。
已声明 rows=2336736、episodes=18685；fields 严格 action float32[rows,2]、state floating[rows,7]、
pixels uint8[rows,224,224,3]、proprio float32[rows,4]、ep_len/ep_offset integer[episodes]、episode_idx/step_idx integer[rows]。
不能在 schema 改变后静默换 archive。
MAIN实际只读build捕获初版required集合遗漏proprio导致严格schema拒绝；prepare前修复并补回归。
proprio仅绑定shape/dtype，不读取其数据或用于控制。该工程错误不构成负实验结果。
第二次实际只读build捕获未注册本机已安装hdf5plugin像素codec的OSError；在读取前显式注册，
封存h5py/hdf5plugin入口源码。未安装、下载或改写H5；这也是prepare前工程错误。

每固定 episode 先核 ep_offset/ep_len 与前后边界精确衔接、无越界。长度<22时 **整 slot 两 lag
均拒绝**，保留四分支分母，不换 episode。合法 slot 读取：

- action local[0:21]，供 lag0 [0:20] 与 lag1 [1:21]；不 clip/tanh/除100/动作球筛选。
- state local[0:21]，其中 state0 用于 native placement，其余仅作 saved labels。
- pixels 仅 local[0,5,10,15,20]；不额外取模型输入或目标。
- episode_idx/step_idx local[0:22] 必须为同 ordinal 与连续0..21；相邻三个 ep_len/ep_offset 也绑定。

所有选中数组/帧记录 dtype、shape、C-order bytes SHA；slice/global row明确记录。整 H5 使用
绝对路径、device/inode/size/mtime_ns、schema、attrs绑定；读取前后stat相同。**这是 stat version
+ selected-slice SHA，不是整 H5 SHA 或认证的数据 revision**。不为此扫描整个大文件。

prepare 无输出数组拷贝，仅<128KiB JSON 元数据，包含固定 contract、source bindings、selected
receipts。verify/run 前和 run 结束后都重新读同 slices、重核 stat/SHA/完整 contract，不刷新采样。
source closure 使用 AST 递归封存仓库 imports，并封存新 source/tests/doc、native env/utils/spaces。
不调用所复用 moving-history source 的 build/episode，也不读取其 sourcepack/outcomes。

## 两 lag 的真实 native 路径

每 branch 单独执行同样的两次 reset，二者 seed 均为 **0**：

1. 首次 native reset 取得它本来的 `goal_state`。不是 archive 未来状态或 checkpoint goal。
2. placement reset 仅传 `options(state=archive_state0, goal_state=first_reset_goal)`，仍 seed0。
   不额外 poststate setter，不恢复 hidden block velocity/角速度，不清零/改写 state7。
3. 从 placement 返回的 **observed0** 保存 PNG/state/full-body snapshot，并对 archive[0]比较。
   不强制 exact：原 native `_set_state` 会应用末两维到 agent velocity、并推进一次 physics，
   所以 observed0 可能已与传入 state0 有差异。不能为追求初态吻合再次 setter。
4. 按 lag 固定20条 literal float32 raw 直接 env.step。native relative=True/action_scale100 保持，
   无额外 /100、不投影、不重新中心化。每次检查 native latest_action dtype/shape/bytes 与原 raw
   exact；在此 gate 前已保存返回 action/state 与新 PNG，故失败不抹去执行证据。
5. 每执行第 k 条实际命令后，只与 archive state[k]/pixels[k]共同 label 比较；lag1 不移动 label。
   正常逐步执行至20；returned terminated/truncated 时立即停，不 padding、不补剩余 commands。
   reset 没有 terminated 返回，因此 `initial_native_success` 仅描述，不新增静默 preterminal gate。

环境 `swm/PushT-v1`：RGB224、with_target=False、render_action=False、relative=True、action_scale100、
dt=.01、control_hz10，固定原 env 源 SHA：
`c6d2104adc368ff1c352edc0e35ac74da9f07fda3a748ddbef0ba6b7a2d1daa2`。
renderer 原生颜色/几何不改；不因蓝像素缺失或 archive RGB 不吻合筛掉分支。仅检查 RGB dtype/shape、
有限 state/body 与 render 不改变 dynamics。原 reward=-distance、terminated=eval_state 仍核对，
goal 不在实际动作后替换。原 native state7 的不足不能由 saved array 恢复完整隐状态。

## 比较与缺失层

在共同 checkpoints0/5/10/15/20保留：state value exact、dtype+bits exact、七维有符号差/绝对误差、
max absolute error；RGB exact、不同像素数、不同通道数、MAE、max absolute channel error。
像素差在 int16 计算，避免 uint8 环绕。state 差为 float64 原坐标逐维相减，不偷偷 wrap angle、
混合不同单位为“任务成功”，不设阈值。每项绑定 archive array SHA 与 actual PNG/file+pixel SHA。

每个实际 step（包括非 checkpoint）保留 state7、两 body position/velocity/unwrappedangle/
angularvelocity/force/torque/sleep 与 float64 bits；记录请求 raw float32 bits、返回 latest_action
dtype/shape/SHA/bits、原 native reward/termination。snapshot 不序列化 contact solver warm-start cache，
不声称隐藏缓存 identity。checkpoint 未到达明确列于 missing_checkpoints；不伪造终态或放进 full20
分母。报告固定 branches=4、expected_checkpoints=20，分别列 horizon_completed/native_stopped/
rejected_short_fixed_slot/engineering_failed/not_started。`alignment_probe_processed`只表示处理完
全部固定分支，**不等于四分支20步成功或对齐成立**；必须看每行状态和实际计数。

## 精确 native budget 来源与失败留证

已只读核对本地原 env.py：reset lines279–282依次 `_set_state(goal)`、`_set_goal_state(goal)`、
`render()`；line300附近再 `_set_state(state)`。`_set_state` lines514–527只设置 agent velocity、
agent/block pose 并真实 space.step一次；`_set_goal_state` lines529–530只赋 goal；step lines312–345
执行10个 internal space.step，无额外 setter/render。故 cap 不是猜测：

| 四分支全执行上限 | attempted cap |
|---|---:|
| reset（各2） | 8 |
| public_step（各20） | 80 |
| physics_step（2×reset+10×public） | 816 |
| explicit_render/PNG（各21） | 84 |
| reset 内 goal_render（各 reset1） | 8 |
| state setter（各 reset2） | 16 |
| **独立 goal_setter（各 reset1）** | **8** |
| 神经/torch/GPU/训练 | 0 |

两类 reset 返回均另记实际 state/goal/full snapshot；全部 render 返回另记 pixel SHA/full snapshot，
内部 goal render 不额外保存 PNG（所以 PNG 上限仍84）。所有资源 dispatch 前 attempted、正常返回后 completed；native failed 子调用和已经返回的内部
physics 留存，不退款。每次 reset 新 space 均在首次 setter前安装 hook。goal setter 独立记录
before/after body快照、requested goal与实际 goal；没有外部 goal attachment。“setter cap16”不含
goal setter，二者不混计。正常完成核 physics=2reset+10public、set_state=2reset、goal_render=reset、
goal_setter=reset；错误路径不强制 full census。factory/close各一次，工厂调用在 hooks前，仅允许
已封存原 native constructor（不推进 physics）；不声称任意 factory可精确计账。

每个 operation attempted/completed/failed event hash-chain+fsync；每步 PNG 和 full snapshot
先持久化，再做返回一致性 gates；commands的原 bits与 archive slice在第一条动作前 commit。
工程异常 fail-fast，首错不会被 failed-event journal/close/restore/final verify盖掉；后续分支
保持 not_started。末尾尽力写 final journal/report；若磁盘最终写失败，不保证 report存在或完整。

固定 wall300s，dispatch前检查，不能抢占单个 native调用，MAIN建议外层timeout350s。
总 output cap8MiB（包括journal/PNG/report/commands/execution_binding），final reserve128KiB，
启动需208MiB可用，后续每写保留200MiB floor。无覆盖、resume、删除或下载，不改封存来源。
Store/Ops/snapshot/image等复用冻结 moving-history source primitives；不运行其协议。

## MAIN 独占执行命令

cwd `/home/gbk/code/LeWorldModel/le-wm-main`；本交付 agent 只执行第一项 RAM/mock测试：

```bash
.venv/bin/python -B -m unittest discover -s tests -p 'test_p1_archive_native_alignment_probe_20261027_v1.py' -v
.venv/bin/python -B -m experiments.spectral_reachability.p1_archive_native_alignment_probe_20261027_v1 prepare --output results/p1_archive_native_alignment_probe_20261027_v1_prepare01.json
.venv/bin/python -B -m experiments.spectral_reachability.p1_archive_native_alignment_probe_20261027_v1 verify --prepared results/p1_archive_native_alignment_probe_20261027_v1_prepare01.json --prepare-sha PREPARE_FILE_SHA
timeout 350s .venv/bin/python -B -m experiments.spectral_reachability.p1_archive_native_alignment_probe_20261027_v1 run --prepared results/p1_archive_native_alignment_probe_20261027_v1_prepare01.json --prepare-sha PREPARE_FILE_SHA --authorize-physics --output results/p1_archive_native_alignment_probe_20261027_v1_main01
```

测试用 sparse logical H5和 fake dynamics，所有数组/Store均RAM；不写大tmp、不造实际物理证据。
测试成功只验证工程/算术/计账合同。实际 MAIN 运行后才知道可重建性，无科学 winner/P1 admission。
