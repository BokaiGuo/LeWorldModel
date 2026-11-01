# Constructive reachable source：独立 saved-data 审计 v1

仅新增此stem的源码、tests、innerdoc及独立小报告。既有producer及其tests/doc、prepare、actual
sourcepack全部不修改。审计器不import producer、conditional generator、gym、Pymunk、原生环境或模型；
不调用producer.verify/read_prepared/episode，不执行reset/render/step/neural。依赖仅stdlib、NumPy、Pillow。
experiment-code用于证据持久化与失败路径测试；不包含训练、绘图或实验执行。

## 输入与独立算法

MAIN prepare01 SHA：`98ffb806018a270f5ef27ffa76be929393a74355c213bc274bf158c72d80b086`。
MAIN actual sourcepack SHA：`9ad3a414fd2148258524d4a7e9d4f070f59532ec2e5943cfa61f9145b3a41bd9`。
prepare envelope由本审计器独立解码（外层<200KB、gzip展开≤2MiB），复算plan semantic SHA。
逐项重hash七项预先固定source closure及历史metadata，特殊metadata_authority_content_id只作语义摘要，
不误当文件路径，不声称独立重建其未保存authority对象。只读历史metadata，不刷新排除表，不按新outcome选根。
当前scope始终unknown_calibration_only。

检查项目：

- 源码配方独立实现：root state7提供radial，切向旋转90°；前5 logit=.5radial、后5=.25radial±.5tangent，
  sign固定随slot交替；float32后tanh，两个5×2 packet各自logit L2≤.75√5。六份commit逐项精确比较。
- 3242条journal逐条index、previous hash、canonical event SHA、prepare与尾锚；API attempt/completion
  用嵌套栈配对，reset必须两setter+一次goalrender，每setter一次physics，每publicstep十次physics。
  attempt不退款；失败栈保留，不能把completed补齐。全量与report及run_finished逐项一致。
- slot与phase严格单向、无重复/跳phase/重试；完整固定分母2–4，不能只报成功者。部分工程失败保留
  engineering_failed/not_started及first error；拒绝保留reason与slot_finished，不能得到blind admission。
- 每张fresh RGB均来自相应phase render完成之后；验证PNG文件SHA、像素SHA、224×224 uint8 RGB、
  canonical agent像素计数、canvas及state/snapshot。逐步own public action顺序和observed frame对应。
- 目标来自construction真实step20 state7+RGB；certificate逐entry核对goal、goalRGB、witness、future
  states、0/5/10 observed history与zero raw history。NPZ禁pickle、限制展开体积、字段白名单和dtype。
- 两次replay从自己的0/5/10 observed history开始，state/RGB精确匹配construction前缀，warmup block
  dynamics也精确匹配。仅允许10warmup后goalbind。未来动作必须独立复算witness，不能使用证书未来帧
  代替实际replay PNG；原生首次terminal后不得继续或padding。
- 原生success独立复算：完整agent+object前四维位置差norm<20；原生角差min(abs(d),2π-abs(d))<π/9。
  reward=-完整state7欧氏距离。另报正确wrapped诊断与旧角差bug，不重写官方success；本批无该bug。
- blind NPZ仅允许rgb_history/raw_action_history/goal_rgb，逐数组绑定replay0自身历史与construction目标；
  禁state、witness、future entry。root identity由root+goal float64 bytes复算，排除历史和本批重复根。

所有精确比较是saved-array与同recipe确定性replay的一致性核验，不外推到跨硬件/环境的数值保证。
不声称独立再模拟了源采样、Pymunk积分或随机goal生成；这些只由冻结source和记录绑定。

## 实际结果与资源

实际输入全链核验一致，2/2固定候选均有两次原生成功replay，各执行10个future步。
6episodes各10warmup+10future=120public；12reset、1224physics、126explicit render、12goal render、
24setter、4goalbind，attempted=completed，errors=[]。126张PNG、6commit、2certificate、2blind payload。

| slot | root SHA | 物体位移(px) | wrapped物体转角(rad) |
| --- | --- | ---: | ---: |
| 0 | f7aec52327fa6a6c8fb444919ba426a467ec2382b57da6d8a4f3701f82c65962 | 94.66802950447014 | 0.6401514987084118 |
| 1 | f47e9958781f7eac4c810f9dfe24232b635f41d5b4c56423d8c2623f9fd5a328 | 149.18110475210253 | 0.8106228472360266 |

这证明本新render/source recipe的真实合法动作见证及其saved重放一致，不代表旧随机goal可达、
模型能解题、两阶段增益或P1通过。目标由witness端点构造是预注册设计本身，不能把重放成功当盲测结果。
hidden state仅限dataset构造/评估；后续部署selector只能得到blind字段，不能把journal/certificate当输入。

## v1证据局限（不改已封存producer）

1. commit JSON在producer源码顺序上先写后执行，但journal没有执行前commit文件hash事件。
   本审计验证内容及动作一致，**不独立证明durable temporal precommit**；文件mtime不能代替证据。
2. goal_bind journal只有API完成标记，无before/after snapshot和逐step info.goal。
   本审计复算state/reward/标签且核对PNG，但**不独立证明attachment瞬间状态不变或info.goal返回值**。
3. 无saved timestamps可独立核实时限/fsync时序；resource计账可验证，wall由封存源码和HOST控制。
4. 某guard失败前没有fresh observation事件；partial/rejected保留原始原因，但不能假称已从数组独立
   复现该失败。缺失必要成功证据则unverified_or_inconsistent，绝不补假capability。

机器报告分别标出上述未证项。`saved_evidence_consistent`只指可核对的saved内容，不是无条件科学准入。
`sourcepack_complete=true`意为固定slots已处理，允许0个reachable；不是survivor-only成功声明。
原P1严格门完全保留：direct困难、非贪心/能力改变、两阶段增量、固定模型独立盲测及新根确认均未由本审计建立。

## 命令与状态

cwd `/home/gbk/code/LeWorldModel/le-wm-main`；测试仅内存mock及saved读取，无tmp/fixture落盘。

```bash
.venv/bin/python -B -m unittest discover -s tests -p 'test_p1_constructive_reachable_saved_audit_20261027_v1.py' -v
.venv/bin/python -B -m experiments.spectral_reachability.p1_constructive_reachable_saved_audit_20261027_v1 \
  --prepared results/p1_constructive_reachable_source_20261027_v1_prepare01.json \
  --prepare-sha 98ffb806018a270f5ef27ffa76be929393a74355c213bc274bf158c72d80b086 \
  --sourcepack results/p1_constructive_reachable_source_20261027_v1_main01/sourcepack.json \
  --sourcepack-sha 9ad3a414fd2148258524d4a7e9d4f070f59532ec2e5943cfa61f9145b3a41bd9 \
  --output results/p1_constructive_reachable_saved_audit_20261027_v1_audit01.json
```

报告exclusive新建，≤200KB且保留200MiB；重复审计时省略output读stdout，或显式NEW report路径，不能覆盖。
无actual时返回actual_sourcepack_missing和exit2，不生成占位成功。证据错误exit2。
saved_partial_consistent_not_admitted表示失败现场账本一致，不准入；P1_admission无条件false。
本版本20项测试包含10项算术/mock与10项actual篡改/回归；actual不存在时其10项明确skip，不冒称实际通过。
