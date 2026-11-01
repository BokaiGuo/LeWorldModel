# 三臂真实saved输入打包器 v1

仅本stem source/tests/本文；不改runner或封存证据。使用experiment-code技能的输入合同和轻量
回归流程，不运行神经/physics/下载，不训练，不派生。不是E1/P1/P2通过，不授予物理权限。

## 接线与身份

复用冻结runner `agent_subgoal_gap_runner_20261028_v1.prepare/verify`。以MAIN02严格消费者
receipt字节SHA `c7d183f5b18d1bfacd81f855025efdbbb403f97e5442650970080766db17088b`为
信任入口，重核其全部依赖/source字节，不重复数组评分审计。继承四根八臂480候选真实CUDA
报告；四根顺序固定，不增根、不选结果。E1保留四根×三臂=12槽，源8臂不是E1三臂结果。

提取完整root/context、原mean/std、原Q配置、实际checkpoint路径和SHA、native evaluator源码，
以及next的controller/native success/restore配置合同。旧配置只作证据，不冒充统一controller。
每根compiled query固定conditional arm候选index0，不读取winner或cost选动作；包含原完整
动作见证及动作生成checkpoint target，不宣称重新调用generator或独立neural replay。

四根原Q不同，分别保存q_0..q_3；runner要求共同Q，因此assets.q保持null，不能把root0的Q
套在其他根或修改runner。Q的原语义摘要与文件字节SHA分别记录（原metric含self SHA）。normalizer为原
canonical字节；encoder与model同为原冻结checkpoint的真实字节绑定，不另造encoder权重。
root_id与root对象canonical SHA分栏，完整history原样保留，不把相同latent维度当配对证明。

真实native goal7/start snapshot缺失：goal文件只描述已保存goal RGB摘要/latent和明确null
native_goal7；其摘要是描述文件身份，不是已认证native taskgoal。renderer outline不是taskgoal。
缺共同controller/observation/budget等时shared身份引用实际写出的unsupported descriptor，
对应assets仍null，绝非未来文件SHA或伪造可用资产。physical/native success保持null。

## fail closed 与缺项

输出spec可直接交原runner prepare，必为blocked；不提供override解除本版缺项。learned
checkpoint/pairing、arbitrary rule、compiled producer rule、adapter source、统一controller、
observation合同、预算资产、restore contract均不伪造。HiLeWM原probe failed_preserved、
unpickled=false独立绑定，失败不是0%成功率；没有usable learned prior就不运行三臂。

caps仅dispatch-disabled上界（callback=1，其余known=0，unknown=null），不是实际零成本、
科学等预算或建议实验budget；预算asset=null确保禁止dispatch。继承成本unknown，未造FLOPs。
native eval_state原语义位置包含agent+object；object-only无改善不能代表完整任务失败。

后续需要MAIN真实共同CUDA adapter/低层controller、精确native状态/goal7/history/warmup/
renderer恢复证据、三份真实proposal输入和配对prior、共同成本上限/hook、P1/freshness权限。
本版仅把已有资产接入prepare，不替代MAIN审核或真实callbacks实现。

## 命令

```bash
cd /home/gbk/code/LeWorldModel/le-wm-main
OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m pytest -q -p no:cacheprovider tests/agent_subgoal_gap_assets_20261028_v1_tests.py
.venv/bin/python -B -m experiments.spectral_reachability.agent_subgoal_gap_assets_20261028_v1 prepare --output /tmp/agent_subgoal_gap_assets_20261028_v1_MAIN01
```

output必须不存在且名字以本stem起始。CLI只写新metadata，不导入任何执行器、模型或环境。
失败保留已写文件，不清理、不重试覆盖。inventory/spec/prepared都有实际字节SHA。
MAIN可将CLI输出的spec.path/spec.sha256直接传给原runner的prepare --input/--sha256；
prepared文件的字节SHA与API canonical SHA仍按runner原合同区分。
本packager verify --input ABS_INVENTORY --sha256 BYTE_SHA重核依赖/生成目标/runner prepare。
本版不是fsync journal或实验执行器，不声称硬wall/output限制或真实执行成本计量。

## 实际验证记录

新14项RAM测试与原runner31项联合 **45 passed in 0.41s**。真实封存metadata已只读解包并
调用原prepare/verify，4 roots/12 E1 slots，status=blocked。首个saved01在JSON键序接缝验证
失败，文件保留；修复仅在本打包器首次prepare前规范化JSON，saved02成功，不改runner。
最终冻结后请用文中MAIN01新目录命令生成绑定最终source/tests/doc的inventory。
测试与saved metadata不构成三臂实验或科学证据；没有执行CUDA、CPU神经或物理。
