# PointMaze temporal OOD V5 协议审计

## 结论

V5 不能回答“短期视觉历史是否对动作排序有用”。它是一个有效的
**强 action-OOD stress test**，而不是隔离了 temporal-memory value 的
repairability test。原始 additive GRU readout 还结构性地禁止了 history
改变 candidate action effect；该实现缺陷已修正为 FiLM interaction。

修复后的 GPU debug replay 仍未通过：recurrent 的 held regret 为
`0.05469`，action-only effect predictor 为 `0.02308`，paired gain 的
95% bootstrap CI 为 `[-0.05328, -0.01066]`。该 replay 使用已经打开过的
outcome panel，只用于定位，不能作为确认性结果。

## 已证实的协议问题

1. 原 readout 是 `b(history) + g(candidate)`。任意两 action 的 effect
   差都与 history 无关，所以无法表达 within-history action
   identifiability。
2. train bank 只有 16 个独立候选序列（尽管有 1,024 行 root×candidate
   样本）；held bank 是 16 个新序列。训练 pulse norm 是 `[0.134, 0.593]`，
   held 是 `[0.282, 0.762]`，14/16 held candidates 含超过 train maximum
   的 pulse。这把 candidate extrapolation 与 memory hypothesis 混合。
3. `action_only` 只是不输入视觉/状态的 **effect predictor**；最终 cost
   仍以 `source_state_raw[:2]` 与 goal 计算。因此所有臂共享特权 XY
   task readout，不能标作纯视觉在线规划。
4. 原 panel 生成阶段就写出了 outcome labels。训练代码没有读取它们，
   但这不等同于 outcome file 在模型选择前不可访问。
5. width/temperature 选择和 authorization CI 复用同一 calibration roots；
   需要在 V6 分成 selection 与 authorization 两段。

## 运行时修复

本机 CUDA allocation 可以通过，但系统 `LD_LIBRARY_PATH` 先加载 cuDNN
9.13，而 PyTorch 2.12+cu130 的 GRU 要求 bundled cuDNN 9.20。新增：

```bash
experiments/run_with_torch_cudnn.sh .venv/bin/python -m \
  experiments.gpu_preflight --device cuda --recurrent-smoke
```

该命令已在 RTX 5060 Ti 上通过，并要求 CUDA allocation 与实际 cuDNN GRU
都成功；常规 allocation-only preflight 不再被误当成 recurrent runtime
成功。

## 代码层防错修复（2026-10-21）

针对“负结果可能由实验设置造成”的审计，V5 runner 现已加入以下强制检查：

- 每个 root 必须拥有完整且相同的 candidate-id 集合；
- `source_frame`、八个历史帧、`[5,40)` 历史动作区间和 35 个严格
  pre-candidate actions 必须逐行一致；
- manifest 记录 train/held 的动作幅度与 pulse-norm 支持，显式暴露
  action-support shift；
- calibration roots 被确定性拆为 selection 与 authorization 两组，
  architecture/temperature 选择不再与放行 CI 使用同一批 root。

这不是对旧 V5 结果的重标定。旧目录缺少当前 runner 要求的 manifest，且
outcome labels 在历史 panel 构建阶段已经物化，因此旧 V5 结果仍只能作为
debug artifact。正式结论必须来自重新生成的、matched-support、outcome
延迟物化的 V6 panel。

本次宿主 GPU 复核也确认了这一点：用修复后的 runner 读取旧 V5 目录会在
manifest 校验阶段 fail closed，而不是静默重跑旧协议；RTX 5060 Ti、CUDA
13.0、cuDNN 9.20 的 recurrent preflight 本身通过。

## V6 的确认性设计

V6 必须使用新 root split：64 train / 48 calibration / 32 outcome，排除
所有旧 outcome roots。primary arm 使用 64 个 train、16 个 held 的
hash-disjoint candidate sequences，二者同为 5×4 pulses、同一幅度范围
`[0.28, 0.78]`；另设 magnitude-OOD stress arm，而不再把它混入 primary。

V6 将使用 pulse-aware candidate encoder 与 history-conditioned interaction，
把 calibration 分成 architecture/temperature selection 和独立
authorization，冻结后才生成 outcome labels。只有在这个 fresh panel 上
再次获得正向 paired CI，才可声称 temporal memory 有 action-ranking
收益。
