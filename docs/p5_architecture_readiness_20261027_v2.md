# P5 独立架构本地身份检查（2026-10-27 v2）

## 已保存结论

报告：`le-wm-main/results/p5_architecture_readiness_20261027_v2/report.json`。
本次仅新增文件，不修改 v1、既有进度文档、封存代码或历史结果；没有模型导入、
神经前向、CUDA、CPU 神经回退、训练、下载或完整解包。

不能继续把本地 JEPA 资产笼统记作“只有 encoder”：三份 `.pth.tar` 实际均为
PyTorch ZIP，归档元数据包含 `predictor`、`module.predictor_embed.weight`、
`module.action_encoder.weight`、`module.predictor_blocks.0.adaLN_modulation.1.weight`、
连续 0–5 六层 predictor block 及 `proprio_encoder` 字面量。
这些是有界 pickle opcode 扫描结果，不是反序列化后的键结构、张量形状或加载成功证明。

| 本地 checkpoint | SHA256（与 v1 完全相同） | 静态身份 |
| --- | --- | --- |
| jepa_wm_pusht.pth.tar | 9beca3eafe0739c3b3adb5d734fa435ccbda0fea8a65d53d4cccec176aaaa0eb | JEPA-WM AdaLN 动作条件动力学候选 |
| jepa_wm_pointmaze.pth.tar | a01d99c4592fbedf44af076cf4c339de230c56f9f377c7559f584b97569b59bc | 同架构、不同任务权重 |
| jepa_wm_wall.pth.tar | 8efb0623cfba1cb3ca210de26f7579c83dd24936635f11989c515afcb23bea1e | 同架构、不同任务权重 |

源码定位：`external/jepa-wms/app/vjepa_wm/utils.py` 的 `pred_type == "AdaLN"`
分支调用 `vit_predictor_AdaLN`；实现位于 `app/plan_common/models/AdaLN_vit.py`。
三份对应 pt/mz/wall YAML 均已作为精确路径和当前字节 SHA 保存。
PushT 配置声明 DINOv2 ViT-S/14、384 维 patch 特征、六层 AdaLN predictor、
内置 action encoder、16 维 proprio embedding、tubelet=1、frameskip=5。
归档参数名与这条源码/配置链一致，支持独立动力学候选身份，而非仅凭文件名判断。
尚未执行形状或严格 state_dict 校验，不能将静态一致写成完整运行认证。

对照四份 stable-worldmodel 原始配置，其 target 都是 `stable_worldmodel.wm.lewm.LeWM`，
使用 192 维 predictor；第二环境权重并不等于第二架构。
三个既有 DINO-WM dynamics 路径仍缺失；缓存 `Dinov2Model` 及 safetensors
只能提供视觉 encoder，不能替代缺失的 DINO-WM dynamics。

## 资产存在，但不是已认证完整原生运行链

JEPA 源码 HEAD 为 `13cf1d9c7e476f53c17714d2e0f1dc239a883ce0`，五个文件有本地修改：
`pusht_dset.py`、`dino.py`、`prop_embedding.py`、`vit.py`、`vit_enc_preds.py`。
报告保存当前源码 SHA，不能用 Git commit 或旧 acquisition manifest 的 clean 状态替代。
PushT 归档 SHA 同本地官方资源 acquisition manifest 记录一致；这不是独立远端真实性核验。

`data/jepa_wm_osf/extracted/pusht_noise/val` 实有 states、rel_actions、velocities、
seq_lengths 文件和 21 个 mp4；本次只哈希小文件、列视频名称，不加载数据内容。
本地包分发元数据：torch 2.12.0、torchvision 0.27.0、tensordict 0.13.0、
transformers 5.9.0、einops 0.8.2、PyYAML 6.0.3、decord 0.6.0、opencv-python 4.13.0.92。
包元数据存在不证明可导入、CUDA 可用或二进制兼容。

原生接口不同：PushT reader 使用 relative action /100，再由 evaluator /ACTION_STD；
每个预测 packet 为五步二维动作。原生 JEPA 还有四维 agent proprio（位置和速度），
不能冒充 LeWM 的 RGB-only 部署，也不能静默移除 proprio。384 维 patch 特征不能
直接套入 LeWM 192 维任务度量/上下文。既有 temporal evaluator 的诊断不等于 P5 Full-D2P。

仍需逐项补齐：

1. 安全的受限 checkpoint / dataset loader，以及完整键、形状对应验证。
   现有 `fetch_checkpoint` 没显式指定 `weights_only`，且模块装载 `strict=False`；
   dataset 存在普通 `pickle.load`。这不等于断言当前 torch 默认不安全，但不能据此
   声称显式受限加载及完整参数认证已成立。
2. 审核冻结上述五份修改源码，验证 HF DINOv2 offline shim 对官方 encoder 的等价性。
   shim 存在、权重缓存存在，不等于官方原生 encoder 已验证。
3. 冻结原生 RGB / proprio / 动作单位与历史 token 绑定，声明特权信息边界。
4. 新的 CUDA-only 安全验证入口，保存严格加载差异、形状、有限且 action-sensitive
   预测及实际调用收据；不得 CPU 神经降级。本 worker 未执行。
5. 独立架构 Full-D2P adapter、匹配协议/任务度量/资源核算及尚未满足的上游科学条件。

因此报告保存 `native_full_d2p_runtime_ready=false`、`cuda_validation_command=null`、
`P5_admission=false`。不提供一个仍需普通 pickle 加载、使用宽松键匹配且只有
temporal 诊断的旧命令作为“可用完整原生 P5 验证命令”。本次不是独立 P5 成功。

## 检查器与实际验证

新增 `le-wm-main/experiments/spectral_reachability/p5_architecture_readiness_20261027_v2.py`。
只使用标准库。ZIP 打开前检查 EOCD/central-directory 大小与成员数；拒绝 traversal、
绝对/Windows 路径、链接/特殊成员、重复成员、加密、过高解压比和超限成员/总大小。
最多 512 MiB 输入、4096 成员、4 MiB central directory、1 GiB 声明解压总量、
64 倍解压比、1 MiB data.pkl 和 200000 opcodes；不支持 tar/旧 pickle 格式时明确拒绝。
不执行 pickle GLOBAL/REDUCE；字符串只能形成候选身份线索，不能抵抗身份元数据伪造。
不打开 tensor storage 成员；全文件 SHA 会顺序读取原始容器字节，和张量装载不同。
输出以 exclusive-create 写入，已有报告会拒绝覆盖。固定路径范围不代表全盘资产穷尽。

在 `le-wm-main` 目录实际运行：

```bash
.venv/bin/python -m pytest -q tests/test_p5_architecture_readiness_20261027_v2.py
.venv/bin/python experiments/spectral_reachability/p5_architecture_readiness_20261027_v2.py --output results/p5_architecture_readiness_20261027_v2/report.json
```

23 项非神经测试通过（0.09 秒），包括恶意 GLOBAL/REDUCE 不执行、路径穿越、符号链接、
重复成员、压缩膨胀、超限 metadata、central-directory 预分配限制、坏 pickle、encoder-only
拒认、伪造 marker 不提升为张量认证，以及无神经/反序列化/解包导入。
首次测试中 AST 检查错误地拒绝安全的 `json.loads`；只修正本轮新测试以限定允许 JSON，
最终完整重跑通过，未放宽 pickle 或架构认证要求。

experiment-code 技能用于可复算记录与结果边界；任务明确禁止训练/神经实验，故未采用其
训练目录或效果图要求。测试是工具安全/逻辑证据，不能替代 CUDA 或效能证据。
