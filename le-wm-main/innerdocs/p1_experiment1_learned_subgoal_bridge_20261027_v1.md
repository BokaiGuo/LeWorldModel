# Experiment1 learned subgoal：无训练接入与资产配对门

2026-10-27。状态：**源码合同可实施，配对资产未就绪，未执行模型或物理**。本文件只新增设计，不改变已封存源码、normalizer、门槛或结果。按 scientific-critical-thinking 区分接口存在、资产认证、真实调用与科学比较；不能以方法名代替真实 generator 调用。

## 1. 可直接决策的结论

最小组件是 **generator + 与它同坐标的 LeWM encoder/projector/低层 dynamics**，不必引入 action prior。固定官方源码已有 `lewm_generator`：Gaussian CEM 对生成的局部 latent goal 评分。完整 SAGE 同时引入 action-prior proposal，不是只改变 learned-subgoal 的干净 Experiment1 baseline。

若要求不额外读取物体真实状态，优先请求 `pusht_visual_generator.pt` 与 `pusht_lewm_object.ckpt` 配对资产；普通 `pusht_generator.pt` 的 lowdim 权限不能默认为 RGB-only。官方 `visual-pusht` suite 仅列 `sage`，所以“visual generator + Gaussian CEM”须标为**新组件消融适配器**，不能冒称原发布 suite 已验证实现。必须检查真实 checkpoint 的 model_config/stats/run_manifest 才能确认视觉权限；把缺失 state 填零不是权限证明。

最短官方已配置且非平凡生成的 PushT goal-offset 是 **50 个环境步，schedule [25,25]**。25 步 schedule [25] 完全走 final-goal fallback；当前 H2=10 步同样不能据此形成 learned-prior 证据。不能强设 tau=5，也不能将任意 procedural placement goal 改名为 dataset future-50。

## 2. 本轮只读证据与精确配对缺口

官方来源为已保存元数据目录：
`/home/gbk/code/LeWorldModel/le-wm-main/results/p2_learned_subgoal_asset_metadata_20261027_v1/`。
其 report SHA256 `4234e1deff0d45c62417c34498ae1cbb3b627a39c4fd23f26f4690e34e8559c6`。
源码 commit `43c0a655bd2bbd33d8047c7588360e193671e9f0`；下述源码路径均相对此目录的 `sources/`，保存为不可导入 `.source.txt`。

| 组件 | 源码声明的文件 SHA256 | 当前证据 |
|---|---|---|
| pusht_lewm_object.ckpt | `4656a13b878d83e417ba97d96257e77fdadc7a9ecc997031eb611c6494a4c270` | 72,334,304 bytes；assets 标 local_only，未找到本地文件 |
| pusht_generator.pt | `0b3647a3a41435969d750ec58176ef5f92a419c4eacae2b5cda74b35e63f90da` | 233,878,994 bytes；源 registry 声明，不是本地/HF实测 |
| pusht_visual_generator.pt | `8c834bac6f63f6dfa092140a9b2687a13cfed35b1c658bbeccd85d41fd297adb` | 229,593,266 bytes；local_only，未找到本地文件 |
| 本地 pusht/lewm/weights.pt | `48938400ae3464c9680731287f583a9cb516f55a8ec64ea13a91be47fb15b607` | 本轮重新读取字节计算；72,290,721 bytes |
| 本地 pusht/lewm/config.json | `2564086e961e7b5c7c04dffc451091115b389a590645ff19653c64fd0bc16e09` | 本轮重新计算；tiny ViT14、224 RGB、192维、3frame、10维 action encoder |

本地两文件完整路径前缀为 `/home/gbk/.stable_worldmodel/checkpoints/pusht/lewm/`。
只读精确 basename 检索范围为 `/home/gbk/.stable_worldmodel`、`/home/gbk/.cache/huggingface`、`/home/gbk/code/LeWorldModel`，maxdepth6；上述三个官方配对文件与 pusht_action_prior.pt 均未命中。这是有界检索，不是全磁盘不存在的证明；空的 `checkpoints/models--pusht--lewm/` 不能充当 HF revision。

**不同容器大小/hash不证明参数不同；同为192维也不证明坐标一致。** 缺失的不是已知官方文件 hash，而是以下配对证明：

1. generator 实际到手字节符合上表 hash，及其可信反序列化后的 `model_config`、`stats`、`run_manifest` 的规范化内容 hash。当前均 unknown，不填写猜测值。
2. generator 训练所用 encoder/projector（及 pred_proj/输出选择）的逐 tensor 语义 hash、模型配置 hash，与选用低层模型对应项相等；若直接采用官方声明的配套 object，仍记录实际加载实例的来源与配置。不得仅比较维数。
3. 图像 resize/uint8→float/ImageNet、pooling、token选择、时间 stride、precision 的合同 hash；固定 RGB probe 的两条编码路径数值核对。SAGE runtime 有 bilinear resize/align_corners=False；不能默认为本地 native transform 完全等价。
4. 权重 immutable revision/授权与依赖 closure。此前 HF API 不可达，revision/真实 inventory/权重 license 尚 unknown；源码 MIT 不自动覆盖模型。Hi-LeWM 2,299,863,924-byte archive 未读取，其内部配对 hash unknown，不是本次小组件方案的可用替代物。

不反序列化本地权重来猜测配对；不拟合 latent mapper。若无法提供上述相等证书，就让所有新比较臂共同使用认证后的官方配套低层，不能复用旧 backbone 下的预测或物理 endpoint 作 matched 比较。

## 3. 精确接入合同：真正生成，而非 final goal 改名

接点 `sage__models__subgoal.py.source.txt` 的 `load_subgoal_prior(path, device)`，要求 `prior_type == pusht_subgoal_prior_v1`，加载 `model_config/model/stats`。官方使用 `torch.load(..., map_location='cpu', weights_only=False)`：先认证来源和 hash，再经 MAIN 许可可信 pickle；CPU 反序列化不是 CPU forward，本方案所有神经 forward 仅 CUDA。

真实调用为：

```python
out = generator(history_latents, goal_latents, normalized_lowdim,
                goal_offset_steps, subgoal_offset_steps)
subgoal = out['prediction']
```

history `[B,Nh,D]`、goal `[B,Ng,D]`、lowdim `[B,L]`、offset 各 `[B]`，prediction 与 goal 同形。真实 D/L/Nh/Ng 由认证 checkpoint 与编码输出核对，不硬塞默认 toy shape。`predict_residual_from='goal'` 时 prediction 已是 goal+delta，不可再加 goal；必须保留 prediction/delta/base 原数组。

接点 `sage__eval__pusht.py.source.txt`：

- `_history_latents` 读取 `_proposal_pixels_raw`，缺失即报错；使用真实三帧 RGB history。
- `_final_goal_latents` 读取 `_proposal_goal_raw`，与训练时 proposal normalization 一致，再走配对 encoder/projector。
- `_normalized_lowdim` 默认 state/proprio；必须按 checkpoint 的 lowdim_keys/mean/std 认证输入，不默许静默零填充。
- `_local_goal_latents` 在 `rem <= tau` 时直接返回 final goal，否则真正调用 generator；缓存 key 是 `(env_id, plan_call, remaining, duration)`。
- `lewm_generator` 的 solver 是 GaussianCEM，目标必须实际来自 prediction；不传 action prior，不混入 action-prior top-mode。

新桥保留 attempted/completed/failed generator forward 与 cache hit/fallback 分账。首次固定查询 rem=50,tau=25，要求 generator_attempted=completed=1、fallback=0；逐 tensor 保存输入输出及 source/root/goal/hash 绑定。hook 在 forward 前记 attempted，返回后保存数组再记 completed，异常不改成 final goal。预测恰好等于 goal 不自动证明没调用；调用 receipt 与数组来源才是调用证据。缓存重复评分不得冒称多次独立生成。

## 4. 正式时域与可比边界

`configs__paper.json.source.txt`：goal-offset horizons 为 `[25,50,75,100,125,150]`；option durations 为 `[15,20,25]` 环境步。

| goal-offset | schedule（环境步） | 首次是否生成 |
|---:|---|---|
| 25 | [25] | 否，rem=tau |
| 50 | [25,25] | 是，50>25；第二阶段通常 final fallback |
| 75 | [15,15,15,15,15] | 是 |
| 100 | [15,15,15,15,15,25] | 是 |
| 125 | 七个15，随后20 | 是 |
| 150 | 十个15 | 是 |

“训练 goal-offset 列表包含30/40”等不等于正式 evaluator 配置支持这些 horizon。50不是50个模型 packet：action_block=5，tau25=5个模型 packet。官方 PushT environment budget=2H，H50 最大100执行步，不是50步；不能隐去超出第一轮 schedule 的 final-goal执行。官方 planner 300 candidates×30 CEM rounds、elites30、history3、frameskip5、bf16；不能与旧80predict/H2称同算力。

官方 ScheduledPolicy 会重复首帧填初始 history，其语义不能冒充当前 source 的三次真实观测。保留 task identity 与 observation-history identity 两层；在所有新臂统一历史规则。当前两个 procedural roots 仅可做明确标记的 OOD 接口诊断，不能未经认证赋予官方 dataset future-50 的时间语义。

## 5. 坐标兼容且不训练的两条可实施路径

**优先路径 A：共同采用官方配对 backbone。** 取得认证 visual generator/object，按 source-native 原始 RGB→配对 latent→generator prediction→同 backbone GaussianCEM。baseline arm 在完全相同 context、schedule、Q/距离函数、candidate pool/seed、raw动作域上只改为 final-goal target；子目标臂保留真实生成 overhead，其他内部方法若加入也必须重编到同一 backbone。生成器 latent 不是 raw RGB，也不是动作，不反解图像、不直接当command。

**路径 B：保留当前本地模型，仅在逐 tensor/预处理等价证书通过后接入。** 配对证书未通过前不执行 learned-prior 科学比较。禁止训练 mapper/重新校准 Q 来弥补坐标差；若不能等价，回到路径 A。

raw relative commands 是物理边界。SAGE train mean `[-0.007726192940026522,0.006876377854496241]`，std `[0.2083914875984192,0.206622913479805]`；本地公共映射 mean `[-.0087,.0068]`、std `[.2019,.2002]` 不同。分别显式实现 `a_raw=a_norm*std+mean` 与逆变换，逐 packet 浮点 roundtrip/时间排列验证；不再 /100，不替换旧模型 normalizer。若原生 env 将 relative command×100，这属于 env 运动尺度，不是第二次标准化。原来的 tanh/logit L2-ball 动作域不是官方 GaussianCEM 域，使用它就是新适配器，应保持所有比较臂一致并明示区别。

## 6. 最小下一次 MAIN 主机实验：一条真实 generator forward

当前**尚不能执行**：缺 generator、配套 object/等价证书与认证的 preencoded packet。下一最小实验不需要物理、action prior、CEM、训练或2.3GB archive；只证明真实 generator 调用与坐标接口。完整官方源码包也未安装；当前 `.source.txt` 不能直接当模块执行。以下命令是资产齐备后可执行的接口 smoke，不是当前已存在的研究 runner，更不构成 Experiment1 性能结果。

先由 MAIN 创建独占小输入目录（非本轮动作），包含 `binding.json` 和 `input.npz`。binding 必须绑定官方 code commit、generator/object/config/预处理/packet 的实测SHA，以及原始 RGB/身份/offset provenance。input 保存由认证配对编码器产生的 `history_latents/goal_latents/normalized_lowdim`，offset 固定50/25。不能用旧未证等价的 latent 冒充。以下调用假设 MAIN 已审查该 binding；`SAGE_ROOT` 是上述固定 commit 的完整可导入源码，`SAGE_PYTHON` 是其审核过的环境，两个 SHA 由 MAIN 从已批准 binding 设置，不是下载所得值的自我背书：

```bash
cd "$SAGE_ROOT"
PYTHONDONTWRITEBYTECODE=1 "$SAGE_PYTHON" -B - "$GENERATOR" "$GENERATOR_SHA256" "$PACKET" "$PACKET_SHA256" "$NEW_SMOKE_OUTPUT" <<'PY'
import hashlib, json, pathlib, sys
import numpy as np
import torch
from sage.models.subgoal import load_subgoal_prior
gen, gen_sha, packet, packet_sha, output = sys.argv[1:]
def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1048576), b''): h.update(chunk)
    return h.hexdigest()
assert sha(gen) == gen_sha and sha(packet) == packet_sha
assert torch.cuda.is_available(), 'No CPU neural fallback permitted'
dst = pathlib.Path(output)
dst.mkdir(parents=False, exist_ok=False)
r = {'generator_attempted': 0, 'generator_completed': 0, 'fallback': 0,
     'errors': [], 'status': 'interface_only', 'goal_offset': 50, 'duration': 25}
try:
    model, stats, checkpoint = load_subgoal_prior(gen, torch.device('cuda'))
    model.eval().requires_grad_(False)
    before = sha(gen)
    with np.load(packet, allow_pickle=False) as data:
        x = [torch.from_numpy(data[k].copy()).float().cuda() for k in
             ('history_latents', 'goal_latents', 'normalized_lowdim')]
    assert all(t.is_cuda for t in x)
    assert all(not p.requires_grad and p.is_cuda for p in model.parameters())
    r['generator_attempted'] = 1
    with torch.inference_mode():
        y = model(*x, torch.tensor([50.], device='cuda'),
                  torch.tensor([25.], device='cuda'))
    torch.cuda.synchronize()
    arrays = {k: y[k].detach().float().cpu().numpy() for k in
              ('prediction', 'delta', 'base')}
    assert arrays['prediction'].shape == tuple(x[1].shape)
    assert all(np.isfinite(v).all() for v in arrays.values())
    np.savez(dst / 'outputs.npz', **arrays)
    r['generator_completed'] = 1
    r['checkpoint_bytes_unchanged'] = sha(gen) == before
except BaseException as exc:
    r['errors'].append({'type': type(exc).__name__, 'message': str(exc)})
    raise
finally:
    (dst / 'report.json').write_text(json.dumps(r, indent=2) + '\n')
PY
```

本最小命令只保存 prediction/delta/base，输入另有已封存 packet；它没有 model in-memory state 全量前后认证、durable attempt journal、逐异常部分 tensor 留存，因此**仅接口 smoke**。生产实验桥还须实现这些机制，不能把 smoke 的 checkpoint_bytes_unchanged 等同 model 参数未变的完整认证。它也不认证输入 packet 的科学来源，配对 binding 审核是不可省的独立门。这里不虚构一个已经存在的 prepare/run CLI。

下一步完整官方 evaluator 的真实入口是 `python -m sage.eval.pusht --method lewm_generator --dataset ... --policy ... --generator ... --manifest ... --out-dir ... --seed 42 --device cuda --bf16`，但它会加载 dataset 并执行物理，且源码 `main()` 在 CUDA 不可用时回退CPU；不得直接作为 CUDA-only、安全物理 consumer 使用。需要 MAIN 新封存 wrapper、预检本地离线 dataset、固定有效 H50 manifest 与全部预算后另行授权。对 RGB-only generator 的该组合还必须明示新适配器身份。没有认证 manifest 时不提供可误执行的具体 dataset/query 路径。

## 7. 最小后续协议与剩余前置条件

先完成上述单调用与保存数组核对，再封存 H50、[25,25] 的共同 backbone 两臂协议。固定数据源 future-frame identity、RGB/history、normalization、同一低层求解器/原始动作域、Q、seed、候选数/迭代和真实成功定义；generator forward 独立记账。资源报告包含 encoder/projector、generator、predict/action encoder、AD/VJP、solve/SVD、replay、物理reset/step/render，不把同 candidate 数当同总算力。结果必须保留失败分母与 fallback，不能挑有利根。

明确还缺：认证可获取的 visual generator+配套低层或本地等价证书；generator checkpoint 内容元数据 hash；许可和 immutable revision；完整依赖/source closure；同坐标 packet；合法固定 H50 query manifest；新 guarded runner。缺项中只有包文件预期 hash 已知，不能写成“全部hash未知”。不需要重新训练，也不需要先取得 action prior。当前证据没有解除 Experiment1 learned-prior 性能缺口，不支持 P1/C1 科学准入。

本轮只读核对 SHA/JSON/文本/有界文件清单；未导入 torch、未运行下载来的测试、未下载任何新资产、未神经/物理执行。没有修改原 metadata/source/results。
