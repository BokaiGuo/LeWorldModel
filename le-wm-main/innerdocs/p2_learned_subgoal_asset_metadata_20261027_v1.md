# Experiment 1：官方资产元数据实际解除审计

2026-10-27。新增独占结果目录 `results/p2_learned_subgoal_asset_metadata_20261027_v1/`，不改任何旧封存或总账。
按citation-check技能区分官方来源存在、源码声明、真实远端文件可用与已验证兼容性。
27次官方API请求均经require_escalated请求权限；累计接收body **419,034 bytes**，包含重复读取和HTTP错误正文，低于5MiB。
不跟随redirect、不发送凭据、不读zip/权重、不执行下载来的代码，不运行神经/物理/训练。

## Hi-LeWM：版本化记录真实可访问，但仅分发大压缩包

官方 `https://zenodo.org/api/records/21353240` HTTP200：题名明确对应Hi-LeWM，
DOI `10.5281/zenodo.21353240`，concept DOI `10.5281/zenodo.21353239`，record revision3。
这是版本特定记录，不是源码commit；记录级许可声明为CC-BY-4.0，不能据此声称读过包内所有许可。

唯一文件 `package.zip`：**2,299,863,924 bytes**；
MD5 `aaf78b8b4fd6f5d1f5e720927951152f`；文件version ID
`79889c57-8c8d-4704-85db-baf33e7d976a`。
元数据描述包内有`code/`和`checkpoints/`，但不是实际包内清单。
files API标`Total uncompressed size limit exceeded`；其官方container清单API实际HTTP500，错误保留。
没有用range/镜像绕过，没有下载压缩包。确切高层/低层checkpoint文件名、配对hash和解包空间仍unknown。
MAIN若选择此路，需要另行批准2.30GB压缩包及未知解包空间，或取得作者分离的小型manifest/组件。

## SAGE：源级精确清单已取得，HF可用性仍未确认

固定commit `43c0a655bd2bbd33d8047c7588360e193671e9f0`，git commit明确指向tree
`da65a63b807d24cc8d59a224f35798e63acc40af`。下载文本保存为不可导入的`.source.txt`，
逐一验证Git blob SHA1与内容SHA256。MIT LICENSE全文已读取；不将源码许可外推为HF权重许可。
HF官方`https://huggingface.co/api/models/CLTRAY/SAGE?blobs=true`实际失败`Network is unreachable`，
未换镜像或重复尝试。因此HF真实inventory、immutable model revision和权重许可仍unknown。

固定源码`configs/checkpoints.json`声明以下文件，完整预期SHA在report.json；**不是本轮HF文件实测**：

| 文件 | bytes | epoch |
|---|---:|---:|
| pusht_generator.pt | 233878994 | 6 |
| pusht_action_prior.pt | 163404922 | 3 |
| pusht_far_action_prior.pt | 163404602 | 3 |
| cube_generator.pt | 233981650 | 6 |
| cube_action_prior.pt | 164984570 | 3 |
| cube_far_action_prior.pt | 164984186 | 3 |

`configs/suites.json`确认LeWM-PushT配套低层为`pusht_lewm_object.ckpt`。
`configs/assets.json`声明其72,334,304 bytes、SHA256
`4656a13b878d83e417ba97d96257e77fdadc7a9ecc997031eb611c6494a4c270`，并标**local_only**。
同一资产表还将RGB-only `pusht_visual_generator.pt`（229,593,266 bytes）、
`pusht_visual_prior.pt`（160,164,534 bytes）标local_only。不能将这个标签写成“已证远端不存在”，
也不能因含HF repo_id就宣称已发布可下。源码列出其它任务published资产不解除PushT的缺项。
RGB-only最小候选是低层object+visual generator，共**301,927,570 bytes**，不必先下action prior；
但这只是源码声明的大小，不是已经可以下载的bundle。

## 真正可复用接口与当前不能直接跑的原因

源码已有`lewm_generator`组件消融：zero-mean Gaussian CEM对生成子目标评分，不使用action prior。
它比将完整SAGE动作proposal混入比较，更接近原Experiment1的共同低层控制器问题。
已读的真实类与签名是：

```text
sage.models.subgoal.PushtSubgoalPrior.forward(
  history_latents, goal_latents, lowdim,
  goal_offset_steps, subgoal_offset_steps
) -> {prediction, delta, base}
```

输入history `[B,Nh,D]`、goal `[B,Ng,D]`、lowdim `[B,L]`、两个offset `[B]`；输出prediction同goal形状。
D/L、精确token坐标、model_config、stats必须来自认证checkpoint，不根据类默认值或测试toy形状猜测。
loader要求`prior_type=pusht_subgoal_prior_v1`与`model_config/model/stats`，源码使用
`torch.load(weights_only=False)`；本轮绝未调用，未来需先认证资产并单独审查可信反序列化。

**关键短时域阻断**：`sage/eval/pusht.py:310`的`rem <= tau`直接使用final goal而不调用generator。
发布训练配置tau为15/20/25原始环境步；当前H2总10步会完全绕过learned subgoal。
强设tau=5属于未认证duration外推，不能叫官方协议；不允许因此暗改现有H2比较。
要真正调用generator需预声明`rem > tau`且使用受支持duration的新长程协议，由MAIN决定。

**权限阻断**：官方`_normalized_lowdim`按checkpoint的lowdim_keys读取state/proprio，默认二者；
普通generator不能自动认定RGB-only。缺数据时源码填零也不等于经训练的visual-only版本。

**坐标阻断**：SAGE训练action mean为[-0.007726192940026522,0.006876377854496241]，
std为[0.2083914875984192,0.206622913479805]，不同于本地冻结公共映射[-.0087,.0068]/[.2019,.2002]。
共同桥应显式以raw relative commands为边界，保留两侧转换。不能擅自替换现有模型normalizer。
其object checkpoint与本地weights.pt序列化SHA/大小不同，**这既不证明tensor不同，也不证明兼容**；
仍需同encoder/projector/config/预处理的实际认证。图像预处理和bf16/float32也不得默认为相同。

## 来源与离线验证

report.json含完整六组件和其它发布资产声明、Zenodo清单、实际阻断与MAIN决策点；
http_receipts.json含27次请求URL/status/bytes/SHA/错误；15份源码仅作为文本保存。
批1/2响应文件末尾可能新增LF，收据原body SHA按received_bytes截取验证；其它批保存精确解码Git blob文本。
初次`trees/<commit>`响应在工具/输出捕获环节被截短：HTTP收据113716 bytes，保存捕获仅56710 bytes含LF，
原body SHA不匹配。这个文件只保留为失败现场，不能认证为官方完整响应，不能据其遗漏或sha推断API异常/源码缺失。
离线验证显式识别并锁定这份partial capture，而不是将其当成功响应。
随后按git commit指向的真实tree请求，确认关键源码blob membership；该请求收据与独立源码blob验证保留。

在inner仓库执行，无网络、tmp、模型或模拟器：

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B results/p2_learned_subgoal_asset_metadata_20261027_v1/verify_saved.py
```

验证只做字节/hash/AST/JSON及短时域分支算术，不运行所下载的模型接口测试。
本交付解除“文件名/预期大小/hash和真实接口完全未知”，不解除HF可用性、低层配套、
RGB权限及短时域协议缺口；不构成模型等价、真实效能或Experiment1科学准入。
