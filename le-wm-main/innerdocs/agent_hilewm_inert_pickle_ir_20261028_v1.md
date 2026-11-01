# Hi-LeWM inert pickle IR v1

本独占source/tests/doc实现的是有界数据IR与字面量模块/参数/buffer元数据，不是反序列化器。worker只运行手写RAM pickle bytes及封存JSON/text读取；没有生产扫描、torch/foreign import、pickle.load/loads/Unpickler、REDUCE/NEWOBJ/BUILD执行或storage解码。使用experiment-code的固定输入/复现记录要求，不使用训练、挑结果或放宽门槛规则。

## 唯一生产输入与前置

固定decoded文件与SHA沿已冻结scanner：`results/agent_hilewm_bounded_deflate_decode_20261028_v1_MAIN01/decoded.unverified.ckpt`，122424517 bytes，SHA `b87805747d40037841877ce7b99b7dda3ebe7a52202c0ba46bf0006ab5d6f008`。固定static report `/home/gbk/code/LeWorldModel/le-wm-main/results/agent_hilewm_pickle_static_inventory_20261028_v1_MAIN01.json` SHA `7c2658ea69caae3117a376d9804357b851d44b4e7babbed9fa3f91fe0fdde5c2`。唯一data.pkl为`hi_lewm_p2_train_hope2_22253175_epoch_15_object/data.pkl`，170471 bytes，SHA `df11e246e2b0e8cd7ab654a4bde0f3e1f090fcf567168f5dd44e73afabd8d480`。

复用冻结scanner SHA `b4fa5b3304a7939ee573f57c16108f839a92fe73bac70d58fc91c33fa2daea67` 的ZIP preflight与decoder只读fd/identity/hash工具，保留原license/config/source/MAIN decode closure。不运行旧scanner的run或decoder的run。生产build由MAIN审后运行；全文件hash只读storage字节，不解释、不解码任何storage；只向ZipFile打开唯一data.pkl（严格长度/SHA/CRC），不提取它到磁盘。

49声明逐位置/opcode/module/name与MAIN完整报告精确相等才可落盘；`original_declarations`保留全部49，不因没有模块路径/不是Tensor而删。它们既不是class安全名单，也不是自动loader allowlist。特别是报告实际类名含`hi_jepa.HiJEPA`、`hi_module.LatentActionEncoder`及`_baseline_lewm_module.*`，不可据曾保存源码路径推断pickle module可可信导入。没有加载许可。

## IR语义：记录语法，不构造Python模型

所有值都是我们定义的JSON节点，引用为整数ID；GLOBAL只保存文字，REDUCE/NEWOBJ/NEWOBJ_EX保存callable/args/kwargs节点、`executed=false`，BUILD只附上state节点，不调用任何构造器、`__setstate__`/`__new__`/persistent_load。memo_table与所有容器edge/node ID保留同一对象/共享引用/自引用。tuple/list/dict/set/字面量、memo、容器mutation只操作这些IR列表，不对文件声明类执行Python方法。

未知opcode/不确定STACK_GLOBAL/EXT/out-of-band buffer/INST/OBJ等当前未实现路径直接incomplete；不猜。未知REDUCE保留opaque调用和全声明，但使整体参数完整性false。固定识别的REDUCE名称只包括`collections.OrderedDict`（严格空参数，用IR entries记录随后SETITEMS）、`builtins/__builtin__.set`（仍opaque record）、`torch._utils._rebuild_tensor_v2`和`_rebuild_parameter`；这不是对这些函数的调用授权，绝不import torch。其他callable出现时无动态添加、无getattr/dispatch到foreign代码。

普通string/int/bool/None/float/bytes保真；float用hex含nan/inf文字而不是JSON非有限值，旧STRING/BINSTRING保留encoding-sensitive语法而非猜解码为unicode。dict重复键原样保留为entries记录，nn遍历遇到重复key拒绝猜覆盖语义。IR表示的是pickle提供的state，不代表执行自定义`__setstate__`后真实模型的`__dict__`。

## nn路径、Tensor和alias元数据

只从根NEWOBJ/NEWOBJ_EX的**一个BUILD state**遍历，支持字面量dict或二元(dict/None,slotdict/None)state，不执行slot赋值。每节点必须显式有`_modules/_parameters/_buffers`三个literal mappings；缺失unknown不填空字典。遍历实际字面量模块名，路径用component数组防`.`名称碰撞，不人为把某条路径认证为low/high。共享module不同路径全部保留；祖先环/深度/计数cap拒绝并记录未闭合原因。显式None逐项保存None，不填0、不作为Tensor alias。输出包括所有已见module class声明、parameter/buffer路径、node alias、module alias与storage-key alias；不筛根、不筛参数。

唯一Tensor布局：`_rebuild_tensor_v2(storage, offset, size_tuple, stride_tuple, requires_grad, hooks)`六参数；Parameter仅`_rebuild_parameter(tensor, requires_grad, hooks)`三参数。五元PID必须是字面量`('storage', GLOBAL storage_type, key, location, numel)`，仅固定FloatStorage/LongStorage记录float32/4字节、int64/8字节声明，不实例化storage。保留storage-ref node、key、location（包括CUDA文字但不访问CUDA）、numel、shape、stride、offset、requires-grad、hooks引用。其他布局/类型/动态参数为unknown，不假装默认兼容。

核对非负shape/stride/offset（<=64维）及view范围；允许合法空/scalar/noncontiguous，绝不改contiguous。storage-key必须映射静态报告的同archive-prefix`data/key`成员，声明字节=declared numel×itemsize；同key冲突声明拒绝。此处使用的是**已封存ZIP目录大小**而非已解码storage，`storage_crc_verified=false/tensor_values_read=false`，不证明字节序、tensor值、storage内容正确、low/high tensor pairing或模型结构运行等价。

完整性表示支持的IR/state语法与路径都闭合，不是安全性或有效权重证书。任一未知路径/REDUCE/布局保持`inert_metadata_incomplete_not_loadable`及非0；已得完整IR/路径和错误仍保存。`nn.complete=true`也只可称“pickle声明的nn state元数据遍历闭合”。状态永远`trusted_loading_allowed=false/tensor_pairing_verified=false/state_dict_extracted=false/baseline_runnable=null/scientific_admission=false`。

## 工程边界与命令

默认工程cap：raw16MiB、100000节点、200000 op/memo、500000 edges、单literal64KiB、20000路径、深度128、报告48MiB。超限拒绝，不静默裁剪为完整。默认120s，可设(0,600]，hash/每op/路径合作检查，非阻塞I/O硬时限。只排他写新本stem JSON报告、O_EXCL/NOFOLLOW/fsync，固定input/caps不可CLI改。计入原asset_resume全树、全部decoder目录、scanner报告和本IR报告的3GiB预算，保留>=10GiB空闲；不复制storage或模型、不删除旧partial/chunk。

```bash
# cwd /home/gbk/code/LeWorldModel/le-wm-main
.venv/bin/python -B -m pytest -q -p no:cacheprovider tests/test_agent_hilewm_inert_pickle_ir_20261028_v1.py
.venv/bin/python -B -m experiments.spectral_reachability.agent_hilewm_inert_pickle_ir_20261028_v1 plan
# 仅MAIN审计后执行；worker未执行build生产文件
.venv/bin/python -B -m experiments.spectral_reachability.agent_hilewm_inert_pickle_ir_20261028_v1 build --output /home/gbk/code/LeWorldModel/le-wm-main/results/agent_hilewm_inert_pickle_ir_20261028_v1_MAIN01.json --wall-seconds 120
```

没有tensor提取或加载命令；未来state_dict导出仍需另行审核storage读取、字节序、typed-storage/alias与精度保持、逐Tensor值hash/low配对/冻结normalizer及运行ABI。MAIN先验IR实物完整性，再决定下一 bounded step；测试不是实物验证，更不是P1/E1准入。三文件冻结后修订新版本，不改MAIN或旧sealed文件。

定向回归：首次46pass/2fail仅为static report 6,392,458 bytes超过旧decoder.small的2MiB上限；未运行生产build，不是科学负结果。新source独立static_metadata读取最多6,392,458+1 bytes，保留固定SHA、JSON重复键/非有限值拒绝、NOFOLLOW与fd/path前后身份检查，不改任何冻结helper或通用metadata cap。RAM回归覆盖大于2MiB报告、越界、错SHA、重复键、symlink，以及冷进程禁止pickle执行/foreign import/生产checkpoint读取。worker只运行RAM与saved JSON绑定测试，生产IR完整性仍unknown，由MAIN执行上述build。
