# E3 项目源码闭包

只读静态闭包覆盖项目根目录的 `.py`/`.so` 文件，以及 `experiments`、`scripts`、`tests`、`external` 四棵树的 `.py`/`.so`，跳过 `__pycache__`、`.git` 和常见测试/类型缓存；拒绝范围内源码或目录符号链接。覆盖这些根中的 Python namespace package 和 `__init__.py`，包括 MAIN04 未封存的 `experiments/research_runtime/__init__.py`。这不是整个项目树所有可能可加载文件：`results`、数据、虚拟环境以外其他目录、`.pyc`、动态生成内容、内存代码、标准库及系统外部动态库均不在项目闭包中；如原 census 观察到其中未绑定的项目内文件，仍按原规则拒绝。已安装依赖使用既有 FIX01 authority，不重建其完整 closure。

collect 使用 `os.walk` 与独立 `scandir` 比对集合、两次哈希并再次确认集合；prepare/verify 再独立 `scandir`、完整逐文件哈希、`os.walk` 和第二次哈希并确认集合，重建 authority 并调用原 HOST authority。所有既有 prepared 与 additional 两表分别检查每个碰撞，并通过原 merge 检查；不修改 census、source guard 或旧产物。collect/prepare 独占写入。跨两个扫描之间瞬态变更再还原无法排除；过程不是锁或原子快照，不支持并发源码修改。若其他智能体正在写入，必须待稳定后重新全部封存并在运行前重新独立 verify；verify 后变化仍有 TOCTOU 风险。

已测 fixture 14 项（项目闭包、安装闭包及原 census）。已发现真实并发漂移：MAIN01 closure 后新增 P2 测试；MAIN02 closure/MAIN01 authority 后新增 P3 源码导致独立 verify 拒绝；MAIN03 和 MAIN04 closure 分别遇到 P3 测试、源码字节改变，prepare 拒绝。旧产物均保留，不能冒称已通过最终 verify。模型、GPU、物理均未执行；未重跑上游 strict，科学准入不变。

从 `le-wm-main` 执行，以下示例中的新输出名不得已存在；在 collect 返回后把实际 SHA 填入后两条，prepare 返回后把新 authority SHA 填入 verify：

```bash
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -m experiments.spectral_reachability.agent_e3_project_closure_20261029_v1 collect --source /home/gbk/code/LeWorldModel/le-wm-main/results/agent_e3_installed_authority_20261029_v1_FIX01.json --source-sha256 c306e309352d35f3ce086aacb5bee781eca28e909f885aa22c615cf48c2599c7 --output /home/gbk/code/LeWorldModel/le-wm-main/results/agent_e3_project_closure_20261029_v1_NEW.json
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -m experiments.spectral_reachability.agent_e3_project_closure_20261029_v1 prepare --source /home/gbk/code/LeWorldModel/le-wm-main/results/agent_e3_installed_authority_20261029_v1_FIX01.json --source-sha256 c306e309352d35f3ce086aacb5bee781eca28e909f885aa22c615cf48c2599c7 --closure /home/gbk/code/LeWorldModel/le-wm-main/results/agent_e3_project_closure_20261029_v1_NEW.json --closure-sha256 REPLACE_WITH_COLLECT_SHA --output /home/gbk/code/LeWorldModel/le-wm-main/results/agent_e3_project_authority_20261029_v1_NEW.json
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -m experiments.spectral_reachability.agent_e3_project_closure_20261029_v1 verify --source /home/gbk/code/LeWorldModel/le-wm-main/results/agent_e3_installed_authority_20261029_v1_FIX01.json --source-sha256 c306e309352d35f3ce086aacb5bee781eca28e909f885aa22c615cf48c2599c7 --closure /home/gbk/code/LeWorldModel/le-wm-main/results/agent_e3_project_closure_20261029_v1_NEW.json --closure-sha256 REPLACE_WITH_COLLECT_SHA --authority /home/gbk/code/LeWorldModel/le-wm-main/results/agent_e3_project_authority_20261029_v1_NEW.json --authority-sha256 REPLACE_WITH_PREPARE_SHA
```
