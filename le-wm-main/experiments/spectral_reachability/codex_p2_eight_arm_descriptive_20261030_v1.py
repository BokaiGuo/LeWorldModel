"""Read-only reconstruction of a completed known-root P2 physical comparison."""
import argparse
import json
import math
from pathlib import Path

from experiments.spectral_reachability import codex_p2_joined_native_physical_20261030_v1 as native

REPORT_SHA = "390cc60677afed00e61d14585921fab5a1b00a3071b212301d01d604e3d21852"
PHYSICAL = native.ROOT / "results/codex_p2_joined_native_physical_20261030_v1_RUN01"
A, B = native.native.ARMS


def rank(a, b):
    return "平" if a == b else ("组合" if b < a else "非组合")


def analyze():
    checked = native.consume(PHYSICAL, REPORT_SHA)
    native.need(checked["status"] == "eight_branches_recorded_not_admitted" and not checked["failures"],
                "complete physical receipt required")
    report = json.loads((PHYSICAL / "report.json").read_text())
    binding = json.loads((PHYSICAL / "binding.json").read_text())
    plans = {(p["root_index"], p["method"]): p for p in binding["plans"]}
    rows = {(r["root_index"], r["method"]): r for r in report["rows"]}
    native.need(len(rows) == len(plans) == 8, "eight unique arms")
    result = []
    for index in range(4):
        planner = native.joiner.host.frontier.read(native.RUN / f"root_{index}" / "report.json")
        arms = {r["arm"]: r for r in planner["results"]}
        pair = []
        for arm in (A, B):
            p, r, saved = plans[index, arm], rows[index, arm], arms[arm]
            winner = saved["winner"]
            witness = winner["witness"]
            actions = witness["receipt"]["actions"]
            native.need(p["commands"] == actions and r["selection_sha256"] == p["selection_sha256"] and
                        r["status"] == "completed" and r["completed_commands"] == 10 and
                        r["endpoint_fresh"] is True and r["full_horizon_completed"] is True,
                        "plan/witness/physical identity")
            events = [json.loads(x) for x in Path(r["journal_path"]).read_text().splitlines()]
            attempts = [e["command"] for e in events if e["kind"] == "attempt"]
            native.need(attempts == actions and len(attempts) == 10, "actual dispatch matches witness")
            metrics = [winner["cost"], r["pose"]["xy_distance"], r["pose"]["angle_distance"],
                       r["official"]["official_position_error"], r["official"]["official_state_distance"]]
            native.need(all(math.isfinite(x) for x in metrics), "finite metrics")
            pair.append(dict(candidate=winner["candidate_id"], predicted=winner["cost"],
                             object_position=r["pose"]["xy_distance"],
                             angle=r["pose"]["angle_distance"],
                             native_position=r["official"]["official_position_error"],
                             native_distance=r["official"]["official_state_distance"],
                             success=r["official"]["official_success"],
                             evaluations=saved["selection_evaluations"],
                             edge_sizes=[len(edge["actions"]) for edge in witness["edges"]],
                             actions=actions))
        native.need(rows[index, A]["context_identity"] == rows[index, B]["context_identity"],
                    "matched physical root and warmup")
        native.need(pair[0]["edge_sizes"] == [10] and pair[1]["edge_sizes"] == [5, 5],
                    "witness topology")
        result.append(dict(root=index, root_sha256=plans[index, A]["root_sha256"],
                           arms=pair, shared_prefix=pair[0]["actions"][:5] == pair[1]["actions"][:5]))
    return result


def render(rows):
    lines = [
        "# P2 八臂 known-root 配对描述分析",
        "",
        "仅为已知四根上的回顾性探索；非独立确认、非等总预算比较，不改变 P1/P2 科学门。",
        "各臂真实物理执行完成 10/10 步；同根两臂复现的 root/goal/warmup context 相同。",
        "本报告重新消费物理 journal 和 RGB，并将每一步实际 dispatch 与 planner winner witness 原样比对。",
        "",
        "N=非组合 joint-H2，C=条件后缀组合；表内依次为 N / C。预测值为原 planner 的 source-Q cost，越小越好；"
        "物体位置为原生状态坐标的物体 xy 欧氏距离，角度为绝对环绕弧度误差；成功为环境原生判定。",
        "预测 cost 与物理误差是不同量纲，不能将数值直接相减。",
        "",
        "| 根 | 预测 cost N / C | 物体位置 N / C | 角度(rad) N / C | 原生成功 N / C | 预测 / 位置 / 角度较优 |",
        "|---:|---:|---:|---:|:---:|:---|",
    ]
    for row in rows:
        a, b = row["arms"]
        lines.append(f"| {row['root']} | {a['predicted']:.6f} / {b['predicted']:.6f} | "
                     f"{a['object_position']:.3f} / {b['object_position']:.3f} | "
                     f"{a['angle']:.4f} / {b['angle']:.4f} | "
                     f"{'是' if a['success'] else '否'} / {'是' if b['success'] else '否'} | "
                     f"{rank(a['predicted'], b['predicted'])} / {rank(a['object_position'], b['object_position'])} / {rank(a['angle'], b['angle'])} |")
    lines += [
        "",
        "物体位置排序与预测排序相反仅见 root 2；角度排序四根均与预测排序一致。"
        "这只是逐根排序描述，不是模型预测误差的数值分解或因果结论。八臂原生成功均为否，成功率没有可排序差异。",
        "",
        "原生整体位置误差（包含 agent 与物体，不等同于上表物体位置）和原生状态距离如下：",
        "",
        "| 根 | 原生位置 N / C | 原生状态距离 N / C | 预测 / 原生位置 / 原生状态距离较优 |",
        "|---:|---:|---:|:---|",
    ]
    for row in rows:
        a, b = row["arms"]
        lines.append(f"| {row['root']} | {a['native_position']:.3f} / {b['native_position']:.3f} | "
                     f"{a['native_distance']:.3f} / {b['native_distance']:.3f} | "
                     f"{rank(a['predicted'], b['predicted'])} / {rank(a['native_position'], b['native_position'])} / "
                     f"{rank(a['native_distance'], b['native_distance'])} |")
    lines += ["", "## 动作来源", "",
              "两臂分别从同一个已保存 CUDA planner 的模型预测候选中选出最小 cost；"
              "非组合为一条 10 步 witness，组合为连续 5+5 步 witness。"
              "每个物理 journal 的 10 个 dispatch 命令与对应 winner receipt 逐项相同，无物理结果选动作。"
              "共同前缀只描述已选动作是否相等，不意味着对所有根进行了仅后缀干预。",
              "", "| 根 | 非组合候选 | 组合候选 | 前 5 步逐项相同 |", "|---:|:---|:---|:---:|"]
    for row in rows:
        a, b = row["arms"]
        lines.append(f"| {row['root']} | {a['candidate']} | {b['candidate']} | {'是' if row['shared_prefix'] else '否'} |")
    lines += [
        "",
        "planner 两臂候选计数分别为 80 和 40；相同物理步数不意味着相同规划计算量。"
        "神经来源 checkpoint SHA-256：`48938400ae3464c9680731287f583a9cb516f55a8ec64ea13a91be47fb15b607`。"
        "模型加载及历史资源全成本、模拟器内部子步未证明相等。",
        "",
        f"物理报告 SHA-256：`{REPORT_SHA}`。"
        "原规划报告各根 SHA 位于原 run_report.json；原始动作、帧及完整验收记录保持原位。",
        "无新模型调用或物理运行；无统计显著性、独立性、等预算或泛化声明。",
        "",
    ]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    native.need(args.output.is_absolute() and args.output.name.startswith("D2P_P2_EIGHT_ARM_DESCRIPTIVE_"),
                "new absolute tagged report")
    text = render(analyze())
    with args.output.open("x", encoding="utf-8") as stream:
        stream.write(text)
    print(json.dumps({"output": str(args.output), "sha256": native.joiner.host.frontier.file_sha(args.output)}))


if __name__ == "__main__":
    main()
