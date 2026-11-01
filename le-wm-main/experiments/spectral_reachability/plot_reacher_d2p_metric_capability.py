"""Complete8-root capability panels from immutable measured analysis arrays.

Reuse the line-template's Agg/PNG+PDF pipeline and the existing complete-panel
renderer. No root selection, smoothing, fitted trend, or efficacy reinterpretation.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from experiments.spectral_reachability import analyze_reacher_d2p_prefix_capability as base


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis",type=Path,required=True)
    parser.add_argument("--output-dir",type=Path,required=True)
    args = parser.parse_args()
    if args.output_dir.exists(): raise FileExistsError("new immutable full-root figure directory required")
    data = json.loads(args.analysis.read_text())
    if (data["schema"] != "reacher_d2p_metric_capability_analysis.v1" or data["P1_admission"] is not False
            or len(data["rows"])!=8 or len({r["seed"] for r in data["rows"]})!=8):
        raise ValueError("complete measured8-root fixed metric analysis required")
    for p,sha in data["input_bindings"].items():
        if base.digest(Path(p)) != sha: raise ValueError("bound physical/model input changed")
    for k,sha in data["implementation_sha256"].items():
        if base.digest(args.analysis.with_suffix(".implementation")/(k+".py")) != sha:
            raise ValueError("independent analyzer archive changed")
    args.output_dir.mkdir(parents=True)
    base.plot(data["original_unweighted_analysis"],args.output_dir)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    styles = (("euclidean_candidate","Euclidean selector","#0072B2","o","-"),
        ("metric_candidate","Fixed metric selector","#E69F00","s","--"),
        ("greedy","Greedy","#CC79A7","x",":"),("zero","Zero prefix","#56B4E9","^","-."))
    seeds = [r["seed"] for r in data["rows"]]
    fig,axes = plt.subplots(3,2,figsize=(11,8),layout="constrained")
    for row,task in enumerate(base.TASKS):
        for col,scale in enumerate(base.SCALES):
            ax = axes[row,col]
            for role,title,color,marker,style in styles:
                values = []
                for root in data["rows"]:
                    t = root["tasks"][task]
                    values.append(t["cells"][str(t["roles"][role])]["probes"][col]["physical_rgb_fd"]["gain_fraction"])
                ax.plot(range(8),values,label=title,color=color,marker=marker,linestyle=style,lw=1,markersize=4)
            ax.set(xticks=range(8),xticklabels=[str(s)[-6:] for s in seeds],ylim=(-.03,1.03),
                ylabel=task.replace("_"," ")+"\nFixed-Q RGB bounded gain fraction",
                xlabel="All eight sealed roots; raw scale "+str(scale))
            ax.grid(axis="y",alpha=.2)
    fig.legend(*axes[0,0].get_legend_handles_labels(),loc="outside upper center",ncol=4,frameon=False)
    for ext in ("png","pdf"): fig.savefig(args.output_dir/("Figure_3."+ext),dpi=300,bbox_inches="tight")
    plt.close(fig)
    caption = args.output_dir/"captions.tex"
    caption.write_text(caption.read_text()+r"""
\begin{figure}[t]
\centering\includegraphics[width=\linewidth]{Figure_3.pdf}
\caption{The same complete eight-root physical RGB response probes under the fixed model-root metric, at both predeclared raw-action scales. The metric is constructed solely from the frozen model's predicted zero-prefix suffix response, not physical labels. Each root's common residual remains anchored to its observed zero-prefix zero-suffix outcome. All three actuator tasks and all four model-sealed roles are retained; coincident roles overplot. This uses the same physical probes as Figures 1--2, not another independent experiment. The transformed RGB gain is distinct from physical joint-task gain, successful reconfiguration, and formal P1 admission.}
\end{figure}
""")
    (args.output_dir/"notes.txt").write_text("Complete8 sealed roots,3 tasks,4 model-only roles; no root/task selection.\n"
        "Figures1/2: original RGB and joint response gains, scales0.1/0.25.\n"
        "Figure3: same physical RGB probes with fixed model-root Q, both scales.\n"
        "No global controllability/P1/independent efficacy claim. Coincident roles overplot.\n")
    for name,path in (("generator.py",Path(__file__)),("base_generator.py",Path(base.__file__))):
        shutil.copyfile(path,args.output_dir/name)
    manifest = {"analysis_path":str(args.analysis.resolve()),"analysis_sha256":base.digest(args.analysis),
        "artifacts":{p.name:base.digest(p) for p in args.output_dir.iterdir()}}
    with (args.output_dir/"manifest.json").open("x") as f: json.dump(manifest,f,indent=2,allow_nan=False)
    print(json.dumps({"output_dir":str(args.output_dir),"roots":seeds,"figures":3,"P1_admission":False}))


if __name__ == "__main__":
    main()
