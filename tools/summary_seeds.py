import json
from pathlib import Path
from statistics import mean, pstdev
from typing import Dict, List

import numpy as np


DEFAULT_GROUPS: Dict[str, List[str]] = {
    "E0_match": ["runs/E0_match"],
    "E5_balanced": ["runs/E5_softmax_balanced_30ep"],
}

METRIC_KEYS = [
    "accuracy",
    "macro_f1",
    "f1_disgust",
    "f1_fear",
]


def load_metric(path: Path) -> Dict[str, float]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def agg(values: List[float]) -> (float, float):
    if len(values) == 1:
        return values[0], 0.0
    return mean(values), pstdev(values)


def summarize(groups: Dict[str, List[str]]) -> Dict[str, Dict[str, float]]:
    summary = {}
    for name, run_dirs in groups.items():
        metrics_list = []
        for rd in run_dirs:
            mpath = Path(rd) / "metrics.json"
            if mpath.exists():
                metrics_list.append(load_metric(mpath))
        if not metrics_list:
            continue
        summary[name] = {}
        for key in METRIC_KEYS:
            vals = [m.get(key, 0.0) for m in metrics_list]
            m, s = agg(vals)
            summary[name][f"{key}_mean"] = m
            summary[name][f"{key}_std"] = s
    return summary


def write_csv(summary: Dict[str, Dict[str, float]], out_path: Path) -> None:
    headers = ["run"]
    for key in METRIC_KEYS:
        headers.extend([f"{key}_mean", f"{key}_std"])
    lines = [",".join(headers)]
    for run, stats in summary.items():
        row = [run]
        for key in METRIC_KEYS:
            row.append(f"{stats.get(f'{key}_mean', 0):.4f}")
            row.append(f"{stats.get(f'{key}_std', 0):.4f}")
        lines.append(",".join(row))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")


def write_latex(summary: Dict[str, Dict[str, float]], out_path: Path) -> None:
    lines = [
        "\\begin{tabular}{lcccc}",
        "\\hline",
        "Run & Acc (mean$\\pm$std) & Macro-F1 & F1 (disgust) & F1 (fear) \\\\",
        "\\hline",
    ]
    for run, stats in summary.items():
        acc = f"{stats['accuracy_mean']:.3f}$\\pm${stats['accuracy_std']:.3f}"
        mf1 = f"{stats['macro_f1_mean']:.3f}$\\pm${stats['macro_f1_std']:.3f}"
        dis = f"{stats['f1_disgust_mean']:.3f}$\\pm${stats['f1_disgust_std']:.3f}"
        fear = f"{stats['f1_fear_mean']:.3f}$\\pm${stats['f1_fear_std']:.3f}"
        lines.append(f"{run} & {acc} & {mf1} & {dis} & {fear} \\\\")
    lines.extend(["\\hline", "\\end{tabular}"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    summary = summarize(DEFAULT_GROUPS)
    runs_dir = Path("runs")
    write_csv(summary, runs_dir / "seed_summary.csv")
    write_latex(summary, runs_dir / "seed_summary_table.tex")
    print("Wrote summary to runs/seed_summary.csv and runs/seed_summary_table.tex")


if __name__ == "__main__":
    main()
