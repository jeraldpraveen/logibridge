"""Task F2/F3 - Pareto frontier of the three variants.

Objectives (all minimised):
  size_KiB                 model file size (Flash footprint)
  mean_latency_ms          laptop, median of 5 rotated trials (benchmark_results.json)
  critical_misses_stream   Critical windows NOT predicted Critical in the fresh transition
                           stream (window end-time truth, transitions included)
Accuracy on validation is 100 % for every variant, so it cannot discriminate and is not an
objective; the fresh-stream Critical misses are the safety metric that does differ.

A variant is Pareto-optimal if no other variant is <= on every objective and < on at least
one. Writes results/pareto.json and results/pareto_chart.png.
Plot: x = size, y = latency (the two hardware costs); the third objective is printed on each
point. One y-axis only.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "optimisation" / "results"
STEM = {"m1_fp32.tflite": "m1_fp32", "m2_ptq_int8.tflite": "m2_ptq_int8",
        "m3_pruned_ptq_int8.tflite": "m3_pruned_ptq_int8"}
SHORT = {"m1_fp32.tflite": "M1 FP32", "m2_ptq_int8.tflite": "M2 PTQ INT8",
         "m3_pruned_ptq_int8.tflite": "M3 Pruned + INT8"}
# categorical slots 1-3 (validated all-pairs, light mode); aqua < 3:1 -> labels on every point
COLORS = {"m1_fp32.tflite": "#2a78d6", "m2_ptq_int8.tflite": "#eb6834",
          "m3_pruned_ptq_int8.tflite": "#1baf7a"}
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e6e5e1", "#fcfcfb"


def dominates(a, b, keys):
    return all(a[k] <= b[k] for k in keys) and any(a[k] < b[k] for k in keys)


if __name__ == "__main__":
    bench = json.load(open(RES / "benchmark_results.json"))
    trans = json.load(open(RES / "transitions_v2_fresh_analysis.json"))
    rows = []
    for v in bench["variants"]:
        t = trans["totals"][STEM[v["file"]]]
        if trans["meta"]["models"][STEM[v["file"]]]["sha256"] != v["sha256"]:
            raise SystemExit(f"{v['file']}: benchmark and transition evaluation used different models")
        rows.append({"file": v["file"], "label": SHORT[v["file"]], "sha256": v["sha256"],
                     "size_KiB": v["size_KiB"], "mean_latency_ms": v["mean_latency_ms"],
                     "latency_trial_range_ms": [min(v["mean_latency_trials_ms"]),
                                                max(v["mean_latency_trials_ms"])],
                     "critical_misses_stream": t["n_critical_all"] - t["critical_correct_all"],
                     "n_critical_stream": t["n_critical_all"],
                     "critical_recall_stream_pct": round(100 * t["critical_correct_all"] / t["n_critical_all"], 2),
                     "critical_recall_steady_pct": round(100 * t["critical_correct_steady"] / t["n_critical_steady"], 2),
                     "under_classified": t["under_classified_all"],
                     "val_accuracy_pct": v["val_accuracy_pct"]})
    keys = ["size_KiB", "mean_latency_ms", "critical_misses_stream"]
    for r in rows:
        r["dominated_by"] = [o["label"] for o in rows if o is not r and dominates(o, r, keys)]
        r["pareto_optimal"] = not r["dominated_by"]
    json.dump({"objectives_minimised": keys, "variants": rows}, open(RES / "pareto.json", "w"), indent=2)

    fig, ax = plt.subplots(figsize=(8.6, 5.4), dpi=170)
    fig.patch.set_facecolor(SURF)
    ax.set_facecolor(SURF)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
        ax.spines[s].set_linewidth(1)
    ax.grid(True, color=GRID, linewidth=1, linestyle="-")
    ax.set_axisbelow(True)
    ax.tick_params(colors=INK2, labelsize=9, length=0)

    us = lambda ms: ms * 1000.0
    # frontier line in the size-latency plane, through the Pareto-optimal points
    front = sorted([r for r in rows if r["pareto_optimal"]], key=lambda r: r["size_KiB"])
    if len(front) > 1:
        ax.plot([r["size_KiB"] for r in front], [us(r["mean_latency_ms"]) for r in front],
                color=INK2, linewidth=1.5, zorder=1)
    for r in rows:
        x, y = r["size_KiB"], us(r["mean_latency_ms"])
        lo, hi = (us(v) for v in r["latency_trial_range_ms"])
        ax.plot([x, x], [lo, hi], color=COLORS[r["file"]], linewidth=2, alpha=0.45,
                solid_capstyle="round", zorder=2)                  # trial range
        ax.scatter([x], [y], s=110, color=COLORS[r["file"]], edgecolors=SURF, linewidths=2, zorder=3)
    # direct labels (selective: name + the safety number that differs)
    # placed in the empty quadrants, away from the frontier line (dx, dy, ha, va)
    offs = {"m1_fp32.tflite": (-14, -14, "right", "top"),
            "m2_ptq_int8.tflite": (-14, -14, "right", "top"),
            "m3_pruned_ptq_int8.tflite": (0, 16, "center", "bottom")}
    for r in rows:
        dx, dy, ha, va = offs[r["file"]]
        status = "Pareto-optimal" if r["pareto_optimal"] else "dominated by " + ", ".join(r["dominated_by"])
        ax.annotate(f"{r['label']}\n{r['n_critical_stream'] - r['critical_misses_stream']}/"
                    f"{r['n_critical_stream']} Critical windows (fresh stream)\n{status}",
                    (r["size_KiB"], us(r["mean_latency_ms"])), xytext=(dx, dy),
                    textcoords="offset points", ha=ha, va=va, fontsize=8.6, color=INK,
                    linespacing=1.35)
    from matplotlib.lines import Line2D
    ax.legend(handles=[Line2D([0], [0], marker="o", linestyle="", markersize=8,
                              markerfacecolor=COLORS[r["file"]], markeredgecolor=SURF,
                              markeredgewidth=1.5, label=r["label"]) for r in rows]
              + [Line2D([0], [0], color=INK2, linewidth=1.5, label="Frontier (visual guide)")],
              loc="upper right", frameon=False, fontsize=8.6, labelcolor=INK)
    ax.set_xlim(3.2, 5.7)
    ax.set_ylim(0, 24)
    ax.set_xlabel("Model file size (KiB)  -  smaller is better", color=INK2, fontsize=9.5)
    ax.set_ylabel("Mean latency per inference (µs, laptop)  -  lower is better", color=INK2, fontsize=9.5)
    fig.suptitle("Pareto frontier: size vs latency, with fresh-stream Critical recall",
                 x=0.06, ha="left", color=INK, fontsize=12.5, fontweight="bold")
    ax.set_title("Vertical bar = range of 5 trial means. All three: 100 % validation accuracy, "
                 "38/38 steady Critical. Laptop CPU, not Raspberry Pi.",
                 loc="left", color=INK2, fontsize=8.6, pad=10)
    fig.text(0.06, 0.012,
             "Unconstrained comparison: none of the three meets the >95 % whole-stream Critical-recall "
             "target (best 43/50 = 86 %).\nThe line joins three measured models as a visual guide; it "
             "says nothing about configurations between them.",
             ha="left", va="bottom", color=INK2, fontsize=8.2, linespacing=1.4)
    fig.tight_layout(rect=(0, 0.075, 1, 1))
    fig.savefig(RES / "pareto_chart.png", facecolor=SURF)
    for r in rows:
        print(f"{r['label']:18s} {r['size_KiB']:.2f} KiB  {us(r['mean_latency_ms']):.1f} us "
              f"(trials {us(r['latency_trial_range_ms'][0]):.1f}-{us(r['latency_trial_range_ms'][1]):.1f})  "
              f"Critical stream {r['n_critical_stream'] - r['critical_misses_stream']}/{r['n_critical_stream']}  "
              f"-> {'PARETO-OPTIMAL' if r['pareto_optimal'] else 'dominated by ' + ', '.join(r['dominated_by'])}")
